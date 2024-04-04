import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from rl_modules.replay_buffer_high import ReplayBuffer
from mpi_utils.normalizer import normalizer
from torch import nn

import torch.nn.functional as F
from torch.distributions import Normal
import copy


"""
ddpg with HER (MPI-version)

"""

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class RMS(object):
    """running mean and std """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


class RND(nn.Module):
    def __init__(self,
                 obs_dim,
                 hidden_dim,
                 rnd_rep_dim,
                 encoder,
                 aug,
                 obs_shape,
                 obs_type,
                 clip_val=5.):
        super().__init__()
        self.clip_val = clip_val
        self.aug = aug

        if obs_type == "pixels":
            self.normalize_obs = nn.BatchNorm2d(obs_dim, affine=False)
        else:
            self.normalize_obs = nn.BatchNorm1d(obs_dim, affine=False)

        self.predictor = nn.Sequential(encoder, nn.Linear(obs_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, rnd_rep_dim))
        self.target = nn.Sequential(copy.deepcopy(encoder),
                                    nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, rnd_rep_dim))

        for param in self.target.parameters():
            param.requires_grad = False

        self.apply(weights_init_)

    def forward(self, obs):
        obs = self.aug(obs)
        obs = self.normalize_obs(obs)
        obs = torch.clamp(obs, -self.clip_val, self.clip_val)
        prediction, target = self.predictor(obs), self.target(obs)
        prediction_error = torch.square(target.detach() - prediction).mean(
            dim=-1, keepdim=True)
        return prediction_error


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim//2)

        self.mean_linear = nn.Linear(hidden_dim//2, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim//2, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        try:
            normal = Normal(mean, std)
        except:
            print(std, mean)

        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class meta_control:
    def __init__(self, args, env, env_params):

        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.args = args
        self.env = env
        self.env_params = env_params
        self.fetch = "Fetch" in self.args.env_name
        self.pusher = "pusher" in self.args.env_name

        # create the network
        self.actor_network = GaussianPolicy(env_params['obs'] + env_params['goal'], env_params['goal'], 256)
        self.critic_network = QNetwork(env_params['obs'] + env_params['goal'], env_params['goal'], 256)
        # build up the target network
        self.critic_target_network = QNetwork(env_params['obs'] + env_params['goal'], env_params['goal'], 256)
        # load the weights into the target networks
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.to(self.device)
            self.critic_network.to(self.device)
            self.critic_target_network.to(self.device)
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_upper_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_upper_critic)
        # create the replay buffer
        self.buffer = ReplayBuffer(
            env_params['obs'] + env_params['goal'],
            env_params['goal'],
            int(args.buffer_size),
            self.device)
        # create the normalizer

        self.target_entropy = -torch.prod(torch.Tensor(env_params['goal']).to(self.device)).item()
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.lr_actor)
        self.alpha = 0.3#self.log_alpha.exp()

        self.gmax = np.zeros(3)
        self.gmin = np.zeros(3)
        self.torch_gmax = None
        self.torch_gmin = None

        self.gmax_distance = np.linalg.norm(self.gmax - self.gmin)

        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

        self.aug = nn.Identity()
        self.encoder = nn.Identity()
        self.rnd = RND(env_params['obs'] + env_params['goal'], 256, 512,
                       self.encoder, self.aug, env_params['obs'] + env_params['goal'],
                       "states").to(self.device)

        # self.d_norm = normalizer(size=1, default_clip_range=self.args.clip_range)

        self.intrinsic_reward_rms = RMS(device=self.device)

        # optimizers
        self.rnd_opt = torch.optim.Adam(self.rnd.parameters(), lr=self.args.lr_rnd_model)
        self.rnd.train()

        self.k = 0.05
        self.target_entropy2 = torch.prod(torch.Tensor([0.00]).to(self.device)).item()
        self.log_alpha2 = torch.tensor([-10.0], requires_grad=True, device=self.device)
        self.alpha2_optim = torch.optim.Adam([self.log_alpha2], lr=self.args.lr_alpha_model)
        self.alpha2 = self.log_alpha2.exp()

        self.alpha1 = 0.0

    def update_rnd(self, obs, step):
        prediction_error = self.rnd(obs)
        loss = prediction_error.mean()
        self.rnd_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.rnd_opt.step()

    def compute_intr_reward(self, obs, step):
        prediction_error = self.rnd(obs)
        _, intr_reward_var = self.intrinsic_reward_rms(prediction_error)
        reward = prediction_error / (
                torch.sqrt(intr_reward_var) + 1e-8)
        return reward

    def gupdate(self, eg):
        if self.gmax.any():
            self.gmax = np.max(np.concatenate([[self.gmax, eg]], axis=1), axis=0)
            self.gmin = np.min(np.concatenate([[self.gmin, eg]], axis=1), axis=0)
            self.torch_gmax = torch.tensor(self.gmax, dtype=torch.float32).to(self.device)
            self.torch_gmin = torch.tensor(self.gmin, dtype=torch.float32).to(self.device)
            self.gmax_distance = np.linalg.norm(self.gmax - self.gmin)
        else:
            self.gmax = eg
            self.gmin = eg

    def unnormalize(self, ig):
        ig = ig * (self.gmax - self.gmin)/2 + (self.gmax + self.gmin)/2
        return ig

    def tensor_unnormalize(self, ig):
        ig = ig.mul((self.torch_gmax - self.torch_gmin)/2) + (self.torch_gmax + self.torch_gmin)/2
        return ig

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, state, evaluate=False):
        # state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor_network.sample(state)
        else:
            _, _, action = self.actor_network.sample(state)
        return action.detach().cpu().numpy()[0]

    def _preproc_og(self, o, g):
        # o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        # g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def compute_distance_reward(self, inputs_norm_tensor, actions_tensor, distance_model):
        if self.fetch:
            ag = inputs_norm_tensor[:, 3:6]
        elif self.pusher:
            ag = inputs_norm_tensor[:, -2 * self.env_params['goal']:-self.env_params['goal']]
        else:
            ag = inputs_norm_tensor[:, -2*self.env_params['goal']:-self.env_params['goal']]

        if distance_model == None:
            distance_reward = torch.norm(ag - actions_tensor, dim=1) - self.k * self.gmax_distance
        else:
            distance_input = torch.cat((ag, actions_tensor), 1)
            distance = distance_model(distance_input)
            distance_reward = distance - self.k

        return distance_reward
        # return 0

    # update the network
    def _update_network(self, distance_model):
        # sample the episodes
        # if self.buffer.__len__() <= self.args.meta_batch_size:
        #     return
        batch_size = min(self.args.meta_batch_size, self.buffer.__len__()+1)
        inputs_norm_tensor, actions_tensor, r_tensor, inputs_next_norm_tensor = self.buffer.sample(batch_size)

        inputs_norm_tensor = inputs_norm_tensor.cuda()
        inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
        actions_tensor = actions_tensor.cuda()
        r_tensor = r_tensor.cuda()

        self.update_rnd(inputs_next_norm_tensor, 0)
        with torch.no_grad():
            intr_reward = self.compute_intr_reward(inputs_next_norm_tensor, 0)

        with torch.no_grad():
            distance_reward = self.compute_distance_reward(inputs_norm_tensor, actions_tensor, distance_model)
            zero_reward = torch.zeros(distance_reward.shape).to(self.device)
            distance_reward = torch.max(distance_reward, zero_reward).view(-1, 1)

        r_tensor = r_tensor + self.alpha1 * intr_reward - self.alpha2 * distance_reward

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor_network.sample(inputs_next_norm_tensor)
            next_state_action = self.tensor_unnormalize(next_state_action)
            qf1_next_target, qf2_next_target = self.critic_target_network(inputs_next_norm_tensor, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = r_tensor + self.args.gamma * min_qf_next_target

        qf1, qf2 = self.critic_network(inputs_norm_tensor,
                               actions_tensor)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.actor_network.sample(inputs_norm_tensor)

        pi = self.tensor_unnormalize(pi)

        qf1_pi, qf2_pi = self.critic_network(inputs_norm_tensor, pi)

        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        with torch.no_grad():
            #distance_reward = self.compute_distance_reward(inputs_norm_tensor, pi)
            distance_reward = self.compute_distance_reward(inputs_norm_tensor, pi, distance_model).view(-1, 1)
            #zero_reward = torch.zeros(distance_reward.shape).to(self.device)
            #distance_reward = torch.max(distance_reward, zero_reward).view(-1, 1)

        # if self.log_alpha2.detach() < -6.9 and distance_reward.detach().mean() < 0:
        #     alpha_loss = 0
        # elif self.log_alpha2.detach() > 3 and distance_reward.detach().mean() > 0:
        #     alpha_loss = 0
        # else:
        #     alpha_loss = -(self.log_alpha2 * (distance_reward - self.target_entropy2).detach()).mean()
        #     self.alpha2_optim.zero_grad()
        #     alpha_loss.backward()
        #     self.alpha2_optim.step()
        #     self.alpha2 = self.log_alpha2.exp()
        #
        # alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        # self.alpha_optim.zero_grad()
        # alpha_loss.backward()
        # self.alpha_optim.step()
        # self.alpha = self.log_alpha.exp()
