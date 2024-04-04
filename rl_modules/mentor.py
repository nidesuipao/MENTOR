import torch
import torch.nn as nn
import copy

import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer_low import replay_buffer
from rl_modules.models import actor, critic
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from rl_modules.reward_model import RewardModel
from rl_modules.high_policy import meta_control
from logger import Logger
import random
from collections import deque
from reward_hot import reward_hotpic_plot
from distance import Distance_model, Distance_buffer
"""
ddpg with HER (MPI-version)

"""


def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def dense_reward(target: np.ndarray, state: np.ndarray) -> float:
    """Generates dense rewards as euclidean error norm of state and target vector

    Args:
        target (np.ndarray): target state vector of dimension (n)
        state (np.ndarray): state vector of dimension (m)

    Returns:
        float: reward
    """

    return -np.linalg.norm(target - state, ord=2)


def dense_reward_insert(state, target):
    target_high = [target[0], target[1], 0.04]

    if state[2] > 0.04:
        return -np.linalg.norm(target_high - state, ord=2) - 0.03

    if np.linalg.norm(target_high[:2] - state[:2], ord=2) < 0.015:
        return -np.linalg.norm(target - state, ord=2)

    return -1


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

        self.apply(weight_init)

    def forward(self, obs):
        obs = self.aug(obs)
        obs = self.normalize_obs(obs)
        obs = torch.clamp(obs, -self.clip_val, self.clip_val)
        prediction, target = self.predictor(obs), self.target(obs)
        prediction_error = torch.square(target.detach() - prediction).mean(
            dim=-1, keepdim=True)
        return prediction_error


class agent:
    def __init__(self, args, env, env_params):
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.args = args
        self.env = env
        self.env_params = env_params
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        self.actor_network_explore = actor(env_params)
        self.critic_network_explore = critic(env_params)
        self.actor_target_network_explore = actor(env_params)
        self.critic_target_network_explore = critic(env_params)
        self.actor_target_network_explore.load_state_dict(self.actor_network_explore.state_dict())
        self.critic_target_network_explore.load_state_dict(self.critic_network_explore.state_dict())

        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()

            self.actor_network_explore.cuda()
            self.critic_network_explore.cuda()
            self.actor_target_network_explore.cuda()
            self.critic_target_network_explore.cuda()

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        self.actor_explore_optim = torch.optim.Adam(self.actor_network_explore.parameters(), lr=self.args.lr_actor)
        self.critic_explore_optim = torch.optim.Adam(self.critic_network_explore.parameters(), lr=self.args.lr_critic)

        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)

        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)

        self.r_norm = normalizer(size=1, default_clip_range=self.args.clip_range)

        self.meta_control = meta_control(args, env, env_params)

        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        self.work_dir = os.getcwd()
        self.logger = Logger(
            self.work_dir + "/exp/" + args.env_name + "/ddpg_final_distance_human_feedback_number/" + str(
                args.max_feedback),
            save_tb=args.log_save_tb,
            log_frequency=args.log_frequency,
            agent="sac",
            seed=str(args.seed),
            reward_batch=str(args.reward_batch),
            num_interact=str(args.num_interact))

        self.reward_model = RewardModel(
            env_params['obs'] + env_params['goal'],
            env_params['goal'],
            ensemble_size=3,
            size_segment=1,
            activation="tanh",
            lr=self.args.lr_reward_model,
            mb_size=args.reward_batch,
            large_batch=10,
            label_margin=0.0,
            teacher_beta=-1,
            teacher_gamma=1,
            teacher_eps_mistake=0,
            teacher_eps_skip=0,
            teacher_eps_equal=0,
            hl=True)

        self.aug = nn.Identity()
        self.encoder = nn.Identity()

        self.rnd = RND(env_params['obs'] + env_params['goal'], 256, 512,
                       self.encoder, self.aug, env_params['obs'] + env_params['goal'],
                       "states").to(self.device)
        self.intrinsic_reward_rms = RMS(device=self.device)

        # optimizers
        self.rnd_opt = torch.optim.Adam(self.rnd.parameters(), lr=self.args.lr_rnd_model)
        self.rnd.train()

        self.alpha = 1
        self.alpha2 = 0

        self.distance_model = Distance_model(env_params['goal'] * 2, 256,
                                             env_params['max_timesteps'] + 1).to(self.device)
        self.distance_optim = torch.optim.Adam(self.distance_model.parameters(), lr=3e-4, weight_decay=0.00001)
        self.distance_buffer = Distance_buffer(args.seed, 10000, env_params['max_timesteps'] + 1)
        self.criterion = torch.nn.MSELoss()

    def distance_learn(self):

        num_epochs = int(len(self.distance_buffer.train_data) / self.args.batch_size)
        total = len(self.distance_buffer.train_data)
        last_index = 0
        ensemble_acc = []
        for epoch in range(num_epochs):
            train_data = self.distance_buffer.train_data[last_index: last_index + self.args.batch_size]
            train_label = self.distance_buffer.train_label[last_index: last_index + self.args.batch_size]
            train_data = torch.tensor(np.array(train_data), dtype=torch.float32).to(self.device)
            train_label = torch.tensor(np.array(train_label), dtype=torch.float32).to(self.device)
            # print(train_data.shape)
            output = self.distance_model(train_data).reshape(-1)
            # print(output[0], train_label[0])
            # print(output.shape, train_label.shape)
            loss = self.criterion(output, train_label)
            # print(loss)

            self.distance_optim.zero_grad()
            loss.backward()
            self.distance_optim.step()
            last_index = last_index + self.args.batch_size

            ensemble_acc.append(loss.item())

    def learn_reward(self, first_flag=0):
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling(self.step)
        else:
            if self.args.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling(self.step)
            elif self.args.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.args.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.args.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.args.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.args.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError

        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.args.reward_update):
                if self.args.label_margin > 0 or self.args.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)

                if total_acc > 0.97:
                    break
        print("Reward function is updated!! ACC: " + str(total_acc))

    def update_rnd(self, obs, step):

        prediction_error = self.rnd(obs)
        loss = prediction_error.mean()
        self.rnd_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.rnd_opt.step()

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        true_episode_reward = 0
        avg_train_true_return = deque([], maxlen=10)
        self.avg_train_success = deque([], maxlen=20)
        interact_count = 0
        self.total_feedback = 0

        self.meta_control.gupdate(np.array([-0.5, -0.5]))
        self.meta_control.gupdate(np.array([0.5, 0.5]))

        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []

                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    eg = observation['desired_goal']
                    ag = observation['achieved_goal']
                    ep_ags = [ag]

                    if self.alpha2 <= 1:
                        self.meta_control.gupdate(eg)
                        self.meta_control.gupdate(ag)
                        inputs = torch.tensor(np.concatenate([obs, eg]), dtype=torch.float32).unsqueeze(0).to(self.device)
                        if self.meta_control.buffer.__len__() > self.meta_control.args.meta_batch_size:
                            ig = self.meta_control._select_actions(inputs)
                        else:
                            ig = np.random.uniform(-1, 1, self.env_params["goal"])

                        g = self.meta_control.unnormalize(ig)
                        meta_obs = np.concatenate([obs, eg])

                        if random.random() > 0.8 and g[0] > 0 and g[1] > 0:
                            g = eg

                        # g_reward = self.env.get_reward(ag, eg)
                        # self.reward_model.add_data(meta_obs, ag, g_reward)

                        self.logger.log('train/g_ag_distance', np.linalg.norm(ag - g), self.step)
                        self.logger.log('train/ag_eg_distance', np.linalg.norm(ag - eg), self.step)
                        self.logger.log('train/g_eg_distance', np.linalg.norm(eg - g), self.step)
                        self.logger.log('train/_g0', g[0], self.step)
                        self.logger.log('train/_g1', g[1], self.step)
                        self.logger.log('train/_ag0', ag[0], self.step)
                        self.logger.log('train/_ag1', ag[1], self.step)
                        self.logger.log('train/_eg0', eg[0], self.step)
                        self.logger.log('train/_eg1', eg[1], self.step)
                        self.logger.dump(self.step, ty='train')

                        if self.total_feedback < self.args.max_feedback and epoch > 2:
                            if interact_count >= self.args.num_interact:
                                # update schedule
                                if self.args.reward_schedule == 1:
                                    frac = (self.args.num_train_steps - self.step) / self.args.num_train_steps
                                    if frac == 0:
                                        frac = 0.01
                                elif self.args.reward_schedule == 2:
                                    frac = self.args.num_train_steps / (self.args.num_train_steps - self.step + 1)
                                else:
                                    frac = 1
                                self.reward_model.change_batch(frac)

                                # update margin --> not necessary / will be updated soon
                                new_margin = np.mean(avg_train_true_return) * (
                                        self.args.segment / self.env_params['max_timesteps'])
                                self.reward_model.set_teacher_thres_skip(new_margin * self.args.teacher_eps_skip)
                                self.reward_model.set_teacher_thres_equal(new_margin * self.args.teacher_eps_equal)

                                # corner case: new total feed > max feed
                                if self.reward_model.mb_size + self.total_feedback > self.args.max_feedback:
                                    self.reward_model.set_batch(self.args.max_feedback - self.total_feedback)
                                self.learn_reward()
                                self.meta_control.buffer.relabel_with_predictor(self.reward_model)
                                interact_count = 0
                                if epoch%100== 101:
                                    self.test_reward_hot(epoch)
                        # else:
                        #     self.meta_control.alpha1 = 0
                    else:
                        g = eg

                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            if random.random() < self.alpha2:
                                pi = self.actor_network(input_tensor)
                            else:
                                pi = self.actor_network_explore(input_tensor)
                            action = self._select_actions(pi)
                        self.step += 1
                        # feed the actions into the environment
                        observation_new, reward, done, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']

                        if self.env.compute_reward(g, ag, 1.0) > -1 and self.alpha2 <= 1:
                            next_meta_obs = np.concatenate([obs, eg])
                            g_reward_hat = self.reward_model.r_hat(np.concatenate([meta_obs, g]))
                            self.r_norm.update(g_reward_hat)
                            self.r_norm.recompute_stats()
                            g_reward_hat = self.r_norm.normalize(g_reward_hat)
                            avg_train_true_return.append(g_reward_hat)
                            meta_ereward = self.env.compute_reward(ag, g, 1.0)
                            self.meta_control.buffer.add(meta_obs, g, g_reward_hat + meta_ereward, next_meta_obs)

                            g_reward = self.env.get_reward(g, eg)
                            self.reward_model.add_data(meta_obs, g, g_reward)

                            # if random.random() < 0.2:
                            #     rg = np.random.uniform(-0.55, 0.55, self.env_params["goal"])
                            #     g_reward = self.env.get_reward(rg, eg)
                            #     self.reward_model.add_data(meta_obs, rg, g_reward)

                            inputs = torch.tensor(np.concatenate([obs, eg]), dtype=torch.float32).unsqueeze(0).to(
                                self.device)
                            g = self.meta_control._select_actions(inputs)
                            g = self.meta_control.unnormalize(g)
                            meta_obs = np.concatenate([obs, eg])

                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        if np.linalg.norm(ag_new-ag) > 0.01:
                            ep_ags.append(ag_new)
                        obs = obs_new
                        ag = ag_new

                    if self.alpha2 <= 1:

                        self.distance_buffer.push(ep_ags, self.env, g)
                        for i in range(30):
                            self.distance_learn()

                        g_reward = self.env.get_reward(g, eg)
                        self.reward_model.add_data(meta_obs, g, g_reward)

                        if random.random() < 0:
                            rg = np.random.uniform(-0.55, 0.55, self.env_params["goal"])
                            g_reward = self.env.get_reward(rg, eg)
                            self.reward_model.add_data(meta_obs, rg, g_reward)

                        interact_count += 1

                        self.avg_train_success.append(self.env.compute_reward(g, ag, 1) + 1)
                        next_meta_obs = np.concatenate([obs, eg])
                        g_reward_hat = self.reward_model.r_hat(np.concatenate([meta_obs, g]))
                        self.r_norm.update(g_reward_hat)
                        self.r_norm.recompute_stats()
                        g_reward_hat = self.r_norm.normalize(g_reward_hat)
                        avg_train_true_return.append(g_reward_hat)
                        meta_ereward = self.env.compute_reward(ag, g, 1.0)
                        self.meta_control.buffer.add(meta_obs, g, g_reward_hat+meta_ereward, next_meta_obs)
                        # if self.meta_control.buffer.__len__() > self.meta_control.args.meta_batch_size:
                        for i in range(100):
                            self.meta_control._update_network(self.distance_model)

                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)



                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)

                self._soft_update_target_network(self.actor_target_network_explore, self.actor_network_explore)
                self._soft_update_target_network(self.critic_target_network_explore, self.critic_network_explore)
            # start to do the evaluation
            if epoch % self.args.test_interval == 0:
                success_rate, avg_reward = self._eval_agent(epoch)
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print(
                        '[{}] epoch is: {}, eval success rate is: {:.3f}, eval avg reward: {:.3f}, meta alpha: {:.3f}, {:.3f}'.format(
                            datetime.now(), epoch, success_rate, avg_reward, self.meta_control.alpha2.item(),
                            self.meta_control.k))
                    torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                                self.actor_network.state_dict()], \
                               self.model_path + '/model.pt')

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                           size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        # o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        # g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def compute_intr_reward(self, obs, step):
        prediction_error = self.rnd(obs)
        _, intr_reward_var = self.intrinsic_reward_rms(prediction_error)
        reward = prediction_error / (
                torch.sqrt(intr_reward_var) + 1e-8)
        return reward

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)

        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()

        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()

        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

        self.update_rnd(inputs_next_norm_tensor, 0)
        with torch.no_grad():
            intr_reward = self.compute_intr_reward(inputs_next_norm_tensor, 0)
        r_tensor = r_tensor + self.alpha * intr_reward

        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network_explore(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network_explore(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        # the q loss
        real_q_value = self.critic_network_explore(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network_explore(inputs_norm_tensor)
        actor_loss = -self.critic_network_explore(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_explore_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network_explore)
        self.actor_explore_optim.step()

        # update the critic_network
        self.critic_explore_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network_explore)
        self.critic_explore_optim.step()
    # do the evaluation

    def test_reward_hot(self, epoch):
        reward_hotpic_plot(self.args.env_name, epoch, self.reward_model, self.distance_model, self.r_norm, self.args,
                           self.meta_control.alpha2, self.meta_control.k)

    def _eval_agent(self, epoch):
        total_success_rate = []
        if epoch%100 == 0:
            n_test_rollouts = 50
        else:
            n_test_rollouts = self.args.n_test_rollouts

        for _ in range(n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            eg = observation['desired_goal']

            # for _ in range(20):
            #     with torch.no_grad():
            #         input_tensor = self._preproc_inputs(obs, eg)
            #         pi = self.actor_network(input_tensor)
            #         actions = pi.detach().cpu().numpy().squeeze()
            #     observation_new, reward, _, info = self.env.step(actions)
            #     obs = observation_new['observation']

            inputs = torch.tensor(np.concatenate([obs, eg]), dtype=torch.float32).unsqueeze(0).to(self.device)
            # ig = self.meta_control._select_actions(inputs, evaluate=bool(n_test_rollouts != 50))
            ig = self.meta_control._select_actions(inputs)
            g = self.meta_control.unnormalize(ig)
            if _ == 0:
                ig = self.meta_control._select_actions(inputs, True)
                g = self.meta_control.unnormalize(ig)
                self.logger.log('eval/_g0', g[0], self.step)
                self.logger.log('eval/_g1', g[1], self.step)
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, reward, _, info = self.env.step(actions)
                obs = observation_new['observation']
                ag = observation_new['achieved_goal']
                per_success_rate.append(self.env.compute_reward(ag, g, 1.0) + 1)
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        test_success_rate = np.mean(total_success_rate[:, -1])

        total_success_rate = []
        avg_reward = 0
        for _ in range(n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, reward, _, info = self.env.step(actions)
                obs = observation_new['observation']
                ag = observation_new['achieved_goal']
                # g = observation_new['desired_goal']
                per_success_rate.append(self.env.compute_reward(g, ag, 1) + 1)
                avg_reward += self.env.compute_reward(g, ag, 1)
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        success_rate = np.mean(total_success_rate[:, 0])

        self.alpha2 = local_success_rate
        if test_success_rate >= 0.6:
            self.meta_control.k = min(self.meta_control.k + 0.01, 1)

        if test_success_rate <= 0.4:
            self.meta_control.k = max(0.1, self.meta_control.k - 0.01)

        # if self.meta_control.k == 1 or self.meta_control.alpha2 == 0:
        #     self.meta_control.alpha1 = 0
        # else:
        #     self.meta_control.alpha1 = 0.6


        if self.step > 0:
            self.logger.log('eval/episode', epoch, self.step)
            self.logger.log('eval/episode_reward', avg_reward/(self.args.n_test_rollouts),
                            self.step)
            self.logger.log('eval/success_rate', local_success_rate,
                            self.step)

            self.logger.log('eval/true_episode_success', success_rate,
                            self.step)
            self.logger.log('eval/k', self.meta_control.k,
                            self.step)
            self.logger.log('eval/train_success', test_success_rate,
                            self.step)

            self.logger.log('eval/meta_alpha', self.meta_control.alpha2,
                            self.step)

            self.logger.log('eval/max_goal_distance', self.meta_control.gmax_distance,
                            self.step)

            self.logger.dump(self.step, ty='eval')

        return local_success_rate, avg_reward/(n_test_rollouts)