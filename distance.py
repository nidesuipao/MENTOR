import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import Dataset


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Distance_buffer(Dataset):
    def __init__(self, seed, capacity, T):
        random.seed(seed)
        self.T = T
        self.capacity = 2000
        self.position = 0

        self.train_data = []
        self.train_label = []
    def push(self, states, env, g):
        # print(states)
        # if env.compute_reward(states[-1], g, 1.0) == -1:
        #     for _ in range(self.T//5):
        #         if len(self.train_data) < self.capacity:
        #             self.train_data.append(None)
        #             self.train_label.append(None)
        #         i = random.randint(0, len(states) - 1)
        #         self.train_data[self.position] = np.concatenate((states[i], g))
        #         self.train_label[self.position] = min(1.0, (self.T - i + 20) * 1.0 / self.T)
        #         # print((self.T - i + 20) * 1.0 / self.T)
        #         self.position = (self.position + 1) % self.capacity


        for _ in range(self.T):
            if len(self.train_data) < self.capacity:
                self.train_data.append(None)
                self.train_label.append(None)
            i = random.randint(0, len(states)-1)
            j = random.randint(i, len(states)-1)
            self.train_data[self.position] = np.concatenate((states[i], states[j]))
            self.train_label[self.position] = 1.0 * (j-i) / self.T
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idx = random.randint(0, len(self.train_data), batch_size)
        return self.train_data[idx], self.train_label[idx]



class Distance_model(nn.Module):
    def __init__(self, num_inputs, hidden_dim, max_distance):
        super(Distance_model, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = F.sigmoid(x)
        return x








