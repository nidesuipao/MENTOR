import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import itertools
import tqdm
import copy
import scipy.stats as st
import os
import time



def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='softmax'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    elif activation == 'softmax':
        net.append(nn.Softmax())
    else:
        net.append(nn.ReLU())

    return net

class RewardModel:
    def __init__(self, ds, da,
                 ensemble_size=3, lr=3e-4, out_size=50, size_segment=1,
                 max_size=100, activation='softmax', capacity=5e5):

        # train data is trajectories, must process to sa and s..
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment

        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds + self.da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.train_batch_size = 128
        self.CEloss = nn.CrossEntropyLoss()

    def construct_ensemble(self):
        for i in range(self.de):
            model = nn.Sequential(*gen_net(in_size=self.ds + self.da,
                                           out_size=50, H=256, n_layers=3,
                                           activation=self.activation)).float().to(self.device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr)

    def predict_distance(self):
        pass

    def distance_reward(self):
        pass

    def get_distance_loss(self):
        pass