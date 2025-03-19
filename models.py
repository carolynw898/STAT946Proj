import math

import random
import torch
import torch.nn as nn
from torch.nn import functional as F


# from SymbolicGPT: https://github.com/mojivalipour/symbolicgpt/blob/master/models.py
class tNet(nn.Module):
    """
    The PointNet structure in the orginal PointNet paper: 
    PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation by Qi et. al. 2017
    """
    def __init__(self, config):
        super(tNet, self).__init__()

        self.activation_func = F.relu
        self.num_units = config.embeddingSize

        self.conv1 = nn.Conv1d(config.numberofVars+config.numberofYs, self.num_units, 1)
        self.conv2 = nn.Conv1d(self.num_units, 2 * self.num_units, 1)
        self.conv3 = nn.Conv1d(2 * self.num_units, 4 * self.num_units, 1)
        self.fc1 = nn.Linear(4 * self.num_units, 2 * self.num_units)
        self.fc2 = nn.Linear(2 * self.num_units, self.num_units)

        #self.relu = nn.ReLU()

        self.input_batch_norm = nn.BatchNorm1d(config.numberofVars+config.numberofYs)
        #self.input_layer_norm = nn.LayerNorm(config.numberofPoints)

        self.bn1 = nn.BatchNorm1d(self.num_units)
        self.bn2 = nn.BatchNorm1d(2 * self.num_units)
        self.bn3 = nn.BatchNorm1d(4 * self.num_units)
        self.bn4 = nn.BatchNorm1d(2 * self.num_units)
        self.bn5 = nn.BatchNorm1d(self.num_units)

    def forward(self, x):
        """
        :param x: [batch, #features, #points]
        :return:
            logit: [batch, embedding_size]
        """
        x = self.input_batch_norm(x)
        x = self.activation_func(self.bn1(self.conv1(x)))
        x = self.activation_func(self.bn2(self.conv2(x)))
        x = self.activation_func(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, dim=2)  # global max pooling
        assert x.size(1) == 4 * self.num_units

        x = self.activation_func(self.bn4(self.fc1(x)))
        x = self.activation_func(self.bn5(self.fc2(x)))
        #x = self.fc2(x)

        return x