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

        self.conv1 = nn.Conv1d(
            config.numberofVars + config.numberofYs, self.num_units, 1
        )
        self.conv2 = nn.Conv1d(self.num_units, 2 * self.num_units, 1)
        self.conv3 = nn.Conv1d(2 * self.num_units, 4 * self.num_units, 1)
        self.fc1 = nn.Linear(4 * self.num_units, 2 * self.num_units)
        self.fc2 = nn.Linear(2 * self.num_units, self.num_units)

        # self.relu = nn.ReLU()

        self.input_batch_norm = nn.BatchNorm1d(config.numberofVars + config.numberofYs)
        # self.input_layer_norm = nn.LayerNorm(config.numberofPoints)

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
        # x = self.fc2(x)

        return x


class NoisePredictionTransformer(nn.Module):
    """Predicts continuous noise in the embedding space for diffusion-based symbolic regression."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        n_layer: int = 6,
        n_head: int = 8,
        n_embd: int = 512,
        max_timesteps: int = 1000,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_embd = n_embd
        self.max_timesteps = max_timesteps

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, n_embd))
        self.time_emb = nn.Embedding(max_timesteps, n_embd)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * 4,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.ln_f = nn.LayerNorm(n_embd)

        self.head = nn.Linear(n_embd, n_embd)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, d: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_t: [B, L] - Noisy expression at time t
            t: [B] - Timestep
            d: [B, n_embd] - T-Net embedding of the dataset
        Returns:
            noise_pred: [B, L, n_embd] - Predicted continuous noise in embedding space
        """
        B, L = x_t.shape

        tok_emb = self.tok_emb(x_t)  # [B, L, n_embd]
        pos_emb = self.pos_emb[:, :L, :]  # [1, L, n_embd]
        time_emb = self.time_emb(t).unsqueeze(1)  # [B, 1, n_embd]
        d = d.unsqueeze(1)  # [B, 1, n_embd]

        x = tok_emb + pos_emb + time_emb + d  # [B, L, n_embd]

        x = self.encoder(x)
        x = self.ln_f(x)

        noise_pred = self.head(x)  # [B, L, n_embd]
        return noise_pred
