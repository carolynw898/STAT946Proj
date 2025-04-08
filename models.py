import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple
from utils import CharDataset
from sympy import sympify, SympifyError
from tqdm import tqdm


# from SymbolicGPT: https://github.com/mojivalipour/symbolicgpt/blob/master/models.py
class PointNetConfig:
    """base PointNet config"""

    def __init__(
        self,
        embeddingSize,
        numberofPoints,
        numberofVars,
        numberofYs,
        method="GPT",
        varibleEmbedding="NOT_VAR",
        **kwargs,
    ):
        self.embeddingSize = embeddingSize
        self.numberofPoints = numberofPoints  # number of points
        self.numberofVars = numberofVars  # input dimension (Xs)
        self.numberofYs = numberofYs  # output dimension (Ys)
        self.method = method
        self.varibleEmbedding = varibleEmbedding

        for k, v in kwargs.items():
            setattr(self, k, v)


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
    def __init__(self, n_embd, max_seq_len, n_layer=6, n_head=8, max_timesteps=1000):
        super().__init__()
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

    def forward(self, x_t, t, condition):
        B, L, _ = x_t.shape
        pos_emb = self.pos_emb[:, :L, :]  # [1, L, n_embd]
        time_emb = self.time_emb(t)
        if time_emb.dim() == 1:  # Scalar t case, [n_embd]
            time_emb = time_emb.unsqueeze(0)  # [1, n_embd]
        time_emb = time_emb.unsqueeze(1)  # [1, 1, n_embd]
        condition = condition.unsqueeze(1)  # [B, 1, n_embd]

        # Expand to match sequence length L
        time_emb = time_emb.expand(B, L, -1)  # [B, L, n_embd]

        x = x_t + pos_emb + time_emb + condition
        return self.encoder(x)


# Symbolic Diffusion with Hybrid Loss
class SymbolicGaussianDiffusion(nn.Module):
    def __init__(
        self,
        tnet_config,
        vocab_size,
        max_seq_len,
        padding_idx: int = 0,
        max_num_vars: int = 9,
        n_layer=6,
        n_head=8,
        n_embd=512,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        ce_weight=1.0,  # Weight for CE loss relative to MSE
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.n_embd = n_embd
        self.timesteps = timesteps
        self.ce_weight = ce_weight

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding layers
        self.tok_emb = nn.Embedding(vocab_size, n_embd, padding_idx=self.padding_idx)
        self.vars_emb = nn.Embedding(max_num_vars, n_embd)

        # Decoder
        self.decoder = nn.Linear(n_embd, vocab_size, bias=False)
        self.decoder.weight = self.tok_emb.weight

        # Models
        self.tnet = tNet(tnet_config)
        self.model = NoisePredictionTransformer(
            n_embd, max_seq_len, n_layer, n_head, timesteps
        )

        # Noise schedule
        self.register_buffer("beta", torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer("alpha", 1.0 - self.beta)
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))

    def q_sample(self, x_start, t, noise=None):
        noise = torch.randn_like(x_start)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1)

        x_t = sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
        return x_t

    def p_mean_variance(self, x, t, t_next, condition):
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        alpha_bar_t_next = self.alpha_bar[t_next]
        beta_t = self.beta[t]

        x_start_pred = self.model(x, t.long(), condition)

        coeff1 = torch.sqrt(alpha_bar_t_next) * beta_t / (1 - alpha_bar_t)
        coeff2 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_next) / (1 - alpha_bar_t)
        mean = coeff1 * x_start_pred + coeff2 * x
        variance = (1 - alpha_bar_t_next) / (1 - alpha_bar_t) * beta_t
        return mean, variance

    @torch.no_grad()
    def p_sample(self, x, t, t_next, condition):
        mean, variance = self.p_mean_variance(x, t, t_next, condition)
        if torch.all(t_next == 0):
            return mean
        noise = torch.randn_like(x)
        return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample(self, points, variables, batch_size=16):
        condition = self.tnet(points) + self.vars_emb(variables)
        shape = (batch_size, self.max_seq_len, self.n_embd)
        x = torch.randn(shape, device=self.device)
        steps = torch.arange(
            self.timesteps - 1, -1, -1, device=self.device
        )  # Fix: start at timesteps-1

        for i in tqdm(
            range(self.timesteps), desc="sampling loop", total=self.timesteps
        ):
            t = steps[i]
            t_next = (
                steps[i + 1]
                if i + 1 < self.timesteps
                else torch.tensor(0, device=self.device)
            )
            x = self.p_sample(x, t, t_next, condition)

        # Map embeddings to token indices via decoder
        logits = self.decoder(x)  # [B, L, vocab_size]
        token_indices = torch.argmax(logits, dim=-1)  # [B, L]
        return token_indices

    def p_losses(
        self, x_start, points, tokens, variables, t, noise=None, mse: bool = True
    ):
        """Hybrid loss: MSE on embeddings + CE on tokens."""
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        condition = self.tnet(points) + self.vars_emb(variables)
        x_start_pred = self.model(x_t, t.long(), condition)

        # MSE loss on embeddings
        if mse:
            mse_loss = F.mse_loss(x_start_pred, x_start, reduction="mean")
        else:
            mse_loss = torch.tensor(0.0, device=self.device)

        # CE loss on tokens
        logits = self.decoder(x_start_pred)  # [B, L, vocab_size]
        ce_loss = (
            F.cross_entropy(
                logits.view(-1, self.vocab_size),  # [B*L, vocab_size]
                tokens.view(-1),  # [B*L]
                ignore_index=self.padding_idx,  # Assuming padding_idx=0
                reduction="none",
            )
            .view(tokens.shape)
            .mean()
        )  # [B]

        # Combine losses
        total_loss = mse_loss + self.ce_weight * ce_loss
        return total_loss, mse_loss, ce_loss

    def forward(self, points, tokens, t):
        token_emb = self.tok_emb(tokens)
        total_loss, mse_loss, ce_loss = self.p_losses(token_emb, tokens, points, t)
        return total_loss, mse_loss, ce_loss
