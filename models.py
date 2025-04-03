import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple


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
        **kwargs
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
    """Predicts continuous noise in the embedding space for diffusion-based symbolic regression."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        padding_idx: int = 0,
        n_layer: int = 6,
        n_head: int = 8,
        n_embd: int = 512,
        max_timesteps: int = 1000,
    ):
        super().__init__()
        # self.tok_emb = nn.Embedding(
        #     vocab_size, n_embd, padding_idx=torch.tensor(padding_idx, dtype=torch.long)
        # )
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

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts noise from noisy token indices, timestep, and condition.

        Args:
            x_t: [B, L] - Noisy expression at time t (token indices)
            t: [B] - Timestep
            condition: [B, n_embd] - Combined T-Net and variable embeddings

        Returns:
            noise_pred: [B, L, n_embd] - Predicted noise in embedding space
        """
        # print(x_t.shape)
        _, L, _ = x_t.shape

        # print(f"x_t min: {x_t.min().item()}, max: {x_t.max().item()}")
        # print(self.tok_emb.weight.shape[0])  # Number of embeddings

        # tok_emb = self.tok_emb(x_t.long())
        pos_emb = self.pos_emb[:, :L, :]
        time_emb = self.time_emb(t).unsqueeze(1)
        condition = condition.unsqueeze(1)

        x = x_t + pos_emb + time_emb + condition
        noise_pred = self.encoder(x)
        return noise_pred


class SymbolicDiffusion(nn.Module):
    def __init__(
        self,
        pconfig,
        vocab_size: int,
        max_seq_len: int,
        padding_idx: int = 0,
        max_num_vars: int = 9,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 512,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        tok_emb_weights: torch.Tensor = None,
        vars_emb_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.n_embd = n_embd
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.tok_emb_weights = tok_emb_weights
        self.vars_emb_weights = vars_emb_weights

        # Initialize embedding layers
        self.tok_emb = nn.Embedding(vocab_size, n_embd, padding_idx=padding_idx)
        self.vars_emb = nn.Embedding(max_num_vars, n_embd)

        # Load and freeze (requires_grad = False ensures they won't be updated) weights if provided
        if tok_emb_weights is not None:
            self.tok_emb.weight = nn.Parameter(tok_emb_weights, requires_grad=False)

        if vars_emb_weights is not None:
            self.vars_emb.weight = nn.Parameter(vars_emb_weights, requires_grad=False)

        self.tnet = tNet(pconfig)
        self.transformer = NoisePredictionTransformer(
            vocab_size,
            max_seq_len,
            padding_idx,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            max_timesteps=timesteps,
        )
        self.decoder = nn.Linear(n_embd, vocab_size)

        self.register_buffer("beta", torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer("alpha", 1.0 - self.beta)
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adds Gaussian noise to the input embeddings at time t.

        Args:
            x0: [B, L, n_embd] - Original expression embeddings
            t: [B] - Timestep

        Returns:
            xt: [B, L, n_embd] - Noisy embeddings at time t
            noise: [B, L, n_embd] - Noise added to the original embeddings
        """
        noise = torch.randn_like(x0, dtype=torch.float)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1)
        sqrt_subone_alpha_bar = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1)
        xt = sqrt_alpha_bar * x0 + sqrt_subone_alpha_bar * noise
        return xt, noise

    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Denoises the noisy embeddings at time t.

        Args:
            x: [B, L, n_embd] - Noisy embeddings at time t
            t: int - Current timestep (scalar)
            condition: [B, n_embd] - Combined T-Net and variable embeddings
            device: str - Device to use (e.g., "cuda")

        Returns:
            x: [B, L, n_embd] - Denoised embeddings
        """
        B = x.shape[0]
        beta_t = self.beta[t].view(-1, 1, 1)
        alpha_t = self.alpha[t].view(-1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        mean = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
        )
        return mean

    def forward(
        self,
        points: torch.Tensor,
        tokens: torch.Tensor,
        variables: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass to predict noise added to embeddings.

        Args:
            points: [B, 2, 250] - Dataset points for conditioning
            tokens: [B, L] - Ground truth expression (token indices)
            variables: [B] - Number of variables per sample
            t: [B] - Timestep

        Returns:
            y_pred: [B, L] - Generated expression (token indices)
            noise_pred: [B, L, n_embd] - Predicted noise
            noise: [B, L, n_embd] - Actual noise added
        """
        B = tokens.shape[0]

        condition = self.tnet(points)
        vars_emb = self.vars_emb(variables)
        condition = condition + vars_emb

        token_emb = self.tok_emb(tokens)
        xt, noise = self.q_sample(token_emb, t)
        noise_pred = self.transformer(xt, t, condition)
        y_pred_emb = self.p_sample(xt, t, noise_pred)
        y_pred_logits = self.decoder(y_pred_emb)
        return y_pred_logits, noise_pred, noise

    @torch.no_grad()
    def sample(
        self,
        points: torch.Tensor,
        variables: torch.Tensor,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Generates a sample by denoising from random noise.

        Args:
            points: [B, 2, 250] - Dataset points for conditioning
            variables: [B] - Number of variables per sample
            device: str - Device to use (e.g., "cuda")

        Returns:
            xt: [B, L] - Generated expression (token indices)
        """
        condition = self.tnet(points) + self.vars_emb(variables)  # [B, n_embd]
        B = condition.shape[0]

        xt_emb = torch.randn(B, self.max_seq_len, self.n_embd, device=device)
        t_tensor = torch.zeros(B, dtype=torch.long, device=device)  # [B]
        for t in range(self.timesteps - 1, -1, -1):
            t_tensor.fill_(t)  # Update in-place: [B] all set to t
            noise_pred = self.transformer(xt_emb, t_tensor, condition)
            xt_emb = self.p_sample(xt_emb, t, noise_pred)  # t as int for p_sample
            xt_logits = self.decoder(xt_emb)
            xt = torch.argmax(xt_logits, dim=-1)
            xt_emb = self.tok_emb(xt)
        return xt

    def loss_fn(
        self,
        noise_pred: torch.Tensor,  # [B, L, n_embd]
        noise: torch.Tensor,  # [B, L, n_embd]
        pred_logits: torch.Tensor,  # [B, L, vocab_size]
        tokens: torch.Tensor,  # [B, L]
        t: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """Computes combined MSE (diffusion) and scheduled CE (token) loss."""
        B, L = tokens.shape
        mse_loss = F.mse_loss(noise_pred, noise)

        ce_weight = 1.0 - (t.float() / self.timesteps)
        ce_loss = F.cross_entropy(
            pred_logits.view(-1, pred_logits.size(-1)),
            tokens.view(-1),
            reduction="none",
            ignore_index=self.padding_idx,
        ).view(B, L)
        weighted_ce_loss = (ce_weight.unsqueeze(1) * ce_loss).mean()

        total_loss = mse_loss + weighted_ce_loss
        return total_loss, mse_loss, weighted_ce_loss
