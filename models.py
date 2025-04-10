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
        train_decoder: bool = True,
        p_uncond: float = 0.2
    ):
        super().__init__()
        self.timesteps = timesteps
        self.n_embd = n_embd
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.tok_emb_weights = tok_emb_weights
        self.vars_emb_weights = vars_emb_weights
        self.p_uncond = p_uncond

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

        self.train_decoder = train_decoder
        if train_decoder:
            self.decoder = nn.Linear(n_embd, vocab_size)

        self.register_buffer("beta", torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer("alpha", 1.0 - self.beta)
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))

    def xtoi(self, xt):
        B, L, _ = xt.shape
        emb_table = self.tok_emb.weight  # [vocab_size, n_embd]

        # Flatten xt to [B*L, n_embd]
        xt_flat = xt.view(B * L, -1)  # [B*L, n_embd]

        # Compute Euclidean distances
        # Expand dimensions to broadcast: [B*L, 1, n_embd] - [1, vocab_size, n_embd]
        diffs = xt_flat.unsqueeze(1) - emb_table.unsqueeze(0)  # [B*L, vocab_size, n_embd]
        distances = torch.norm(diffs, p=2, dim=2)  # [B*L, vocab_size], L2 norm along n_embd

        # Find the nearest embedding indices (smallest distance)
        nearest_indices = torch.argmin(distances, dim=1)  # [B*L]
        nearest_indices = nearest_indices.view(B, L)  # Reshape to [B, L]

        return nearest_indices


    def round(self, xt):
        emb_table = self.tok_emb.weight
        idx = self.xtoi(xt)
        return emb_table[idx]

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
        t: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Denoises the noisy embeddings at time t.

        Args:
            x: [B, L, n_embd] - Noisy embeddings at time t
            t: torch.Tensor - Current timestep (scalar)
            condition: [B, n_embd] - Combined T-Net and variable embeddings
            device: str - Device to use (e.g., "cuda")

        Returns:
            x: [B, L, n_embd] - Denoised embeddings
        """
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
        
        tnet_output = self.tnet(points)  # [B, n_embd]
        vars_emb = self.vars_emb(variables)  # [B, n_embd]
        condition = tnet_output + vars_emb  # Full condition: points + variables
        
        # Randomly drop the points' contribution with probability p_uncond
        mask = torch.rand(B, device=condition.device) < self.p_uncond
        condition = torch.where(mask[:, None], vars_emb, condition)  # Unconditional: only variables
        
        token_emb = self.tok_emb(tokens)
        xt, noise = self.q_sample(token_emb, t)
        noise_pred = self.transformer(xt, t, condition)
        y_pred = self.p_sample(xt, t, noise_pred)

        if self.train_decoder:
            y_pred = self.decoder(y_pred)

        return y_pred, noise_pred, noise

    @torch.no_grad()
    def sample_progressive(
            self,
            points: torch.Tensor,
            variables: torch.Tensor,
            device: str = "cuda",
            guidance_scale: float = 0.5,  # Added parameter for guidance strength
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates a sample by denoising from random noise with classifier-free guidance.
        
        Args:
        points: [B, 2, 250] - Dataset points for conditioning
        variables: [B] - Number of variables per sample
        device: str - Device to use (e.g., "cuda")
        guidance_scale: float - Strength of guidance (default 1.0, no extra guidance)
        
        Returns:
        xt: [B, L] - Generated expression (token indices)
        """
        B = points.shape[0]
        
        # Precompute conditions outside the loop
        condition_cond = self.tnet(points) + self.vars_emb(variables)  # [B, n_embd], conditional
        condition_uncond = self.vars_emb(variables)  # [B, n_embd], unconditional
        condition_batch = torch.cat([condition_cond, condition_uncond], dim=0)  # [2B, n_embd]
        
        xt_emb = torch.randn(B, self.max_seq_len, self.n_embd, device=device)
        t_tensor = torch.zeros(B, dtype=torch.long, device=device)  # [B]
        prog_embs = []
        for t in range(self.timesteps - 1, -1, -1):
            t_tensor.fill_(t)  # Update in-place: [B] all set to t
            # Batch conditional and unconditional inputs
            xt_emb_batch = xt_emb.repeat(2, 1, 1)  # [2B, L, n_embd]
            t_batch = t_tensor.repeat(2)  # [2B]
            noise_pred_batch = self.transformer(xt_emb_batch, t_batch, condition_batch)  # [2B, L, n_embd]
            
            # Split predictions
            noise_pred_cond = noise_pred_batch[:B]  # [B, L, n_embd]
            noise_pred_uncond = noise_pred_batch[B:]  # [B, L, n_embd]
            
            # Apply classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            xt_emb = self.p_sample(xt_emb, t, noise_pred)  # t as int for p_sample
            if self.train_decoder:
                xt_logits = self.decoder(xt_emb)
                xt = torch.argmax(xt_logits, dim=-1)
                xt_emb = self.tok_emb(xt)

            prog_embs.append(xt_emb)
                
        if not self.train_decoder:
            xt = self.xtoi(xt_emb)

        return xt, prog_embs


    @torch.no_grad()
    def sample(
            self,
            points: torch.Tensor,
            variables: torch.Tensor,
            device: str = "cuda",
            guidance_scale: float = 0.5,  # Added parameter for guidance strength
    ) -> torch.Tensor:
        result, _ = self.sample_progressive(points, variables, device, guidance_scale)
        return result


    @torch.no_grad()
    def sample_ddim_progressive(
        self,
        points: torch.Tensor,
        variables: torch.Tensor,
        device: str = "cuda",
        guidance_scale: float = 0.5,
        num_steps: int = 50,  # Number of DDIM steps (fewer than timesteps)
        eta: float = 0.2,     # Controls stochasticity (0 = deterministic, 1 = DDPM-like)
    ) -> torch.Tensor:
        """Generates a sample using DDIM with classifier-free guidance.

        Args:
            points: [B, 2, 250] - Dataset points for conditioning
            variables: [B] - Number of variables per sample
            device: str - Device to use (e.g., "cuda")
            guidance_scale: float - Strength of guidance
            num_steps: int - Number of sampling steps (subset of timesteps)
            eta: float - Stochasticity parameter (0 for deterministic, 1 for DDPM-like)

        Returns:
            xt: [B, L] - Generated expression (token indices)
        """
        B = points.shape[0]

        # Precompute conditions
        condition_cond = self.tnet(points) + self.vars_emb(variables)  # [B, n_embd]
        condition_uncond = self.vars_emb(variables)  # [B, n_embd]
        condition_batch = torch.cat([condition_cond, condition_uncond], dim=0)  # [2B, n_embd]

        # Select subset of timesteps (e.g., linearly spaced from 0 to timesteps-1)
        step_indices = torch.linspace(0, self.timesteps - 1, steps=num_steps + 1, device=device).long()
        tau = step_indices[1:]  # Skip t=0 for the loop, use it at the end

        # Initialize noisy embeddings
        xt_emb = torch.randn(B, self.max_seq_len, self.n_embd, device=device)
        prog_embs = []

        # DDIM sampling loop
        for i in range(len(tau) - 1, -1, -1):  # Reverse from last tau to first
            t = tau[i]  # Current timestep
            t_prev = tau[i - 1] if i > 0 else torch.tensor(0, device=device)  # Previous timestep (0 at end)

            # Batch conditional and unconditional inputs
            xt_emb_batch = xt_emb.repeat(2, 1, 1)  # [2B, L, n_embd]
            t_batch = t.repeat(B).repeat(2)  # [2B]

            # Predict noise
            noise_pred_batch = self.transformer(xt_emb_batch, t_batch, condition_batch)
            noise_pred_cond = noise_pred_batch[:B]  # [B, L, n_embd]
            noise_pred_uncond = noise_pred_batch[B:]  # [B, L, n_embd]
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Compute predicted x_0
            sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1)
            pred_x0 = (xt_emb - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
            pred_x0 = self.round(pred_x0)

            # Compute direction pointing to x_t-1
            sqrt_alpha_bar_t_prev = torch.sqrt(self.alpha_bar[t_prev]).view(-1, 1, 1)
            sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1 - self.alpha_bar[t_prev]).view(-1, 1, 1)

            # Sigma (stochasticity control)
            sigma_t = eta * torch.sqrt((1 - self.alpha_bar[t_prev]) / (1 - self.alpha_bar[t])) * torch.sqrt(1 - self.alpha_bar[t] / self.alpha_bar[t_prev])

            # Deterministic direction
            direction = sqrt_one_minus_alpha_bar_t_prev - sigma_t * noise_pred

            # Update xt_emb
            xt_emb = sqrt_alpha_bar_t_prev * pred_x0 + direction * noise_pred
            if eta > 0 and t > 0:  # Add noise if eta > 0 and not final step
                xt_emb = xt_emb + sigma_t * torch.randn_like(xt_emb)

            prog_embs.append(xt_emb)

        # Final mapping to token indices
        xt = self.xtoi(xt_emb)
        return xt, prog_embs


    @torch.no_grad()
    def sample_ddim(
        self,
        points: torch.Tensor,
        variables: torch.Tensor,
        device: str = "cuda",
        guidance_scale: float = 0.5,
        num_steps: int = 50,  # Number of DDIM steps (fewer than timesteps)
        eta: float = 0.0,     # Controls stochasticity (0 = deterministic, 1 = DDPM-like)
    ) -> torch.Tensor:
        result, _ = self.sample_ddim_progressive(points, variables, device, guidance_scale, num_steps, eta)

        return result


    def loss_fn(
        self,
        noise_pred: torch.Tensor,  # [B, L, n_embd]
        noise: torch.Tensor,  # [B, L, n_embd]
        pred_logits: torch.Tensor,  # [B, L, vocab_size] or [B, L, n_embd]
        tokens: torch.Tensor,  # [B, L]
        t: torch.Tensor,  # [B]
        verbose: bool = False,
    ) -> Tuple[torch.Tensor]:
        """Computes combined MSE (diffusion) and scheduled CE (token) loss."""
        B, L = tokens.shape
        if self.train_decoder:
            assert pred_logits.shape == (
                B,
                L,
                self.vocab_size,
            ), f"Prediction is not in the correct shape: Expected {(B,L,self.vocab_size)} got {pred_logits.shape}"
        else:
            assert pred_logits.shape == (
                B,
                L,
                self.n_embd,
            ), f"Prediction is not in the correct shape: Expected {(B,L,self.n_embd)} got {pred_logits.shape}\n Note: expected size is different if we are not training the decoder."

        mse_loss = F.mse_loss(noise_pred, noise)

        ce_weight = 1.0 - (t.float() / self.timesteps)
        if self.train_decoder:
            ce_loss = F.cross_entropy(
                pred_logits.view(-1, pred_logits.size(-1)),
                tokens.view(-1),
                reduction="none",
                ignore_index=self.padding_idx,
            ).view(B, L)
            weighted_ce_loss = (ce_weight.unsqueeze(1) * ce_loss).mean()
            rounding_loss = torch.tensor(0.0, device=tokens.device)
        else:
            weighted_ce_loss = torch.tensor(0.0, device=tokens.device)
            rounding_loss = F.mse_loss(pred_logits, self.round(pred_logits))

        loss_components = torch.stack(
            [mse_loss,
             weighted_ce_loss,
             rounding_loss])

        if verbose:
            print(loss_components)

        total_loss = mse_loss + weighted_ce_loss + rounding_loss
        return total_loss, *loss_components


class VelocityPredictionTransformer(nn.Module):
    """Predicts the velocity field for flow matching with continuous time."""
    def __init__(self, max_seq_len: int,
                 n_layer: int = 4,
                 n_head: int = 4,
                 n_embd: int = 512):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, n_embd))
        self.time_emb = nn.Linear(1, n_embd)  # Continuous time input
        self.n_embd = n_embd
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=n_embd * 4,
            activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

    def sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes sinusoidal embedding for continuous t in [0, 1].
        
        Args:
            t: [B] - Continuous time between 0 and 1
            
        Returns:
            [B, n_embd] - Sinusoidal embedding
        """
        half_dim = self.n_embd // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)  # Frequency scaling
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)  # [half_dim]
        emb = t[:, None] * emb[None, :]  # [B, half_dim]
        return torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)  # [B, n_embd]

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Predicts velocity from current state, continuous time, and condition.

        Args:
            x_t: [B, L, n_embd] - Current state in the flow
            t: [B] - Continuous time between 0 and 1
            condition: [B, n_embd] - Combined T-Net and variable embeddings
            mask: [B, L] - Optional padding mask (True for valid positions)

        Returns:
            velocity: [B, L, n_embd] - Predicted velocity field
        """
        B, L, _ = x_t.shape
        pos_emb = self.pos_emb[:, :L, :]  # [1, L, n_embd]
        time_emb = self.sinusoidal_embedding(t).unsqueeze(1)  # [B, 1, n_embd]
        condition = condition.unsqueeze(1)  # [B, 1, n_embd]

        time_cond = torch.cat((time_emb, condition), dim=-1)  # [B, 2 * n_embd]
        proj = nn.Linear(2 * self.n_embd, self.n_embd).to(x_t.device)  # Project to n_embd
        time_cond_emb = proj(time_cond)

        x = x_t + pos_emb + time_cond_emb 
        velocity = self.encoder(x, src_key_padding_mask=mask)
        return velocity


class SymbolicFlowMatching(nn.Module):
    def __init__(
        self, pconfig,
        vocab_size: int,
        max_seq_len: int,
        padding_idx: int = 0,
        max_num_vars: int = 9, n_layer: int = 4, n_head: int = 4, n_embd: int = 512,
        tok_emb_weights: torch.Tensor = None, vars_emb_weights: torch.Tensor = None,
        train_decoder: bool = False, p_uncond: float = 0.2
    ):

        super().__init__()
        self.n_embd = n_embd
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.p_uncond = p_uncond

        # Embedding layers
        self.tok_emb = nn.Embedding(vocab_size, n_embd, padding_idx=padding_idx)
        self.vars_emb = nn.Embedding(max_num_vars, n_embd)
        if tok_emb_weights is not None:
            self.tok_emb.weight = nn.Parameter(tok_emb_weights, requires_grad=False)
        if vars_emb_weights is not None:
            self.vars_emb.weight = nn.Parameter(vars_emb_weights, requires_grad=False)

        # Networks
        self.tnet = tNet(pconfig)  # Assumed to be defined externally
        self.transformer = VelocityPredictionTransformer(max_seq_len, n_layer, n_head, n_embd)
        self.train_decoder = train_decoder
        if train_decoder:
            raise NotImplementedError("Training Decoder Not Supported yet")
            self.decoder = nn.Linear(n_embd, vocab_size)

    def xtoi(self, xt):
        B, L, _ = xt.shape
        emb_table = self.tok_emb.weight
        xt_flat = xt.view(B * L, -1)
        diffs = xt_flat.unsqueeze(1) - emb_table.unsqueeze(0)
        distances = torch.norm(diffs, p=2, dim=2)
        return torch.argmin(distances, dim=1).view(B, L)

    def round(self, xt):
        return self.tok_emb.weight[self.xtoi(xt)]

    def forward(self,
                points: torch.Tensor,
                tokens: torch.Tensor,
                variables: torch.Tensor,
                t:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass to predict velocity field.

        Args:
            points: [B, 2, 250] - Dataset points for conditioning
            tokens: [B, L] - Ground truth expression (token indices)
            variables: [B] - Number of variables per sample

        Returns:
            pred_emb: [B, L, n_embd] or [B, L, vocab_size] - Predicted embeddings or logits
            velocity_pred: [B, L, n_embd] - Predicted velocity
            target_velocity: [B, L, n_embd] - Target velocity (x_1 - x_0)
        """
        B = tokens.shape[0]
        device = tokens.device

        # Condition
        tnet_output = self.tnet(points)
        vars_emb = self.vars_emb(variables)
        condition = tnet_output + vars_emb
        mask = torch.rand(B, device=device) < self.p_uncond
        condition = torch.where(mask[:, None], vars_emb, condition)

        # Sample time
        x_0 = torch.randn(B, self.max_seq_len, self.n_embd, device=device)
        x_1 = self.tok_emb(tokens)
        x_t = (1 - t.view(-1, 1, 1)) * x_0 + t.view(-1, 1, 1) * x_1

        # Target velocity
        target_velocity = x_1 - x_0

        # Predict velocity with padding mask
        pad_mask = (tokens == self.padding_idx)  # [B, L]
        velocity_pred = self.transformer(x_t, t, condition, mask=pad_mask)

        # Predict embeddings or logits
        pred_emb = x_t + velocity_pred
        if self.train_decoder:
            pred_emb = self.decoder(pred_emb)

        return pred_emb, velocity_pred, target_velocity

    @torch.no_grad()
    def sample(self, points: torch.Tensor, variables: torch.Tensor, device: str = "cuda", guidance_scale: float = 0.5, num_steps: int = 50) -> torch.Tensor:
        """
        Generates a sample using Euler integration with classifier-free guidance.

        Args:
            points: [B, 2, 250] - Dataset points
            variables: [B] - Number of variables
            guidance_scale: float - Strength of guidance
            num_steps: int - Number of Euler steps

        Returns:
            xt: [B, L] - Generated token indices
        """
        B = points.shape[0]
        condition_cond = self.tnet(points) + self.vars_emb(variables)
        condition_uncond = self.vars_emb(variables)
        xt = torch.randn(B, self.max_seq_len, self.n_embd, device=device)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = torch.full((B,), step * dt, device=device)
            xt_batch = xt.repeat(2, 1, 1)
            t_batch = t.repeat(2)
            condition_batch = torch.cat([condition_cond, condition_uncond], dim=0)
            velocity_batch = self.transformer(xt_batch, t_batch, condition_batch)  # No mask during sampling
            v_cond, v_uncond = velocity_batch[:B], velocity_batch[B:]
            velocity = v_uncond + guidance_scale * (v_cond - v_uncond)
            xt = xt + velocity * dt

        return self.xtoi(xt)

    def loss_fn(self,
                velocity_pred: torch.Tensor,
                target_velocity: torch.Tensor,
                pred_emb: torch.Tensor,
                tokens: torch.Tensor,
                t: torch.Tensor,
                verbose=False):
        """
        Computes the loss function for end-to-end flow matching training,
        calculating xt and token_emb from inputs.

        Args:
            velocity_pred: [B, L, n_embd] - Predicted velocity
            target_velocity: [B, L, n_embd] - Target velocity (x_1 - x_0)
            pred_emb: [B, L, n_embd] - Predicted embeddings (unused here)
            tokens: [B, L] - Input token indices
            t: [B] - Continuous time between 0 and 1
            verbose: bool - Whether to print loss components

        Returns:
            total_loss, mse_loss, zero, zero - Tuple of loss values
        """
        if self.train_decoder:
            raise NotImplementedError("Training Decoder Not Supported yet")


        token_emb = self.tok_emb(tokens)  # Shape: [B, L, n_embd], true embeddings (x_1)
        x_0 = torch.randn_like(token_emb)  # Noise, shape: [B, L, n_embd]
        t = t.view(-1, 1, 1)  # [B, 1, 1] for broadcasting
        xt = (1 - t) * x_0 + t * token_emb  # [B, L, n_embd]

        mse_loss = F.mse_loss(velocity_pred, target_velocity)

        remaining_time = (1 - t)  # [B, 1, 1]
        x_1_pred = xt + remaining_time * velocity_pred  # [B, L, n_embd]

        loss_x1 = F.mse_loss(x_1_pred, token_emb)

        total_loss = mse_loss + 0.1 * loss_x1  # Adjust weight as needed

        if verbose:
            print(f"MSE Velocity Loss: {mse_loss.item()}, Loss_x1: {loss_x1.item()}")

        return total_loss, mse_loss, loss_x1, torch.tensor(0.0, device=tokens.device)


if __name__ == "__main__":
    # setting hyperparameters
    n_embd = 64
    timesteps = 1000
    batch_size = 64
    learning_rate = 6e-4
    num_epochs = 5
    blockSize = 32
    numVars = 1
    numYs = 1
    numPoints = 250
    target = "Skeleton"
    const_range = [-2.1, 2.1]
    trainRange = [-3.0, 3.0]
    decimals = 8
    addVars = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 50

    import numpy as np
    import glob
    import random

    pconfig = PointNetConfig(
        embeddingSize=n_embd,
        numberofPoints=numPoints,
        numberofVars=numVars,
        numberofYs=numYs,
    )

    model = SymbolicDiffusion(
        pconfig=pconfig,
        vocab_size=50,
        max_seq_len=blockSize,
        padding_idx=1,
        max_num_vars=9,
        n_layer=4,
        n_head=4,
        n_embd=n_embd,
        timesteps=timesteps,
        beta_start=0.0001,
        beta_end=0.02,
    ).to(device)

    tokens = torch.randint(0, vocab_size, (batch_size, blockSize), device=device)
    points = torch.randn((batch_size, 2, numPoints), device=device)
    variables = torch.randint(0, 9, (batch_size,), device=device)

    print(variables.shape)

    t = torch.randint(0, timesteps, (tokens.shape[0],), device=device)

    y_pred_emb, noise_pred, noise = model(points, tokens, variables, t)
    generated_tokens = model.sample_ddim(points, variables, device)

    loss, *loss_comps = model.loss_fn(noise_pred, noise, y_pred_emb, tokens, t)
    print("Predicted denoise: ------------- \n", y_pred_emb)
    print("Predicted noise  : ------------- \n", noise_pred)
    print("Actual noise     : ------------- \n", noise)
    print("Generated tokens : ------------- \n", generated_tokens)


    model = SymbolicFlowMatching(
        pconfig=pconfig,
        vocab_size=50,
        max_seq_len=blockSize,
        padding_idx=1,
        max_num_vars=9,
        n_layer=4,
        n_head=4,
        n_embd=n_embd,
    ).to(device)

    tokens = torch.randint(0, vocab_size, (batch_size, blockSize), device=device)
    points = torch.randn((batch_size, 2, numPoints), device=device)
    variables = torch.randint(0, 9, (batch_size,), device=device)

    print(variables.shape)

    t = torch.rand((tokens.shape[0],), device=device)

    y_pred_emb, noise_pred, noise = model(points, tokens, variables, t)
    generated_tokens = model.sample(points, variables, device)

    loss, *loss_comps = model.loss_fn(noise_pred, noise, y_pred_emb, tokens, t)
    print("Predicted denoise: ------------- \n", y_pred_emb)
    print("Predicted noise  : ------------- \n", noise_pred)
    print("Actual noise     : ------------- \n", noise)
    print("Generated tokens : ------------- \n", generated_tokens)
