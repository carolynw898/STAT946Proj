import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from utils import get_predicted_skeleton


# from SymbolicGPT: https://github.com/mojivalipour/symbolicgpt/blob/master/models.py
class PointNetConfig:
    """base PointNet config"""

    def __init__(
        self,
        embeddingSize,
        numberofPoints,
        numberofVars,
        numberofYs,
        **kwargs,
    ):
        self.embeddingSize = embeddingSize
        self.numberofPoints = numberofPoints  # number of points
        self.numberofVars = numberofVars  # input dimension (Xs)
        self.numberofYs = numberofYs  # output dimension (Ys)

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


# from https://github.com/juho-lee/set_transformer/blob/master/modules.py
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


# from https://github.com/juho-lee/set_transformer/blob/master/models.py
class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input,
        num_outputs,
        dim_output,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=False,
    ):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return self.dec(self.enc(X)).squeeze(1)


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
        _, L, _ = x_t.shape
        pos_emb = self.pos_emb[:, :L, :]  # [1, L, n_embd]
        time_emb = self.time_emb(t)
        if time_emb.dim() == 1:  # Scalar t case, [n_embd]
            time_emb = time_emb.unsqueeze(0)  # [1, n_embd]
        time_emb = time_emb.unsqueeze(1)  # [1, 1, n_embd]
        condition = condition.unsqueeze(1)  # [B, 1, n_embd]

        x = x_t + pos_emb + time_emb + condition
        return self.encoder(x)


# influenced by https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/simple_diffusion.py
class SymbolicGaussianDiffusion(nn.Module):
    def __init__(
        self,
        tnet_config: PointNetConfig,
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
        set_transformer=True,
        ce_weight=1.0,  # Weight for CE loss relative to MSE
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.n_embd = n_embd
        self.timesteps = timesteps
        self.set_transformer = set_transformer
        self.ce_weight = ce_weight

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tok_emb = nn.Embedding(vocab_size, n_embd, padding_idx=self.padding_idx)
        self.vars_emb = nn.Embedding(max_num_vars, n_embd)

        self.decoder = nn.Linear(n_embd, vocab_size, bias=False)
        self.decoder.weight = self.tok_emb.weight

        if set_transformer:
            dim_input = tnet_config.numberofVars + tnet_config.numberofYs
            self.tnet = SetTransformer(
                dim_input=dim_input,
                num_outputs=1,
                dim_output=n_embd,
                num_inds=tnet_config.numberofPoints,
                dim_hidden=tnet_config.embeddingSize,
                num_heads=4,
            )
        else:
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
    def sample(self, points, variables, train_dataset, batch_size=16, ddim_step=0):
        if self.set_transformer:
            points = points.transpose(1, 2)

        condition = self.tnet(points) + self.vars_emb(variables)
        shape = (batch_size, self.max_seq_len, self.n_embd)
        x = torch.randn(shape, device=self.device)
        steps = torch.arange(self.timesteps - 1, -1, -1, device=self.device)

        for i in range(0, self.timesteps, ddim_step):
            t = steps[i]
            t_next = (
                steps[i + ddim_step]
                if i + ddim_step < self.timesteps
                else torch.tensor(0, device=self.device)
            )
            x = self.p_sample(x, t, t_next, condition)

            # Print prediction every 250 steps
            # if (i + 1) % 250 == 0:
            #    logits = self.decoder(x)  # [B, L, vocab_size]
            #    token_indices = torch.argmax(logits, dim=-1)  # [B, L]
            #    for j in range(batch_size):
            #       token_indices_j = token_indices[j]  # [L]
            #        predicted_skeleton = get_predicted_skeleton(
            #            token_indices_j, train_dataset
            #        )
            #        tqdm.write(f" sample {j}: predicted_skeleton: {predicted_skeleton}")

        logits = self.decoder(x)  # [B, L, vocab_size]
        token_indices = torch.argmax(logits, dim=-1)  # [B, L]
        predicted_skeletons = []
        for j in range(batch_size):
            token_indices_j = token_indices[j]  # [L]
            predicted_skeleton = get_predicted_skeleton(token_indices_j, train_dataset)
            predicted_skeletons.append(predicted_skeleton)
        return predicted_skeletons

    def p_losses(
        self, x_start, points, tokens, variables, t, noise=None, mse: bool = False
    ):
        """Hybrid loss: MSE on embeddings + CE on tokens."""
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)

        if self.set_transformer:
            points = points.transpose(1, 2)

        condition = self.tnet(points) + self.vars_emb(variables)
        x_start_pred = self.model(x_t, t.long(), condition)

        # MSE loss on embeddings
        if mse:
            mse_loss = F.mse_loss(x_start_pred, x_start)
        else:
            mse_loss = torch.tensor(0.0, device=self.device)

        # CE loss on tokens
        logits = self.decoder(x_start_pred)  # [B, L, vocab_size]
        ce_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),  # [B*L, vocab_size]
            tokens.view(-1),  # [B*L]
            ignore_index=self.padding_idx,
            reduction="mean",
        )

        total_loss = mse_loss + self.ce_weight * ce_loss
        return total_loss, mse_loss, ce_loss

    def forward(self, points, tokens, variables, t, mse=False):
        token_emb = self.tok_emb(tokens)
        total_loss, mse_loss, ce_loss = self.p_losses(
            token_emb, points, tokens, variables, t, mse=mse
        )
        return total_loss, mse_loss, ce_loss
