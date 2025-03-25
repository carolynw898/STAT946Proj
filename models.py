import torch
import torch.nn as nn
from torch.nn import functional as F


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


# ++++++++++++++++++++++++++++++++++++++++
# DIFFUSION PART

# Image noise model

# class ConditionalTimedUnet(nn.Module):
#     checkpoint_name = "timeattnunet"

#     def __init__(self, enc_chs=(3,32,64,128,256), dec_chs=(256,128,64,32), out_channels=3, time_emb_dim = 4):
#         super().__init__()
#         self.pool = nn.MaxPool2d((2,2))

#         self.encoder = nn.ModuleList([
#             TimeDoubleConv(enc_chs[i], enc_chs[i+1],time_emb_dim=time_emb_dim)
#             for i in range(len(enc_chs)-1)])

#         self.cond_encoder = nn.ModuleList([
#             TimeDoubleConv(enc_chs[i], enc_chs[i+1],time_emb_dim=time_emb_dim)
#             for i in range(len(enc_chs)-1)])


#         self.decoder = nn.ModuleList([
#             TimeDoubleConv(dec_chs[i], dec_chs[i+1],time_emb_dim=time_emb_dim, up=True)
#             for i in range(len(dec_chs)-1)])

#         self.output_layer = nn.Sequential(
#           AttentionBlock(dec_chs[-1]),
#           nn.ConvTranspose2d(dec_chs[-1], out_channels, 4, 2, 1)
#         )

#     def forward(self, x, c, t):
#         ftrs = []
#         for block in self.encoder:
#             x = block(x,t)
#             # x = self.pool(x)
#             ftrs.append(x)

#         cftrs = []
#         for block in self.encoder:
#             c = block(c,t)
#             # x = self.pool(x)
#             cftrs.append(c)

#         for up, skip_con, cond_skip_con in zip(self.decoder, ftrs[::-1], cftrs[::-1]):
#             skip_con = skip_con.to(x.device)
#             x = torch.cat([x, skip_con, cond_skip_con], dim=1)
#             x = up(x,t)

#         out = self.output_layer(x)
#         return out


# class DiffusionConfig:
#     """ base PointNet config """
#     def __init__(self, model, timesteps = 1000,
#                  beta_start=0.0001, beta_end=0.02,
#                     **kwargs):
#         self.model = model
#         self.timesteps = timesteps
#         self.beta_start = beta_start # number of points
#         self.beta_end = beta_end # input dimension (Xs)

#         for k,v in kwargs.items():
#             setattr(self, k, v)


# class Diffusion(nn.Module):
#     def __init__(self, dconfig, pconfig = None):
#         super().__init__()
#         self.timesteps = dconfig.timesteps
#         self.model = dconfig.model
#         self.checkpoint_name = 'diffusion-' + self.model.checkpoint_name

#         beta = torch.linspace(dconfig.beta_start, dconfig.beta_end, timesteps)
#         alpha = 1.0 - beta
#         alpha_bar = torch.cumprod(alpha, dim=0)

#         if pconfig is not None:
#             self.pconfig = pconfig
#             self.pnet = tNet(pconfig)

#         self.register_buffer("beta", beta)
#         self.register_buffer("alpha", alpha)
#         self.register_buffer("alpha_bar", alpha_bar)

#     def q_sample(self, x0, t):
#         "Forward process"
#         noise = torch.randn_like(x0)
#         sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
#         sqrt_subone_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]
#         return sqrt_alpha_bar * x0 + sqrt_subone_alpha_bar * noise, noise


#     @torch.no_grad()
#     def p_sample(self, x, t, condition = None, device="cuda"):
#         "Backward process, t is an integer"
#         num_samples = condition.size(0)
#         t_batch = torch.tensor([t] * num_samples, device=device)  # Time conditioning
#         beta_t = self.beta[t].view(-1, 1, 1, 1)
#         alpha_t = self.alpha[t].view(-1, 1, 1, 1)
#         alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

#         pred_noise = self.model(x, condition, t_batch)  # Predict noise
#         mean = (1 / torch.sqrt(alpha_t)) \
#           * (x - beta_t * pred_noise / torch.sqrt(1 - alpha_bar_t))

#         return mean


#     @torch.no_grad()
#     def sample(self, condition, img_size=(3, 32, 32), device="cuda"):
#         """Generate images from pure noise"""
#         assert torch.is_grad_enabled() == False

#         num_samples = condition.size(0)

#         self.model.eval()
#         x = torch.randn((num_samples, *img_size), device=device)  # Start with Gaussian noise

#         for t in range(self.timesteps - 1, -1, -1):

#             mean = self.p_sample(condition, x, t)
#             if t > 0:
#                 noise = torch.randn_like(x, device = device)
#                 x = mean + torch.sqrt(beta_t) * noise
#             else:
#                 x = mean

#         return x


#     @torch.no_grad()
#     def sample_progressive(self, condition, img_size=(3, 32, 32), device="cuda"):
#         """Generate images from pure noise"""
#         assert torch.is_grad_enabled() == False

#         num_samples = condition.size(0)
#         self.model.eval()
#         x = torch.randn((num_samples, *img_size), device=device)
#         xts=[]

#         for t in range(self.timesteps - 1, -1, -1):
#             mean = self.p_sample(x,condition, t)
#             mean = torch.clamp(mean, -1.0, 1.0)
#             if t > 0:
#                 # noise = torch.randn_like(x, device = device)
#                 # x = mean + torch.sqrt(beta_t) * noise
#                 x = mean
#             else:
#                 x = mean

#             xts.append(x * 0.5 + 0.5)

#         return xts

#     # TODO: Maybe rescale diffusion
#     def train_step(self,
#                    x, # Input equation
#                    p = None, # Points
#                    v = None # Variables
#                    tokenizer = None
#                    ):

#         c = p

#         if self.pnet is not None:
#             c = self.pnet(p)

#         t = torch.randint(0, model.timesteps, (x0.shape[0],), device=device)

#         # Sample for diffusion forward process.
#         xt, noise = self.q_sample(x0, t)
#         noise_hat = model(xt, c, t) # train the noise predictor
#         loss = criterion(noise_hat, noise)  # Assuming reconstruction of images

#         return loss(noise_hat, noise)

#     def forward(self, condition, *args, **kwargs):
#         return self.sample(condition)
