from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
import torch.nn as nn


@dataclass
class DiffusionConfig:
    T: int = 1000                 # number of diffusion steps
    schedule: str = "cosine"      # "linear" or "cosine"
    beta_start: float = 1e-4      # used by linear schedule
    beta_end: float = 0.02        # used by linear schedule
    cosine_s: float = 0.008       # used by cosine schedule (Nichol & Dhariwal)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Baisicly just makes how noisy each step is
def make_beta_schedule(cfg: DiffusionConfig) -> torch.Tensor:
    """
    Create a length-T tensor of betas in (0, 1) controlling noise each step.

    - Linear: betas increase linearly from beta_start to beta_end.
    - Cosine: betas are derived from a cosine-shaped cumulative alpha schedule
              (often trains better/stabler than linear).

    Returns:
        betas: torch.FloatTensor of shape [T] on cfg.device
    """
    if cfg.schedule == "linear":
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T, dtype=torch.float32)

    elif cfg.schedule == "cosine":
        # Cosine alpha_bar_t from Nichol & Dhariwal (Improved DDPM)
        s = cfg.cosine_s
        steps = cfg.T + 1
        t = torch.linspace(0, cfg.T, steps, dtype=torch.float64)
        f = torch.cos(((t / cfg.T + s) / (1 + s)) * math.pi / 2) ** 2  # monotonically decreasing
        alpha_bar = f / f[0]                                           # normalize so alpha_bar[0] = 1
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])                   # convert cumulative alphas to per-step betas
        betas = betas.to(torch.float32)

    else:
        raise ValueError(f"Unknown schedule: {cfg.schedule!r}. Use 'linear' or 'cosine'.")

    # Numeric safety: avoid zeros/ones which can explode logs or divisions later
    betas = betas.clamp(1e-8, 0.999).to(cfg.device)
    return betas

# calculates all the helper values we need to quickly mix an image with the right amount of noise at any timestep
def precompute_alphas(betas: torch.Tensor):
    """
    Given betas[0..T-1], precompute all alpha terms we need:

    Returns a dict with:
      - alphas:            1 - betas                      [T]
      - alpha_bar:         cumprod(alphas)                [T]
      - sqrt_alpha_bar:    sqrt(alpha_bar)                [T]
      - sqrt_1m_alpha_bar: sqrt(1 - alpha_bar)            [T]
      - alphas_cumprod_prev: alpha_bar shifted by 1 (with 1.0 at start)  [T]
      - one_over_sqrt_alphas: 1 / sqrt(alphas)            [T]
    Shapes are all [T] float tensors on the same device as betas.
    """
    assert betas.ndim == 1, "betas must be 1-D [T]"
    device = betas.device
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    # numeric helpers used by training/sampling formulas
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_1m_alpha_bar = torch.sqrt(1.0 - alpha_bar)
    one_over_sqrt_alphas = torch.rsqrt(alphas)

    # shift(alpha_bar, right, fill=1.0) -> [1, alpha_bar[0], ..., alpha_bar[T-2]]
    alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]], dim=0)

    return {
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_1m_alpha_bar": sqrt_1m_alpha_bar,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "one_over_sqrt_alphas": one_over_sqrt_alphas,
    }

# Takes a clean image and a timestep, and returns the image with the right amount of noise added for that step.
def q_sample(x0: torch.Tensor, t: torch.Tensor, alphas_dict: dict, noise: torch.Tensor = None):
    """
    Forward noising process q(x_t | x_0):
    Given a clean image x0, pick a timestep t and return its noisy version x_t.

    Args:
        x0:    Clean images [batch, C, H, W], values in [-1, 1].
        t:     LongTensor [batch], each entry is a timestep (0..T-1).
        alphas_dict: precomputed values from precompute_alphas().
        noise: Optional Gaussian noise [batch, C, H, W].
               If None, random N(0,1) noise is sampled.

    Returns:
        x_t:   Noisy images at timestep t [batch, C, H, W]
        noise: The actual noise that was added (needed for training loss).
    """
    if noise is None:
        noise = torch.randn_like(x0)

    # gather correct alpha values per batch index
    sqrt_alpha_bar_t = alphas_dict["sqrt_alpha_bar"].gather(0, t).view(-1, 1, 1, 1)
    sqrt_1m_alpha_bar_t = alphas_dict["sqrt_1m_alpha_bar"].gather(0, t).view(-1, 1, 1, 1)

    # formula: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    x_t = sqrt_alpha_bar_t * x0 + sqrt_1m_alpha_bar_t * noise
    return x_t, noise


# picks random diffusion steps for each image in the batch.
def sample_timesteps(batch_size: int, T: int, device: str):
    """
    Uniformly sample integer timesteps in [0, T-1] for a batch.
    """
    return torch.randint(0, T, (batch_size,), device=device, dtype=torch.long)

# trains the model by making it predict the exact noise we added at timestep t (using plain MSE).
def diffusion_loss(model, x0: torch.Tensor, t: torch.Tensor, alphas_dict: dict):
    """
    DDPM epsilon-prediction loss (MSE):
      1) Create x_t by adding the correct noise to x0 at timestep t (q_sample).
      2) Ask the model to predict that noise given (x_t, t).
      3) Minimize MSE between predicted noise and the true noise.
    """
    # step 1: generate x_t and the actual noise used
    x_t, noise = q_sample(x0, t, alphas_dict)

    # step 2: model predicts noise; model must accept (x_t, t)
    # expected shape: pred_noise same as x_t
    pred_noise = model(x_t, t)

    # step 3: MSE between predicted and true noise
    loss = torch.mean((pred_noise - noise) ** 2)
    return loss


# --- timestep embedding (sinusoidal) ---
import torch
import torch.nn as nn
import math

def sinusoidal_time_emb(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    timesteps: [B] int64 in [0, T-1]
    returns:   [B, dim]
    """
    assert timesteps.dtype == torch.long
    device = timesteps.device
    half = dim // 2
    # frequencies
    freq = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / half
    )  # [half]
    # outer product: [B, half]
    args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, dim]
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0,1))
    return emb


# --- building blocks ---
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        # add time conditioning
        temb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        h = self.norm1(h + temb)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(h + self.skip(x))


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block1 = ResidualBlock(in_ch, out_ch, time_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, time_dim)
        self.pool = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        skip = x
        x = self.pool(x)
        return x, skip


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block1 = ResidualBlock(out_ch + skip_ch, out_ch, time_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # [B, out_ch + skip_ch, H, W]
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        return x



class UNetTiny(nn.Module):
    def __init__(self, in_ch=3, base=64, time_dim=128):
        super().__init__()
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim*4),
            nn.SiLU(),
            nn.Linear(time_dim*4, time_dim)
        )

        # encoder
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down1 = Down(base, base*2, time_dim)    # out: base*2 @ 16x16, skip1: base*2
        self.down2 = Down(base*2, base*4, time_dim)  # out: base*4 @ 8x8,  skip2: base*4

        # bottleneck
        self.bot1 = ResidualBlock(base*4, base*4, time_dim)
        self.bot2 = ResidualBlock(base*4, base*4, time_dim)

        # decoder (note the skip_ch args!)
        self.up1 = Up(in_ch=base*4, out_ch=base*2, skip_ch=base*4, time_dim=time_dim)  # 8->16
        self.up2 = Up(in_ch=base*2, out_ch=base,   skip_ch=base*2, time_dim=time_dim)  # 16->32

        self.out_norm = nn.GroupNorm(8, base)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = sinusoidal_time_emb(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x = self.in_conv(x)
        x, skip1 = self.down1(x, t_emb)  # skip1: base*2
        x, skip2 = self.down2(x, t_emb)  # skip2: base*4

        x = self.bot1(x, t_emb)
        x = self.bot2(x, t_emb)

        x = self.up1(x, skip2, t_emb)    # expects skip_ch=base*4
        x = self.up2(x, skip1, t_emb)    # expects skip_ch=base*2

        x = self.out_norm(x)
        x = self.out_act(x)
        return self.out_conv(x)

    
# ---------- utils ----------
def denorm_to_01(x: torch.Tensor) -> torch.Tensor:
    """Map from [-1,1] back to [0,1] for saving/visualizing."""
    return (x.clamp(-1, 1) + 1) * 0.5

def gather_t(vec: torch.Tensor, t: torch.Tensor):
    """
    vec: [T], t: [B] -> returns vec[t] shaped [B,1,1,1] for broadcasting.
    """
    return vec.gather(0, t).view(-1, 1, 1, 1)


# ---------- one reverse step ----------
@torch.no_grad()
def p_sample(model, x_t: torch.Tensor, t: torch.Tensor, alphas_dict: dict):
    """
    One DDPM reverse step: p(x_{t-1} | x_t) using ε-prediction.
    """
    alphas = alphas_dict["alphas"]                    # α_t
    alpha_bar = alphas_dict["alpha_bar"]              # \bar{α}_t
    alpha_bar_prev = alphas_dict["alphas_cumprod_prev"]  # \bar{α}_{t-1}

    alpha_t = gather_t(alphas, t)                     # [B,1,1,1]
    abar_t = gather_t(alpha_bar, t)
    abar_prev_t = gather_t(alpha_bar_prev, t)

    beta_t = 1.0 - alpha_t
    # posterior variance \tilde{β}_t
    posterior_var = beta_t * (1.0 - abar_prev_t) / (1.0 - abar_t)
    posterior_log_var = torch.log(posterior_var.clamp(min=1e-20))

    # predict noise ε
    eps_pred = model(x_t, t)

    # x0 prediction from ε
    x0_pred = (x_t - torch.sqrt(1.0 - abar_t) * eps_pred) / torch.sqrt(abar_t)
    x0_pred = x0_pred.clamp(-1, 1)

    # posterior mean \tilde{μ}_t(x_t, x0)
    coef_x0 = torch.sqrt(abar_prev_t) * beta_t / (1.0 - abar_t)
    coef_xt = torch.sqrt(alpha_t) * (1.0 - abar_prev_t) / (1.0 - abar_t)
    mean = coef_x0 * x0_pred + coef_xt * x_t

    # add noise except at t=0
    noise = torch.randn_like(x_t)
    nonzero = (t > 0).float().view(-1, 1, 1, 1)
    x_prev = mean + nonzero * torch.exp(0.5 * posterior_log_var) * noise
    return x_prev


# ---------- full sampling loop ----------
@torch.no_grad()
def sample_loop(model, cfg: DiffusionConfig, alphas_dict: dict,
                batch_size=16, img_size=32, channels=3):
    """
    Start from pure noise and iterate t=T-1..0 to get images in [-1,1].
    """
    device = cfg.device
    x = torch.randn(batch_size, channels, img_size, img_size, device=device)
    model.eval()
    for t_step in reversed(range(cfg.T)):
        t = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
        x = p_sample(model, x, t, alphas_dict)
    return x  # [-1,1]

# ---------- utils ----------
def denorm_to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) * 0.5

def gather_t(vec: torch.Tensor, t: torch.Tensor):
    return vec.gather(0, t).view(-1, 1, 1, 1)

# ---------- one reverse step ----------
@torch.no_grad()
def p_sample(model, x_t: torch.Tensor, t: torch.Tensor, alphas_dict: dict):
    alphas = alphas_dict["alphas"]
    alpha_bar = alphas_dict["alpha_bar"]
    alpha_bar_prev = alphas_dict["alphas_cumprod_prev"]

    alpha_t = gather_t(alphas, t)
    abar_t = gather_t(alpha_bar, t)
    abar_prev_t = gather_t(alpha_bar_prev, t)

    beta_t = 1.0 - alpha_t
    posterior_var = beta_t * (1.0 - abar_prev_t) / (1.0 - abar_t)
    posterior_log_var = torch.log(posterior_var.clamp(min=1e-20))

    eps_pred = model(x_t, t)
    x0_pred = (x_t - torch.sqrt(1.0 - abar_t) * eps_pred) / torch.sqrt(abar_t)
    x0_pred = x0_pred.clamp(-1, 1)

    coef_x0 = torch.sqrt(abar_prev_t) * beta_t / (1.0 - abar_t)
    coef_xt = torch.sqrt(alpha_t) * (1.0 - abar_prev_t) / (1.0 - abar_t)
    mean = coef_x0 * x0_pred + coef_xt * x_t

    noise = torch.randn_like(x_t)
    nonzero = (t > 0).float().view(-1, 1, 1, 1)
    x_prev = mean + nonzero * torch.exp(0.5 * posterior_log_var) * noise
    return x_prev

# ---------- full sampling loop ----------
@torch.no_grad()
def sample_loop(model, cfg: DiffusionConfig, alphas_dict: dict,
                batch_size=16, img_size=32, channels=3):
    device = cfg.device
    x = torch.randn(batch_size, channels, img_size, img_size, device=device)
    model.eval()
    for t_step in reversed(range(cfg.T)):
        t = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
        x = p_sample(model, x, t, alphas_dict)
    return x



# ...existing code...
if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms

    # Load CelebA dataset (faces, 64x64)
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])
    dataset = torchvision.datasets.CelebA(root="./data", split="train", download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    show_batch_images(train_loader, num_images=8)
# ...existing code...
