import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class PatchDiffusionDecoder(nn.Module):
    def __init__(self, embed_dim, patch_dim, timesteps=1000, hidden_dim=1024):
        super().__init__()
        self.patch_dim = patch_dim
        self.timesteps = timesteps

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(patch_dim + embed_dim + hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, patch_dim)
        )

        # DDPMパラメタ (ノイズスケジューリング)
        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer('alpha_bar', alpha_bar)

    def forward(self, x, t, c):
        """
        x: [N, patch_dim] - ノイズ付きの画像パッチ
        t: [N] - タイムステップ
        c: [N, embed_dim] - 条件（V-JEPAからの予測特徴量次元）
        """
        t_emb = self.time_mlp(t)
        h = torch.cat([x, t_emb, c], dim=-1)
        return self.mlp(h)
