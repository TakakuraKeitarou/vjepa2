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

    @torch.no_grad()
    def sample(self, c, shape):
        """
        c: [N, embed_dim] - 条件（V-JEPAからの予測特徴量次元）
        shape: [N, patch_dim] - 生成するパッチの形状
        """
        device = c.device
        x = torch.randn(shape, device=device) # 初期状態は純粋なノイズ
        
        beta = torch.linspace(1e-4, 0.02, self.timesteps, device=device)
        alpha = 1. - beta
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            # ノイズを予測
            pred_noise = self.forward(x, t, c)
            
            alpha_t = alpha[i]
            alpha_bar_t = self.alpha_bar[i]
            
            # X_{t-1} を計算（デノイジング）
            mean = (1. / torch.sqrt(alpha_t)) * (x - ((1. - alpha_t) / torch.sqrt(1. - alpha_bar_t)) * pred_noise)
            
            # 発散（値の爆発）を防ぐためのクリッピング (ImageNetの正規化範囲は -2.1 ~ 2.6 程度)
            mean = torch.clamp(mean, -3.0, 3.0)
            
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta[i])
                x = mean + sigma * noise
            else:
                x = mean
                
        return x

    def forward(self, x, t, c):
        """
        x: [N, patch_dim] - ノイズ付きの画像パッチ
        t: [N] - タイムステップ
        c: [N, embed_dim] - 条件（V-JEPAからの予測特徴量次元）
        """
        t_emb = self.time_mlp(t)
        h = torch.cat([x, t_emb, c], dim=-1)
        return self.mlp(h)
