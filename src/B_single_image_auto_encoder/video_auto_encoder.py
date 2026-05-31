import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class AutoEncoderConfig():

    embedding_dim: int = 256
    max_temporal_length: int = 32

    n_heads: int = 16
    mlp_ratio: float = 3.0
    depth: int = 8

    drop_rate: int = 0.1
    att_drop_rate: int = 0.1
    att_logit_soft_cap: float = 50

    patch_size: int = 16
    img_input_H: int = 64
    img_input_W: int = 64

    n_latents: int = 4
    bottleneck_dim: int = 32
    


class Attention(nn.Module):
    def __init__(self, config: AutoEncoderConfig, is_encoder):
        super().__init__()
        # input is of shape (Batch, n_tokens, Token_dim)
        self.n_heads = config.n_heads
        self.head_dim = config.embedding_dim / self.n_heads
        self.n_latents = config.n_latents

        assert int(self.head_dim) == self.head_dim
        self.head_dim = int(self.head_dim)

        self.scale = 1/(self.head_dim ** 0.5)
        
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.soft_cap_act = nn.Tanh()
        self.soft_cap = config.att_logit_soft_cap

        self.is_encoder = is_encoder
        self.dropout = nn.Dropout(config.drop_rate)
        self.att_droput = nn.Dropout(config.att_drop_rate)
        self.qkv = nn.Linear(config.embedding_dim, 3*config.embedding_dim)
        self.proj = nn.Linear(config.embedding_dim, config.embedding_dim)


    
    def forward(self, x: torch.Tensor):
        # input is of shape (Batch, n_tokens, Token_dim)
        B, N, C = x.shape

        qkv: torch.Tensor = self.qkv(x)
        # shape (Batch, n_tokens, 3*Token_dim)
        # we want it in (Batch, n_tokens, 3, heads, rest)
        qkv = qkv.reshape((B, N, 3, self.n_heads, -1)).permute((2, 3, 0, 1, 4))

        
        q, k, v = qkv[0], qkv[1], qkv[2]
        # shape: (heads, B, S, head_dim)

        q, k = self.q_norm(q), self.k_norm(k)


        mask = torch.zeros((N, N), device=x.device).bool()
        if self.is_encoder:
            mask[:-self.n_latents, -self.n_latents:] = True
        else:
            mask[-self.n_latents:, :-self.n_latents] = True

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = self.soft_cap_act(attn_scores / self.soft_cap) * self.soft_cap
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(attn_scores, dim=-1)

        attn = self.att_droput(attn)
        # shape: (heads, B, S, S)

        x = attn @ v
        # shape: (heads, B, S, head_dim)

        # zurück zu (B, S, C)
        x = x.permute(1, 2, 0, 3).reshape(B, N, C)

        return self.dropout(self.proj(x))
    


class SwiGLU(nn.Module):
    def __init__(self, config: AutoEncoderConfig) -> None:
        super().__init__()
        self.ll1_1 = nn.Linear(config.embedding_dim, int(config.embedding_dim * config.mlp_ratio))
        self.ll1_2 = nn.Linear(config.embedding_dim, int(config.embedding_dim * config.mlp_ratio))
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(config.drop_rate)
        self.ll2 = nn.Linear(int(config.embedding_dim * config.mlp_ratio), config.embedding_dim)

    def forward(self, x):
        x = self.ll1_1(x) * self.act(self.ll1_2(x))
        x = self.dropout(x)
        x = self.ll2(x)
        x = self.dropout(x)
        return x



class TransformerBlock(nn.Module):
    def __init__(self, config: AutoEncoderConfig, is_encoder) -> None:
        super().__init__()
        self.att = Attention(config, is_encoder)
        self.ln1 = nn.RMSNorm(config.embedding_dim)
        self.mlp = SwiGLU(config)
        self.ln2 = nn.RMSNorm(config.embedding_dim)


    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x



class Encoder(nn.Module):
    def __init__(self, config: AutoEncoderConfig) -> None:
        super().__init__()

        assert not (config.img_input_H % config.patch_size) and not (config.img_input_W % config.patch_size)

        self.patch_size = config.patch_size
        self.n_patches = (config.img_input_H // config.patch_size) * (config.img_input_W // config.patch_size)

        self.patch_proj = nn.Linear(config.patch_size * config.patch_size * 3, config.embedding_dim)
        self.learned_position_embedding = nn.Parameter(0.05*torch.rand((self.n_patches, config.embedding_dim)))

        self.learned_latents = nn.Parameter(0.05*torch.rand((config.n_latents, config.embedding_dim)))
        self.n_latents = config.n_latents

        self.blocks = nn.ModuleList([TransformerBlock(config, is_encoder=True) for _ in range(config.depth)])

        self.norm = nn.RMSNorm(config.embedding_dim)
        self.bottleneck_proj = nn.Linear(config.embedding_dim, config.bottleneck_dim)
        self.act = nn.Tanh()
    
    
    def patchify(self, x):
        B, H, W, C = x.shape

        # (B, H, W, C) → (B, C, H, W) for easier patching
        x = x.permute((0, 3, 1, 2))

        # (B, C, H, W) → (B, C, n_h, patch_size, n_w, patch_size)
        x = x.reshape((B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size))

        # → (B, n_h, n_w, patch_size, patch_size, C)
        x = x.permute((0, 2, 4, 3, 5, 1))

        # → (B, n_patches, patch_size*patch_size*C)
        return x.reshape((B, self.n_patches, self.patch_size * self.patch_size * C))

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.patchify(x)

        x = self.patch_proj(x) + self.learned_position_embedding

        # TODO: add dropout I think

        latents = self.learned_latents.unsqueeze(0).expand(B, -1, -1)
        x = torch.concat((x, latents), dim=1)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x[:, -self.n_latents:, :])

        return self.act(self.bottleneck_proj(x))



class Decoder(nn.Module):
    def __init__(self, config: AutoEncoderConfig) -> None:
        super().__init__()
        assert not (config.img_input_H % config.patch_size) and not (config.img_input_W % config.patch_size)

        self.patch_size = config.patch_size
        self.h_patches = config.img_input_H // config.patch_size
        self.w_patches = config.img_input_W // config.patch_size
        self.n_patches = (config.img_input_H // config.patch_size) * (config.img_input_W // config.patch_size)

        self.from_bottleneck_proj = nn.Linear(config.bottleneck_dim, config.embedding_dim)
        self.learned_patch_tokens = nn.Parameter(0.05*torch.rand((self.n_patches, config.embedding_dim)))
        
        self.blocks = nn.ModuleList([TransformerBlock(config, is_encoder=False) for _ in range(config.depth)])
        
        self.norm = nn.RMSNorm(config.embedding_dim)
        self.patch_token_to_img_patch_proj = nn.Linear(config.embedding_dim, config.patch_size*config.patch_size*3)
        self.act  = nn.Sigmoid()

    def imagify_patches(self, x):
        # x.shape = (B, self.n_patches, self.patch_size * self.patch_size * C)
        B, N, d = x.shape

        # (B, n_H, n_W, 16, 16, 3)
        x = x.reshape((B, self.h_patches, self.w_patches, self.patch_size, self.patch_size, 3))

        # (B, n_H, 16, n_W, 16, 3)
        x = x.permute((0, 1, 3, 2, 4, 5))
        
        x = x.reshape((B, self.h_patches*self.patch_size, self.w_patches*self.patch_size, 3))

        return x


    def forward(self, x):
        # Input are just the low dimensional latent tokens
        B, N, d = x.shape

        x = self.from_bottleneck_proj(x)

        learned_patch_tokens = self.learned_patch_tokens.unsqueeze(0).expand(B, -1, -1)
        
        x = torch.concat((learned_patch_tokens, x), dim=1)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x[:, :self.n_patches, :])

        x = self.imagify_patches(self.act(self.patch_token_to_img_patch_proj(x)))

        return x



class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

