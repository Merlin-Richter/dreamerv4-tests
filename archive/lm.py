import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelConfig():

    vokab_size: int = 128
    embedding_dim: int = 128
    max_sequence_length: int = 64

    n_heads: int = 8
    mlp_ratio: float = 3.0
    depth: int = 16

    drop_rate: int = 0.1
    att_drop_rate: int = 0.1
    att_logit_soft_cap: float = 50
    


class TokenEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()  # ? not sure
        self.embedding = nn.Embedding(config.vokab_size, config.embedding_dim)
        self.max_sequence_length = config.max_sequence_length
    
    def forward(self, x):
        return self.embedding(x)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # input is of shape (Batch, Sequence, Token_dim)
        self.n_heads = config.n_heads
        self.head_dim = config.embedding_dim / self.n_heads

        assert int(self.head_dim) == self.head_dim
        self.head_dim = int(self.head_dim)

        self.scale = 1/(self.head_dim ** 0.5)
        
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.soft_cap_act = nn.Tanh()
        self.soft_cap = config.att_logit_soft_cap

        self.dropout = nn.Dropout(config.drop_rate)
        self.att_droput = nn.Dropout(config.att_drop_rate)
        self.qkv = nn.Linear(config.embedding_dim, 3*config.embedding_dim)
        self.proj = nn.Linear(config.embedding_dim, config.embedding_dim)

        
        d_half = self.head_dim // 2
        

        cos = 10_000 ** (-2 * torch.arange(d_half, dtype=torch.float32) / d_half)
        sin = 10_000 ** (-2 * torch.arange(d_half, dtype=torch.float32) / d_half)

        cos = torch.outer(torch.arange(config.max_sequence_length, dtype=torch.float32), cos)
        sin = torch.outer(torch.arange(config.max_sequence_length, dtype=torch.float32), sin)
        cos = torch.cos(cos)
        sin = torch.sin(sin)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    
    def forward(self, x: torch.Tensor):
        # input is of shape (Batch, Sequence, Token_dim)
        B, S, C = x.shape

        qkv: torch.Tensor = self.qkv(x)
        # shape (Batch, Sequence, 3*Token_dim)
        # we want it in (Batch, Sequence, 3, heads, rest)
        qkv = qkv.reshape((B, S, 3, self.n_heads, -1)).permute((2, 3, 0, 1, 4))

        
        q, k, v = qkv[0], qkv[1], qkv[2]
        # shape: (heads, B, S, head_dim)

        q, k = self.q_norm(q), self.k_norm(k)

        # RoPE
        q_first, q_second = q[:, :, :, :self.head_dim // 2], q[:, :, :, self.head_dim // 2:]
        k_first, k_second = k[:, :, :, :self.head_dim // 2], k[:, :, :, self.head_dim // 2:]

        q = torch.concat((q_first * self.cos[:S] + -q_second * self.sin[:S], q_second * self.cos[:S] + q_first * self.sin[:S]), dim=-1)
        k = torch.concat((k_first * self.cos[:S] + -k_second * self.sin[:S], k_second * self.cos[:S] + k_first * self.sin[:S]), dim=-1)

        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = self.soft_cap_act(attn_scores / self.soft_cap) * self.soft_cap
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(attn_scores, dim=-1)

        attn = self.att_droput(attn)
        # shape: (heads, B, S, S)

        x = attn @ v
        # shape: (heads, B, S, head_dim)

        # zurück zu (B, S, C)
        x = x.permute(1, 2, 0, 3).reshape(B, S, C)

        return self.dropout(self.proj(x))
    

class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ll1 = nn.Linear(config.embedding_dim, int(config.embedding_dim * config.mlp_ratio))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.drop_rate)
        self.ll2 = nn.Linear(int(config.embedding_dim * config.mlp_ratio), config.embedding_dim)
    
    def forward(self, x):
        x = self.ll1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.ll2(x)
        x = self.dropout(x)
        return x

class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
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
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.att = Attention(config)
        self.ln1 = nn.RMSNorm(config.embedding_dim)
        self.mlp = SwiGLU(config)
        self.ln2 = nn.RMSNorm(config.embedding_dim)


    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.token_embed = TokenEmbedding(config)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.depth)])

        self.norm = nn.RMSNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.vokab_size)

    def forward(self, x):
        x = self.token_embed(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    config = ModelConfig(max_sequence_length=32, embedding_dim=128)
    t = Transformer(config)
