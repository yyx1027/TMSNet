import torch.nn as nn
from Code.IntmdSequential import IntermediateSequential

# -------------------------
# CrossAttention
# -------------------------
class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x_q, x_kv):
        B, N_q, C = x_q.shape
        B, N_k, C = x_kv.shape

        # query
        q = self.to_q(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # key/value
        kv = self.to_kv(x_kv).reshape(B, N_k, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# -------------------------
# Residual
# -------------------------
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, x_kv=None):
        try:
            return self.fn(x, x_kv) + x
        except TypeError:
            return self.fn(x) + x

# -------------------------
# PreNorm / PreNormDrop / FeedForward
# -------------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x_kv=None):
        if x_kv is not None:
            return self.fn(self.norm(x), x_kv)
        else:
            return self.fn(self.norm(x))

class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x, x_kv=None):
        if x_kv is not None:
            return self.dropout(self.fn(self.norm(x), self.norm(x_kv)))
        else:
            return self.dropout(self.fn(self.norm(x)))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x, x_kv=None):
        return self.net(x)

# -------------------------
# TransformerCrossModel
# -------------------------
class TransformerCrossModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            # CrossAttention + Dropout
            layers.append(
                Residual(
                    PreNormDrop(
                        dim,
                        dropout_rate,
                        CrossAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                    )
                )
            )
            # FeedForward
            layers.append(
                Residual(
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                )
            )
        self.net = IntermediateSequential(*layers, return_intermediate=False)

    def forward(self, x_q, x_kv):
        x = x_q
        for layer in self.net:
            x = layer(x, x_kv)
        return x
 