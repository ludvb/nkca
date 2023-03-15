"""
Implements U-Net with residual blocks and linear self-attention. Borrows heavily
from Hoogeboom et al. (2021):
https://github.com/ehoogeboom/multinomial_diffusion/blob/main/segmentation_diffusion/layers/layers.py
"""

import math

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn, *, dim, dim_out=None):
        if dim_out is None:
            dim_out = dim

        super().__init__()

        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1, dim_out, 1, 1))
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) * self.g + self.res_conv(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.rescale_steps = float(rescale_steps)

    def forward(self, t):
        x = t * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(8, dim_out),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class EmbeddingBlock(nn.Module):
    def __init__(self, dim, dim_out, *, dim_emb):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Conv2d(dim_emb, dim_out, 1))
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)

    def forward(self, x, emb):
        y = self.block1(x)
        y += self.mlp(emb)
        y = self.block2(y)
        return y


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        *_, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class UNet(nn.Module):
    def __init__(
        self,
        dim,
        dim_cond=None,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        if dim_cond is None:
            dim_cond = dim

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        def _make_attn_block(dim):
            return nn.Sequential(
                LinearAttention(dim),
                nn.GroupNorm(8, dim),
                nn.Mish(),
            )

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        Residual(
                            EmbeddingBlock(dim_in, dim_out, dim_emb=dim_cond),
                            dim=dim_in,
                            dim_out=dim_out,
                        ),
                        Residual(_make_attn_block(dim_out), dim=dim_out),
                        Residual(
                            EmbeddingBlock(dim_out, dim_out, dim_emb=dim_cond),
                            dim=dim_out,
                        ),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = Residual(
            EmbeddingBlock(mid_dim, mid_dim, dim_emb=dim_cond), dim=mid_dim
        )
        self.mid_attn = Residual(_make_attn_block(mid_dim), dim=mid_dim)
        self.mid_block2 = Residual(
            EmbeddingBlock(mid_dim, mid_dim, dim_emb=dim_cond), dim=mid_dim
        )

        for ind, (dim_out, dim_in) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        Residual(
                            EmbeddingBlock(2 * dim_in, dim_out, dim_emb=dim_cond),
                            dim=2 * dim_in,
                            dim_out=dim_out,
                        ),
                        Residual(_make_attn_block(dim_out), dim=dim_out),
                        Residual(
                            EmbeddingBlock(dim_out, dim_out, dim_emb=dim_cond),
                            dim=dim_out,
                        ),
                        Upsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

    def forward(self, x, c):
        h = []
        for resnet, attn, resnet2, downsample in self.downs:
            x = resnet(x, c)
            x = attn(x)
            x = resnet2(x, c)
            h.append((x, c))
            x = downsample(x)
            c = TF.resize(c, x.shape[2:])

        x = self.mid_block1(x, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, c)

        for resnet, attn, resnet2, upsample in self.ups:
            x_, c = h.pop()
            x = torch.cat((x, x_), dim=1)
            x = resnet(x, c)
            x = attn(x)
            x = resnet2(x, c)
            x = upsample(x)

        return x
