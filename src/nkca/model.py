import abc
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nkca import session

from . import distribution as distr
from .primitives import Block, SinusoidalPosEmb, UNet


def _noise_schedule_continuous(noise_level: torch.Tensor):
    b = noise_level
    b = torch.clamp_min(b, 1e-7)
    a = 1 - b
    return a, b


def _compute_transition_distribution_continuous(
    x: torch.Tensor,
    pzx: distr.Distribution,
    noise_level_x: torch.Tensor,
    noise_level_y: torch.Tensor,
):
    """Computes the transition model for continuous data (Eq. 19)."""
    match pzx:
        case distr.Delta(z):
            return _compute_transition_distribution_continuous(
                x, distr.Normal(z, torch.zeros_like(z)), noise_level_x, noise_level_y
            )
        case distr.Normal(zloc, zscale):
            try:
                # Modify p(z | x) if there is an inpainting mask
                cond_mask = session.get("cond_mask")
                cond_z = session.get("cond_z")
                zloc = torch.where(cond_mask, cond_z, zloc)
                zscale = torch.where(cond_mask, torch.zeros_like(zscale), zscale)
            except ValueError:
                pass

            w = session.get_or_set("lag", 0.5)
            ax, bx = _noise_schedule_continuous(noise_level_x)
            ax, bx = ax.view(-1, 1, 1, 1), bx.view(-1, 1, 1, 1)
            ay, by = _noise_schedule_continuous(noise_level_y)
            ay, by = ay.view(-1, 1, 1, 1), by.view(-1, 1, 1, 1)
            wz = ay - ax * w
            loc = w * x + wz * zloc
            var = by - w**2 * bx + wz**2 * zscale**2
            scale = torch.sqrt(var)
            scale = torch.clamp(scale, min=1e-6)
            return distr.Normal(loc, scale)
        case _:
            raise NotImplementedError(
                f"Denoising distribution {type(pzx)} for continuous data not implemented."
            )


def _noise_schedule_discrete(noise_level: torch.Tensor):
    b = noise_level
    b = torch.clamp_min(b, 1e-7)
    return b


def _compute_transition_distribution_discrete(
    x: torch.Tensor,
    pzx: distr.Distribution,
    noise_level_x: torch.Tensor,
    noise_level_y: torch.Tensor,
):
    """Computes the transition model for categorical data (Eq. 27)."""
    num_classes = session.get("num_classes")
    match pzx:
        case distr.Delta(z):
            zlogits = torch.where(F.one_hot(z, num_classes).bool(), 0.0, -1e9)
            return _compute_transition_distribution_discrete(
                x, distr.Categorical(zlogits), noise_level_x, noise_level_y
            )
        case distr.Categorical(zlogits):
            try:
                # Modify p(z | x) if there is an inpainting mask
                cond_mask = session.get("cond_mask")
                cond_z = session.get("cond_z")
                cond_zlogits = torch.where(
                    F.one_hot(cond_z, num_classes).bool(), 0.0, -1e9
                )
                zlogits = torch.where(cond_mask.unsqueeze(-1), cond_zlogits, zlogits)
            except ValueError:
                pass

            w = session.get_or_set("lag", 0.95)
            bx = _noise_schedule_discrete(noise_level_x).view(-1, 1, 1, 1, 1)
            by = _noise_schedule_discrete(noise_level_y).view(-1, 1, 1, 1, 1)
            by = (by - w * bx) / (1 - w * bx)
            by = torch.clamp(by, min=0, max=1)
            xlogits = F.one_hot(x, num_classes + 1).float().log()
            zlogits = torch.cat([zlogits, torch.zeros_like(zlogits[..., :1]).log()], -1)
            elogits = (
                F.one_hot(torch.full_like(x, num_classes), num_classes + 1)
                .float()
                .log()
            )
            logits = torch.logsumexp(
                torch.stack(
                    [
                        torch.log(by) + elogits,
                        # ^ pixel is scrambled
                        torch.log(1 - by) + np.log(w) + xlogits,
                        # ^ pixel is not scrambled but retained from previous state
                        torch.log(1 - by) + np.log(1 - w) + zlogits,
                        # ^ pixel is not scrambled and sampled from pzx
                    ]
                ),
                axis=0,
            )
            return distr.Categorical(logits)
        case _:
            raise NotImplementedError(
                f"Denoising distribution {type(pzx)} for discrete data not implemented"
            )


def compute_transition_distribution(
    x: torch.Tensor,
    pzx: distr.Distribution,
    noise_level_x: torch.Tensor,
    noise_level_y: torch.Tensor,
):
    """Computes the transition model p(y | x) given a denoising distribution
    p(z | x).

    Args:
        x (torch.Tensor): previous state
        pzx (distr.Distribution): denoising distribution
        noise_level_x (torch.Tensor): noise level of previous state
        noise_level_y (torch.Tensor): noise level of next state

    Returns:
        distr.Distribution: transition model
    """
    if torch.is_floating_point(x):
        return _compute_transition_distribution_continuous(
            x, pzx, noise_level_x, noise_level_y
        )
    else:
        return _compute_transition_distribution_discrete(
            x, pzx, noise_level_x, noise_level_y
        )


def add_noise(x: torch.Tensor, noise_level: torch.Tensor):
    return compute_transition_distribution(
        x, distr.Delta(x), torch.zeros_like(noise_level), noise_level
    )


class Kernel(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def pyx(
        self,
        x: torch.Tensor,
        noise_level_x: torch.Tensor,
        noise_level_y: torch.Tensor,
    ) -> tuple[distr.Distribution, Any]:
        """Computes the transition model p_{\theta}(y | x).

        Args:
            x (torch.Tensor): previous state
            noise_level_x (torch.Tensor): noise level of previous state
            noise_level_y (torch.Tensor): noise level of next state

        Returns:
            tuple[distr.Distribution, Any]: Tuple of transition model and
                additional, free-form data returned by the model (e.g., the
                inferred denoised state)
        """

    def forward(
        self,
        x: torch.Tensor,
        noise_level_x: torch.Tensor,
        noise_level_y: torch.Tensor | None = None,
    ) -> tuple[distr.Distribution, Any]:
        if noise_level_y is None:
            noise_level_y = noise_level_x
        return self.pyx(x, noise_level_x, noise_level_y)


class CategoricalKernel(Kernel):
    """Implements the categorical noise kernel described in Sec. 2.4."""

    def __init__(
        self,
        num_channels: int,
        hidden_size: int,
        num_classes: int,
        dim_mults: list[int] | None = None,
    ):
        super().__init__()

        if dim_mults is None:
            dim_mults = [2**i for i in range(4)]

        self.num_classes = num_classes
        self.num_channels = num_channels

        self.noise_embedding = nn.Sequential(
            SinusoidalPosEmb(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.class_embedding_x = nn.Embedding(num_classes + 1, hidden_size)
        self.channel_embedding_x = nn.Embedding(num_channels, hidden_size)

        self.unet = UNet(
            dim=hidden_size,
            dim_cond=hidden_size,
            dim_mults=dim_mults,
        )
        self.proj = nn.Sequential(
            Block(hidden_size, hidden_size),
            nn.Conv2d(hidden_size, num_channels * num_classes, 1),
        )

    def pyx(self, x, noise_level_x, noise_level_y):
        cemb = self.class_embedding_x(x)
        W_channel = self.channel_embedding_x(torch.arange(x.shape[1], device=x.device))
        h = torch.einsum("bchwd,cd->bdhw", cemb, W_channel)

        nemb = self.noise_embedding(noise_level_x)
        nemb = nemb.view(*nemb.shape, *(1 for _ in h.shape[nemb.ndim :]))

        logits = self.proj(self.unet(h, nemb))
        logits = rearrange(
            logits, "b (c n) h w -> b c h w n", c=self.num_channels, n=self.num_classes
        )
        logits = F.log_softmax(logits, dim=-1)
        pzx = distr.Categorical(logits)
        pyx = compute_transition_distribution(x, pzx, noise_level_x, noise_level_y)

        return pyx, {"z": distr.mode(pzx)}


class ContinuousKernel(Kernel):
    """Implements the continuous noise kernel described in Sec. 2.3."""

    def __init__(
        self,
        num_channels: int,
        hidden_size: int,
        dim_mults: list[int] | None = None,
    ):
        super().__init__()

        if dim_mults is None:
            dim_mults = [2**i for i in range(4)]

        self.noise_embedding = nn.Sequential(
            SinusoidalPosEmb(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.intensity_embedder = nn.Sequential(
            nn.Conv2d(num_channels, hidden_size, 1),
        )
        self.unet = UNet(
            dim=hidden_size,
            dim_cond=hidden_size,
            dim_mults=dim_mults,
        )
        self.proj = nn.Sequential(
            Block(hidden_size, hidden_size), nn.Conv2d(hidden_size, 2 * num_channels, 1)
        )

    def pyx(
        self,
        x: torch.Tensor,
        noise_level_x: torch.Tensor,
        noise_level_y: torch.Tensor,
    ):
        h = self.intensity_embedder(x)

        emb = self.noise_embedding(noise_level_x)
        emb = emb.view(*emb.shape, *(1 for _ in h.shape[emb.ndim :]))

        params = self.proj(self.unet(h, emb))
        loc = params[:, 0::2]
        scale = F.softplus(params[:, 1::2])
        pzx = distr.Normal(loc, scale)
        pyx = compute_transition_distribution(x, pzx, noise_level_x, noise_level_y)

        return pyx, {"z": distr.mode(pzx)}
