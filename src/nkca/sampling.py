from typing import Any, Callable, Iterator

import torch
from tqdm import tqdm

from . import distribution as distr
from .util import to_device


def run_markov_chain(
    noise_kernel: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[distr.Distribution, Any],
    ],
    x0: torch.Tensor,
    noise_iterator: Iterator[float],
):
    def _broadcast_noise_level(noise_level: float):
        return torch.tensor(noise_level, device=x0.device).repeat(x0.shape[0])

    x = x0
    noise_level_t = _broadcast_noise_level(next(noise_iterator))
    progress_bar = tqdm(noise_iterator)
    for noise_level_tp1 in progress_bar:
        progress_bar.set_description(f"Noise level: {noise_level_tp1:.3f}")
        noise_level_tp1 = _broadcast_noise_level(noise_level_tp1)
        with torch.no_grad():
            pyx, extra = noise_kernel(x, noise_level_t, noise_level_tp1)
        x = distr.sample(pyx)
        noise_level_t = noise_level_tp1
        yield x.cpu(), to_device("cpu", extra)
