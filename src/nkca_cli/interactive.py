import sys

import fire
import functorch as ft
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from effecthandlers_logging import TextLogger, log_info
from tabulate import tabulate
from torchvision.utils import make_grid

from nkca import session
from nkca.sampling import run_markov_chain
from nkca.session import Session
from nkca.util import Checkpoint

from .datasets import to_image
from .model import get_model
from .util import list_settings


class _AppendableIterator:
    def __init__(self, *items):
        self.q = []
        self.append(*items)

    def append(self, *items):
        self.q.extend(items)
        return self

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.q.pop()
        except IndexError:
            raise StopIteration()


def interactive(
    checkpoint,
    *,
    visualize="x",
    max_fps=32,
    batch_size=16,
    rescale=1.0,
    **hparams,
):
    """Samples interactively from a noise kernel transition model.

    Entrypoint for the `nkca-interactive` command. Additional flags are
    interpreted as session variables.

    Args:
        checkpoint (str): Path to model checkpoint.
        visualize (str): Which output to visualize. Must be one of "x" (the
            noisy state of the chain) and "z" (the mode of the denoising model).
        max_fps (int): Maximum frames per second.
        batch_size (int): Batch size.
        rescale (float): Rescaling factor for the visualization.
    """
    checkpoint = Checkpoint(**torch.load(checkpoint))

    with (
        TextLogger([sys.stderr]),
        Session(**checkpoint.session),
        Session(**hparams),
    ):
        noise_kernel = get_model()
        noise_kernel = noise_kernel.cuda()
        noise_kernel = nn.DataParallel(noise_kernel)
        noise_kernel.load_state_dict(checkpoint.ema)
        noise_kernel.eval()

        settings_tbl = list_settings()
        log_info(tabulate(settings_tbl))

        data_shape = session.get("data_shape")
        discrete = session.get_or_set("discrete", False)
        if discrete:
            x0 = torch.zeros(size=(batch_size, *data_shape), dtype=torch.long)
        else:
            x0 = torch.zeros(size=(batch_size, *data_shape), dtype=torch.float32)
        x0 = x0.cuda()

        next_noise_level = 1.0
        noise_iterator = _AppendableIterator(1.0, 1.0)
        chain = run_markov_chain(
            noise_kernel=noise_kernel, x0=x0, noise_iterator=noise_iterator
        )

        pygame.init()
        clock = pygame.time.Clock()
        display_size = None

        for d in ({"x": x, **e} for x, e in chain):
            for event in pygame.event.get():
                match event.type:
                    case pygame.QUIT:
                        break

            if pygame.key.get_pressed()[pygame.K_UP]:
                next_noise_level = min(1.00, next_noise_level + 0.01)
            if pygame.key.get_pressed()[pygame.K_DOWN]:
                next_noise_level = max(0.01, next_noise_level - 0.01)
            noise_iterator.append(next_noise_level)

            val = d[visualize].float()
            val = F.interpolate(val, scale_factor=rescale, mode="bilinear")
            val = ft.vmap(to_image)(val)
            num_col = int(np.ceil(np.sqrt(batch_size)))
            grid = make_grid(val, nrow=num_col)
            grid = grid.permute(1, 2, 0)
            grid = grid.transpose(0, 1)
            grid = (255 * grid).round().clamp(0, 255).byte()
            grid = grid.numpy()

            grid_size = [x + 40 for x in grid.shape[:2]]
            if grid_size != display_size:
                display_size = grid_size
                display = pygame.display.set_mode(display_size)

            # Clear screen
            display.fill((0, 0, 0))
            surf = pygame.surfarray.make_surface(grid)
            display.blit(surf, (20, 20))

            # Display noise level
            font = pygame.font.SysFont("monospace", 12)
            text_surface = font.render(
                f"noise: {next_noise_level:0.2f}", True, (255, 0, 0)
            )
            display.blit(text_surface, (0, 0))

            # Display fps
            text_surface = font.render(f"fps: {clock.get_fps():.2f}", True, (255, 0, 0))
            display.blit(text_surface, (0, display_size[1] - text_surface.get_height()))

            # Update display
            pygame.display.update()
            clock.tick(max_fps)


def main():
    fire.Fire(interactive)


if __name__ == "__main__":
    main()
