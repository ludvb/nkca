import datetime
import itertools as it
import os
import sys

import fire
import functorch as ft
import imageio as ii
import numpy as np
import torch
import torch.nn as nn
from effecthandlers_logging import TextLogger, log_info
from tabulate import tabulate
from torchvision.utils import make_grid
from tqdm import tqdm

from nkca import datasets, session
from nkca.sampling import run_markov_chain
from nkca.session import Session
from nkca.util import Checkpoint, zip_dicts

from .datasets import to_image
from .model import get_model
from .util import batch_iterator, list_settings


def generate_samples(
    noise_kernel,
    initial_state: torch.Tensor | None = None,
    outputs: list[str] | None = None,
    num_samples: int | None = None,
    save_intermediate: bool = False,
):
    """Generates samples from a noise kernel transition model. The noise level
    is annealed linearly from "max_sampling_noise" to "min_sampling_noise" over
    "num_sampling_steps" steps (set as session variables).

    Args:
        noise_kernel: The transition model
        initial_state (torch.Tensor | None): The initial state of the chain.
            If None, an all-zeros state will be used.
        outputs (list[str] | None): Outputs to accumulate. The string "x"
            corresponds to noisy data and "z" to denoised data. If `None`,
            defaults to ["z"].
        num_samples (int | None): Number of samples to generate
        save_intermediate (bool): Whether to accumulate intermediate states or
            only the final state of the chain.

    Yields:
        torch.Tensor: A minibatch of outputs of shape (len(outputs),
            num_sampling_steps, batch_size, *data_shape) if save_intermediate is
            True, or (len(outputs), batch_size, *data_shape) otherwise.
    """
    batch_size = session.get_or_set("batch_size", 32)
    try:
        min_noise = session.get("min_noise")
    except ValueError:
        min_noise = 0.01
    min_noise = session.get_or_set("min_sampling_noise", min_noise)
    max_noise = session.get_or_set("max_sampling_noise", 1.0)
    num_steps = session.get_or_set("num_sampling_steps", 100)

    if outputs is None:
        outputs = ["z"]

    if initial_state is not None:
        initial_state_generator = iter(initial_state)
    else:
        data_shape = session.get("data_shape")
        discrete = session.get("discrete")
        num_samples = num_samples or batch_size
        initial_state_generator = (
            torch.zeros(
                size=data_shape,
                dtype=torch.long if discrete else torch.float32,
            )
            for _ in range(num_samples)
        )

    for batch in batch_iterator(initial_state_generator, batch_size):
        x0 = torch.stack(batch).cuda()
        noise_iterator = iter(
            [0.0]
            + np.linspace(
                max_noise, min_noise, num=num_steps, dtype=np.float32, endpoint=False
            ).tolist()
        )
        chain = run_markov_chain(
            noise_kernel=noise_kernel,
            x0=x0,
            noise_iterator=noise_iterator,
        )
        res = zip_dicts(*[{"x": x, **e} for x, e in chain])
        res = torch.stack([torch.stack(res[k]) for k in outputs])
        if not save_intermediate:
            res = res[:, -1]
        yield res


def _save_image(filename: str, x: torch.Tensor, nrow=None):
    match x.ndim:
        case 3:
            x = to_image(x)
        case 4:
            if nrow is None:
                nrow = int(np.sqrt(x.shape[0]))
            x = ft.vmap(to_image)(x)
            x = make_grid(x, nrow=nrow)
        case n:
            raise ValueError(f"Invalid number of dimensions: {n}")

    img_data = x.permute(1, 2, 0).detach().cpu().numpy()
    img_data = (img_data * 255).astype(np.uint8)

    log_info(f"Saving image to {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    ii.imwrite(filename, img_data)


def sample(
    checkpoint: str,
    *initial_images: str,
    batch_size: int = 64,
    num_samples: int | None = None,
    save_grid: bool = False,
    save_intermediate: bool = False,
    save_dir: str | None = None,
    seed: int | None = None,
    inpainting_mask: str | None = None,
    **hparams,
):
    """Samples from a noise kernel transition model.

    Entrypoint for the `nkca-sample` command. Additional flags are interpreted
    as session variables.

    Args:
        checkpoint (str): Path to model checkpoint.
        initial_images (str): Optional paths to images used as initial states
            when sampling variants or as conditionals when inpainting.
        batch_size (int): Batch size.
        num_samples (int | None): Number of samples to generate. Only used when
            no `initial_images` are passed. If `None`, defaults to `batch_size`.
        save_grid (bool): Whether to save a grid of all final states.
        save_intermediate (bool): Whether to save intermediate states of the
            Markov chain or only the final state.
        save_dir (str | None): Directory to save outputs and logging data to.
            If `None`, will save to a new directory "nkca_samples_<date>" in the
            current directory.
        seed (int | None): Optional PyTorch RNG seed.
        inpainting_mask (str | None): Path to an optional inpainting mask. The
            mask should have the same size as the `initial_images`. Non-zero
            regions of the mask specify regions of the `initial_images` to
            inpaint.
    """
    if num_samples is None:
        num_samples = batch_size

    if save_dir is None:
        save_dir = f"nkca_samples_{datetime.datetime.now().strftime('%Y%m%d')}"
    save_dir = next(
        x
        for x in it.chain((save_dir,), (save_dir + f"_{i}" for i in it.count(1)))
        if not os.path.exists(x)
    )
    os.makedirs(save_dir)

    checkpoint = Checkpoint(**torch.load(checkpoint))

    with (
        open(os.path.join(save_dir, "log.txt"), "w") as log_file,
        TextLogger([log_file, sys.stderr]),
        Session(**checkpoint.session),
        Session(**hparams),
        Session(batch_size=batch_size),
    ):
        log_info(tabulate(list_settings()))

        if seed is not None:
            torch.random.manual_seed(seed)

        discrete = session.get("discrete")
        if len(initial_images) == 0:
            data_shape = session.get("data_shape")
            initial_state = None
            cond_zs = torch.zeros(
                data_shape, dtype=torch.long if discrete else torch.float32
            )
            cond_zs = cond_zs.cuda()
            cond_mask = torch.zeros_like(cond_zs, dtype=torch.bool)
        else:
            initial_state = []
            for src_img in initial_images:
                src_img = ii.imread(src_img)
                if src_img.dtype == np.uint8:
                    src_img = src_img / 255.0
                src_img = torch.as_tensor(src_img).float()
                src_img = src_img.permute(2, 0, 1)
                initial_state.append(src_img)
            initial_state = torch.stack(initial_state)
            initial_state = initial_state.cuda()

            if discrete:
                num_classes = session.get("num_classes")
                initial_state = datasets.discretize(num_classes, 0.0, 1.0)(
                    initial_state
                )
            else:
                initial_state = 2 * initial_state - 1

            if inpainting_mask is not None:
                cond_mask = ii.imread(inpainting_mask) > 0
                cond_mask = torch.as_tensor(cond_mask)
                cond_mask = cond_mask.permute(2, 0, 1)
                cond_zs = initial_state
            else:
                cond_mask = torch.zeros_like(initial_state, dtype=torch.bool)
                cond_zs = torch.zeros_like(initial_state).cuda()
            cond_mask = cond_mask.cuda()

        with Session(cond_mask=cond_mask, cond_z=cond_zs):
            noise_kernel = get_model()
            noise_kernel = noise_kernel.cuda()
            noise_kernel = nn.DataParallel(noise_kernel)
            noise_kernel.load_state_dict(checkpoint.ema)
            noise_kernel.eval()

            xs_all, zs_all = [], []
            for sample_idx, (xs, zs) in enumerate(
                (
                    (xs, zs)
                    for xss, zss in tqdm(
                        generate_samples(
                            noise_kernel,
                            initial_state=initial_state,
                            outputs=["x", "z"],
                            num_samples=num_samples,
                            save_intermediate=save_intermediate,
                        ),
                        total=num_samples // batch_size,
                        position=1,
                        ncols=100,
                    )
                    for xs, zs in zip(
                        xss.transpose(0, 1) if save_intermediate else xss.unsqueeze(1),
                        zss.transpose(0, 1) if save_intermediate else zss.unsqueeze(1),
                    )
                ),
                1,
            ):
                for t, (x, z) in enumerate(zip(xs, zs), 1):
                    _save_image(
                        os.path.join(save_dir, f"z_{sample_idx}_{t:04d}.png"), z
                    )
                    _save_image(
                        os.path.join(save_dir, f"x_{sample_idx}_{t:04d}.png"), x
                    )

                if save_grid:
                    xs_all.append(xs[-1].cpu())
                    zs_all.append(zs[-1].cpu())

            if save_grid:
                _save_image(os.path.join(save_dir, f"x_grid.png"), torch.stack(xs_all))
                _save_image(os.path.join(save_dir, f"z_grid.png"), torch.stack(zs_all))


def main():
    fire.Fire(sample)


if __name__ == "__main__":
    main()
