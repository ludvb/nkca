import datetime
import itertools as it
import os
import sys
from typing import Any

import attr
import fire
import functorch as ft
import torch
import torch.nn as nn
from effecthandlers_logging import TextLogger, log_info
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm

from nkca import session
from nkca.session import Session
from nkca.util import Checkpoint

from .datasets import get_dataset, to_image
from .model import get_model
from .sample import generate_samples
from .util import list_settings


@attr.define
class Metric:
    def __repr__(self):
        return str(self)

    def __str__(self):
        return show(self)


@attr.define
class FID(Metric):
    state: Any

    def __init__(self, state=None):
        if state is None:
            state = FrechetInceptionDistance().cuda()
        self.state = state


@attr.define
class IS(Metric):
    state: Any

    def __init__(self, state=None):
        if state is None:
            state = InceptionScore().cuda()
        self.state = state


def update(metric: Metric, x: torch.Tensor, is_real: bool):
    match metric:
        case FID(state):
            state.update(x.cuda(), is_real)
            return FID(state)
        case IS(state):
            if not is_real:
                state.update(x.cuda())
            return IS(state)
        case _:
            raise NotImplementedError()


def show(metric: Metric, precision: int = 2):
    match metric:
        case FID(state):
            format_string = "{{:.{prec}f}}".format(prec=precision)
            return format_string.format(state.compute().item())
        case IS(state):
            m, s = state.compute()
            format_string = "{{:.{prec}f}} +/- {{:.{prec}f}}".format(prec=precision)
            return format_string.format(m.item(), s.item())
        case _:
            raise NotImplementedError()


def as_numeric(metric: Metric):
    match metric:
        case FID(state):
            return state.compute().item()
        case IS(state):
            return state.compute()[0].item()
        case _:
            raise NotImplementedError()


def compare(metric1: Metric, metric2: Metric):
    match metric1, metric2:
        case FID(), FID():
            return as_numeric(metric1) < as_numeric(metric2)
        case IS(), IS():
            return as_numeric(metric1) > as_numeric(metric2)
        case _:
            raise NotImplementedError()


def compute_metrics(
    noise_kernel,
    dataset,
    num_samples=None,
    output_name="z",
):
    batch_size = session.get_or_set("batch_size", 32)

    if num_samples is None:
        num_samples = len(dataset)

    metric_initializers = session.get_or_set(
        "metrics",
        {
            "FID": FID,
            "IS": IS,
        },
        include_in_checkpoint=False,
    )
    metrics = {k: v() for k, v in metric_initializers.items()}

    def _normalize(x):
        return (255 * ft.vmap(to_image)(x)).byte()

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    for xs in tqdm(dataloader, ncols=100):
        xs = _normalize(xs)
        for k, m in metrics.items():
            metrics[k] = update(m, xs, True)

    for i in tqdm(range((num_samples - 1) // batch_size + 1), position=1, ncols=100):
        n = min(num_samples - i * batch_size, batch_size)
        ((xs,),) = generate_samples(
            noise_kernel,
            outputs=[output_name],
            num_samples=n,
        )
        xs = _normalize(xs)
        for k, m in metrics.items():
            metrics[k] = update(m, xs, False)

    return metrics


def run(
    checkpoint: str,
    dataset: str,
    *,
    save_dir: str | None = None,
    seed: int | None = None,
    **hparams,
):
    """Evaluates a noise kernel transition model.

    Entrypoint for the `nkca-metrics` command. Additional flags are interpreted
    as session variables.

    Args:
        checkpoint (str): Path to model checkpoint.
        dataset (str): Path to reference dataset to evaluate against.
        save_dir (str | None): Directory to save results and logging data to.
            If `None`, will save to a new directory "nkca_metrics_<date>" in the
            current directory.
        seed (int | None): Optional PyTorch RNG seed.
    """
    if save_dir is None:
        save_dir = f"nkca_metrics_{datetime.datetime.now().strftime('%Y%m%d')}"
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
        Session(**(checkpoint.session | hparams)),
    ):
        log_info(tabulate(list_settings()))

        if seed is not None:
            torch.random.manual_seed(seed)

        dataset = get_dataset(dataset)

        noise_kernel = get_model()
        noise_kernel = noise_kernel.cuda()
        noise_kernel = nn.DataParallel(noise_kernel)
        noise_kernel.load_state_dict(checkpoint.ema)
        noise_kernel.eval()

        num_samples = session.get_or_set("num_samples", None)
        output_name = session.get_or_set("output_name", "z")

        result = compute_metrics(
            noise_kernel,
            dataset,
            num_samples=num_samples,
            output_name=output_name,
        )

        log_info("Results:")
        log_info("========")
        for k, v in result.items():
            log_info(f"{k}: {str(v)}")
            with open(os.path.join(save_dir, f"metrics-{k}.txt"), "w") as f:
                f.write(show(v, precision=10))


def main():
    fire.Fire(run)


if __name__ == "__main__":
    main()
