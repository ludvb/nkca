import datetime
import itertools as it
import os
import sys
from copy import deepcopy

import fire
import functorch as ft
import torch
import torch.nn as nn
import torchvision.transforms as T
from effecthandlers_logging import TextLogger, log_info
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

import nkca.distribution as distr
from nkca import session
from nkca.model import add_noise
from nkca.session import Session
from nkca.util import Checkpoint, eval, map_tree

from . import metrics as metrics
from .datasets import get_dataset, to_image
from .model import get_model
from .sample import generate_samples
from .util import list_settings


def save_checkpoint(filename, noise_kernel, ema, optim):
    log_info(f"Saving checkpoint to {filename}...")
    session.set("rng_state", torch.random.get_rng_state())
    torch.save(
        dict(
            model_state=noise_kernel.state_dict(),
            ema=ema,
            optimizer_state=optim.state_dict(),
            session=session.checkpoint(),
        ),
        filename,
    )


def train(
    dataset,
    *,
    checkpoint_freq: int | None = None,
    checkpoint: str | None = None,
    data_workers: int = 4,
    epochs: int = 500,
    eval_freq: int | None = None,
    save_dir: str | None = None,
    viz_freq: int = 10,
    **hparams,
):
    """Trains a noise kernel transition model using contrastive adjustment.

    Entrypoint for the `nkca-train` command. Please note that additional flags
    are interpreted as session variables. For example, to set the hyperparameter
    w (Eqs. 15 and 23), the flag "--lag" can be used. Refer to the places in the
    code where session variables are used to see which flags are available.

    Args:
        dataset (str): Path to dataset to train on.
        checkpoint_freq (str | None): How often to save checkpoints (epochs).
            If `None`, will be set to `viz_freq`.
        checkpoint (str | None): Optional checkpoint to resume training from.
        data_workers (int | None): Number of dataloader workers.
        epochs (int | None): Number of epochs to train for.
        eval_freq (int | None): How often to evaluate the model (epochs). If
            `None`, evaluates the model only at the end of training.
        save_dir (str | None): Directory to save checkpoints and logging data
            to.  If `None`, will save to a new directory "nkca_<date>" in the
            current directory.
        viz_freq (int | None): How often to log samples from the model (epochs).
    """
    if checkpoint_freq is None:
        checkpoint_freq = viz_freq

    if eval_freq is None:
        eval_freq = epochs

    if save_dir is None:
        save_dir = f"nkca_{datetime.datetime.now().strftime('%Y%m%d')}"
    save_dir = next(
        x
        for x in it.chain((save_dir,), (save_dir + f"_{i}" for i in it.count(1)))
        if not os.path.exists(x)
    )
    os.makedirs(save_dir)

    if checkpoint is not None:
        checkpoint = Checkpoint(**torch.load(checkpoint, map_location="cpu"))
        session_init = checkpoint.session
    else:
        session_init = {}

    with (
        open(os.path.join(save_dir, "log.txt"), "w") as log_file,
        TextLogger([log_file, sys.stderr]),
        Session(**(session_init | hparams)),
    ):
        dataset = get_dataset(dataset)
        example_data = next(iter(dataset))
        session.set("data_shape", tuple(example_data.shape))

        data_augmentation = session.get_or_set("data_augmentation", [])
        dataset_augmented = dataset
        for augmentation in data_augmentation:
            match augmentation:
                case "fliph":
                    dataset_augmented = dataset_augmented >> T.RandomHorizontalFlip()
                case _:
                    raise ValueError(f"Unknown augmentation {augmentation}")

        batch_size = session.get_or_set("batch_size", 32)
        dataloader = DataLoader(
            dataset_augmented,
            batch_size=batch_size,
            num_workers=data_workers,
            shuffle=True,
        )

        noise_kernel = get_model()
        noise_kernel = noise_kernel.cuda()
        noise_kernel = nn.DataParallel(noise_kernel)

        lr = session.get_or_set("lr", 1e-4)
        adam_b1 = session.get_or_set("adam_b1", 0.9)
        adam_b2 = session.get_or_set("adam_b2", 0.999)
        optim = torch.optim.Adam(
            noise_kernel.parameters(), lr=lr, betas=(adam_b1, adam_b2)
        )

        min_noise = session.get_or_set("min_noise", 0.01)
        ema_decay = session.get_or_set("ema_decay", 0.999)
        epoch = session.get_or_set("epoch", 0)
        global_step = session.get_or_set("global_step", 0)

        if checkpoint is not None:
            noise_kernel.load_state_dict(checkpoint.model_state)
            optim.load_state_dict(checkpoint.optimizer_state)
            ema = {k: v.cuda() for k, v in checkpoint.ema.items()}
        else:
            ema = deepcopy(dict(noise_kernel.state_dict()))

        summary_writer = SummaryWriter(save_dir)
        settings_tbl = list_settings()
        log_info(tabulate(settings_tbl))
        summary_writer.add_text("settings", tabulate(settings_tbl, tablefmt="html"))

        try:
            torch.random.set_rng_state(session.get("rng_state"))
        except ValueError:
            try:
                torch.random.manual_seed(session.get("seed"))
            except ValueError:
                pass

        best_scores = {}
        epoch_progress = tqdm(
            range(epoch, epochs),
            desc="Training progress",
            unit="epoch",
            initial=epoch,
            total=epochs,
            position=1,
            ncols=100,
        )
        for epoch in epoch_progress:
            noise_kernel.train()
            batch_progress = tqdm(
                dataloader, unit="batch", total=len(dataloader), ncols=100
            )
            batch_loss, m = 0.0, 0
            for x in batch_progress:
                x = x.cuda()

                # Contrastive adjustment step (Algorithm 2)
                noise_level = torch.rand(size=(x.shape[0],), device=x.device)
                noise_level = noise_level * (1 - min_noise) + min_noise
                p_noisy = add_noise(x, noise_level)
                x = distr.sample(p_noisy)

                with torch.no_grad():
                    pyx, _ = noise_kernel(x, noise_level)
                    y = distr.sample(pyx)
                pxy, _ = noise_kernel(y, noise_level)
                logp = distr.logp(pxy, x)
                loss = -logp.reshape(logp.shape[0], -1).sum(1).mean()

                optim.zero_grad()
                loss.backward()
                optim.step()

                # Update EMA
                ema = map_tree(
                    lambda ema, v: ema_decay * ema + (1 - ema_decay) * v,
                    ema,
                    dict(noise_kernel.state_dict()),
                )

                # Bookkeeping
                global_step = session.set("global_step", global_step + 1)
                loss = loss.item()
                summary_writer.add_scalar("step/loss", loss, global_step)
                batch_loss, m = batch_loss + loss, m + 1
                batch_progress.set_description(
                    "  /  ".join(
                        [f"Epoch {epoch + 1:4d}", f"Loss {batch_loss / m:+0.3e}"]
                    )
                )

            epoch = session.set("epoch", epoch + 1)
            summary_writer.add_scalar("epoch/loss", batch_loss / m, global_step=epoch)

            if epoch % checkpoint_freq == 0:
                save_checkpoint(
                    os.path.join(save_dir, f"checkpoint_latest.pth"),
                    noise_kernel,
                    ema,
                    optim,
                )

            if epoch % viz_freq == 0:
                with eval(noise_kernel, state_dict=ema):
                    outputs = ["x", "z"]
                    (samples,) = generate_samples(
                        noise_kernel,
                        outputs=outputs,
                        save_intermediate=True,
                    )
                    for k, vs in zip(outputs, samples):
                        imgs = ft.vmap(ft.vmap(to_image))(vs)
                        summary_writer.add_image(
                            f"samples/final-{k}", make_grid(imgs[-1]), global_step=epoch
                        )

                        anim_frames = torch.stack(
                            [
                                make_grid(img)
                                for img in imgs[:: max(1, len(imgs) // (8 * 16))]
                            ]
                        )
                        anim_frames = anim_frames.unsqueeze(0)
                        summary_writer.add_video(
                            f"samples/anim-{k}", anim_frames, global_step=epoch, fps=16
                        )

            if epoch % eval_freq == 0:
                with eval(noise_kernel, state_dict=ema):
                    ms = metrics.compute_metrics(
                        noise_kernel=noise_kernel, dataset=dataset
                    )
                for k, m in ms.items():
                    log_info(f"Current {k} score is {str(m)}")
                    summary_writer.add_scalar(
                        f"epoch/{k}", metrics.as_numeric(m), global_step=epoch
                    )
                    try:
                        is_best = metrics.compare(m, best_scores[k])
                    except KeyError:
                        is_best = True
                    if is_best:
                        best_scores[k] = m
                        save_checkpoint(
                            os.path.join(save_dir, f"checkpoint_best_{k}.pth"),
                            noise_kernel,
                            ema,
                            optim,
                        )

        save_checkpoint(
            os.path.join(save_dir, f"checkpoint_final.pth"), noise_kernel, ema, optim
        )


def main():
    fire.Fire(train)


if __name__ == "__main__":
    main()
