import dataclasses

import numpy as np
import torch

from .util import iterable_dataclass


class Distribution:
    pass


@iterable_dataclass
@dataclasses.dataclass
class Categorical(Distribution):
    logits: torch.Tensor


@iterable_dataclass
@dataclasses.dataclass
class Normal(Distribution):
    mu: torch.Tensor
    sd: torch.Tensor


@iterable_dataclass
@dataclasses.dataclass
class Delta(Distribution):
    x: torch.Tensor


def logp(d: Distribution, x: torch.Tensor) -> torch.Tensor:
    match d:
        case Categorical(logits):
            return torch.distributions.Categorical(logits=logits).log_prob(x)
        case Normal(mu, sd):
            energy = (x - mu) ** 2 / (2 * sd**2)
            lognorm = -0.5 * np.log(2 * np.pi) - torch.log(sd)
            return lognorm - energy
        case _:
            raise NotImplementedError(f"Logp of {d} not implemented")


def sample(d: Distribution) -> torch.Tensor:
    match d:
        case Categorical(logits):
            return torch.distributions.Categorical(logits=logits).sample()
        case Normal(mu, sd):
            mu, sd = torch.broadcast_tensors(mu, sd)
            return mu + sd * torch.randn_like(sd)
        case Delta(x):
            return x
        case _:
            raise NotImplementedError(f"Sample of {d} not implemented")


def mean(d: Distribution) -> torch.Tensor:
    match d:
        case Categorical(logits):
            return (torch.exp(logits) * torch.arange(logits.shape[-1]).to(logits)).sum(
                -1
            )
        case Normal(mu, _):
            return mu
        case Delta(x):
            return x
        case _:
            raise NotImplementedError(f"Mean of {d} not implemented")


def mode(d: Distribution) -> torch.Tensor:
    match d:
        case Categorical(logits):
            return torch.argmax(logits, -1)
        case Normal(mu, _):
            return mu
        case Delta(x):
            return x
        case _:
            raise NotImplementedError(f"Mode of {d} not implemented")
