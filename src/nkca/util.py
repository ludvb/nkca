import dataclasses
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, OrderedDict

import attr
import torch
import torch.nn as nn


def zip_dicts(*ds):
    return {k: tuple(d[k] for d in ds) for k in ds[0]}


def map_tree(fn: Callable, x1: Any, *xs: Any) -> Any:
    match x1:
        case dict():
            return {k: map_tree(fn, *vs) for k, vs in zip_dicts(x1, *xs).items()}
        case tuple():
            return tuple(map_tree(fn, *args) for args in zip(x1, *xs))
        case list():
            return [map_tree(fn, *args) for args in zip(x1, *xs)]
        case x:
            return fn(x, *xs)


def to_device(device: str | torch.device, x: Any):
    def _apply(v):
        if isinstance(v, torch.Tensor):
            return v.to(device)
        return v

    return map_tree(_apply, x)


def iterable_dataclass(cls):
    """Decorator that modifies a `dataclasses.dataclass` to allow the decorated
    class to be used as a return value in `torch.nn.DataParallel` modules."""
    init_fn = cls.__init__

    def __init__(self, *args, **kwargs):
        try:
            if isinstance(args[0], map):
                assert len(args) == 1
                (args,) = args
        except IndexError:
            pass
        return init_fn(self, *args, **kwargs)

    def __iter__(self):
        for f in dataclasses.fields(self):
            yield getattr(self, f.name)

    cls.__init__ = __init__
    cls.__iter__ = __iter__
    return cls


@attr.define
class Checkpoint:
    model_state: OrderedDict[str, torch.Tensor]
    ema: OrderedDict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    session: dict[str, Any]


@contextmanager
def eval(model: nn.Module, state_dict: dict | None = None):
    training_state = model.training
    model.eval()

    original_state_dict = deepcopy(model.state_dict())
    if state_dict is not None:
        model.load_state_dict(state_dict)

    try:
        yield model
    finally:
        model.train(training_state)
        model.load_state_dict(original_state_dict)
