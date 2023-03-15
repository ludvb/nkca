import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder

from .util import map_tree


class TransformedDataset(data.Dataset):
    def __init__(self, dataset, transform=None):
        if transform is None:
            transform = lambda x: x

        self.dataset = dataset
        self.transform = transform

    def __rshift__(self, transform):
        return TransformedDataset(self, transform)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for x in iter(self.dataset):
            yield self.transform(x)

    def __getitem__(self, index):
        x = self.dataset[index]
        return self.transform(x)


def map_all(transform):
    def _transform(xs):
        return map_tree(transform, xs)

    return _transform


def discretize(discretization_steps, min_val, max_val):
    def _discretize(x):
        x = torch.searchsorted(
            torch.as_tensor(
                np.linspace(
                    min_val,
                    max_val,
                    num=discretization_steps - 1,
                    endpoint=False,
                ),
                device=x.device,
            ),
            x,
        )
        x = x.long()
        return x

    return _discretize


def resize(size):
    def _resize(x):
        orig_w, orig_h = TF.get_image_size(x)
        orig_cropped_size = min(orig_h, orig_w)
        x_cropped = TF.center_crop(x, orig_cropped_size)
        x_resized = TF.resize(x_cropped, size, antialias=True)
        return x_resized

    return _resize


class RecursiveImageFolder(ImageFolder):
    def find_classes(self, directory: str):
        classes = {}
        for dirpath, _, filenames in os.walk(directory):
            if filenames != []:
                classes[dirpath] = dirpath
        return list(classes), classes


def imagedataset(path, **kwargs):
    dataset = RecursiveImageFolder(path, **kwargs)
    return TransformedDataset(dataset, lambda x: T.ToTensor()(x[0]))
