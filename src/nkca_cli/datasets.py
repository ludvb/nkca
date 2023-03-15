import torch

from nkca import datasets, session


def to_image(x: torch.Tensor) -> torch.Tensor:
    """Converts data to floating point image with values in [0,1]"""
    if not session.get("discrete"):
        return torch.clamp((x + 1) / 2, 0, 1)
    num_classes = session.get("num_classes")
    return x.float() / (num_classes - 1)


def get_dataset(dataset_path: str):
    """Constructs dataset from image directory path"""

    dataset = datasets.imagedataset(dataset_path)

    try:
        image_size = session.get("resize")
        dataset = dataset >> datasets.resize(image_size)
    except ValueError:
        pass

    discrete = session.get_or_set("discrete", False)
    if discrete:
        num_classes = session.get_or_set("num_classes", 10)
        dataset = dataset >> datasets.discretize(num_classes, 0.0, 1.0)
    else:
        dataset = dataset >> datasets.map_all(lambda x: 2 * x.float() - 1)

    return dataset
