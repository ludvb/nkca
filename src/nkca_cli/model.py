from nkca import session
from nkca.model import CategoricalKernel, ContinuousKernel


def _get_continuous_model():
    hidden_size = session.get_or_set("hidden_size", 128)
    dim_mults = session.get_or_set("dim_mults", [1, 2, 4, 8])
    num_channels, *_ = session.get("data_shape")
    return ContinuousKernel(
        num_channels=num_channels,
        hidden_size=hidden_size,
        dim_mults=dim_mults,
    )


def _get_categorical_model():
    hidden_size = session.get_or_set("hidden_size", 128)
    dim_mults = session.get_or_set("dim_mults", [1, 2, 4, 8])
    num_channels, *_ = session.get("data_shape")
    num_classes = session.get("num_classes")
    return CategoricalKernel(
        num_channels=num_channels,
        hidden_size=hidden_size,
        num_classes=num_classes,
        dim_mults=dim_mults,
    )


def get_model():
    """Constructs noise kernel based on the current session settings."""
    if session.get("discrete"):
        return _get_categorical_model()
    else:
        return _get_continuous_model()
