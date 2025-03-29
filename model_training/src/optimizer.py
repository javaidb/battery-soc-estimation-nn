from enum import Enum

from torch import optim


class _OptimizerChoices(str, Enum):
    """Supported optimizer options used for data validation."""

    adam: str = "adam"
    sgd: str = "sgd"


def get_optimizer(name: str) -> optim.Optimizer:
    """
    Get the uinstantiated Optimizer class from PyTorch.

    Parameters
    ----------
    name : str
        Name of optimizer

    Returns
    -------
    optim.Optimizer
        Uninstantiated Optimizer class

    Raises
    ------
    ValueError
        If the name of optimizer is not a valid option.
    NotImplementedError
        If the optimizer is not implemented. This is a development error.
    """

    match name.lower():
        case "adam":
            return optim.Adam

        case "sgd":
            return optim.SGD

        case _:
            _supported_choices = [option.value for option in _OptimizerChoices]
            if name not in _supported_choices:
                raise ValueError(
                    f"{name} optimizer not supported. Please select from {_supported_choices}"
                )

            raise NotImplementedError(f"{name} optimizer not implemented by developer.")