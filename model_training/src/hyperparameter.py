from abc import ABC
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel


class Hyperparameter(BaseModel, ABC):
    """
    Hyperparameter base class.

    Parameters
    ----------
    name : str
        Name of hyperparameter
    """

    name: str


class IntegerHyperparameter(Hyperparameter):
    """
    Hyperparameter for integer type.

    Parameters
    ----------
    low : int
        Lowest value in tuning range
    high : int
        Highest value in tuning range
    log : bool, optional
        Whether to sample from log domain, by default False
    """

    low: int
    high: int
    log: Optional[bool] = False


class FloatHyperparameter(Hyperparameter):
    """
    Hyperparameter for float type.

    Parameters
    ----------
    low : float
        Lowest value in tuning range
    high : float
        Highest value in tuning range
    log : bool, optional
        Whether to sample from log domain, by default False
    """

    low: float
    high: float
    log: Optional[bool] = False


class CategoricalHyperparameter(Hyperparameter):
    """
    Hyperparameter for float type.

    Parameters
    ----------
    choices : bool | int | float | str
        Categorical choices
    """

    choices: list[Union[bool, int, float, str]]


class _HyperparameterTypes(str, Enum):
    """Supported hyperparameter types used for data validation."""

    integer: str = "integer"
    float: str = "float"
    categorical: str = "categorical"


def get_hyperparameter(type: str) -> Hyperparameter:
    """
    Get hyperparameter class.

    Parameters
    ----------
    type : str
        Hyperparameter type

    Returns
    -------
    Hyperparameter
        Hyperparameter

    Raises
    ------
    ValueError
        If the hyperparameter type is not a valid option.
    NotImplementedError
        If the hyperparameter type is not implemented. This is a development error.
    """

    match type.lower():
        case "integer":
            return IntegerHyperparameter

        case "float":
            return FloatHyperparameter

        case "categorical":
            return CategoricalHyperparameter

        case _:
            _supported_choices = [option.value for option in _HyperparameterTypes]
            if type not in _supported_choices:
                raise ValueError(
                    f"Hyperparameter {type} type not supported. Please select from {_supported_choices}"
                )

            raise NotImplementedError(
                f"Hyperparameter {type} not implemented by developer."
            )