from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch import Tensor, nn


class LossFunction(nn.Module, ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        raise NotImplementedError


class RMSE(LossFunction):
    """Root mean square error loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        mse_loss_fn = nn.MSELoss()
        rmse = torch.sqrt(mse_loss_fn(y_pred, y_true))
        return rmse


class MAE(LossFunction):
    """Mean absolute error loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return torch.mean(torch.abs(y_pred - y_true))


class MSE(LossFunction):
    """Mean square error loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        mse_loss_fn = nn.MSELoss()
        return mse_loss_fn(y_pred, y_true)


class WeightedRMSE(LossFunction):
    """
    Weighted Root Mean Square Error loss function.
    
    This loss function applies higher weights to errors in critical SOC regions
    (very low and very high SOC values) which are important for battery management.
    
    Parameters
    ----------
    low_soc_threshold : float
        SOC value below which errors receive higher weight (default: 0.2 or 20%)
    high_soc_threshold : float
        SOC value above which errors receive higher weight (default: 0.8 or 80%)
    critical_weight : float
        Weight multiplier for errors in critical regions (default: 2.0)
    """

    def __init__(
        self, 
        low_soc_threshold: float = 0.2, 
        high_soc_threshold: float = 0.8, 
        critical_weight: float = 2.0
    ):
        super().__init__()
        self.low_soc_threshold = low_soc_threshold
        self.high_soc_threshold = high_soc_threshold
        self.critical_weight = critical_weight

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # Calculate squared errors
        squared_errors = (y_pred - y_true) ** 2
        
        # Create weight mask based on true SOC values
        weights = torch.ones_like(y_true)
        
        # Apply higher weights to low SOC regions
        low_soc_mask = (y_true <= self.low_soc_threshold)
        weights = torch.where(low_soc_mask, weights * self.critical_weight, weights)
        
        # Apply higher weights to high SOC regions
        high_soc_mask = (y_true >= self.high_soc_threshold)
        weights = torch.where(high_soc_mask, weights * self.critical_weight, weights)
        
        # Apply weights to squared errors
        weighted_squared_errors = weights * squared_errors
        
        # Calculate weighted RMSE
        return torch.sqrt(torch.mean(weighted_squared_errors))


class _LossFunctionChoices(str, Enum):
    """Supported loss function options used for data validation."""

    rmse: str = "rmse"
    mae: str = "mae"
    mse: str = "mse"
    weighted_rmse: str = "weighted_rmse"


def get_loss_function(name: str) -> LossFunction:
    """
    Get loss fcuntion class.

    Parameters
    ----------
    name : str
        Name of loss function

    Returns
    -------
    LossFunction
        Loss function

    Raises
    ------
    ValueError
        If the name of loss function is not a valid option.
    NotImplementedError
        If the loss function is not implemented. This is a development error.
    """

    match name.lower():
        case "rmse":
            return RMSE()

        case "mae":
            return MAE()

        case "mse":
            return MSE()
            
        case "weighted_rmse":
            return WeightedRMSE()

        case _:
            _supported_choices = [option.value for option in _LossFunctionChoices]
            if name not in _supported_choices:
                raise ValueError(
                    f"{name} loss function not supported. Please select from {_supported_choices}"
                )

            raise NotImplementedError(
                f"{name} loss function not implemented by developer."
            )