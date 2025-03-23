from typing import Any

import lightning.pytorch as pl
from pydantic import BaseModel
from torch import Tensor, nn, optim
import torch


class StepOutput(BaseModel, protected_namespaces=(), arbitrary_types_allowed=True):
    """
    Interface for training, validation and test step output. PyTorch callbacks can
    leverage this interface.

    Parameters
    ----------
    loss : Tensor
        Step loss value (e.g., training loss/validation error)
    true_output : Tensor
        True output variable
    model_output : Tensor
        Model output variable
    """

    loss: Tensor
    true_output: Tensor
    model_output: Tensor


class TrainingModule(pl.LightningModule):
    """
    Training module (based on Lightning) that initializes training and implements
    training, validation and test steps.

    Parameters
    ----------
    model : nn.Module
        PyTorch model
    loss_function : nn.Module
        Function to compute error between true vs model output
    optimizer : optim.Optimizer
        Training optimizer
    """

    def __init__(
        self,
        model: nn.Module,
        loss_function: nn.Module,
        optimizer: optim.Optimizer,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_function
        self.optimizer = optimizer

    def calculate_accuracy(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Calculate percentage accuracy within a certain voltage tolerance."""
        # Define acceptable voltage error (e.g., 0.05V)
        voltage_tolerance = 0.05
        # Calculate absolute error in actual voltage space
        abs_error = torch.abs(y_true - y_pred)
        # Calculate percentage of predictions within tolerance
        accuracy = torch.mean((abs_error <= voltage_tolerance).float()) * 100
        return accuracy

    def training_step(self, batch: list, batch_idx: int) -> dict[str, Any]:
        """Step for training datasets."""

        X, Y = batch
        Y_pred = self.model(X)
        
        # Calculate metrics
        training_error = self.loss_fn(Y, Y_pred)
        training_accuracy = self.calculate_accuracy(Y, Y_pred)
        
        # Log metrics with proper reduction
        self.log("train_loss", training_error, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_accuracy", training_accuracy, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        step_output = StepOutput(
            loss=training_error, true_output=Y, model_output=Y_pred
        )

        return {
            "loss": training_error,
            "train_accuracy": training_accuracy,
            "step_output": step_output,
        }

    def validation_step(self, batch: list, batch_idx: int) -> dict[str, Any]:
        """Step for validation datasets."""

        X, Y = batch
        Y_pred = self.model(X)
        
        # Calculate metrics
        validation_error = self.loss_fn(Y, Y_pred)
        validation_accuracy = self.calculate_accuracy(Y, Y_pred)
        
        # Log metrics with proper reduction
        self.log("val_loss", validation_error, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_accuracy", validation_accuracy, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        step_output = StepOutput(
            loss=validation_error, true_output=Y, model_output=Y_pred
        )

        return {
            "val_loss": validation_error,
            "val_accuracy": validation_accuracy,
            "step_output": step_output,
        }

    def test_step(self, batch: list, batch_idx: int) -> dict[str, Any]:
        """Step for test datasets."""

        X, Y = batch
        Y_pred = self.model(X)
        
        # Calculate metrics
        test_error = self.loss_fn(Y, Y_pred)
        test_accuracy = self.calculate_accuracy(Y, Y_pred)
        
        # Log metrics with proper reduction
        self.log("test_loss", test_error, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_accuracy", test_accuracy, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        step_output = StepOutput(
            loss=test_error, true_output=Y, model_output=Y_pred
        )

        return {
            "test_loss": test_error,
            "test_accuracy": test_accuracy,
            "step_output": step_output,
        }

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure optimizer for training."""

        return self.optimizer