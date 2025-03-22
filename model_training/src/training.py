from typing import Any

import lightning.pytorch as pl
from pydantic import BaseModel
from torch import Tensor, nn, optim


class StepOutput(BaseModel, protected_namespaces=(), arbitrary_types_allowed=True):
    """
    Interface for training, validation and test step output. PyTorch callbacks can
    leverage this interface.

    Parameters
    ----------
    loss : Tensor
        Step loss value (e.g., training loss/validation accuracy)
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
        Function to compute accuracy between true vs model output
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

    def training_step(self, batch: list, batch_idx: int) -> dict[str, Any]:
        """Step for training datasets."""

        X, Y = batch
        Y_pred = self.model(X)
        training_loss = self.loss_fn(Y, Y_pred)
        step_output = StepOutput(
            loss=training_loss, true_output=Y_pred, model_output=Y_pred
        )

        self.log("training_loss", training_loss)

        return {
            "loss": training_loss,
            "step_output": step_output,
        }

    def validation_step(self, batch: list, batch_idx: int) -> dict[str, Any]:
        """Step for validation datasets."""

        X, Y = batch
        Y_pred = self.model(X)
        validation_accuracy = self.loss_fn(Y, Y_pred)
        step_output = StepOutput(
            loss=validation_accuracy, true_output=Y_pred, model_output=Y_pred
        )

        self.log("validation_accuracy", validation_accuracy)

        return {
            "loss": validation_accuracy,
            "step_output": step_output,
        }

    def test_step(self, batch: list, batch_idx: int) -> dict[str, Any]:
        """Step for test datasets."""

        X, Y = batch
        Y_pred = self.model(X)
        test_accuracy = self.loss_fn(Y, Y_pred)
        step_output = StepOutput(
            loss=test_accuracy, true_output=Y_pred, model_output=Y_pred
        )

        self.log("test_accuracy", test_accuracy)

        return {
            "loss": test_accuracy,
            "step_output": step_output,
        }

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure optimizer for training."""

        return self.optimizer