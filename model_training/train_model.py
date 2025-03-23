import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch

import optuna
from pathlib import Path
from datetime import datetime

import yaml
from src.training_config import process_user_config
from src.hyperparameter_tuning import run_hyperparameter_tuning
from src.data_module import DataModule
from src.cell_model import LSTM, CellModel

# # Disable cuDNN to avoid potential issues with tensor layouts
# torch.backends.cudnn.enabled = False

now = datetime.now()
formatted_datetime = now.strftime('%Y%m%d%H%M%S')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlflow.set_tracking_uri("http://mlflow:5000")

def build_model(trial, training_config):
    """Build model based on trial parameters and training configuration.
    
    Args:
        trial: Optuna trial containing hyperparameters
        training_config: TrainingConfig object
    """
    model_config = training_config.model
    model_class = model_config.model
    
    # Build model using configuration
    model = model_class(
        input_size=model_config.input_size,
        hidden_size=trial.params["hidden_size"],
        output_size=model_config.output_size,
        num_lstm_layers=trial.params["num_lstm_layers"],
    )

    # Add optional parameters only if they exist in config
    if "dropout" in model_config:
        model.dropout = model_config.dropout
    if "bidirectional" in model_config:
        model.bidirectional = model_config.bidirectional
    if "use_temperature_aware" in model_config:
        model.use_temperature_aware = model_config.use_temperature_aware

    return model

if __name__ == "__main__":

    
    with open("config.yaml", "r") as confg_file:
        user_config = yaml.safe_load(confg_file)

    user_config["data_module"].update({"data_module": DataModule})
    user_config["model"].update({"model": LSTM})

    training_config = process_user_config(user_config)
    study = run_hyperparameter_tuning(training_config=training_config)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Create output directory
    Path("./training/output").mkdir(parents=True, exist_ok=True)
    
    # Save best model using proper configuration
    best_model = build_model(trial, training_config)
    torch.save(best_model.state_dict(), "./training/output/best_model.pth")
