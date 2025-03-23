import os
from typing import Callable

import lightning as L
import pandas as pd
import torch
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset, random_split


class LGHG2Dataset(Dataset):
    time_col = "Time [min]"
    voltage_col = "Voltage [V]"
    current_col = "Current [A]"
    temperature_col = "Temperature [degC]"
    capacity_col = "Capacity [Ah]"
    soc_col = "SOC [-]"
    ocv_col = "Open Circuit Voltage [V]"
    overpotential_col = "Overpotential [V]"

    norm_settings = {
        current_col: (-20, 20),
        temperature_col: (-30, 50),
        soc_col: (0, 1),
        overpotential_col: (-1, 1),
    }

    def __init__(self, data_directory: str):
        self.data = []
        self.dataset_names = []

        for T in ["25degC"]:
            T_directory = f"{data_directory}/{T}"
            file_names = list(
                filter(lambda f: f.endswith("parquet"), os.listdir(T_directory))
            )

            for f in file_names:
                df = pd.read_parquet(f"{T_directory}/{f}")
                df = self.calculate_overpotential(df)
                df = self.normalize_data(df)
                self.data.append(df)
                self.dataset_names.append(f"{T}/{f}")

    def calculate_overpotential(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate overpotential with temperature compensation
        df[self.overpotential_col] = df[self.voltage_col] - df[self.ocv_col]
        
        # Add temperature compensation factor
        temp_factor = 1 + 0.002 * (df[self.temperature_col] - 25)  # 0.2% per degree from 25°C
        df[self.overpotential_col] *= temp_factor
        
        return df

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, norm_range in self.norm_settings.items():
            min_val, max_val = norm_range
            df[col] = (df[col] - min_val) / (max_val - min_val) * 2 - 1
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        df = self.data[index]

        X_cols = [self.soc_col, self.current_col, self.temperature_col]
        Y_col = self.overpotential_col
        data_length = len(df[self.time_col])

        # Ensure all required columns exist
        for col in X_cols + [Y_col]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Create tensors and handle NaN values
        X = torch.stack(
            [torch.tensor(df[col], dtype=torch.float32) for col in X_cols], dim=1
        )
        Y = torch.tensor(df[Y_col], dtype=torch.float32).view(data_length, 1)

        return X, Y


class DataModule(L.LightningDataModule):
    """Data module that splits dataset into train, validation and test."""

    def __init__(
        self,
        data_directory: str,
        train_split: float = 0.7,
        val_split: float = 0.2,
        test_split: float = 0.1,
        batch_size: int = 32
    ) -> None:
        super().__init__()

        if round(sum([train_split, val_split, test_split]), 6) != 1:
            raise ValueError("All of train/val/test splits must sum up to 1.")

        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size

        self.dataset = LGHG2Dataset(data_directory=data_directory)

    def setup(self, stage: str) -> None:
        num_dataset = len(self.dataset)
        num_training_set = round(self.train_split * num_dataset)
        num_validation_set = round(self.val_split * num_dataset)
        num_test_set = round(self.test_split * num_dataset)

        self.training_set, self.validation_set, self.test_set = random_split(
            self.dataset, [num_training_set, num_validation_set, num_test_set]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.training_set, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_set, 
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )