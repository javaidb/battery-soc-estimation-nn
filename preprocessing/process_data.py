import os
import sys

sys.path.append("../")
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
from typing import List, Dict, Any
from helpers import plot, config

# Get configuration for the data label
data_label = "mendeley"
data_config = config.config.data_labels.mendeley
cols = data_config.column_names
dirs = data_config.directories

# Setting up directories
raw_data_dir = dirs.raw_data
formatted_data_dir = dirs.formatted_data
processed_data_dir = dirs.processed_data

def get_pOCV_SOC_interp_fn(file_path: str, x_parameter: str) -> interp1d:
    """
    Create pseudo OCV-SOC interpolation function from slow discharge data.

    Parameters
    ----------
    file_path : str
        Path to slow discharge data
    x_parameter : str
        X parameter, either "OCV" or "SOC"

    Returns
    -------
    interp1d
        Interpolation function that takes OCV/SOC as input and returns the other
    """
    if x_parameter.upper() not in ["OCV", "SOC"]:
        raise ValueError("x_parameter needs to be one of ['OCV', 'SOC']")

    df = pd.read_csv(file_path)
    df = df[df[cols.current] < 0]
    df[cols.capacity] = df[cols.capacity] - df[cols.capacity].iloc[0]
    df[cols.soc] = 1 - abs(df[cols.capacity] / df[cols.capacity].iloc[-1])

    match x_parameter.upper():
        case "OCV":
            return interp1d(df[cols.voltage], df[cols.soc])

        case "SOC":
            return interp1d(df[cols.soc], df[cols.voltage])

    return


def estimate_soc_and_ocv(
    df: pd.DataFrame, get_soc_fn: interp1d, get_ocv_fn: interp1d
) -> pd.DataFrame:
    """
    Create a new SOC column with estimated values.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    get_soc_fn : interp1d
        OCV-SOC interpolation function

    Returns
    -------
    pd.DataFrame
        Time series data with estimated SOC values
    """

    df[cols.capacity] = df[cols.capacity] - df[cols.capacity].iloc[0]

    final_soc = float(get_soc_fn(df[cols.voltage].iloc[-1]))
    est_total_capacity = abs(df[cols.capacity].iloc[-1]) / (1 - final_soc)
    df[cols.soc] = 1 - abs(df[cols.capacity]) / est_total_capacity
    df[cols.ocv] = df[cols.soc].apply(get_ocv_fn).apply(float)

    return df


def generate_and_save_plot(
    data_df: pd.DataFrame,
    save_file_path: str,
    fig_title: str = "",
) -> None:
    """
    Generate a plot for parsed raw data and saves figure as png.

    Parameters
    ----------
    data_df : pd.DataFrame
        Parsed data for plotting.
    save_file_path : str
        File path of saved figure.
    fig_title : str, optional
        Figure title, by default ''
    """

    fig, _ = plot(
        xy_data=[
            {
                "x_data": data_df[cols.time],
                "y_data": data_df[cols.voltage],
                "label": "Voltage",
            },
            {
                "x_data": data_df[cols.time],
                "y_data": data_df[cols.current],
                "label": "Current",
                "plot_num": 2,
            },
            {
                "x_data": data_df[cols.time],
                "y_data": data_df[cols.temperature],
                "label": "Temperature",
                "plot_num": 3,
            },
            {
                "x_data": data_df[cols.time],
                "y_data": data_df[cols.capacity],
                "label": "Capacity",
                "plot_num": 4,
            },
            {
                "x_data": data_df[cols.time],
                "y_data": data_df[cols.soc],
                "label": "SOC",
                "plot_num": 5,
            },
        ],
        x_label=cols.time,
        y_label={
            1: cols.voltage,
            2: cols.current,
            3: cols.temperature,
            4: cols.capacity,
            5: cols.soc,
        },
        title=fig_title,
        fig_size=(10, 12.5),
        show_plt=False,
    )

    fig.savefig(save_file_path)

    return


if __name__ == "__main__":
    temperatures = filter(
        lambda folder: "degC" in folder, os.listdir(formatted_data_dir)
    )

    for T in temperatures:
        parsed_data_T_directory = f"{formatted_data_dir}/{T}"
        processed_data_T_directory = f"{processed_data_dir}/{T}"

        if not os.path.exists(processed_data_T_directory):
            os.makedirs(processed_data_T_directory)

        C20_file_name = next(
            filter(
                lambda f: f.endswith(".csv") and "C20" in f,
                os.listdir(parsed_data_T_directory),
            )
        )
        C20_file_path = f"{parsed_data_T_directory}/{C20_file_name}"
        get_soc = get_pOCV_SOC_interp_fn(C20_file_path, x_parameter="OCV")
        get_ocv = get_pOCV_SOC_interp_fn(C20_file_path, x_parameter="SOC")

        csv_files = filter(
            lambda f: f.endswith(".csv") and "C20" not in f,
            os.listdir(parsed_data_T_directory),
        )

        for csv_file in csv_files:
            try:
                csv_file_name = csv_file.split("_parsed.csv")[0]

                df = pd.read_csv(f"{parsed_data_T_directory}/{csv_file}")
                df = estimate_soc_and_ocv(df, get_soc_fn=get_soc, get_ocv_fn=get_ocv)
                df.to_parquet(
                    f"{processed_data_T_directory}/{csv_file_name}.parquet", index=False
                )

                generate_and_save_plot(
                    data_df=df,
                    save_file_path=f"{processed_data_T_directory}/{csv_file_name}_plot.png",
                    fig_title=f"{csv_file_name} @ {T}",
                )
                plt.close()
                print(f"Processed {csv_file_name} @ {T}")

            except Exception as e:
                print(f"Error processing {csv_file} - {e}")