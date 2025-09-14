from pathlib import Path

import numpy as np
import pandas as pd
import torch


def get_device_from_str(device_str: str) -> torch.device:
    if device_str not in ["cpu", "cuda", "mps"]:
        raise ValueError("Device must be one of 'cpu', 'cuda', or 'mps'")
    if device_str == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS device is not available on this machine.")
    if device_str == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device is not available on this machine.")
    return torch.device(device_str)


def test_submission_maker(
    input_data_path: Path, predictions: np.ndarray, sigma: float, multiplier: float = 1.0
) -> pd.DataFrame:
    result = np.concatenate(
        [predictions.clip(min=0) * multiplier, np.full(predictions.shape, sigma)], axis=1
    )
    assert result.shape == (predictions.shape[0], 283 * 2), (
        f"Result shape {result.shape} is not correct"
    )
    submission_df = pd.read_csv(input_data_path / "sample_submission.csv")
    submission_df.iloc[:, 1:] = result
    return submission_df


def train_submission_maker(
    input_data_path: Path, predictions: np.ndarray, sigma: float, multiplier: float = 1.0
) -> pd.DataFrame:
    result = np.concatenate(
        [predictions.clip(min=0.0) * multiplier, np.full(predictions.shape, sigma)], axis=1
    )
    assert result.shape == (predictions.shape[0], 283 * 2), (
        f"Result shape {result.shape} is not correct"
    )
    sample_submission_df = pd.read_csv(input_data_path / "sample_submission.csv")
    planet_ids = pd.read_csv(input_data_path / "train.csv")["planet_id"].values

    submission_df = pd.DataFrame(
        np.concatenate([planet_ids[:, np.newaxis], result], axis=1), # type: ignore
        columns=sample_submission_df.columns,
    )
    return submission_df
