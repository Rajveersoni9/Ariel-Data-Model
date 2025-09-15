import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer

from ariel_pred.dataset import DataLoaderAndCalibrator
from ariel_pred.features import WavelengthsGroupsMultiplierFinder
from ariel_pred.models import SValuesCNNTrainer

app = typer.Typer()


@app.command()
def main(
    input_data_folder: str = "../data/raw",
    calibrated_data_folder: str = "../data/calibrated/full",
    submission_file: str = "./submission.csv",
    trained_model_folder: str = "../models/s_values_cnn",
    data_cutoff: int | None = None,
    binning: int = 4,
    device: str = "cpu",
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    force_retrain: bool = False,
    train_multiplier: float = 1e3,
    stop_at_calibration: bool = False,
    stop_at_training: bool = False,
):
    torch_device = torch.device(device)
    # Path setup
    input_data_path = Path(input_data_folder)
    calibrated_data_path = Path(calibrated_data_folder)
    trained_model_path = Path(trained_model_folder)

    if not input_data_path.exists():
        raise ValueError(f"Input data folder {input_data_path} does not exist")

    os.makedirs(calibrated_data_path, exist_ok=True)

    # Calibration and preprocessing
    data_loader = DataLoaderAndCalibrator(
        data_path=Path(input_data_path),
        output_path=Path(calibrated_data_path),
        force_recalibration=False,
        cut_airs_channels=True,
        binning=binning,
        n_jobs=4,
        data_cutoff=data_cutoff,
    )

    train_data, train_labels = data_loader.load_all_train_data()

    if stop_at_calibration:
        print("Stopping at calibration")
        return

    feature_extractor = WavelengthsGroupsMultiplierFinder()
    train_features = feature_extractor.extract_features(train_data, average_cross_groups=False)
    train_features = np.clip(train_features, a_min=0.0, a_max=10.0).transpose(
        (0, 2, 1)
    )  # (n_samples, n_channels, n_features)

    cnn_trainer = SValuesCNNTrainer(
        models_save_path=trained_model_path,
        device=torch_device,
        in_channels=train_features.shape[1],
        num_channels=283,
        train_multiplier=train_multiplier,
    )

    cnn_trainer.train(
        train_features,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        force_retrain=force_retrain,
        return_predictions=False,
    )
    if stop_at_training:
        print("Stopping at training")
        return

    # Prediction on test set
    test_data = data_loader.load_all_test_data()
    test_features = feature_extractor.extract_features(test_data, average_cross_groups=False)
    test_features = np.clip(test_features, a_min=0.0, a_max=10.0).transpose(
        (0, 2, 1)
    )  # (n_samples, n_channels, n_features)
    spectrum, sigma = cnn_trainer.predict(test_features)

    result = np.concatenate([spectrum, sigma], axis=1)
    assert result.shape == (test_data.shape[0], 283 * 2), (
        f"Result shape {result.shape} is not correct"
    )
    submission_df = pd.read_csv(input_data_path / "sample_submission.csv")
    submission_df.iloc[:, 1:] = result
    submission_df.to_csv(submission_file, index=False)
    print(f"Submission saved to {submission_file}")


if __name__ == "__main__":
    app()
