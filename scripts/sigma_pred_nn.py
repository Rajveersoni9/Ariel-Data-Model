import os
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from ariel_pred.dataset import DataLoaderAndCalibrator
from ariel_pred.features import WavelengthsGroupsMultiplierFinder
from ariel_pred.modeling.nn_with_sigma_pred import ResNetAndCNNWithSigmaModelTrainer
from ariel_pred.sigma import OGSignalVarBasedSigmaCalculator

app = typer.Typer()


@app.command()
def main(
    input_data_folder: str = "../data/raw",
    calibrated_data_folder: str = "../data/calibrated/full",
    submission_file: str = "./submission.csv",
    trained_model_folder: str = "../models/sigma_pred_nn",
    binning: int = 4,
    mean_fgs_sigma: float = 9.0e-4,
    mean_airs_sigma: float = 6.5e-4,
    epochs: int = 300,
    stop_at_calibration: bool = False,
    stop_at_training: bool = False,
):
    # Path setup
    input_data_path = Path(input_data_folder)
    calibrated_data_path = Path(calibrated_data_folder)

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
    )

    train_data, train_labels = data_loader.load_all_train_data()

    if stop_at_calibration:
        print("Stopping at calibration")
        return

    features_extractor = WavelengthsGroupsMultiplierFinder()

    train_features = features_extractor.extract_features(
        train_data,
        average_cross_groups=False,
        wavelengths_groups=[1, 2, 4, 8, 16, 32, 64],
        weights=[1, 1, 1, 1, 1, 1, 1],
    )

    model_data = train_features.transpose(0, 2, 1)
    train_star_info_df = pd.read_csv(input_data_path / "train_star_info.csv")
    train_star_info = train_star_info_df[["Rs", "i"]].values

    trainer = ResNetAndCNNWithSigmaModelTrainer(
        Path(trained_model_folder),
        n_splits=5,
        num_channels=model_data.shape[1],
        wavelengths=model_data.shape[2],
        num_star_features=train_star_info.shape[1],
    )

    trainer.train(model_data * 1e3, train_star_info, train_labels * 1e3, epochs=epochs)

    if stop_at_training:
        print("Stopping at training")
        return

    test_data = data_loader.load_all_test_data()

    test_features = features_extractor.extract_features(
        test_data,
        average_cross_groups=False,
        wavelengths_groups=[1, 2, 4, 8, 16, 32, 64],
        weights=[1, 1, 1, 1, 1, 1, 1],
    )

    test_model_data = test_features.transpose(0, 2, 1)
    test_star_info_df = pd.read_csv(input_data_path / "test_star_info.csv")
    test_star_info = test_star_info_df[["Rs", "i"]].values
    predictions, sigma = trainer.predict(test_model_data * 1e3, test_star_info)
    predictions = predictions / 1e3
    sigma = sigma / 1e3
    spectrum = predictions

    old_sigma_calculator = OGSignalVarBasedSigmaCalculator(
        mean_fgs_sigma=mean_fgs_sigma,
        mean_airs_sigma=mean_airs_sigma,
    )
    old_sigma = old_sigma_calculator.get_sigma(test_data)
    
    merged_sigma = (old_sigma + np.repeat(sigma[:, np.newaxis], 283, axis=1)) / 2

    result = np.concatenate([spectrum, merged_sigma], axis=1)
    assert result.shape == (test_data.shape[0], 283 * 2), (
        f"Result shape {result.shape} is not correct"
    )
    submission_df = pd.read_csv(input_data_path / "sample_submission.csv")
    submission_df.iloc[:, 1:] = result
    submission_df.to_csv(submission_file, index=False)
    print(f"Submission saved to {submission_file}")


if __name__ == "__main__":
    app()
