import os
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from ariel_pred.dataset import DataLoaderAndCalibrator
from ariel_pred.features import WavelengthsGroupsMultiplierFinder

app = typer.Typer()


@app.command()
def main(
    input_data_folder: str = "../data/raw_subset",
    calibrated_data_folder: str = "../data/processed/s_value/calibrated",
    submission_file: str = "./submission.csv",
    binning: int = 4,
    sigma: float = 0.000715,
    multiplier: float = 0.945,
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

    test_data = data_loader.load_all_test_data()

    feature_extractor = WavelengthsGroupsMultiplierFinder()

    features = feature_extractor.extract_features(test_data)
    features = np.clip(features, a_min=0.0, a_max=10.0)

    result = np.concatenate([features * multiplier, np.full(features.shape, sigma)], axis=1)
    assert result.shape == (test_data.shape[0], 283 * 2), f"Result shape {result.shape} is not correct"
    submission_df = pd.read_csv(input_data_path / "sample_submission.csv")
    submission_df.iloc[:, 1:] = result
    submission_df.to_csv(submission_file, index=False)
    print(f"Submission saved to {submission_file}")


if __name__ == "__main__":
    app()
