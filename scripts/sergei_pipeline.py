import os
from pathlib import Path

import numpy as np
import torch
import typer

from ariel_pred.config import CalibrationConfig
from ariel_pred.dataset import DataLoaderAndCalibrator, LabelsLoader
from ariel_pred.features import SergeiOldFeaturesExtractor
from ariel_pred.models import SegeiOldCNNTrainer
from ariel_pred.preprocessing import SergeiDataSmoother
from ariel_pred.transit import WindowBasedPhaseDetector

app = typer.Typer()


@app.command()
def main(
    input_data_folder: str = "../data/raw_subset",
    output_data_folder: str = "../data/processed/sergei",
    output_model_folder: str = "../models/sergei",
    stop_at_calibration: bool = False,
    stop_at_feature_extraction: bool = False,
    stop_at_model_training: bool = False,
):
    # Path setup
    input_data_path = Path(input_data_folder)
    output_data_path = Path(output_data_folder)
    output_model_path = Path(output_model_folder)

    if not input_data_path.exists():
        raise ValueError(f"Input data folder {input_data_path} does not exist")

    os.makedirs(output_data_path, exist_ok=True)
    os.makedirs(output_model_path, exist_ok=True)

    calibrated_train_data_file = output_data_path / "calibrated_train_data.npy"
    calibrated_test_data_file = output_data_path / "calibrated_test_data.npy"
    train_features_file = output_data_path / "train_features.npy"
    test_features_file = output_data_path / "test_features.npy"
    train_labels_file = output_data_path / "train_labels.npy"

    # Calibration and preprocessing
    calibration_config = CalibrationConfig(
        data_path=input_data_path,
        binning=1,
        airs_lower_channel=0,
        airs_upper_channel=356,
        preprocessing_n_jobs=4,
    )
    signal_processor = DataLoaderAndCalibrator(cfg=calibration_config)
    data_smoother = SergeiDataSmoother(window_size=3)
    if not calibrated_train_data_file.exists():
        print("Calibrating and saving train data...")
        train_data = signal_processor.process_all_data("train")
        train_data = np.array([data_smoother.smooth(signal) for signal in train_data])
        np.save(calibrated_train_data_file, train_data)
    else:
        print("Loading calibrated train data...")
        train_data = np.load(calibrated_train_data_file, allow_pickle=True)

    test_data = signal_processor.process_all_data("test")
    test_data = np.array([data_smoother.smooth(signal) for signal in test_data])
    np.save(calibrated_test_data_file, test_data)
    if stop_at_calibration:
        return

    # Feature Extraction
    transit_detector = WindowBasedPhaseDetector()
    feature_extractor = SergeiOldFeaturesExtractor(phase_detector=transit_detector)
    if not train_features_file.exists():
        print("Extracting and saving train features...")
        train_features = feature_extractor.extract_features(train_data)
        np.save(train_features_file, train_features)
    else:
        print("Loading train features...")
        train_features = np.load(train_features_file, allow_pickle=True)
    if stop_at_feature_extraction:
        return

    print("Extracting and saving test features...")
    test_features = feature_extractor.extract_features(test_data)
    np.save(test_features_file, test_features)

    # Labels loading
    if not train_labels_file.exists():
        print("Loading and saving train labels...")
        labels_loader = LabelsLoader(base_data_path=str(input_data_path))
        train_labels = labels_loader.load_labels()
        np.save(train_labels_file, train_labels)
    else:
        print("Loading train labels...")
        train_labels = np.load(train_labels_file, allow_pickle=True)

    # Model Training
    cnn_train_data = train_features.transpose(0, 2, 1)
    cnn_test_data = test_features.transpose(0, 2, 1)
    if len(os.listdir(output_model_path)) == 0:
        print("Training model...")
        model_trainer = SegeiOldCNNTrainer(device=torch.device("mps"))
        model_trainer.train(cnn_train_data, train_labels, output_model_path)
    if stop_at_model_training:
        return


if __name__ == "__main__":
    app()
