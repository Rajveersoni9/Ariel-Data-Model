import itertools
import os
from pathlib import Path
from typing import Literal

from astropy.stats import sigma_clip
import numpy as np
import pandas as pd
from pqdm.threads import pqdm

from ariel_pred.config import CalibrationConfig


# Based on Vitaly Kudelya's Baseline
class OldDataLoaderAndCalibrator:
    def __init__(self, cfg: CalibrationConfig, data_cutoff: int | None = None):
        self.cfg = cfg
        self.adc_info = pd.read_csv(f"{self.cfg.DATA_PATH}/adc_info.csv")
        self.data_cutoff = data_cutoff

    def _apply_linear_corr(self, linear_corr, signal):
        linear_corr_flipped = np.flip(linear_corr, axis=0)
        corrected_signal = signal.copy()

        for x, y in itertools.product(range(signal.shape[1]), range(signal.shape[2])):
            poly = np.poly1d(linear_corr_flipped[:, x, y])
            corrected_signal[:, x, y] = poly(corrected_signal[:, x, y])

        return corrected_signal

    def _calibrate_single_signal(self, planet_id, dataset, sensor):
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]

        signal = pd.read_parquet(
            f"{self.cfg.DATA_PATH}/{dataset}/{planet_id}/{sensor}_signal_0.parquet"
        ).to_numpy()
        dark = pd.read_parquet(
            f"{self.cfg.DATA_PATH}/{dataset}/{planet_id}/{sensor}_calibration_0/dark.parquet"
        ).to_numpy()
        dead = pd.read_parquet(
            f"{self.cfg.DATA_PATH}/{dataset}/{planet_id}/{sensor}_calibration_0/dead.parquet"
        ).to_numpy()
        flat = pd.read_parquet(
            f"{self.cfg.DATA_PATH}/{dataset}/{planet_id}/{sensor}_calibration_0/flat.parquet"
        ).to_numpy()
        linear_corr = (
            pd.read_parquet(
                f"{self.cfg.DATA_PATH}/{dataset}/{planet_id}/{sensor}_calibration_0/linear_corr.parquet"
            )
            .values.astype(np.float64)
            .reshape(sensor_cfg["linear_corr_shape"])
        )

        signal = signal.reshape(sensor_cfg["raw_shape"])
        gain = self.adc_info[f"{sensor}_adc_gain"].iloc[0]
        offset = self.adc_info[f"{sensor}_adc_offset"].iloc[0]
        signal = signal / gain + offset

        hot = sigma_clip(dark, sigma=5, maxiters=5).mask  # type: ignore

        if sensor == "AIRS-CH0":
            signal = signal[:, :, self.cfg.AIRS_LOWER_CHANNEL : self.cfg.AIRS_UPPER_CHANNEL]
            linear_corr = linear_corr[
                :, :, self.cfg.AIRS_LOWER_CHANNEL : self.cfg.AIRS_UPPER_CHANNEL
            ]
            dark = dark[:, self.cfg.AIRS_LOWER_CHANNEL : self.cfg.AIRS_UPPER_CHANNEL]
            dead = dead[:, self.cfg.AIRS_LOWER_CHANNEL : self.cfg.AIRS_UPPER_CHANNEL]
            flat = flat[:, self.cfg.AIRS_LOWER_CHANNEL : self.cfg.AIRS_UPPER_CHANNEL]
            hot = hot[:, self.cfg.AIRS_LOWER_CHANNEL : self.cfg.AIRS_UPPER_CHANNEL]

        base_dt, increment = sensor_cfg["dt_pattern"]
        dt = np.ones(len(signal)) * base_dt
        dt[1::2] += increment

        signal = signal.clip(0)
        signal = self._apply_linear_corr(linear_corr, signal)
        signal -= dark * dt[:, np.newaxis, np.newaxis]

        flat = flat.reshape(sensor_cfg["calibrated_shape"])
        flat[dead.reshape(sensor_cfg["calibrated_shape"])] = np.nan
        flat[hot.reshape(sensor_cfg["calibrated_shape"])] = np.nan

        signal = signal / flat

        return signal

    def _preprocess_calibrated_signal(self, calibrated_signal, sensor):
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]
        binning = sensor_cfg["binning"]

        if sensor == "AIRS-CH0":
            signal_roi = calibrated_signal[:, 10:22, :]
        elif sensor == "FGS1":
            signal_roi = calibrated_signal[:, 10:22, 10:22]
            signal_roi = signal_roi.reshape(signal_roi.shape[0], -1)

        mean_signal = np.nanmean(signal_roi, axis=1)  # type: ignore

        cds_signal = mean_signal[1::2] - mean_signal[0::2]

        n_bins = cds_signal.shape[0] // binning
        binned = np.array(
            [cds_signal[j * binning : (j + 1) * binning].mean(axis=0) for j in range(n_bins)]
        )

        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))

        return binned

    def _process_planet_sensor(self, args):
        planet_id, sensor, dataset = args["planet_id"], args["sensor"], args["dataset"]
        calibrated = self._calibrate_single_signal(planet_id, dataset, sensor)
        preprocessed = self._preprocess_calibrated_signal(calibrated, sensor)
        return preprocessed

    def process_all_data(self, dataset: Literal["train", "test"]) -> np.ndarray:
        """
        Process and preprocess signals for all planets and sensors.

        Returns data of shape (num_planets, num_time_bins, total_channels)

        Currently, num_time_bins is 187 (notice the binning) and total channels is 283 with the first being FGS1 and the rest being AIRS-CH0.

        Returns:
            np.ndarray: Preprocessed signals with shape (num_planets, num_time_bins, total_channels)
        """

        # NOTE: The order of sensors is important here for the shape of the output array.
        # NOTE: Currently, process only the first observation (index 0) of each planet.
        planet_ids = pd.read_csv(f"{self.cfg.DATA_PATH}/{dataset}_star_info.csv")[
            "planet_id"
        ].values.astype(int)
        if self.data_cutoff is not None:
            planet_ids = planet_ids[: self.data_cutoff]

        args_fgs1 = [
            dict(planet_id=planet_id, dataset=dataset, sensor="FGS1") for planet_id in planet_ids
        ]

        preprocessed_fgs1 = pqdm(
            args_fgs1, self._process_planet_sensor, n_jobs=self.cfg.PREPROCESSING_N_JOBS
        )

        args_airs_ch0 = [
            dict(planet_id=planet_id, dataset=dataset, sensor="AIRS-CH0")
            for planet_id in planet_ids
        ]
        preprocessed_airs_ch0 = pqdm(
            args_airs_ch0, self._process_planet_sensor, n_jobs=self.cfg.PREPROCESSING_N_JOBS
        )

        preprocessed_signal = np.concatenate(
            [np.stack(preprocessed_fgs1), np.stack(preprocessed_airs_ch0)], axis=2
        )
        return preprocessed_signal


class LabelsLoader:
    def __init__(self, base_data_path: str, data_cutoff: int | None = None):
        self.base_data_path = Path(base_data_path)
        self.data_cutoff = data_cutoff

    def load_labels(self) -> np.ndarray:
        labels_df = pd.read_csv(self.base_data_path / "train.csv")
        labels = labels_df.drop(columns=["planet_id"]).to_numpy()
        if self.data_cutoff is not None:
            labels = labels[: self.data_cutoff]
        return labels


class DataLoaderAndCalibrator:
    def __init__(
        self,
        data_path: Path,
        output_path: Path,
        force_recalibration: bool = False,
        data_cutoff: int | None = None,
        binning: int = 4,
        cut_airs_channels: bool = True,
        n_jobs: int = 4,
    ):
        assert data_path.exists(), f"Data path {data_path} does not exist."
        self.data_path = data_path
        self.adc_info = pd.read_csv(data_path / "adc_info.csv")
        self.data_cutoff = data_cutoff
        os.makedirs(output_path, exist_ok=True)
        self.train_data_file = output_path / "calibrated_train_data.npy"
        self.test_data_file = output_path / "calibrated_test_data.npy"
        self.force_recalibration = force_recalibration
        self.binning = binning
        self.cut_airs_channels = cut_airs_channels
        self.n_jobs = n_jobs
        self.airs_first_channel_index = 39
        self.airs_last_channel_index = 321
        self.sensor_config = {
            "AIRS-CH0": {
                "raw_shape": [11250, 32, 356],
                "calibrated_shape": [1, 32, 282 if cut_airs_channels else 356],
                "linear_corr_shape": (6, 32, 356),
                "dt_pattern": (0.1, 4.5),
                "binning": binning,
            },
            "FGS1": {
                "raw_shape": [135000, 32, 32],
                "calibrated_shape": [1, 32, 32],
                "linear_corr_shape": (6, 32, 32),
                "dt_pattern": (0.1, 0.1),
                "binning": binning * 12,
            },
        }

    def _apply_linear_corr(self, linear_corr, signal):
        linear_corr_flipped = np.flip(linear_corr, axis=0)
        corrected_signal = signal.copy()

        for x, y in itertools.product(range(signal.shape[1]), range(signal.shape[2])):
            poly = np.poly1d(linear_corr_flipped[:, x, y])
            corrected_signal[:, x, y] = poly(corrected_signal[:, x, y])

        return corrected_signal

    def _calibrate_single_signal(self, planet_id, dataset, sensor):
        sensor_cfg = self.sensor_config[sensor]

        signal = pd.read_parquet(
            f"{self.data_path}/{dataset}/{planet_id}/{sensor}_signal_0.parquet"
        ).to_numpy()
        dark = pd.read_parquet(
            f"{self.data_path}/{dataset}/{planet_id}/{sensor}_calibration_0/dark.parquet"
        ).to_numpy()
        dead = pd.read_parquet(
            f"{self.data_path}/{dataset}/{planet_id}/{sensor}_calibration_0/dead.parquet"
        ).to_numpy()
        flat = pd.read_parquet(
            f"{self.data_path}/{dataset}/{planet_id}/{sensor}_calibration_0/flat.parquet"
        ).to_numpy()
        linear_corr = (
            pd.read_parquet(
                f"{self.data_path}/{dataset}/{planet_id}/{sensor}_calibration_0/linear_corr.parquet"
            )
            .values.astype(np.float64)
            .reshape(sensor_cfg["linear_corr_shape"])
        )

        signal = signal.reshape(sensor_cfg["raw_shape"])
        gain = self.adc_info[f"{sensor}_adc_gain"].iloc[0]
        offset = self.adc_info[f"{sensor}_adc_offset"].iloc[0]
        signal = signal / gain + offset

        hot = sigma_clip(dark, sigma=5, maxiters=5).mask  # type: ignore

        if sensor == "AIRS-CH0" and self.cut_airs_channels:
            signal = signal[:, :, self.airs_first_channel_index : self.airs_last_channel_index]
            linear_corr = linear_corr[
                :, :, self.airs_first_channel_index : self.airs_last_channel_index
            ]
            dark = dark[:, self.airs_first_channel_index : self.airs_last_channel_index]
            dead = dead[:, self.airs_first_channel_index : self.airs_last_channel_index]
            flat = flat[:, self.airs_first_channel_index : self.airs_last_channel_index]
            hot = hot[:, self.airs_first_channel_index : self.airs_last_channel_index]

        base_dt, increment = sensor_cfg["dt_pattern"]
        dt = np.ones(len(signal)) * base_dt
        dt[1::2] += increment

        signal = signal.clip(0)
        signal = self._apply_linear_corr(linear_corr, signal)
        signal -= dark * dt[:, np.newaxis, np.newaxis]

        flat = flat.reshape(sensor_cfg["calibrated_shape"])
        flat[dead.reshape(sensor_cfg["calibrated_shape"])] = np.nan
        flat[hot.reshape(sensor_cfg["calibrated_shape"])] = np.nan

        signal = signal / flat

        return signal

    def _preprocess_calibrated_signal(self, calibrated_signal, sensor):
        sensor_cfg = self.sensor_config[sensor]
        binning = sensor_cfg["binning"]

        if sensor == "AIRS-CH0":
            signal_roi = calibrated_signal[:, 10:22, :]
        elif sensor == "FGS1":
            signal_roi = calibrated_signal[:, 10:22, 10:22]
            signal_roi = signal_roi.reshape(signal_roi.shape[0], -1)

        mean_signal = np.nanmean(signal_roi, axis=1)  # type: ignore

        cds_signal = mean_signal[1::2] - mean_signal[0::2]

        n_bins = cds_signal.shape[0] // binning
        binned = np.array(
            [cds_signal[j * binning : (j + 1) * binning].mean(axis=0) for j in range(n_bins)]
        )

        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))

        return binned

    def _process_planet_sensor(self, args):
        planet_id, sensor, dataset = args["planet_id"], args["sensor"], args["dataset"]
        calibrated = self._calibrate_single_signal(planet_id, dataset, sensor)
        preprocessed = self._preprocess_calibrated_signal(calibrated, sensor)
        return preprocessed

    def _process_all_data(self, dataset: Literal["train", "test"]) -> np.ndarray:
        """
        Process and preprocess signals for all planets and sensors.

        Returns data of shape (num_planets, num_time_bins, total_channels)

        Currently, num_time_bins is 187 (notice the binning) and total channels is 283 with the first being FGS1 and the rest being AIRS-CH0.

        Returns:
            np.ndarray: Preprocessed signals with shape (num_planets, num_time_bins, total_channels)
        """

        # NOTE: The order of sensors is important here for the shape of the output array.
        # NOTE: Currently, process only the first observation (index 0) of each planet.
        planet_ids = pd.read_csv(f"{self.data_path}/{dataset}_star_info.csv")[
            "planet_id"
        ].values.astype(int)
        if self.data_cutoff is not None:
            planet_ids = planet_ids[: self.data_cutoff]

        args_fgs1 = [
            dict(planet_id=planet_id, dataset=dataset, sensor="FGS1") for planet_id in planet_ids
        ]

        preprocessed_fgs1 = pqdm(args_fgs1, self._process_planet_sensor, n_jobs=self.n_jobs)

        args_airs_ch0 = [
            dict(planet_id=planet_id, dataset=dataset, sensor="AIRS-CH0")
            for planet_id in planet_ids
        ]
        preprocessed_airs_ch0 = pqdm(
            args_airs_ch0, self._process_planet_sensor, n_jobs=self.n_jobs
        )

        preprocessed_signal = np.concatenate(
            [np.stack(preprocessed_fgs1), np.stack(preprocessed_airs_ch0)], axis=2
        )
        return preprocessed_signal

    def _load_train_labels(self) -> np.ndarray:
        labels_df = pd.read_csv(self.data_path / "train.csv")
        labels = labels_df.drop(columns=["planet_id"]).to_numpy()
        if self.data_cutoff is not None:
            labels = labels[: self.data_cutoff]
        return labels

    def load_all_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.train_data_file.exists() or self.force_recalibration:
            print("Calibrating and saving train data...")
            train_data = self._process_all_data("train")
            np.save(self.train_data_file, train_data)
        else:
            print("Loading calibrated train data...")
            train_data = np.load(self.train_data_file, allow_pickle=True)
            if self.data_cutoff is not None:
                train_data = train_data[: self.data_cutoff]
        labels = self._load_train_labels()
        assert len(train_data) == len(labels), (
            "Mismatch between data and labels lengths. Please recheck."
        )
        assert train_data.shape[2] == 283 if self.cut_airs_channels else 357, (
            "Unexpected number of channels in train data. This is probably due to previously saved data "
            "with different channel cutting settings. Please delete the existing calibrated data file or set "
            "force_recalibration to True."
        )
        return train_data, labels

    def load_all_test_data(self) -> np.ndarray:
        test_data = self._process_all_data("test")
        np.save(self.test_data_file, test_data)
        assert test_data.shape[2] == 283 if self.cut_airs_channels else 357, (
            "Unexpected number of channels in test data. Unknown reason. Please recheck."
        )
        return test_data
