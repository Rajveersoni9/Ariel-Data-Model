import itertools
from typing import Literal

from astropy.stats import sigma_clip
import numpy as np
import pandas as pd
from pqdm.threads import pqdm

from ariel_pred.config import Config


# Based on Vitaly Kudelya's Baseline
class DataLoaderAndCalibrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.adc_info = pd.read_csv(f"{self.cfg.DATA_PATH}/adc_info.csv")

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
