from functools import partial

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from tqdm.auto import tqdm

from ariel_pred.models import TransitMultiplicationFactorFinder
from ariel_pred.preprocessing import SGSmoothing
from ariel_pred.transit import FunctionFittingBasedPhaseDetector, WindowBasedPhaseDetector


class SignalPoly:
    def __init__(self, deg):
        self.deg = deg
        self.poly_features = PolynomialFeatures(degree=deg, include_bias=True)
        self.model = LinearRegression()

    def fit(self, x, y):
        X_poly = self.poly_features.fit_transform(np.array(x).reshape(-1, 1))
        self.model.fit(X_poly, y)

    def predict(self, x):
        X_poly = self.poly_features.transform(np.array(x).reshape(-1, 1))
        return self.model.predict(X_poly)


class SergeiOldFeaturesExtractor:
    def __init__(
        self, phase_detector: WindowBasedPhaseDetector, sg_window: int = 300, sg_poly: int = 3
    ):
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.phase_detector = phase_detector

    def _smooth_data(self, signal: np.ndarray) -> np.ndarray:
        return savgol_filter(signal, self.sg_window, self.sg_poly)

    def _f_coef(self, px, y, s):
        q = np.abs(px - y * s).mean()
        return q

    def _try_s_alpenglow2(self, px, y, y2, y3, s):
        q = np.abs(px - y * (y2 + y3 * s)).mean()
        return q

    def _calibrate_train_alpenglow(
        self,
        signal: np.ndarray,
        p0: int,
        p1: int,
        p2: int,
        p3: int,
        max_deg: int = 6,
        min_deg: int = 1,
    ):
        best_deg, best_score, best_s, best_poly = 1, 1e12, 0, None
        out = np.concatenate((np.arange(p0), np.arange(p3, signal.shape[0])))
        x, y = out, signal[out]
        x2 = np.arange(p1, p2)
        y2 = signal[p1:p2]

        x = np.concatenate((x, x2))
        y = np.concatenate((y, y2))
        y2 = np.ones_like(y)
        y3 = np.zeros_like(y)
        y3[out.shape[0] :] = 1.0
        for deg in range(min_deg, max_deg + 1):
            p = SignalPoly(deg)
            p.fit(x[: len(out)], y[: len(out)])
            px = p.predict(x)

            f = partial(self._try_s_alpenglow2, px, y, y2, y3)
            r = minimize_scalar(f)
            s = r.x  # type: ignore
            q = r.fun  # type: ignore
            if q < best_score:
                best_score = q
                best_poly = p
                best_s = s
                best_deg = deg

        return best_s, best_deg, best_poly, best_score

    def extract_features(self, data: np.ndarray) -> np.ndarray:
        assert len(data.shape) == 3, (
            "Expecting 3D array: (num_planets, num_time_steps, num_wavelengths)"
        )
        assert data.shape[2] == 283, "Expecting the AIRS channels to be cut"
        feats = np.zeros((len(data), 283, 9))
        max_degs = np.zeros(len(data)).astype(int)
        for i in tqdm(range(len(data))):
            train = data[i]

            p0, p1, p2, p3 = 0, 0, 0, 0

            ranges = [(0, 283)]
            for r1, r2 in ranges:
                signal = train[:, r1:r2].mean(axis=1)
                signal = self._smooth_data(signal)
                p0, p1, p2, p3 = self.phase_detector.phase_detect(signal)
                s, max_deg, _, _ = self._calibrate_train_alpenglow(signal, p0, p1, p2, p3)
                max_degs[i] = max_deg
                feats[i, r1:r2, 0] = s

            x_out = np.concatenate((np.arange(p0), np.arange(p3, signal.shape[0])))  # type: ignore
            x_in = np.arange(p1, p2)

            ranges = [(0, 133), (133, 283)]
            for r1, r2 in ranges:
                signal = train[:, r1:r2].mean(axis=1)
                signal = self._smooth_data(signal)
                s, _, _, _ = self._calibrate_train_alpenglow(
                    signal, p0, p1, p2, p3, max_deg=max_degs[i]
                )
                feats[i, r1:r2, 1] = s

            ranges = [(0, 62), (62, 133), (133, 200), (200, 283)]
            for r1, r2 in ranges:
                signal = train[:, r1:r2].mean(axis=1)
                signal = self._smooth_data(signal)
                s, _, p, _ = self._calibrate_train_alpenglow(signal, p0, p1, p2, p3, max_degs[i])
                feats[i, r1:r2, 2] = s

                px = p.predict(np.arange(data.shape[1]))  # type: ignore

                for l in range(r1, r2):  # noqa: E741
                    signal = self._smooth_data(train[:, l])

                    f = partial(self._f_coef, px[x_out], signal[x_out])
                    r = minimize_scalar(f)
                    a = r.x  # type: ignore

                    f = partial(self._f_coef, px[x_in], signal[x_in])
                    r = minimize_scalar(f)
                    b = r.x  # type: ignore

                    feats[i, l, 7] = 1.0 - a / b

            ranges = [(j * 50, 50 + j * 50) for j in range(283 // 50 + 1)]
            for r1, r2 in ranges:
                signal = train[:, r1:r2].mean(axis=1)
                signal = self._smooth_data(signal)
                s, _, p, _ = self._calibrate_train_alpenglow(signal, p0, p1, p2, p3, 4)
                feats[i, r1:r2, 3] = s
                px = p.predict(np.arange(data.shape[1]))  # type: ignore
                if r2 > 283:
                    r2 = 283
                for l in range(r1, r2):  # noqa: E741
                    signal = self._smooth_data(train[:, l])

                    f = partial(self._f_coef, px[x_out], signal[x_out])
                    r = minimize_scalar(f)
                    a = r.x  # type: ignore

                    f = partial(self._f_coef, px[x_in], signal[x_in])
                    r = minimize_scalar(f)
                    b = r.x  # type: ignore

                    feats[i, l, 8] = 1.0 - a / b

            ranges = [
                (0, 1),
                (1, 16),
                (16, 31),
                (31, 46),
                (46, 62),
                (62, 77),
                (77, 97),
                (97, 112),
                (112, 133),
                (133, 149),
                (149, 168),
                (168, 185),
                (185, 200),
                (200, 215),
                (215, 240),
                (240, 255),
                (255, 283),
            ]
            for r1, r2 in ranges:
                signal = train[:, r1:r2].mean(axis=1)
                signal = self._smooth_data(signal)
                s, _, _, _ = self._calibrate_train_alpenglow(signal, p0, p1, p2, p3, 3)
                feats[i, r1:r2, 4] = s

            ranges = [(j * 8, 8 + j * 8) for j in range(283 // 8 + 1)]
            for r1, r2 in ranges:
                signal = train[:, r1:r2].mean(axis=1)
                signal = self._smooth_data(signal)
                s, _, _, _ = self._calibrate_train_alpenglow(signal, p0, p1, p2, p3, 3)
                feats[i, r1:r2, 5] = s

            q_train = self._smooth_data(train.transpose(1, 0))
            q_train = q_train / q_train.mean(axis=1, keepdims=True)

            px = p.predict(np.arange(data.shape[1]))  # type: ignore

            for l in range(283):  # noqa: E741
                signal = q_train[l]

                f = partial(self._f_coef, px[x_out], signal[x_out])
                r = minimize_scalar(f)
                a = r.x  # type: ignore

                f = partial(self._f_coef, px[x_in], signal[x_in])
                r = minimize_scalar(f)
                b = r.x  # type: ignore
                feats[i, l, 6] = 1.0 - a / b

        return feats


class WavelengthsGroupsMultiplierFinder:
    def __init__(
        self,
        transit_finder: FunctionFittingBasedPhaseDetector = FunctionFittingBasedPhaseDetector(),
        multiplier: TransitMultiplicationFactorFinder = TransitMultiplicationFactorFinder(),
        smoother: SGSmoothing | None = SGSmoothing(window_size=150, poly_order=2),
    ):
        self.transit_finder = transit_finder
        self.multiplier = multiplier
        self.smoother = smoother

    def extract_features(
        self,
        all_data: np.ndarray,
        wavelengths_groups: list[int] = [1, 2, 4, 8, 16, 32, 64],
        weights: list[float] = [1, 1, 1, 1, 1, 1, 1],
        average_cross_groups: bool = True,
        return_transit_locations: bool = False,
    ) -> np.ndarray:
        assert len(all_data.shape) == 3, (
            "Expecting 3D array: (num_planets, num_time_steps, num_wavelengths)"
        )
        assert all_data.shape[2] == 283, "Expecting the AIRS channels to be cut"
        assert len(wavelengths_groups) == len(weights), (
            "wavelengths_groups and weights must have the same length"
        )

        num_planets = all_data.shape[0]
        num_wavelengths = all_data.shape[2]

        num_groups = len(wavelengths_groups)
        feats = np.zeros((num_planets, num_wavelengths, num_groups))
        ranges_list = [np.array_split(np.arange(num_wavelengths), cb) for cb in wavelengths_groups]

        transit_location = (
            np.zeros((num_planets, 4), dtype=int) if return_transit_locations else None
        )

        for i in tqdm(range(num_planets)):
            data = all_data[i]
            t1, t2, t3, t4 = self.transit_finder.phase_detect(
                self.smoother.smooth(data.mean(axis=1)) if self.smoother else data.mean(axis=1)
            )

            if return_transit_locations:
                transit_location[i] = (t1, t2, t3, t4)  # type: ignore

            for j, ranges in enumerate(ranges_list):
                for r in ranges:
                    signal = data[:, r].mean(axis=1)
                    signal = self.smoother.smooth(signal) if self.smoother else signal
                    s = self.multiplier.predict(signal, t1, t2, t3, t4)
                    feats[i, r, j] = s

        if average_cross_groups:
            feats = np.average(feats, axis=2, weights=weights)

        if return_transit_locations:
            return feats, transit_location  # type: ignore

        return feats


class PerChannelFluctuationsFinder:
    def __init__(self, smoother: SGSmoothing | None = SGSmoothing(window_size=150, poly_order=2)):
        self.smoother = smoother

    def extract_features(self, all_data: np.ndarray, transit_locations: np.ndarray) -> np.ndarray:
        assert len(all_data.shape) == 3, (
            "Expecting 3D array: (num_planets, num_time_steps, num_wavelengths)"
        )
        assert len(transit_locations.shape) == 2 and transit_locations.shape[1] == 4, (
            "Expecting transit_locations to be of shape (num_planets, 4)"
        )
        assert all_data.shape[0] == transit_locations.shape[0], (
            "num_planets in all_data and transit_locations must be the same"
        )

        num_planets = all_data.shape[0]
        num_wavelengths = all_data.shape[2]

        feats = np.zeros((num_planets, num_wavelengths))

        for i in tqdm(range(num_planets)):
            signal = all_data[i].copy()
            t1, t2, t3, t4 = transit_locations[i]

            for w in range(signal.shape[1]):  # noqa: E741
                signal[:, w] = (
                    self.smoother.smooth(signal[:, w]) if self.smoother else signal[:, w]
                )
            x_out = np.concatenate((np.arange(t1), np.arange(t4, signal.shape[0])))
            feats[i] = (signal[x_out].mean(axis=0) - signal[x_out].mean()) / signal[x_out].mean()

        return feats
