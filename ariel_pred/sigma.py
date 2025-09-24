import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from ariel_pred.preprocessing import SGSmoothing
from ariel_pred.transit import FunctionFittingBasedPhaseDetector


class SpectrumVariationScaler:
    def __init__(
        self, mean_sigma: float = 0.0007, num_channels: int = 283, min_sigma: float = 1e-6
    ):
        self.mean_sigma = mean_sigma
        self.num_channels = num_channels
        self.min_sigma = min_sigma

    def get_sigma(self, spectrum: np.ndarray):
        assert spectrum.ndim == 2
        per_planet_sigma = (spectrum.std(axis=1) / spectrum.std(axis=1).mean()) * self.mean_sigma
        per_planet_sigma = per_planet_sigma.clip(min=self.min_sigma)
        return np.repeat(per_planet_sigma[:, np.newaxis], repeats=self.num_channels, axis=1)


class OGSignalVarBasedSigmaCalculator:
    def __init__(
        self,
        mean_fgs_sigma: float = 8.5e-4,
        mean_airs_sigma: float = 5.0e-4,
        fgs_min_sigma: float = 1e-6,
        airs_min_sigma: float = 1e-6,
        smoother: SGSmoothing = SGSmoothing(window_size=150, poly_order=2),
        transit_finder: FunctionFittingBasedPhaseDetector = FunctionFittingBasedPhaseDetector(),
        eps: float = 1e-12,
    ):
        self.mean_fgs_sigma = mean_fgs_sigma
        self.mean_airs_sigma = mean_airs_sigma
        self.fgs_min_sigma = fgs_min_sigma
        self.airs_min_sigma = airs_min_sigma
        self.smoother = smoother
        self.transit_finder = transit_finder
        self.eps = eps

    def _get_airs_sigma(self, data: np.ndarray, transit_locations: np.ndarray):
        airs_sigma = np.zeros(data.shape[0])

        for i in tqdm(range(data.shape[0]), desc="Calculating AIRS sigma"):
            signal = data[i]
            t1, t2, t3, t4 = transit_locations[i]

            airs = signal[:, 1:].mean(axis=1)

            oot = np.concatenate([airs[:t1], airs[t4:]], axis=0)
            inn = airs[t2:t3]

            var_oot = np.var(oot)
            var_inn = np.var(inn)
            oot_mean = np.mean(oot)

            airs_sigma[i] = np.sqrt((var_inn / len(inn) + var_oot / len(oot)).mean()) / max(
                oot_mean, self.eps
            )

        airs_sigma_scaled = airs_sigma / np.mean(airs_sigma) * self.mean_airs_sigma
        airs_sigma_scaled = airs_sigma_scaled.clip(min=self.airs_min_sigma)
        return airs_sigma_scaled

    def _get_fgs_sigma(self, data: np.ndarray, transit_locations: np.ndarray):
        fgs_sigma = np.zeros(data.shape[0])

        for i in tqdm(range(data.shape[0]), desc="Calculating FGS sigma"):
            signal = data[i]
            t1, t2, t3, t4 = transit_locations[i]

            fgs = signal[:, 0]
            oot = np.concatenate([fgs[:t1], fgs[t4:]])
            inn = fgs[t2:t3]

            var_oot = np.var(oot)
            var_inn = np.var(inn)
            oot_mean = np.mean(oot)

            fgs_sigma[i] = np.sqrt(var_inn / len(inn) + var_oot / len(oot)) / max(
                oot_mean, self.eps
            )

        fgs_sigma_scaled = fgs_sigma / np.mean(fgs_sigma) * self.mean_fgs_sigma
        fgs_sigma_scaled = fgs_sigma_scaled.clip(min=self.fgs_min_sigma)
        return fgs_sigma_scaled

    def get_sigma(self, data: np.ndarray):
        assert data.ndim == 3 and data.shape[2] == 283

        smoothed_white_curves = np.zeros((data.shape[0], data.shape[1]))
        for i in range(data.shape[0]):
            smoothed_white_curves[i] = self.smoother.smooth(data[i, :, 1:].mean(axis=1))
        transit_locations = self.transit_finder.phase_detect_multiple_planets(
            smoothed_white_curves
        )

        fgs_sigma_scaled = self._get_fgs_sigma(data.copy(), transit_locations)
        airs_sigma_scaled = self._get_airs_sigma(data.copy(), transit_locations)

        sigma = np.concatenate(
            [
                fgs_sigma_scaled[:, np.newaxis],
                np.repeat(airs_sigma_scaled[:, np.newaxis], 282, axis=1),
            ],
            axis=1,
        )

        return sigma


class WhiteCurveVarBasedSigmaCalculator:
    def __init__(
        self,
        mean_fgs_sigma: float = 8.5e-4,
        mean_airs_sigma: float = 5.0e-4,
        fgs_min_sigma: float = 1e-6,
        airs_min_sigma: float = 1e-6,
        smoother: SGSmoothing = SGSmoothing(window_size=150, poly_order=2),
        transit_finder: FunctionFittingBasedPhaseDetector = FunctionFittingBasedPhaseDetector(),
        eps: float = 1e-12,
    ):
        self.mean_fgs_sigma = mean_fgs_sigma
        self.mean_airs_sigma = mean_airs_sigma
        self.fgs_min_sigma = fgs_min_sigma
        self.airs_min_sigma = airs_min_sigma
        self.smoother = smoother
        self.transit_finder = transit_finder
        self.eps = eps

    def _get_sigma(self, data: np.ndarray, transit_locations: np.ndarray):
        sigma = np.zeros(data.shape[0])

        for i in tqdm(range(data.shape[0]), desc="Calculating AIRS sigma"):
            signal = data[i]
            t1, t2, t3, t4 = transit_locations[i]

            unsmoothed_white_curve = signal[:, 1:].mean(axis=1)

            oot = np.concatenate(
                [unsmoothed_white_curve[:t1], unsmoothed_white_curve[t4:]], axis=0
            )
            inn = unsmoothed_white_curve[t2:t3]

            var_oot = np.var(oot)
            var_inn = np.var(inn)
            oot_mean = np.mean(oot)

            sigma[i] = np.sqrt((var_inn / len(inn) + var_oot / len(oot)).mean()) / max(
                oot_mean, self.eps
            )

        sigma = np.repeat(sigma[:, np.newaxis], 283, axis=1)
        sigma[:, 0] = sigma[:, 0] / np.mean(sigma[:, 0]) * self.mean_fgs_sigma
        sigma[:, 0] = sigma[:, 0].clip(min=self.fgs_min_sigma)
        sigma[:, 1:] = sigma[:, 1:] / np.mean(sigma[:, 1:]) * self.mean_airs_sigma
        sigma[:, 1:] = sigma[:, 1:].clip(min=self.airs_min_sigma)
        return sigma

    def get_sigma(self, data: np.ndarray):
        assert data.ndim == 3 and data.shape[2] == 283, (
            "Data should be of shape (num_planets, num_time_steps, 283)"
        )

        smoothed_white_curves = np.zeros((data.shape[0], data.shape[1]))
        for i in range(data.shape[0]):
            smoothed_white_curves[i] = self.smoother.smooth(data[i, :, 1:].mean(axis=1))
        transit_locations = self.transit_finder.phase_detect_multiple_planets(
            smoothed_white_curves
        )

        sigma = self._get_sigma(data.copy(), transit_locations)

        return sigma


class LROnVariousFeaturesSigmaCalculator:
    def __init__(
        self,
        mean_fgs_sigma: float = 9.0e-4,
        mean_airs_sigma: float = 5.5e-4,
        fgs_min_sigma: float = 1e-6,
        airs_min_sigma: float = 1e-6,
        eps: float = 1e-12,
        smoother: SGSmoothing = SGSmoothing(window_size=150, poly_order=2),
        transit_finder: FunctionFittingBasedPhaseDetector = FunctionFittingBasedPhaseDetector(),
    ):
        self.mean_fgs_sigma = mean_fgs_sigma
        self.mean_airs_sigma = mean_airs_sigma
        self.fgs_min_sigma = fgs_min_sigma
        self.airs_min_sigma = airs_min_sigma
        self.eps = eps
        self.smoother = smoother
        self.transit_finder = transit_finder
        self.models = []

    def _create_model(self):
        model = LinearRegression()
        return model

    def _get_white_curve_var(
        self,
        data: np.ndarray,
        transit_locations: np.ndarray,
        mean_sigma: float = 5.5e-4,
        min_sigma: float = 1e-6,
        eps=1e-12,
    ) -> np.ndarray:
        sigma = np.zeros(data.shape[0])

        for i in tqdm(range(data.shape[0]), desc="Calculating AIRS sigma"):
            signal = data[i]
            t1, t2, t3, t4 = transit_locations[i]

            airs = signal[:, 1:].mean(axis=1)

            oot = np.concatenate([airs[:t1], airs[t4:]], axis=0)
            inn = airs[t2:t3]

            var_oot = np.var(oot)
            var_inn = np.var(inn)
            oot_mean = np.mean(oot)

            sigma[i] = np.sqrt((var_inn / len(inn) + var_oot / len(oot)).mean()) / max(
                oot_mean, eps
            )

        sigma_scaled = sigma / np.mean(sigma) * mean_sigma
        sigma_scaled = sigma_scaled.clip(min=min_sigma)
        return sigma_scaled

    def _extract_features(self, data: np.ndarray, spectrum: np.ndarray):
        spectrum_mean = spectrum.mean(axis=1)
        spectrum_std = spectrum.std(axis=1)
        # spectrum_var = spectrum.var(axis=1)

        smoothed_white_curves = np.zeros((data.shape[0], data.shape[1]))
        for i in range(data.shape[0]):
            smoothed_white_curves[i] = self.smoother.smooth(data[i, :, 1:].mean(axis=1))
        transit_locations = self.transit_finder.phase_detect_multiple_planets(
            smoothed_white_curves
        )

        white_curve_var = self._get_white_curve_var(
            data=data.copy(),
            transit_locations=transit_locations,
            mean_sigma=self.mean_airs_sigma,
            eps=self.eps,
        )

        features = np.concatenate(
            [
                spectrum_mean[:, np.newaxis],
                spectrum_std[:, np.newaxis],
                # spectrum_var[:, np.newaxis],
                white_curve_var[:, np.newaxis],
            ],
            axis=1,
        )
        return features

    def _get_optimal_sigma(self, spectrum: np.ndarray, labels: np.ndarray):
        def cost_function(params, x, mu):
            sigma = params[0]
            return (
                0.5 * (np.log(2 * np.pi) + np.log(sigma**2) + ((x - mu) ** 2) / (sigma**2))
            ).mean()

        optimal_sigma = np.zeros(spectrum.shape[0])

        for i in tqdm(range(spectrum.shape[0])):
            res = minimize(
                cost_function, [0.00060], args=(labels[i], spectrum[i]), bounds=[(1e-6, 0.01)]
            )
            optimal_sigma[i] = res.x[0]
        
        return optimal_sigma

    def train(self, data: np.ndarray, spectrum: np.ndarray, labels: np.ndarray):
        X = self._extract_features(data, spectrum)
        y = self._get_optimal_sigma(spectrum, labels)

        kf = KFold(n_splits=5, shuffle=True)
        cv_preds = np.zeros_like(y)
        self.models = []
        for i, (train_index, val_index) in enumerate(kf.split(X)):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model = self._create_model()
            model.fit(X_train, y_train)
            self.models.append(model)
            cv_preds[val_index] = model.predict(X_val)

            print(
                f"Fold {i + 1}, Validation RMSE: {np.sqrt(np.mean((y_val - cv_preds[val_index]) ** 2))}"
            )

        sigma = np.repeat(cv_preds[:, np.newaxis], 283, axis=1)
        sigma[:, 0] = sigma[:, 0] / max(np.mean(sigma[:, 0]), self.eps) * self.mean_fgs_sigma
        sigma[:, 0] = sigma[:, 0].clip(min=self.fgs_min_sigma)
        sigma[:, 1:] = sigma[:, 1:] / max(np.mean(sigma[:, 1:]), self.eps) * self.mean_airs_sigma
        sigma[:, 1:] = sigma[:, 1:].clip(min=self.airs_min_sigma)
        return sigma

    def get_sigma(self, data: np.ndarray, spectrum: np.ndarray):
        assert data.ndim == 3 and data.shape[2] == 283, (
            "Data should be of shape (num_planets, num_time_steps, 283)"
        )
        assert len(self.models) > 0, "Models are not trained. Please call the train method first."

        X = self._extract_features(data, spectrum)
        preds = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            preds[:, i] = model.predict(X)
        sigma_pred = preds.mean(axis=1)

        sigma = np.repeat(sigma_pred[:, np.newaxis], 283, axis=1)
        sigma[:, 0] = sigma[:, 0] / max(np.mean(sigma[:, 0]), self.eps) * self.mean_fgs_sigma
        sigma[:, 0] = sigma[:, 0].clip(min=self.fgs_min_sigma)
        sigma[:, 1:] = sigma[:, 1:] / max(np.mean(sigma[:, 1:]), self.eps) * self.mean_airs_sigma
        sigma[:, 1:] = sigma[:, 1:].clip(min=self.airs_min_sigma)
        return sigma
