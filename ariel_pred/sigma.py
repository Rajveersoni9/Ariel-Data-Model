import numpy as np


class SpectrumVariationScaler:
    def __init__(self, mean_sigma: float = 0.0007, num_channels: int = 283):
        self.mean_sigma = mean_sigma
        self.num_channels = num_channels

    def get_sigma(self, spectrum: np.ndarray):
        assert spectrum.ndim == 2
        per_planet_sigma = (spectrum.std(axis=1) / spectrum.std(axis=1).mean()) * self.mean_sigma
        return np.repeat(per_planet_sigma[:, np.newaxis], repeats=self.num_channels, axis=1)
