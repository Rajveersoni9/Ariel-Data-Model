import numpy as np


class SergeiDataSmoother:
    def __init__(
        self, window_size: int, airs_first_channel: int = 39, airs_last_channel: int = 321
    ):
        self.airs_first_channel = airs_first_channel
        self.airs_last_channel = airs_last_channel
        self.window_size = window_size

    def smooth(self, signal: np.ndarray) -> np.ndarray:
        q = signal[
            :,
            self.airs_first_channel + 1 - self.window_size : self.airs_last_channel
            + 1
            + self.window_size,
        ]  # +1 to skip FGS1
        q = q / signal[
            :,
            self.airs_first_channel + 1 - self.window_size : self.airs_last_channel
            + 1
            + self.window_size,
        ].mean(axis=1, keepdims=True)
        q_coef = q.mean(axis=0)
        gauss_coefs = np.array(
            [0.01227215, 0.07819333, 0.23753036, 0.34400831, 0.23753036, 0.07819333, 0.01227215]
        )

        t_smooth = signal[
            :,
            self.airs_first_channel + 1 - self.window_size : self.airs_last_channel
            + 1
            + self.window_size,
        ].copy()
        for l in range(self.window_size, t_smooth.shape[1] - self.window_size):
            coefs = q_coef[l - self.window_size : l + self.window_size + 1] / q_coef[l]

            t_smooth[:, l] = np.dot(
                signal[
                    :,
                    self.airs_first_channel
                    + 1
                    - self.window_size
                    + l
                    - self.window_size : self.airs_first_channel
                    + 1
                    - self.window_size
                    + l
                    + self.window_size
                    + 1,
                ]
                * coefs,
                gauss_coefs,
            )

        if self.window_size > 0:
            t_smooth = t_smooth[:, self.window_size : -self.window_size][:, ::-1]
        else:
            t_smooth = t_smooth[:, ::-1]
        return np.concatenate([signal[:, 0:1], t_smooth], axis=1)
