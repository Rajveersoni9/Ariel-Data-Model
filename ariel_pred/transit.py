import numpy as np
from scipy.optimize import minimize


class WindowBasedPhaseDetector:
    def __init__(
        self,
        window_size=90,
        margin=30,
        scale=30,
        transit_begin_search_start=40,
        transit_begin_search_end=82,
        transit_end_search_start=105,
        transit_end_search_end=147,
        binning_factor=1,
    ):
        self.window_size = window_size // binning_factor
        self.margin = margin // binning_factor
        self.scale = scale // binning_factor
        self.transit_begin_search_start = transit_begin_search_start // binning_factor
        self.transit_begin_search_end = transit_begin_search_end // binning_factor
        self.transit_end_search_start = transit_end_search_start // binning_factor
        self.transit_end_search_end = transit_end_search_end // binning_factor

    def phase_detect(self, signal: np.ndarray) -> tuple[int, int, int, int]:
        assert len(signal.shape) == 1, "Expecting White Curve. Average over wavelengths first."
        phase1, phase2 = (1, 1), (1, 1)
        best_drop = 0

        for i in range(
            self.transit_begin_search_start * self.scale,
            self.transit_begin_search_end * self.scale,
        ):
            t1 = (
                signal[i - self.window_size : i + self.window_size].max()
                - signal[i - self.window_size : i + self.window_size].min()
            )
            if t1 > best_drop:
                i_max = (
                    i
                    - self.window_size
                    + np.argmax(signal[i - self.window_size : i + self.window_size])
                )
                i_min = (
                    i
                    - self.window_size
                    + np.argmin(signal[i - self.window_size : i + self.window_size])
                )

                if i_min < i_max:  # Here we're looking for a drop. If it's a rise, skip it.
                    continue

                best_drop = t1
                phase1 = (i_max - self.margin, i_min + self.margin)

        best_drop = 0
        for i in range(
            self.transit_end_search_start * self.scale,
            self.transit_end_search_end * self.scale,
        ):
            t1 = (
                signal[i - self.window_size : i + self.window_size].max()
                - signal[i - self.window_size : i + self.window_size].min()
            )
            if t1 > best_drop:
                i_max = (
                    i
                    - self.window_size
                    + np.argmax(signal[i - self.window_size : i + self.window_size])
                )
                i_min = (
                    i
                    - self.window_size
                    + np.argmin(signal[i - self.window_size : i + self.window_size])
                )

                if i_max < i_min:  # Here we're looking for a rise. If it's a drop, skip it.
                    continue

                best_drop = t1

                phase2 = (i_min - self.margin, i_max + self.margin)

        return int(phase1[0]), int(phase1[1]), int(phase2[0]), int(phase2[1])


class FunctionFittingBasedPhaseDetector:
    def __init__(self, window_size: int = 100, width: int = 300):
        self.window_size = window_size
        self.width = width

    def _get_middle_of_phases_estimate(self, data: np.ndarray) -> tuple[int, int]:
        kernel = np.ones(self.window_size) / self.window_size
        moving_average = np.convolve(data, kernel, mode="valid")
        diff = np.diff(moving_average)
        moving_average_diff = np.convolve(diff, kernel, mode="valid")
        min_idx = np.argmin(moving_average_diff)
        max_idx = np.argmax(moving_average_diff)
        min_idx_on_data = min_idx + self.window_size
        max_idx_on_data = max_idx + self.window_size
        return int(min_idx_on_data), int(max_idx_on_data)

    def _cost_function(self, params: tuple[float, float, float, float], data: np.ndarray, is_drop=True) -> float:
        t1, t2, a, b = params
        t1 = int(t1)
        t2 = int(t2)
        # Constrain Violation
        if t1 > t2:
            return (t1 - t2) * 1e9
        if is_drop and a < b:
            return (b - a) * 1e9
        if not is_drop and a > b:
            return (a - b) * 1e9
        y = np.full((data.shape[0],), a)
        y[t1:t2] = np.linspace(a, b, t2 - t1)
        y[t2:] = b
        cost = np.sum((data - y) ** 2)
        return cost

    def _get_phase_boundaries(self, data: np.ndarray, region_estimate: int, is_drop=True) -> tuple[int, int]:
        data = data[
            max(0, region_estimate - self.width) : min(len(data), region_estimate + self.width)
        ]
        initial_params = [
            max(0, self.width // 2),
            min(self.width * 3 // 2, len(data) - 1),
            np.mean(data[: max(1, self.width // 2)]),
            np.mean(data[min(len(data) - 1, self.width * 3 // 2) :]),
        ]
        bounds = [
            (1, len(data) - 2),
            (2, len(data) - 1),
            (min(data), max(data)),
            (min(data), max(data)),
        ]
        result = minimize(
            self._cost_function, initial_params, args=(data, is_drop), bounds=bounds, method="Nelder-Mead"
        )
        phase_begin, phase_end, _, _ = result.x
        phase_begin = int(phase_begin) + max(0, region_estimate - self.width)
        phase_end = int(phase_end) + max(0, region_estimate - self.width)
        return phase_begin, phase_end

    def phase_detect(self, data: np.ndarray) -> tuple[int, int, int, int]:
        assert len(data.shape) == 1, "Expecting White Curve. Average over wavelengths first."
        min_idx, max_idx = self._get_middle_of_phases_estimate(data)
        drop_begin, drop_end = self._get_phase_boundaries(data.copy(), min_idx, is_drop=True)
        rise_begin, rise_end = self._get_phase_boundaries(data.copy(), max_idx, is_drop=False)
        return drop_begin, drop_end, rise_begin, rise_end
