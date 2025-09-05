import numpy as np


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
    ):
        self.window_size = window_size
        self.margin = margin
        self.scale = scale
        self.transit_begin_search_start = transit_begin_search_start
        self.transit_begin_search_end = transit_begin_search_end
        self.transit_end_search_start = transit_end_search_start
        self.transit_end_search_end = transit_end_search_end

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
