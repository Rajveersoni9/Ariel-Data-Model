import matplotlib.pyplot as plt
import numpy as np


def plot_white_curve(
    signal: np.ndarray, title: str = "White Curve", color: str = "blue", alpha: float = 1.0
) -> None:
    """
    Plots the white curve (average flux over all wavelengths) of the given signal.

    Args:
        signal (np.ndarray): 1D or 2D array with shape (num_time_steps) or (num_time_steps, num_wavelengths).
        title (str): Title of the plot. Default is "White Curve".

    Returns:
        None
    """
    assert len(signal.shape) <= 2, "Signal should be 1D or 2D array"
    if len(signal.shape) == 2:
        signal = signal.mean(axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(signal, color=color, alpha=alpha)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.grid()
