from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Configuration for the project paths and settings."""

    PROJ_ROOT = Path(__file__).resolve().parents[1]
    ROOT_DATA_DIR = PROJ_ROOT / "data"
    RAW_DATA_DIR = ROOT_DATA_DIR / "raw"
    INTERIM_DATA_DIR = ROOT_DATA_DIR / "interim"
    PROCESSED_DATA_DIR = ROOT_DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = ROOT_DATA_DIR / "external"

    MODELS_DIR = PROJ_ROOT / "models"

    REPORTS_DIR = PROJ_ROOT / "reports"
    FIGURES_DIR = REPORTS_DIR / "figures"

    DATA_PATH = PROJ_ROOT / "data" / "raw_subset"

    PLANET_IDS = []


class CalibrationConfig:
    def __init__(
        self,
        data_path: Path,
        binning: int = 1,
        airs_lower_channel: int = 0,
        airs_upper_channel: int = 356,
        preprocessing_n_jobs: int = 4,
    ):
        self.DATA_PATH = data_path
        self.SENSOR_CONFIG = {
            "AIRS-CH0": {
                "raw_shape": [11250, 32, 356],
                "calibrated_shape": [1, 32, airs_upper_channel - airs_lower_channel],
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
        self.AIRS_LOWER_CHANNEL = airs_lower_channel
        self.AIRS_UPPER_CHANNEL = airs_upper_channel
        self.PREPROCESSING_N_JOBS = preprocessing_n_jobs


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
