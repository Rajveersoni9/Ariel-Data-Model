import numpy as np
from pathlib import Path
import pandas as pd

class SubmissionMaker:
    def __init__(self, input_data_path: Path, submission_file_path: Path, data_cutoff: int | None = None):
        self.input_data_path = input_data_path
        self.submission_file_path = submission_file_path
        self.data_cutoff = data_cutoff
        
        
    def create_constant_sigma_submission(self, spectras: np.ndarray, constant_sigma: float = 0.001):
        