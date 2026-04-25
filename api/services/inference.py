import numpy as np
import logging
from typing import List, Tuple
import os
import sys

# Ensure ariel_pred can be imported if running from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import existing ML models/classes
try:
    from ariel_pred.modeling.s_values_cnn_with_star_info import SValuesCNNWithStarInfoTrainer
    from ariel_pred.features import WavelengthsGroupsMultiplierFinder
    from ariel_pred.sigma import LROnVariousFeaturesSigmaCalculator
    MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Failed to import ML classes from ariel_pred: {e}. Inference will run in mock mode.")
    MODELS_AVAILABLE = False


logger = logging.getLogger(__name__)

class InferenceService:
    _instance = None

    def __new__(cls):
        """ Singleton pattern to ensure models are loaded only once """
        if cls._instance is None:
            cls._instance = super(InferenceService, cls).__new__(cls)
            cls._instance.initialize_models()
        return cls._instance

    def initialize_models(self):
        """ Load and instantiate models locally """
        logger.info("Initializing models...")
        if not MODELS_AVAILABLE:
            logger.info("Running without actual model files (Mock Mode).")
            self.trainer = None
            self.feature_finder = None
            self.sigma_calculator = None
            return

        try:
            # NOTE: These paths/initializations might need adjusting based on exact requirements.
            # Assuming these classes can be initialized with default parameters or point to specific config paths.
            self.feature_finder = WavelengthsGroupsMultiplierFinder()
            
            # SValuesCNNWithStarInfoTrainer requires a weights_dir path
            weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'lr_sigma_with_nn'))
            self.trainer = SValuesCNNWithStarInfoTrainer(
                weights_dir=weights_path, 
                num_channels=7, 
                num_star_features=2
            ) 
            
            self.sigma_calculator = LROnVariousFeaturesSigmaCalculator()
            
            logger.info("Models initialized successfully.")
        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}")
            raise e

    def run_prediction(self, data: List[List[List[float]]], star_info: List[float]) -> Tuple[List[float], List[float]]:
        """
        Executes the prediction pipeline.
        Steps:
        1. Convert input to NumPy
        2. Run feature extraction
        3. Predict spectrum
        4. Predict sigma
        """
        if not MODELS_AVAILABLE:
            # Return dummy data matching the 283-values requirement for UI testing if models aren't ready
            logger.info("Returning dummy prediction data.")
            dummy_spectrum = np.random.uniform(0.1, 0.9, 283).tolist()
            dummy_sigma = np.random.uniform(0.01, 0.05, 283).tolist()
            return dummy_spectrum, dummy_sigma

        try:
            # 1. Convert input to NumPy
            np_data = np.array(data)
            np_star_info = np.array(star_info)
            
            # np_data is already 3D from the request: (1, time_steps, 283)
            # np_star_info is 1D: (2,), needs batch dimension
            np_star_info = np.expand_dims(np_star_info, axis=0)
            
            # 2. Feature Extraction
            # The model expects 7 channels, so we do not average across groups.
            # output shape: (batch, wavelengths, num_channels) -> transpose to (batch, num_channels, wavelengths)
            features = self.feature_finder.extract_features(np_data, average_cross_groups=False)
            features = features.transpose(0, 2, 1)
            
            # 3. Model Prediction (Spectrum)
            spectrum_pred = self.trainer.predict(features, np_star_info)
            
            # 4. Sigma Prediction
            try:
                # This requires self.sigma_calculator to have been trained or models loaded
                sigma_pred = self.sigma_calculator.get_sigma(np_data, spectrum_pred)
            except AssertionError as err:
                logger.warning(f"Sigma calculation skipped due to missing models: {err}. Using dummy sigma.")
                sigma_pred = np.random.uniform(0.01, 0.05, 283).reshape(1, 283)

            # Ensure output is flattened to a simple list of 283 values
            spectrum = spectrum_pred.flatten().tolist()
            sigma = sigma_pred.flatten().tolist()
            
            if len(spectrum) != 283 or len(sigma) != 283:
                logger.warning(f"Expected 283 values, but got {len(spectrum)} for spectrum and {len(sigma)} for sigma.")

            return spectrum, sigma

        except Exception as e:
            logger.error(f"Prediction pipeline failed: {str(e)}")
            raise e

# Instantiate the singleton so it can be imported by the router
inference_service = InferenceService()
