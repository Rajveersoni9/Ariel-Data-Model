from fastapi import APIRouter, HTTPException
import logging
from ..schemas.request_response import PredictRequest, PredictResponse
from ..services.inference import inference_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)

@router.post("/", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """
    Accepts 3D array of data and star info, and returns the predicted spectrum and sigma.
    """
    try:
        # Run the inference pipeline
        spectrum, sigma = inference_service.run_prediction(request.data, request.star_info)
        
        return PredictResponse(
            spectrum=spectrum,
            sigma=sigma
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
