from pydantic import BaseModel, validator
from typing import List, Any

class PredictRequest(BaseModel):
    data: List[List[List[float]]]  # 3D array representation
    star_info: List[float]         # Expected to be length 2: [Rs, i]

    @validator('star_info')
    def validate_star_info(cls, v):
        if len(v) != 2:
            raise ValueError('star_info must be a list of exactly 2 float values: [Rs, i]')
        return v

    @validator('data')
    def validate_data(cls, v):
        if not v or not v[0] or not v[0][0]:
            raise ValueError('data must be a non-empty 3D array')
        return v

class PredictResponse(BaseModel):
    spectrum: List[float]
    sigma: List[float]
