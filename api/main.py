from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import predict

app = FastAPI(
    title="Ariel Exoplanet Predictor API",
    description="Backend API for Ariel Exoplanet predictions using CNN and feature extractors.",
    version="1.0.0",
)

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.get("/")
async def root():
    return {"message": "Welcome to the Ariel Exoplanet Predictor API."}
