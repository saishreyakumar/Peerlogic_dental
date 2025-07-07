from fastapi import FastAPI, HTTPException
from common.models import (
    PredictionRequest, 
    PredictionResponse, 
    HealthResponse, 
    ModelInfoResponse
)
from core.prediction import prediction_service

app = FastAPI(title="Dental No-Show Prediction API")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Dental No-Show Prediction API", "status": "healthy"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=prediction_service.is_model_loaded()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_no_show(request: PredictionRequest):
    """
    Predict the likelihood of a patient not showing up for their appointment.
    
    Returns:
    - pts: Probability of no-show (0.0 to 1.0)
    - bucket: Risk category (High < 0.50, Medium 0.50-0.79, Low ≥ 0.80)
    """
    try:
        return prediction_service.predict(request)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get information about the loaded model"""
    try:
        info = prediction_service.get_model_info()
        return ModelInfoResponse(**info)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/example")
async def example_request():
    """Get an example prediction request"""
    return {
        "example_request": {
            "patient_id": 1,
            "phone_number": "+14155551234",
            "date_of_birth": "1985-03-15",
            "gender": "Female",
            "zip_code": "94102",
            "insurance_type": "PPO",
            "payment_on_file": True,
            "outstanding_balance": 125.50,
            "appointment_type": "Hygiene",
            "booking_channel": "Online",
            "copay_amount": 25.00,
            "practice_zip_code": "94102",
            "no_show_rate": 0.1,
            "cancellation_rate": 0.05,
            "total_appointments": 5,
            "call_count": 2,
            "avg_call_duration": 180.0,
            "recorded_calls": 1
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 