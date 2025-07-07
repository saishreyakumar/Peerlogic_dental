from pydantic import BaseModel
from typing import Optional, List

class PredictionRequest(BaseModel):
    """Request model for no-show prediction"""
    # Patient information
    patient_id: int
    phone_number: str
    date_of_birth: str  # Format: YYYY-MM-DD
    gender: str
    zip_code: str
    insurance_type: str
    payment_on_file: bool
    outstanding_balance: float
    
    # Appointment information
    appointment_type: str
    booking_channel: str
    copay_amount: float
    practice_zip_code: str
    
    # Historical data (optional, defaults to 0)
    no_show_rate: Optional[float] = 0.0
    cancellation_rate: Optional[float] = 0.0
    total_appointments: Optional[int] = 0
    call_count: Optional[int] = 0
    avg_call_duration: Optional[float] = 0.0
    recorded_calls: Optional[int] = 0

class PredictionResponse(BaseModel):
    """Response model for no-show prediction"""
    pts: float
    bucket: str

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_type: str
    feature_count: int
    features: List[str]
    encoders: List[str] 