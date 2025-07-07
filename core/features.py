import pandas as pd
from typing import List
from common.models import PredictionRequest
from common.utils import calculate_age, calculate_distance_score

def safe_transform(encoder, value, default_value=0):
    """Safely transform a categorical value, returning default if unseen"""
    try:
        return encoder.transform([value])[0]
    except ValueError:
        # Return default encoding for unseen labels
        return default_value

def create_features(request: PredictionRequest, encoders: dict) -> pd.DataFrame:
    """Create features from prediction request"""
    
    # Calculate derived features
    age = calculate_age(request.date_of_birth)
    age_risk = 1 if 16 <= age <= 55 else 0
    distance_score = calculate_distance_score(request.zip_code, request.practice_zip_code)
    high_distance = 1 if distance_score >= 50 else 0
    booking_risk = 1 if request.booking_channel in ['Online', 'SMS'] else 0
    insurance_risk = 1 if request.insurance_type == 'Self-Pay' else 0
    payment_risk = 1 if (not request.payment_on_file and request.outstanding_balance > 0) else 0
    appointment_risk = 1 if request.appointment_type not in ['Restorative', 'Root Canal', 'Extraction'] else 0
    
    # Create feature vector
    features = {
        'age': age,
        'age_risk': age_risk,
        'distance_score': distance_score,
        'high_distance': high_distance,
        'booking_risk': booking_risk,
        'insurance_risk': insurance_risk,
        'payment_risk': payment_risk,
        'appointment_risk': appointment_risk,
        'no_show_rate': request.no_show_rate,
        'cancellation_rate': request.cancellation_rate,
        'total_appointments': request.total_appointments,
        'call_count': request.call_count,
        'avg_call_duration': request.avg_call_duration,
        'recorded_calls': request.recorded_calls,
        'outstanding_balance': request.outstanding_balance,
        'copay_amount': request.copay_amount,
        'gender_encoded': safe_transform(encoders['gender'], request.gender, 0),
        'insurance_encoded': safe_transform(encoders['insurance'], request.insurance_type, 0),
        'booking_encoded': safe_transform(encoders['booking'], request.booking_channel, 0),
        'appointment_encoded': safe_transform(encoders['appointment'], request.appointment_type, 0)
    }
    
    return pd.DataFrame([features])

def get_feature_names() -> List[str]:
    """Get list of all feature names"""
    return [
        'age', 'age_risk', 'distance_score', 'high_distance',
        'booking_risk', 'insurance_risk', 'payment_risk', 'appointment_risk',
        'no_show_rate', 'cancellation_rate', 'total_appointments',
        'call_count', 'avg_call_duration', 'recorded_calls',
        'outstanding_balance', 'copay_amount',
        'gender_encoded', 'insurance_encoded', 'booking_encoded', 'appointment_encoded'
    ] 