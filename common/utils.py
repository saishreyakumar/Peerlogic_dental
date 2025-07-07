from datetime import datetime
from typing import Union

def calculate_age(date_of_birth: Union[str, datetime], reference_date: datetime = None) -> int:
    """Calculate age from date of birth (string or datetime object)"""
    if reference_date is None:
        reference_date = datetime.now()
    
    # Handle both string and datetime inputs
    if isinstance(date_of_birth, str):
        try:
            birth_date = datetime.strptime(date_of_birth, '%Y-%m-%d')
        except:
            return 30  # Default age if parsing fails
    else:
        birth_date = date_of_birth
    
    return (reference_date - birth_date).days // 365

def calculate_distance_score(patient_zip: str, practice_zip: str) -> float:
    """Calculate actual geographic distance between ZIP codes in miles"""
    try:
        import zipcodes
        from haversine import haversine, Unit
        import math
        
        # Get coordinate data for both ZIP codes
        patient_data = zipcodes.matching(patient_zip)
        practice_data = zipcodes.matching(practice_zip)
        
        # If either ZIP code is not found, return a moderate distance
        if not patient_data or not practice_data:
            return 100  # Default high distance for invalid ZIP codes
        
        # Extract coordinates with validation
        patient_lat = float(patient_data[0]['lat'])
        patient_long = float(patient_data[0]['long'])
        practice_lat = float(practice_data[0]['lat'])
        practice_long = float(practice_data[0]['long'])
        
        # Validate coordinates are within reasonable bounds
        if (abs(patient_lat) > 90 or abs(patient_long) > 180 or 
            abs(practice_lat) > 90 or abs(practice_long) > 180):
            return 100
        
        # Check for invalid coordinates (0,0 or NaN)
        if (patient_lat == 0 and patient_long == 0) or (practice_lat == 0 and practice_long == 0):
            return 100
        
        if (math.isnan(patient_lat) or math.isnan(patient_long) or 
            math.isnan(practice_lat) or math.isnan(practice_long)):
            return 100
        
        patient_coords = (patient_lat, patient_long)
        practice_coords = (practice_lat, practice_long)
        
        # Calculate distance using haversine formula
        distance_miles = haversine(patient_coords, practice_coords, unit=Unit.MILES)
        
        # Ensure distance is finite and reasonable (max 5000 miles for continental US)
        if math.isnan(distance_miles) or math.isinf(distance_miles) or distance_miles > 5000:
            return 100
        
        return round(max(0, distance_miles), 2)
        
    except Exception as e:
        # If any error occurs, return a moderate distance
        return 100

def predict_bucket(probability: float) -> str:
    """Convert Propensity-To-Show (PTS) to risk bucket
    
    Args:
        probability: Propensity-To-Show (0.0 = won't show, 1.0 = will definitely show)
    
    Returns:
        Risk bucket based on likelihood of showing up:
        - High risk: < 50% chance of showing up
        - Medium risk: 50-79% chance of showing up  
        - Low risk: ≥ 80% chance of showing up
    """
    if probability < 0.50:
        return "High"     # Low PTS = High risk of no-show
    elif probability < 0.80:
        return "Medium"   # Medium PTS = Medium risk of no-show
    else:
        return "Low"      # High PTS = Low risk of no-show 