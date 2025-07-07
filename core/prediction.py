import joblib
import pandas as pd
from common.models import PredictionRequest, PredictionResponse
from core.features import create_features
from common.utils import predict_bucket

class PredictionService:
    """Service for handling no-show predictions"""
    
    def __init__(self, model_path: str = 'models/prediction_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.feature_columns = None
        self.encoders = None
        self.load_model()
    
    def load_model(self):
        import os
        """Load the trained model and encoders"""
        try:
            if os.path.exists(self.model_path):
                model_artifacts = joblib.load(self.model_path)
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = model_artifacts['model']
            self.feature_columns = model_artifacts['feature_columns']
            self.encoders = model_artifacts['encoders']
            print(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            print(f"Model file not found: {self.model_path}")
            print("Please train the model first using model_training.py")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make prediction for a patient"""
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded. Please train the model first.")
        
        try:
            # Create features
            feature_df = create_features(request, self.encoders)
            
            # Ensure correct column order
            feature_df = feature_df.reindex(columns=self.feature_columns, fill_value=0)
            
            # Make prediction
            no_show_probability = self.model.predict_proba(feature_df)[0][1]
            # Convert no-show probability to Propensity-To-Show (PTS)
            pts = 1.0 - no_show_probability
            bucket = predict_bucket(pts)
            
            return PredictionResponse(pts=round(pts, 4), bucket=bucket)
            
        except Exception as e:
            raise RuntimeError(f"Prediction error: {str(e)}")
    
    def get_model_info(self) -> dict:
        """Get model information"""
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        return {
            "model_type": "RandomForestClassifier",
            "feature_count": len(self.feature_columns),
            "features": self.feature_columns,
            "encoders": list(self.encoders.keys())
        }

# Global instance
prediction_service = PredictionService() 