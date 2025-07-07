"""
Unified Model Training & Optimization for Dental No-Show Prediction

This script automatically:
1. Loads and prepares data
2. Tests multiple classifiers using hyperparameter optimization
3. Selects the best performing model
4. Trains the final model and saves it for API use
"""

import warnings
import sys
import os

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training import run_training_pipeline, ModelRegistry

warnings.filterwarnings('ignore')


def main():
    """Unified training and optimization pipeline"""
    print("Dental No-Show Prediction - Unified Training & Optimization")
    print("=" * 70)
    
    # Get all available model configurations for reference
    model_configs = ModelRegistry.get_all_configs()
    
    print(f"Testing {len(model_configs)} different classifiers:")
    for config in model_configs:
        print(f"  - {config.name}")
    
    print(f"\nRunning comprehensive optimization...")
    
    # Run comprehensive optimization and training using the new pipeline
    model, auc_score, results = run_training_pipeline(
        mode='full',  # Full optimization mode
        n_trials=50   # 50 trials per classifier
    )
    
    if model is not None and results:
        # Get best model info from results
        best_model_name = max(results.keys(), key=lambda k: results[k]['best_score'])
        best_score = results[best_model_name]['best_score']
        
        print(f"\nTRAINING COMPLETE!")
        print(f"Best Classifier: {best_model_name}")
        print(f"Final AUC-ROC Score: {best_score:.4f}")
        print(f"Model saved to: models/prediction_model.pkl")
        print(f"Ready for API deployment!")
    else:
        print("\nTraining failed. Please check the data and try again.")


if __name__ == "__main__":
    main() 