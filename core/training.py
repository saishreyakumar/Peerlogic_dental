#!/usr/bin/env python3
"""
Comprehensive No-Show Prediction Training System
Combines basic training and advanced optimization in one modular framework.
"""

import pandas as pd
import numpy as np
import optuna
import joblib
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from common.utils import calculate_age, calculate_distance_score, predict_bucket

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')


class DataLoader:
    """Handles all data loading operations"""
    
    @staticmethod
    def load_data():
        """Load all CSV files and return as DataFrames"""
        patients = pd.read_csv('data/patients.csv')
        providers = pd.read_csv('data/providers.csv')
        appointments = pd.read_csv('data/appointments.csv')
        phone_calls = pd.read_csv('data/phone_calls.csv')
        
        # Strip whitespace from all string columns
        patients = patients.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        appointments = appointments.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        phone_calls = phone_calls.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        # Convert date columns with error handling
        patients['date_of_birth'] = pd.to_datetime(patients['date_of_birth'], errors='coerce')
        patients['created_at'] = pd.to_datetime(patients['created_at'], errors='coerce')
        appointments['scheduled_start'] = pd.to_datetime(appointments['scheduled_start'], errors='coerce')
        appointments['scheduled_end'] = pd.to_datetime(appointments['scheduled_end'], errors='coerce')
        appointments['created_at'] = pd.to_datetime(appointments['created_at'], errors='coerce')
        phone_calls['start_time'] = pd.to_datetime(phone_calls['start_time'], errors='coerce')
        
        return patients, providers, appointments, phone_calls


class FeatureEngineer:
    """Handles feature creation and data preprocessing"""
    
    def __init__(self):
        self.feature_columns = [
            'age', 'age_risk', 'distance_score', 'high_distance',
            'booking_risk', 'insurance_risk', 'payment_risk', 'appointment_risk',
            'no_show_rate', 'cancellation_rate', 'total_appointments',
            'call_count', 'avg_call_duration', 'recorded_calls',
            'outstanding_balance', 'copay_amount'
        ]
        self.encoders = {}
    
    def create_training_features(self, patients, providers, appointments, phone_calls):
        """Create features for no-show prediction training"""
        
        # Create target variable: 1 if No-Show, 0 otherwise
        appointments['no_show'] = (appointments['status'] == 'No-Show').astype(int)
        
        # Filter only completed appointments for historical analysis
        historical_appointments = appointments[appointments['status'].isin(['Completed', 'No-Show', 'Cancelled'])]
        
        # Patient historical features
        patient_history = historical_appointments.groupby('patient_id').agg({
            'no_show': ['sum', 'count'],
            'status': lambda x: (x == 'Cancelled').sum(),
            'appointment_id': 'count'
        }).reset_index()
        
        patient_history.columns = ['patient_id', 'no_show_count', 'total_appointments', 'cancelled_count', 'total_appts_check']
        patient_history['no_show_rate'] = patient_history['no_show_count'] / patient_history['total_appointments'].clip(lower=1)
        patient_history['cancellation_rate'] = patient_history['cancelled_count'] / patient_history['total_appointments'].clip(lower=1)
        
        # Phone call features
        phone_features = phone_calls.groupby('phone_number').agg({
            'call_id': 'count',
            'duration_secs': 'mean',
            'recording_url': lambda x: x.notna().sum()
        }).reset_index()
        phone_features.columns = ['phone_number', 'call_count', 'avg_call_duration', 'recorded_calls']
        
        # Merge patients with phone features
        patients_with_calls = patients.merge(phone_features, on='phone_number', how='left')
        patients_with_calls = patients_with_calls.fillna(0)
        
        # Create training dataset from historical appointments
        historical_appointments = appointments[appointments['status'].isin(['Completed', 'No-Show', 'Cancelled'])].copy()
        
        # Merge all data
        training_data = historical_appointments.merge(patients_with_calls, on='patient_id', how='left')
        training_data = training_data.merge(providers, on='provider_id', how='left')
        training_data = training_data.merge(patient_history, on='patient_id', how='left')
        
        # Fill missing historical data (new patients)
        training_data = training_data.fillna(0)
        
        # Calculate derived features
        training_data['age'] = training_data['date_of_birth'].apply(lambda x: calculate_age(x))
        training_data['age_risk'] = ((training_data['age'] >= 16) & (training_data['age'] <= 55)).astype(int)
        
        # Distance features
        training_data['distance_score'] = training_data.apply(
            lambda row: calculate_distance_score(row['zip_code'], row['practice_zip_code']), axis=1
        )
        training_data['high_distance'] = (training_data['distance_score'] >= 50).astype(int)
        
        # Risk features
        training_data['booking_risk'] = training_data['booking_channel'].isin(['Online', 'SMS']).astype(int)
        training_data['insurance_risk'] = (training_data['insurance_type'] == 'Self-Pay').astype(int)
        training_data['payment_risk'] = ((training_data['payment_on_file'] == False) & 
                                       (training_data['outstanding_balance'] > 0)).astype(int)
        training_data['appointment_risk'] = training_data['appointment_type'].isin(['Restorative', 'Root Canal', 'Extraction']).astype(int)
        
        return training_data
    
    def prepare_features_and_target(self, training_data):
        """Prepare features and target for model training"""
        # Prepare numerical features
        X = training_data[self.feature_columns].fillna(0).copy()
        y = training_data['no_show']
        
        # Prepare categorical encoders
        self.encoders = {
            'gender': LabelEncoder(),
            'insurance': LabelEncoder(),
            'booking': LabelEncoder(),
            'appointment': LabelEncoder()
        }
        
        # Add encoded categorical features
        X['gender_encoded'] = self.encoders['gender'].fit_transform(training_data['gender'])
        X['insurance_encoded'] = self.encoders['insurance'].fit_transform(training_data['insurance_type'])
        X['booking_encoded'] = self.encoders['booking'].fit_transform(training_data['booking_channel'])
        X['appointment_encoded'] = self.encoders['appointment'].fit_transform(training_data['appointment_type'])
        
        return X, y


class ModelConfig:
    """Configuration for a model with its hyperparameter search space"""
    
    def __init__(self, name: str, model_class: type, param_generator: Callable, 
                 default_params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.model_class = model_class
        self.param_generator = param_generator
        self.default_params = default_params or {}


class ModelRegistry:
    """Registry of available models and their configurations"""
    
    @staticmethod
    def get_random_forest_config() -> ModelConfig:
        """Random Forest configuration"""
        def param_generator(trial):
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
        
        default_params = {'random_state': 42, 'class_weight': 'balanced'}
        return ModelConfig('RandomForest', RandomForestClassifier, param_generator, default_params)
    
    @staticmethod
    def get_xgboost_config() -> ModelConfig:
        """XGBoost configuration"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        def param_generator(trial):
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
            }
        
        default_params = {
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'verbosity': 0
        }
        return ModelConfig('XGBoost', XGBClassifier, param_generator, default_params)
    
    @staticmethod
    def get_logistic_regression_config() -> ModelConfig:
        """Logistic Regression configuration"""
        def param_generator(trial):
            # Use fixed combinations to avoid dynamic categorical distributions
            penalty_solver_combo = trial.suggest_categorical('penalty_solver', [
                'l1_liblinear', 'l1_saga', 
                'l2_liblinear', 'l2_lbfgs', 'l2_saga',
                'elasticnet_saga',
                'none_lbfgs', 'none_saga'
            ])
            
            penalty, solver = penalty_solver_combo.split('_')
            
            params = {
                'C': trial.suggest_float('C', 1e-4, 100, log=True),
                'penalty': penalty,
                'solver': solver
            }
            
            # Add l1_ratio only for elasticnet penalty
            if penalty == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)
            
            return params
        
        default_params = {'random_state': 42, 'class_weight': 'balanced', 'max_iter': 1000}
        return ModelConfig('LogisticRegression', LogisticRegression, param_generator, default_params)
    
    @staticmethod
    def get_gradient_boosting_config() -> ModelConfig:
        """Gradient Boosting configuration"""
        def param_generator(trial):
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=25),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
        
        default_params = {'random_state': 42}
        return ModelConfig('GradientBoosting', GradientBoostingClassifier, param_generator, default_params)
    
    @staticmethod
    def get_ada_boost_config() -> ModelConfig:
        """AdaBoost configuration"""
        def param_generator(trial):
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=25),
                'learning_rate': trial.suggest_float('learning_rate', 0.1, 2.0),
                'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])
            }
        
        default_params = {'random_state': 42}
        return ModelConfig('AdaBoost', AdaBoostClassifier, param_generator, default_params)
    

    
    @staticmethod
    def get_all_configs() -> List[ModelConfig]:
        """Get all available model configurations"""
        configs = [
            ModelRegistry.get_random_forest_config(),
            ModelRegistry.get_logistic_regression_config(),
            ModelRegistry.get_gradient_boosting_config(),
            ModelRegistry.get_ada_boost_config(),
        ]
        
        if XGBOOST_AVAILABLE:
            configs.append(ModelRegistry.get_xgboost_config())
        
        return configs
    
    @staticmethod
    def get_quick_configs() -> List[ModelConfig]:
        """Get a subset of models for quick optimization"""
        return [
            ModelRegistry.get_random_forest_config(),
            ModelRegistry.get_logistic_regression_config(),
            ModelRegistry.get_ada_boost_config()
        ]


class BasicTrainer:
    """Simple, quick model training with default parameters"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_engineer = FeatureEngineer()
    
    def train_simple_model(self, model_type: str = 'random_forest'):
        """Train a simple model with default parameters"""
        print("Loading data...")
        data_loader = DataLoader()
        patients, providers, appointments, phone_calls = data_loader.load_data()
        
        print("Creating features...")
        training_data = self.feature_engineer.create_training_features(patients, providers, appointments, phone_calls)
        X, y = self.feature_engineer.prepare_features_and_target(training_data)
        
        print(f"Training dataset shape: {X.shape}")
        print(f"No-show rate: {y.mean():.3f}")
        print(f"No-show counts: {y.value_counts().to_dict()}")
        
        # Choose model
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif model_type == 'logistic_regression':
            model = LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Split data
        if len(y.unique()) > 1 and len(y) > 10:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        
        # Train model
        print(f"Training {model_type} model...")
        model.fit(X_train, y_train)
        
        # Evaluate
        auc_score = None
        if len(X_test) > 0 and len(y_test.unique()) > 1:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            print(f"AUC-ROC Score: {auc_score:.4f}")
        else:
            print("AUC-ROC Score: Unable to calculate (insufficient test data)")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nTop 10 Feature Importances:")
            print(feature_importance.head(10))
        
        # Save model
        model_artifacts = {
            'model': model,
            'feature_columns': list(X.columns),
            'encoders': self.feature_engineer.encoders,
            'model_type': model_type,
            'auc_score': auc_score
        }
        
        joblib.dump(model_artifacts, 'models/prediction_model.pkl')
        print("\nModel saved to models/prediction_model.pkl")
        
        return model, auc_score


class AdvancedOptimizer:
    """Advanced model optimization using Optuna"""
    
    def __init__(self, cv_folds: int = 5, n_trials: int = 50, random_state: int = 42):
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        self.feature_engineer = FeatureEngineer()
        self.results = {}
    
    def create_objective(self, model_config: ModelConfig, X: pd.DataFrame, y: pd.Series) -> Callable:
        """Create an objective function for a specific model"""
        def objective(trial):
            # Generate parameters using the model's parameter generator
            params = model_config.param_generator(trial)
            
            # Add default parameters
            params.update(model_config.default_params)
            
            # Create and evaluate model
            model = model_config.model_class(**params)
            scores = cross_val_score(model, X, y, cv=self.skf, scoring='roc_auc')
            return scores.mean()
        
        return objective
    
    def optimize_model(self, model_config: ModelConfig, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize a specific model"""
        print(f"\nOptimizing {model_config.name}...")
        
        study = optuna.create_study(
            direction='maximize',
            study_name=f'{model_config.name}_optimization',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        objective = self.create_objective(model_config, X, y)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        best_score = study.best_value
        best_params = study.best_params
        
        print(f"{model_config.name} - Best AUC-ROC: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        
        result = {
            'model_config': model_config,
            'best_score': best_score,
            'best_params': best_params,
            'study': study
        }
        
        self.results[model_config.name] = result
        return result
    
    def run_optimization(self, optimization_type: str = 'full'):
        """Run optimization pipeline"""
        print("Loading data...")
        data_loader = DataLoader()
        patients, providers, appointments, phone_calls = data_loader.load_data()
        
        print("Creating features...")
        training_data = self.feature_engineer.create_training_features(patients, providers, appointments, phone_calls)
        X, y = self.feature_engineer.prepare_features_and_target(training_data)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        print(f"No-show rate: {y.mean():.3f}")
        
        # Choose model configurations
        if optimization_type == 'quick':
            model_configs = ModelRegistry.get_quick_configs()
            print("Starting Quick Model Optimization")
        else:
            model_configs = ModelRegistry.get_all_configs()
            print("Starting Full Model Optimization")
        
        print("=" * 60)
        
        # Run optimization for each model
        for model_config in model_configs:
            try:
                self.optimize_model(model_config, X, y)
            except Exception as e:
                print(f"Error optimizing {model_config.name}: {str(e)}")
                continue
        
        # Get best model
        if self.results:
            best_model_name, best_score, best_params, best_config = self.get_best_model()
            
            # Train final model
            final_model = self.train_final_model(best_config, best_params, X, y)
            
            # Save optimized model (exclude non-serializable optimization artifacts)
            optimization_summary = {
                model_name: {
                    'best_score': result['best_score'],
                    'best_params': result['best_params']
                }
                for model_name, result in self.results.items()
            }
            
            model_artifacts = {
                'model': final_model,
                'feature_columns': list(X.columns),
                'encoders': self.feature_engineer.encoders,
                'model_type': best_model_name,
                'auc_score': best_score,
                'best_params': best_params,
                'optimization_summary': optimization_summary
            }
            
            joblib.dump(model_artifacts, 'models/prediction_model.pkl')
            print(f"\nOptimized model saved to models/prediction_model.pkl")
            
            # Return cleaned results (serializable)
            clean_results = {
                model_name: {
                    'best_score': result['best_score'],
                    'best_params': result['best_params']
                }
                for model_name, result in self.results.items()
            }
            
            return final_model, best_score, clean_results
        else:
            print("No successful optimizations completed")
            return None, None, {}
    
    def get_best_model(self) -> Tuple[str, float, Dict[str, Any], ModelConfig]:
        """Get the overall best performing model"""
        if not self.results:
            raise ValueError("No optimization results available. Run optimization first.")
        
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['best_score'])
        best_result = self.results[best_model_name]
        
        print(f"\nBEST MODEL: {best_model_name}")
        print(f"Best AUC-ROC Score: {best_result['best_score']:.4f}")
        print(f"Best Parameters: {best_result['best_params']}")
        
        return (best_model_name, best_result['best_score'], 
                best_result['best_params'], best_result['model_config'])
    
    def train_final_model(self, model_config: ModelConfig, best_params: Dict[str, Any], 
                         X: pd.DataFrame, y: pd.Series) -> Any:
        """Train final model with optimized parameters"""
        print(f"\nTraining final {model_config.name} model...")
        
        # Merge best parameters with default parameters
        final_params = {**model_config.default_params, **best_params}
        
        model = model_config.model_class(**final_params)
        model.fit(X, y)
        
        return model


class TrainingPipeline:
    """Main training pipeline that provides both basic and advanced options"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def run_basic_training(self, model_type: str = 'random_forest'):
        """Run basic training with default parameters"""
        print("BASIC TRAINING MODE")
        print("=" * 50)
        
        trainer = BasicTrainer(random_state=self.random_state)
        model, auc_score = trainer.train_simple_model(model_type)
        
        # Test bucket prediction
        test_probabilities = [0.23, 0.65, 0.85]
        print("\nBucket prediction examples:")
        for prob in test_probabilities:
            bucket = predict_bucket(prob)
            print(f"Probability: {prob:.2f} -> Bucket: {bucket}")
        
        return model, auc_score
    
    def run_advanced_optimization(self, optimization_type: str = 'full', n_trials: int = 50):
        """Run advanced optimization with hyperparameter tuning"""
        print("ADVANCED OPTIMIZATION MODE")
        print("=" * 50)
        
        optimizer = AdvancedOptimizer(
            n_trials=n_trials, 
            random_state=self.random_state
        )
        model, auc_score, results = optimizer.run_optimization(optimization_type)
        
        if model is not None:
            # Test bucket prediction
            test_probabilities = [0.23, 0.65, 0.85]
            print("\nBucket prediction examples:")
            for prob in test_probabilities:
                bucket = predict_bucket(prob)
                print(f"Probability: {prob:.2f} -> Bucket: {bucket}")
            
            # Print optimization summary
            print("\nOPTIMIZATION SUMMARY:")
            print("=" * 40)
            for model_name, result in results.items():
                print(f"{model_name}: {result['best_score']:.4f}")
        
        return model, auc_score, results


def run_training_pipeline(mode: str = 'basic', **kwargs):
    """Main entry point for training pipeline
    
    Args:
        mode: 'basic' for simple training, 'quick' for fast optimization, 'full' for complete optimization
        **kwargs: Additional arguments for training
    """
    pipeline = TrainingPipeline()
    
    if mode == 'basic':
        model_type = kwargs.get('model_type', 'random_forest')
        return pipeline.run_basic_training(model_type)
    elif mode in ['quick', 'full']:
        n_trials = kwargs.get('n_trials', 50 if mode == 'full' else 20)
        return pipeline.run_advanced_optimization(mode, n_trials)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'basic', 'quick', or 'full'")


if __name__ == "__main__":
    # Example usage
    print("No-Show Prediction Training System")
    print("=" * 60)
    
    # Run basic training
    model, auc_score = run_training_pipeline(mode='basic', model_type='random_forest') 