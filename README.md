# Dental No-Show Prediction API

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

## Problem

This system provides an early warning by predicting which patients are likely to skip, enabling practices to proactively intervene.

## Solution Overview

**What It Does:**
Predicts no-show risk using machine learning models trained on historical patient, appointment, and engagement data. Each patient receives a clear risk category:

- **High Risk** (<50% probability): Prioritize with calls and reminders
- **Medium Risk** (50–79%): Standard reminders likely sufficient
- **Low Risk** (≥80%): Minimal reminders needed

**Key Features:**

- **Automated Model Selection**: Compares multiple ML algorithms, picks best performer
- **Real Geography**: Calculates actual driving distance, not ZIP approximations
- **Self-Improving**: Retrains automatically with new data
- **Production Ready**: Professional FastAPI service with Docker deployment
- **20+ Predictors**: Age, insurance, payment history, booking method, and more

## Project Structure

```
Peerlogic_dental/
├── api/                     # FastAPI REST service
│   └── model_prediction.py  # Main API endpoints
├── core/                    # ML pipeline & algorithms
│   ├── features.py          # Feature engineering
│   ├── prediction.py        # Prediction service
│   └── training.py          # Model training & optimization
├── common/
│   ├── models.py            # Data models & schemas
│   └── utils.py             # Helper functions
├── scripts/                 # Training scripts
│   └── model_training.py
├── tests/                   # Tests
├── models/                  # Saved model artifacts
├── dental_no_show_prediction.ipynb  # Analysis notebook
└── summary.md
```

## Quick Start

### 1. Setup Environment

```bash
uv sync
```

### 2. Train Your Model

**Full Optimization (Recommended - 20-30 minutes):**

```bash
uv run python scripts/model_training.py
```

**Quick Training (2-3 minutes):**

```bash
uv run python -c "from core.training import run_training_pipeline; run_training_pipeline(mode='basic')"
```

### 3. Start the API

```bash
uv run uvicorn api.model_prediction:app --host 0.0.0.0 --port 8000
```

### 4. Test the System

```bash
uv run python test_api.py
```

## API Usage

### Core Prediction Endpoint

**`POST /predict`** - Get no-show risk prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": 1,
    "phone_number": "+14155551234",
    "date_of_birth": "1985-03-15",
    "gender": "Female",
    "zip_code": "94102",
    "insurance_type": "PPO",
    "payment_on_file": true,
    "outstanding_balance": 125.50,
    "appointment_type": "Hygiene",
    "booking_channel": "Online",
    "copay_amount": 25.00,
    "practice_zip_code": "94102"
  }'
```

**Response:**

```json
{
  "pts": 0.23,
  "bucket": "High"
}
```

### Additional Endpoints

| Endpoint          | Purpose                        |
| ----------------- | ------------------------------ |
| `GET /health`     | System health check            |
| `GET /model-info` | Model details and feature list |
| `GET /example`    | Sample request format          |
| `GET /docs`       | Interactive API documentation  |

## Model Comparison & Selection

The system automatically compares multiple machine learning algorithms and selects the best performer based on **cross-validated AUC-ROC scores**:

### Models Tested by Training Mode

| Mode      | Models Compared                                                                | Selection Criteria    |
| --------- | ------------------------------------------------------------------------------ | --------------------- |
| **Basic** | Random Forest (default parameters)                                             | Single model training |
| **Quick** | Random Forest • Logistic Regression • AdaBoost                                 | Highest CV AUC-ROC    |
| **Full**  | Random Forest • Logistic Regression • Gradient Boosting • AdaBoost • XGBoost\* | Highest CV AUC-ROC    |

### How Best Model is Selected

1. **Hyperparameter Optimization**: Each model optimized using Optuna with 20-50 trials
2. **Cross-Validation**: 5-fold stratified cross-validation for robust evaluation
3. **AUC-ROC Scoring**: Models ranked by mean cross-validated AUC-ROC score
4. **Automatic Selection**: Highest scoring model automatically saved for production

### Training Performance

| Mode      | Time      | Optimization   | Best For         |
| --------- | --------- | -------------- | ---------------- |
| **Basic** | 2-3 min   | None           | Quick testing    |
| **Quick** | 5-10 min  | 20 trials each | Development      |
| **Full**  | 20-30 min | 50 trials each | Production level |

### Custom Training

```python
from core.training import run_training_pipeline

# Quick optimization
model, score, results = run_training_pipeline(mode='quick', n_trials=20)

# Custom model
model, score = run_training_pipeline(mode='basic', model_type='logistic_regression')
```

## Key Predictors (In Order of Importance)

1. **Past No-Show History** - Strongest predictor
2. **Driving Distance** - Geographic convenience factor
3. **Outstanding Balance Without Payment Method** - Financial barriers
4. **Age 16–55 & Self-Pay Insurance** - Demographic risk factors
5. **Booking Channel** - Online/SMS riskier than phone

## Docker Deployment

### Local Development

```bash
docker build -t dental-api .
docker run -p 8000:8000 dental-api
```

### Production Ready

- Health checks at `/health`
- Comprehensive error handling
- Structured logging
- Graceful degradation

## Testing & Validation

### Automated Testing

```bash

uv run uvicorn api.model_prediction:app --host 0.0.0.0 --port 8000

# Run test suite
uv run python tests/test_api.py
```

### Model Performance

- Target: ≥75% AUC-ROC accuracy
- Cross-validation with stratified sampling
- Handles class imbalance automatically

## Success Metrics

- **Accuracy**: ≥75% AUC-ROC score

---
