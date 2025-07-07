# Dental No-Show Prediction System – Executive Summary

## Problem Framing

This ML system predicts appointment no-show risk to reduce revenue loss.

**Solution:** AdaBoost model (AUC-ROC: 0.851) selected after comprehensive testing of 5 algorithms with Optuna optimization. Categorizes patients into three risk buckets: High (<50% PTS), Medium (50-79% PTS), Low (≥80% PTS) for targeted intervention strategies.

## Key Predictive Drivers

1. **Historical No-Show Rate** - Past behavior predicts future behavior
2. **Geographic Distance** - Real driving distance from practice
3. **Financial Barriers** - Outstanding balance without payment method
4. **Demographics** - Age 16-55, self-pay insurance highest risk
5. **Booking Channel** - Online/SMS bookings riskier than phone

## Key Model Assumptions

**Feature Engineering:**

- Age 16-55 represents highest no-show risk demographic
- ZIP centroid distance sufficiently approximates actual driving distance
- Distance >50 miles threshold indicates "high distance" risk factor
- Online/SMS booking channels indicate higher risk than phone bookings
- Self-pay insurance correlates with higher financial barriers

**Behavioral Patterns:**

- Historical no-show behavior is strongest predictor of future behavior
- Outstanding balance + no payment method creates significant financial barrier
- Phone engagement patterns (frequency, duration) indicate patient commitment
- Past appointment frequency and cancellations predict future reliability

**Business Context:**

- Practices can act on predictions with targeted interventions within 24-48 hours
- Cost of false positives (over-reminding) < cost of false negatives (missed revenue)
- Staff can interpret risk scores in daily workflows
- Binary show/no-show classification sufficient for intervention decisions

## Model Performance

**Validation Results:** AUC-ROC 0.851 (exceeds 0.75 production threshold), 20% test set validation (2,400 samples) with robust error handling for production deployment.

**Technical Implementation:** FastAPI REST service with 20+ engineered features, automated model selection from 5 algorithms, and containerized deployment.

## Recommended Next Steps

**Model Enhancement & Optimization:**

- Expand feature set with weather, local events, or economic indicators
- Fine-tune AdaBoost hyperparameters based on production feedback
- Explore ensemble methods combining top-performing models

**Data Collection & Analysis:**

- Analyze call recordings and extract transcripts to identify verbal cues and communication patterns that predict no-shows (would be a good addition for the company)
- Add real-time feedback loop to capture intervention effectiveness
- Monitor prediction accuracy against actual no-show outcomes
