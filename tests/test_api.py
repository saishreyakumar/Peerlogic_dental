#!/usr/bin/env python3
"""
Simple test script for the Dental No-Show Prediction API
Tests the modular API structure.
"""

import requests
import json
import sys

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict():
    """Test the predict endpoint"""
    print("\nTesting predict endpoint...")
    
    # Test case 1: Low risk patient
    low_risk_patient = {
        "patient_id": 1,
        "phone_number": "+14155551234",
        "date_of_birth": "1985-03-15",
        "gender": "Female",
        "zip_code": "94102",
        "insurance_type": "PPO",
        "payment_on_file": True,
        "outstanding_balance": 0.0,
        "appointment_type": "Hygiene",
        "booking_channel": "Phone",
        "copay_amount": 25.00,
        "practice_zip_code": "94102",
        "no_show_rate": 0.0,
        "cancellation_rate": 0.0,
        "total_appointments": 10,
        "call_count": 3,
        "avg_call_duration": 180.0,
        "recorded_calls": 2
    }
    
    # Test case 2: High risk patient
    high_risk_patient = {
        "patient_id": 2,
        "phone_number": "+14155552345",
        "date_of_birth": "1995-01-15",
        "gender": "Male",
        "zip_code": "94999",  # Far away
        "insurance_type": "Self-Pay",
        "payment_on_file": False,
        "outstanding_balance": 500.0,
        "appointment_type": "Root Canal",
        "booking_channel": "Online",
        "copay_amount": 200.00,
        "practice_zip_code": "94102",
        "no_show_rate": 0.5,
        "cancellation_rate": 0.3,
        "total_appointments": 2,
        "call_count": 0,
        "avg_call_duration": 0.0,
        "recorded_calls": 0
    }
    
    test_cases = [
        ("Low Risk Patient", low_risk_patient),
        ("High Risk Patient", high_risk_patient)
    ]
    
    for name, patient_data in test_cases:
        print(f"\n{name}:")
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=patient_data,
                headers={"Content-Type": "application/json"}
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Prediction: {result['pts']:.4f} -> Bucket: {result['bucket']}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error: {e}")

def test_example():
    """Test the example endpoint"""
    print("\nTesting example endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/example")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("Example request format retrieved successfully")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run all tests"""
    print("=== Dental No-Show Prediction API Test ===")
    
    # Test health endpoint
    if not test_health():
        print("Health check failed. Make sure the API server is running.")
        sys.exit(1)
    
    # Test predict endpoint
    test_predict()
    
    # Test example endpoint
    test_example()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main() 