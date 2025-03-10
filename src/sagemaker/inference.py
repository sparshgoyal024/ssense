"""
SageMaker Inference Script for Fraud Detection

This script is used by SageMaker to load the model and handle inference requests.
"""

import os
import json
import logging
import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# These are the features our model expects
EXPECTED_FEATURES = ['amount', 'device_type', 'location', 'is_vpn', 'card_type', 'status']

# Location risk mapping
LOCATION_RISK = {
    'California, USA': 0.2,
    'New York, USA': 0.2,
    'Texas, USA': 0.2,
    'Florida, USA': 0.3,
    'Illinois, USA': 0.2,
    'London, UK': 0.3,
    'Paris, France': 0.4,
    'Berlin, Germany': 0.4,
    'Tokyo, Japan': 0.5,
    'Sydney, Australia': 0.5,
    'Unknown': 0.9
}

# Pre-trained model - this could be a SageMaker built-in model
# or loaded from a model.tar.gz file
def model_fn(model_dir):
    """
    Load the model for inference
    
    Args:
        model_dir: Directory where model artifacts are stored
        
    Returns:
        Loaded model object
    """
    logger.info("Loading fraud detection model")
    
    try:
        # For pretrained SageMaker model, you could load it here
        # model = joblib.load(os.path.join(model_dir, 'model.joblib'))
        
        # For simplicity, we'll use a rule-based model
        logger.info("Using rule-based fraud detection model")
        return "rule-based-model"
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body, request_content_type="application/json"):
    """
    Deserialize and prepare the prediction input
    
    Args:
        request_body: Request data
        request_content_type: Data content type
        
    Returns:
        Preprocessed input data
    """
    logger.info("Processing input data")
    
    if request_content_type == "application/json":
        # Parse input JSON
        input_data = json.loads(request_body)
        
        # Convert to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = pd.DataFrame(input_data)
            
        # Validate input features
        for feature in EXPECTED_FEATURES:
            if feature not in df.columns:
                logger.warning(f"Missing expected feature: {feature}")
                if feature == 'amount':
                    df[feature] = 0
                elif feature == 'is_vpn':
                    df[feature] = False
                else:
                    df[feature] = 'unknown'
        
        # Add location risk
        df['location_risk'] = df['location'].map(lambda loc: LOCATION_RISK.get(loc, 0.5))
        
        # Convert device_type to numerical
        df['device_risk'] = df['device_type'].map({
            'mobile': 0.7,
            'desktop': 0.3, 
            'tablet': 0.5,
            'unknown': 0.8
        }).fillna(0.8)
        
        # Convert card_type to numerical
        df['card_risk'] = df['card_type'].map({
            'credit': 0.3,
            'debit': 0.2,
            'gift': 0.7,
            'unknown': 0.8
        }).fillna(0.8)
        
        # Convert status to numerical
        df['status_risk'] = df['status'].map({
            'approved': 0.2,
            'pending': 0.5,
            'declined': 0.8,
            'unknown': 0.8
        }).fillna(0.8)
        
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Make prediction with the loaded model
    
    Args:
        input_data: Preprocessed input data
        model: Loaded model object
        
    Returns:
        Model predictions
    """
    logger.info("Making prediction")
    
    try:
        # For a simple rule-based model
        if model == "rule-based-model":
            # Calculate risk score based on input features
            risk_score = (
                (input_data['amount'].clip(10, 3000) / 3000 * 0.3) +
                (input_data['location_risk'] * 0.2) +
                (input_data['device_risk'] * 0.15) +
                (input_data['card_risk'] * 0.15) +
                (input_data['status_risk'] * 0.1) +
                (input_data['is_vpn'].astype(int) * 0.1)
            )
            
            # Determine if transaction is fraudulent based on risk score
            is_fraud = risk_score > 0.65
            
            # Return prediction results
            return {
                'is_fraud': bool(is_fraud.iloc[0]),
                'fraud_probability': float(risk_score.iloc[0]),
                'risk_score': float(risk_score.iloc[0])
            }
        else:
            # For an ML model, you would do something like:
            # fraud_probability = model.predict_proba(input_data)[0, 1]
            # is_fraud = model.predict(input_data)[0]
            # return {'is_fraud': bool(is_fraud), 'fraud_probability': float(fraud_probability)}
            
            logger.error("Unsupported model type")
            return {
                'is_fraud': False,
                'fraud_probability': 0.0,
                'risk_score': 0.0,
                'error': 'Unsupported model type'
            }
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return {
            'is_fraud': False,
            'fraud_probability': 0.0,
            'risk_score': 0.0,
            'error': str(e)
        }

def output_fn(prediction, response_content_type="application/json"):
    """
    Serialize the prediction result
    
    Args:
        prediction: Prediction result
        response_content_type: Response content type
        
    Returns:
        Serialized prediction
    """
    logger.info("Generating output")
    
    if response_content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")