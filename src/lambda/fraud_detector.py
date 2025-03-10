import json
import base64
import boto3
import os
import logging
import time
# Add these imports
import pandas as pd
from io import BytesIO
import joblib

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Get environment variables
S3_BUCKET = os.environ.get('S3_BUCKET', '')
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', '')

# Path to the ML model in S3
MODEL_PATH = 'models/fraud_detection_model.joblib'

# Global variable to cache the model
model = None

def get_model():
    """Loads the ML model from S3 or returns cached model"""
    global model
    if model is None:
        try:
            # Check if SageMaker endpoint exists
            sagemaker_endpoint = os.environ.get('SAGEMAKER_ENDPOINT', '')
            
            if sagemaker_endpoint and len(sagemaker_endpoint) > 0:
                # Download model from S3
                logger.info(f"Downloading model from s3://{S3_BUCKET}/{MODEL_PATH}")
                obj = s3_client.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
                model_bytes = obj['Body'].read()
                
                # Load the model
                model = joblib.load(BytesIO(model_bytes))
                logger.info("Model loaded successfully")
            else:
                # Return a dummy model for rule-based approach
                logger.info("Using rule-based model (no SageMaker endpoint)")
                model = "rule-based"
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Return a dummy model for rule-based approach
            logger.info("Falling back to rule-based model")
            model = "rule-based"
    return model

def rule_based_prediction(transaction):
    """Simple rule-based fraud prediction"""
    # Extract transaction details
    amount = transaction.get('amount', 0)
    device_type = transaction.get('device_type', 'unknown')
    location = transaction.get('location', 'unknown')
    is_vpn = transaction.get('is_vpn', False)
    card_type = transaction.get('card_type', 'unknown')
    status = transaction.get('status', 'unknown')
    
    # Calculate risk score based on simple rules
    risk_score = 0.0
    
    # Amount-based risk
    if amount > 1000:
        risk_score += 0.4
    elif amount > 500:
        risk_score += 0.2
    elif amount > 200:
        risk_score += 0.1
        
    # Device type risk
    if device_type == 'mobile':
        risk_score += 0.1
        
    # Location risk
    high_risk_locations = ['Tokyo, Japan', 'Berlin, Germany', 'Paris, France', 'London, UK']
    if location in high_risk_locations:
        risk_score += 0.2
        
    # VPN risk
    if is_vpn:
        risk_score += 0.1
        
    # Card type risk
    if card_type == 'gift':
        risk_score += 0.2
        
    # Status risk
    if status == 'declined':
        risk_score += 0.3
        
    # Determine if fraudulent based on threshold
    threshold = float(os.environ.get('RISK_THRESHOLD', 0.7))
    is_fraud = risk_score >= threshold
    
    # Return prediction results
    return {
        'transaction_id': transaction['transaction_id'],
        'is_fraud': is_fraud,
        'fraud_probability': risk_score,
        'risk_score': risk_score,
        'method': 'rule-based'
    }

def predict_fraud(transaction):
    """Fraud prediction using rule-based approach or ML model"""
    try:
        # Get model (will be either ML model or "rule-based")
        model = get_model()
        
        # If we have a real model, use it
        if model != "rule-based":
            try:
                # Convert to DataFrame with required format
                df = pd.DataFrame([{
                    'amount': transaction.get('amount', 0),
                    'device_type': transaction.get('device_type', 'unknown')
                }])
                
                # Make prediction
                fraud_probability = float(model.predict_proba(df)[0, 1])
                is_fraud = bool(model.predict(df)[0])
                
                return {
                    'transaction_id': transaction['transaction_id'],
                    'is_fraud': is_fraud,
                    'fraud_probability': fraud_probability,
                    'risk_score': fraud_probability,
                    'method': 'ml-model'
                }
            except Exception as e:
                logger.error(f"Error using ML model: {str(e)}")
                # Fall back to rule-based approach
                return rule_based_prediction(transaction)
        else:
            # Use rule-based approach
            return rule_based_prediction(transaction)
            
    except Exception as e:
        logger.error(f"Error predicting fraud: {str(e)}")
        # Return default values in case of error
        return {
            'transaction_id': transaction['transaction_id'],
            'is_fraud': False,
            'fraud_probability': 0.0,
            'risk_score': 0.0,
            'error': str(e)
        }

def store_transaction(transaction, prediction):
    """Store transaction and prediction in DynamoDB"""
    try:
        table = dynamodb.Table(DYNAMODB_TABLE)
        
        # Add prediction results to transaction
        transaction_item = {**transaction, **prediction}
        
        # Add timestamp for TTL (30 days from now)
        transaction_item['ttl'] = int(time.time()) + (30 * 24 * 60 * 60)
        
        # Store in DynamoDB
        table.put_item(Item=transaction_item)
        logger.info(f"Stored transaction {transaction['transaction_id']} in DynamoDB")
            
    except Exception as e:
        logger.error(f"Error storing transaction: {str(e)}")

def handler(event, context):
    """Lambda handler function"""
    logger.info(f"Processing {len(event['Records'])} records")
    
    for record in event['Records']:
        try:
            # Decode Kinesis data
            payload = base64.b64decode(record['kinesis']['data']).decode('utf-8')
            transaction = json.loads(payload)
            
            logger.info(f"Processing transaction {transaction['transaction_id']}")
            
            # Predict fraud
            prediction = predict_fraud(transaction)
            
            # Store transaction and prediction
            store_transaction(transaction, prediction)
            
            # Log prediction results
            logger.info(f"Transaction {transaction['transaction_id']} - Fraud probability: {prediction['fraud_probability']}")
            
            # If fraudulent, take immediate action
            if prediction['is_fraud']:
                logger.warning(f"FRAUD DETECTED: Transaction {transaction['transaction_id']} from user {transaction['user_id']}")
                
        except Exception as e:
            logger.error(f"Error processing record: {str(e)}")
    
    return {
        'statusCode': 200,
        'body': json.dumps(f"Processed {len(event['Records'])} transactions")
    }