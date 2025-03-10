"""
Fraud Detection Lambda Function

This Lambda function processes transaction data from Kinesis,
makes fraud predictions using a rule-based approach,
and stores results in DynamoDB.
"""

import json
import base64
import boto3
import os
import logging
import time
from decimal import Decimal

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
sns = boto3.client('sns')

# Get environment variables
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'fraud-detection-results')
ALERT_TOPIC_ARN = os.environ.get('ALERT_TOPIC_ARN', '')
RISK_THRESHOLD = float(os.environ.get('RISK_THRESHOLD', '0.7'))

def predict_fraud(transaction):
    """Fraud prediction using rule-based approach or ML model"""
    try:
        # Check if SageMaker endpoint exists and should be used
        sagemaker_endpoint = os.environ.get('SAGEMAKER_ENDPOINT', '')
        
        if sagemaker_endpoint and len(sagemaker_endpoint) > 0:
            # Use ML model if available
            try:
                model = get_model()
                
                # Convert to DataFrame with required format
                df = pd.DataFrame([{
                    'amount': transaction.get('amount', 0),
                    'device_type': transaction.get('device_type', 'unknown')
                }])
                
                # Make prediction - this assumes our simple model structure
                fraud_probability = float(model.predict_proba(df)[0, 1])
                is_fraud = bool(model.predict(df)[0])
                
                return {
                    'transaction_id': transaction['transaction_id'],
                    'is_fraud': is_fraud,
                    'fraud_probability': fraud_probability,
                    'risk_score': fraud_probability
                }
            except Exception as model_error:
                logger.warning(f"Error using ML model: {str(model_error)}. Falling back to rule-based approach.")
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
        
def store_transaction(transaction, prediction):
    """
    Store transaction and prediction in DynamoDB
    
    Args:
        transaction: Transaction data dictionary
        prediction: Prediction results dictionary
        
    Returns:
        None
    """
    try:
        table = dynamodb.Table(DYNAMODB_TABLE)
        
        # Combine transaction and prediction
        item = {**transaction, **prediction}
        
        # Convert float values to Decimal for DynamoDB
        for key, value in item.items():
            if isinstance(value, float):
                item[key] = Decimal(str(value))
        
        # Add timestamp for TTL (30 days from now)
        item['ttl'] = int(time.time()) + (30 * 24 * 60 * 60)
        
        # Store in DynamoDB
        table.put_item(Item=item)
        logger.info(f"Stored transaction {transaction['transaction_id']} in DynamoDB")
            
    except Exception as e:
        logger.error(f"Error storing transaction: {str(e)}")

def send_alert(transaction, prediction):
    """
    Send alert for fraudulent transaction
    
    Args:
        transaction: Transaction data dictionary
        prediction: Prediction results dictionary
        
    Returns:
        None
    """
    if not ALERT_TOPIC_ARN:
        return
        
    try:
        # Create alert message
        message = {
            'alert_type': 'FRAUD_DETECTED',
            'transaction_id': transaction['transaction_id'],
            'user_id': transaction['user_id'],
            'amount': transaction['amount'],
            'timestamp': transaction['timestamp'],
            'location': transaction['location'],
            'fraud_probability': prediction['fraud_probability'],
            'risk_score': prediction['risk_score']
        }
        
        # Send to SNS
        sns.publish(
            TopicArn=ALERT_TOPIC_ARN,
            Message=json.dumps(message),
            Subject=f"FRAUD ALERT: Transaction {transaction['transaction_id']}"
        )
        logger.info(f"Sent fraud alert for transaction {transaction['transaction_id']}")
        
    except Exception as e:
        logger.error(f"Error sending alert: {str(e)}")

def handler(event, context):
    """
    Lambda handler function
    
    Args:
        event: AWS Lambda event object
        context: AWS Lambda context object
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing {len(event.get('Records', []))} records")
    
    processed_count = 0
    fraud_count = 0
    
    for record in event.get('Records', []):
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
            
            # If fraudulent and above threshold, send alert
            if prediction['is_fraud'] or prediction.get('risk_score', 0) > RISK_THRESHOLD:
                fraud_count += 1
                logger.warning(f"FRAUD DETECTED: Transaction {transaction['transaction_id']} from user {transaction['user_id']}")
                send_alert(transaction, prediction)
                
            processed_count += 1
                
        except Exception as e:
            logger.error(f"Error processing record: {str(e)}")
    
    logger.info(f"Processed {processed_count} transactions, detected {fraud_count} fraudulent")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'processed': processed_count,
            'fraudulent': fraud_count
        })
    }