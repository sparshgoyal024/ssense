"""
Fraud Detection Lambda Function

This Lambda function processes transaction data from Kinesis,
invokes the SageMaker endpoint for fraud prediction,
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
sagemaker_runtime = boto3.client('sagemaker-runtime')
dynamodb = boto3.resource('dynamodb')
sns = boto3.client('sns')

# Get environment variables
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'fraud-detection-results')
SAGEMAKER_ENDPOINT = os.environ.get('SAGEMAKER_ENDPOINT', 'fraud-detection-endpoint')
ALERT_TOPIC_ARN = os.environ.get('ALERT_TOPIC_ARN', '')
RISK_THRESHOLD = float(os.environ.get('RISK_THRESHOLD', '0.7'))

def predict_fraud(transaction):
    """
    Makes fraud prediction using SageMaker endpoint
    
    Args:
        transaction: Transaction data dictionary
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Format the transaction for the SageMaker endpoint
        # Focus on features that matter for fraud detection
        payload = {
            'amount': transaction.get('amount', 0),
            'device_type': transaction.get('device_type', 'unknown'),
            'location': transaction.get('location', 'unknown'),
            'is_vpn': transaction.get('is_vpn', False),
            'card_type': transaction.get('card_type', 'unknown'),
            'status': transaction.get('status', 'unknown')
        }
        
        # Invoke SageMaker endpoint
        logger.info(f"Invoking SageMaker endpoint: {SAGEMAKER_ENDPOINT}")
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse the response
        result = json.loads(response['Body'].read().decode())
        
        # Return prediction results with transaction ID
        return {
            'transaction_id': transaction['transaction_id'],
            'is_fraud': result.get('is_fraud', False),
            'fraud_probability': result.get('fraud_probability', 0.0),
            'risk_score': result.get('risk_score', 0.0)
        }
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