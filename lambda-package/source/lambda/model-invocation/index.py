import json
import base64
import boto3
import os
import logging
import time
import datetime
from decimal import Decimal

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sagemaker_runtime = boto3.client('sagemaker-runtime')
sns = boto3.client('sns')

# Get environment variables
S3_BUCKET = os.environ.get('S3_BUCKET', '')
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'ssense-fraud-transactions')
SOLUTION_PREFIX = os.environ.get('SOLUTION_PREFIX', 'ssense-fraud')
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', '')

# SageMaker endpoint name
ENDPOINT_NAME = f"{SOLUTION_PREFIX}-xgb"

def calculate_location_risk(location):
    """Calculate risk score based on location"""
    location_risk = {
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
    return location_risk.get(location, 0.7)  # Default risk for unknown locations

def extract_features(transaction):
    """Extract and compute features for fraud detection"""
    # Basic features from transaction
    features = {
        'amount': float(transaction.get('amount', 0)),
        'device_type': transaction.get('device_type', 'unknown'),
        'is_vpn': 1 if transaction.get('is_vpn', False) else 0,
        'card_type': transaction.get('card_type', 'unknown'),
        'status': transaction.get('status', 'approved')
    }
    
    # Extract time-based features
    try:
        timestamp = datetime.datetime.strptime(
            transaction.get('timestamp', datetime.datetime.now().isoformat()),
            '%Y-%m-%dT%H:%M:%SZ'
        )
        features['hour_of_day'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
        features['is_night'] = 1 if (timestamp.hour < 6 or timestamp.hour >= 22) else 0
    except Exception as e:
        logger.warning(f"Error parsing timestamp: {str(e)}")
        features['hour_of_day'] = 0
        features['day_of_week'] = 0
        features['is_weekend'] = 0
        features['is_night'] = 0
    
    # Calculate location risk
    features['location_risk'] = calculate_location_risk(transaction.get('location', 'Unknown'))
    
    # User transaction count (mock value in real system would query history)
    features['user_transaction_count'] = 5
    
    # Amount z-score (mock calculation)
    features['amount_zscore'] = (features['amount'] - 200) / 100 if features['amount'] > 0 else 0
    
    # Calculate risk score
    features['transaction_risk_score'] = (
        (features['amount'] / 1000) * 0.3 +
        features['is_vpn'] * 0.2 +
        features['location_risk'] * 0.3 +
        (1 if transaction.get('status') == 'declined' else 0) * 0.2
    )
    
    return features

def format_features_for_prediction(features):
    """Format features for the SageMaker endpoint"""
    # Map categorical features to numeric
    device_type_mapping = {
        'mobile': 0,
        'desktop': 1,
        'tablet': 2,
        'unknown': 3
    }
    
    device_type_value = device_type_mapping.get(features['device_type'], 3)
    
    # Prepare all features in correct order as used during training
    feature_list = [
        str(features['amount']),
        str(features.get('hour_of_day', 0)),
        str(features.get('day_of_week', 0)),
        str(features.get('is_weekend', 0)),
        str(features.get('is_night', 0)),
        str(features.get('location_risk', 0.5)),
        str(features.get('is_vpn', 0)),
        str(features.get('user_transaction_count', 1)),  # Default to 1
        str(features.get('amount_zscore', 0)),
        str(features.get('transaction_risk_score', 0))
    ]
    
    # Add one-hot encoded values for device_type
    # Assuming these match the order in your training data
    device_types = ['mobile', 'desktop', 'tablet']
    for dtype in device_types:
        feature_list.append('1' if features.get('device_type') == dtype else '0')
    
    # Add one-hot encoded values for card_type
    card_types = ['credit', 'debit', 'gift']
    for ctype in card_types:
        feature_list.append('1' if features.get('card_type') == ctype else '0')
    
    # Add one-hot encoded values for status
    statuses = ['approved', 'pending', 'declined']
    for status in statuses:
        feature_list.append('1' if features.get('status') == status else '0')
    
    return ','.join(feature_list)

def predict_fraud(transaction, features):
    """Get fraud prediction from SageMaker endpoint"""
    try:
        # Format features for prediction
        features_csv = format_features_for_prediction(features)
        
        # Invoke SageMaker endpoint
        logger.info(f"Invoking endpoint: {ENDPOINT_NAME} with features: {features_csv}")
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=features_csv
        )
        
        # Parse response
        result = response['Body'].read().decode('utf-8')
        fraud_probability = float(result)
        is_fraud = fraud_probability > 0.5
        
        logger.info(f"Prediction: probability={fraud_probability}, is_fraud={is_fraud}")
        
        return {
            'transaction_id': transaction['transaction_id'],
            'fraud_probability': fraud_probability,
            'is_fraud': is_fraud,
            'prediction_timestamp': datetime.datetime.now().isoformat(),
            'feature_values': features
        }
    except Exception as e:
        logger.error(f"Error invoking SageMaker endpoint: {str(e)}")
        # Fallback to rule-based prediction using risk score
        return {
            'transaction_id': transaction['transaction_id'],
            'fraud_probability': features['risk_score'],
            'is_fraud': features['risk_score'] > 0.7,
            'prediction_timestamp': datetime.datetime.now().isoformat(),
            'error': str(e),
            'feature_values': features
        }
def store_transaction(transaction, prediction):
    """Store transaction and prediction in DynamoDB"""
    try:
        table = dynamodb.Table(DYNAMODB_TABLE)
        
        # Add prediction results to transaction
        item = {**transaction, **prediction}
        
        # Handle non-serializable types
        item = json.loads(json.dumps(item, default=str), parse_float=Decimal)
        
        # Add timestamp for TTL (30 days from now)
        item['ttl'] = int(time.time()) + (30 * 24 * 60 * 60)
        
        # Store in DynamoDB
        table.put_item(Item=item)
        logger.info(f"Stored transaction {transaction['transaction_id']} in DynamoDB")
        return True
    except Exception as e:
        logger.error(f"Error storing transaction in DynamoDB: {str(e)}")
        return False

def send_fraud_alert(transaction, prediction):
    """Send SNS notification for fraudulent transactions"""
    try:
        if not SNS_TOPIC_ARN:
            logger.warning("SNS_TOPIC_ARN not configured. Skipping alert.")
            return False
        
        if prediction.get('is_fraud', False):
            # Format the message
            message = {
                'subject': f"FRAUD ALERT: Transaction {transaction['transaction_id']}",
                'transaction_id': transaction['transaction_id'],
                'user_id': transaction['user_id'],
                'amount': transaction['amount'],
                'timestamp': transaction['timestamp'],
                'fraud_probability': prediction['fraud_probability'],
                'risk_factors': [
                    f"Amount: ${transaction['amount']:.2f}",
                    f"Location: {transaction['location']}",
                    f"Device: {transaction['device_type']}",
                    f"VPN: {transaction['is_vpn']}",
                    f"Card Type: {transaction['card_type']}"
                ]
            }
            
            # Send the notification
            sns.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject=f"FRAUD ALERT: Transaction {transaction['transaction_id']}",
                Message=json.dumps(message, indent=2)
            )
            
            logger.info(f"Sent fraud alert for transaction {transaction['transaction_id']}")
            return True
    except Exception as e:
        logger.error(f"Error sending fraud alert: {str(e)}")
        return False

def lambda_handler(event, context):
    """Lambda handler function"""
    logger.info(f"Processing event: {json.dumps(event, default=str)}")
    
    # Process Kinesis records
    if 'Records' in event and event.get('Records'):
        logger.info(f"Processing {len(event['Records'])} Kinesis records")
        
        processed_count = 0
        fraud_count = 0
        
        for record in event['Records']:
            try:
                # Decode Kinesis data
                payload = base64.b64decode(record['kinesis']['data']).decode('utf-8')
                transaction = json.loads(payload)
                
                logger.info(f"Processing transaction {transaction['transaction_id']}")
                
                # Extract features
                features = extract_features(transaction)
                
                # Get fraud prediction
                prediction = predict_fraud(transaction, features)
                
                # Store transaction and prediction
                store_transaction(transaction, prediction)
                
                # Send alert if fraudulent
                if prediction['is_fraud']:
                    fraud_count += 1
                    send_fraud_alert(transaction, prediction)
                    logger.warning(f"FRAUD DETECTED: Transaction {transaction['transaction_id']} from user {transaction['user_id']}")
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing Kinesis record: {str(e)}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f"Processed {processed_count} transactions, detected {fraud_count} fraudulent transactions",
                'processed_count': processed_count,
                'fraud_count': fraud_count
            })
        }
    
    # Direct invocation (e.g., from API Gateway)
    elif 'transaction' in event:
        transaction = event['transaction']
        
        # Extract features
        features = extract_features(transaction)
        
        # Get fraud prediction
        prediction = predict_fraud(transaction, features)
        
        # Store transaction and prediction
        store_transaction(transaction, prediction)
        
        # Send alert if fraudulent
        if prediction['is_fraud']:
            send_fraud_alert(transaction, prediction)
            logger.warning(f"FRAUD DETECTED: Transaction {transaction['transaction_id']} from user {transaction['user_id']}")
        
        return {
            'statusCode': 200,
            'body': {
                'transaction_id': transaction['transaction_id'],
                'fraud_prediction': {
                    'is_fraud': prediction['is_fraud'],
                    'fraud_probability': prediction['fraud_probability']
                }
            }
        }
    
    # Invalid event structure
    else:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'message': "Invalid event structure. Expected Kinesis records or transaction data."
            })
        }