import json
import random
import datetime
import uuid
import boto3
import logging
import io
import os
import math
from datetime import timedelta

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Get environment variables
S3_BUCKET = "fraudassessmentssensemodel"
DATA_PREFIX = os.environ.get('DATA_PREFIX', 'historical-data')
NUM_TRANSACTIONS = int(os.environ.get('NUM_TRANSACTIONS', 50000))
FRAUD_RATIO = float(os.environ.get('FRAUD_RATIO', 0.1))  # 10% fraud

# AWS clients
s3_client = boto3.client('s3')

def log_normal_random(mean=4.5, sigma=1.0):
    """Custom log-normal distribution implementation without NumPy"""
    # Box-Muller transform to generate normally distributed random number
    u1 = random.random()
    u2 = random.random()
    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    
    # Transform to log-normal distribution
    x = math.exp(math.log(mean) + sigma * z0)
    return x

def check_data_exists():
    """Check if historical data already exists in S3 bucket"""
    try:
        # Check if the bucket exists
        s3_client.head_bucket(Bucket=S3_BUCKET)
        logger.info(f"Bucket {S3_BUCKET} exists")
        
        # Check if historical data file exists
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=f"{DATA_PREFIX}/transactions.json")
            logger.info(f"Historical data already exists at s3://{S3_BUCKET}/{DATA_PREFIX}/transactions.json")
            return True
        except Exception:
            logger.info(f"Historical data doesn't exist in bucket {S3_BUCKET}")
            return False
    except Exception:
        logger.info(f"Bucket {S3_BUCKET} doesn't exist")
        # Create the bucket if it doesn't exist
        try:
            s3_client.create_bucket(
                Bucket=S3_BUCKET,
                CreateBucketConfiguration={'LocationConstraint': boto3.session.Session().region_name}
            )
            logger.info(f"Created bucket {S3_BUCKET}")
        except Exception as e:
            logger.error(f"Error creating bucket {S3_BUCKET}: {str(e)}")
            raise
        return False

def generate_timestamp(days_back=365):
    """Generate a random timestamp within the specified days back from now"""
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days_back)
    time_delta = end_date - start_date
    random_seconds = random.randrange(int(time_delta.total_seconds()))
    return start_date + datetime.timedelta(seconds=random_seconds)

def generate_transaction(is_fraud=False):
    """Generate a realistic transaction based on the schema"""
    # Generate a transaction ID
    transaction_id = f"T{uuid.uuid4().hex[:8].upper()}"
    
    # Generate a user ID
    user_id = f"U{random.randint(10000, 99999)}"
    
    # Generate timestamp
    timestamp = generate_timestamp()
    timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Generate transaction amount
    if is_fraud:
        # Fraudulent transactions tend to be larger or very small (testing with small amounts)
        amount_pattern = random.random()
        if amount_pattern < 0.7:  # 70% chance of high amount
            amount = round(random.uniform(500, 3000), 2)
        else:  # 30% chance of suspicious micro-transaction
            amount = round(random.uniform(0.5, 10), 2)
    else:
        # Normal transactions follow a log-normal distribution using custom implementation
        amount = round(log_normal_random(mean=4.5, sigma=1.0), 2)
        # Clamp to reasonable range
        amount = min(max(amount, 5.0), 500.0)
    
    # Generate device type with realistic distribution
    device_options = ['mobile', 'desktop', 'tablet']
    if is_fraud:
        # Fraudulent transactions more likely from mobile
        device_weights = [0.65, 0.25, 0.1]
    else:
        # Normal distribution favors desktop slightly
        device_weights = [0.4, 0.5, 0.1]
    device_type = random.choices(device_options, weights=device_weights)[0]
    
    # Generate location
    locations = [
        'California, USA', 'New York, USA', 'Texas, USA', 'Florida, USA', 
        'Illinois, USA', 'London, UK', 'Paris, France', 'Berlin, Germany', 
        'Tokyo, Japan', 'Sydney, Australia'
    ]
    
    if is_fraud:
        # Fraudulent transactions more likely from unusual locations
        location_weights = [0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.2, 0.15, 0.15, 0.1]
    else:
        # Normal transactions more likely from US
        location_weights = [0.25, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]
    
    location = random.choices(locations, weights=location_weights)[0]
    
    # Generate VPN usage
    if is_fraud:
        # Fraudulent transactions more likely to use VPN
        is_vpn = random.choices([True, False], weights=[0.7, 0.3])[0]
    else:
        # Normal transactions less likely to use VPN
        is_vpn = random.choices([True, False], weights=[0.08, 0.92])[0]
    
    # Generate card type
    card_options = ['credit', 'debit', 'gift']
    if is_fraud:
        # Fraudulent transactions more likely with credit cards or gift cards
        card_weights = [0.5, 0.2, 0.3]
    else:
        # Normal distribution favors debit and credit
        card_weights = [0.45, 0.5, 0.05]
    
    card_type = random.choices(card_options, weights=card_weights)[0]
    
    # Generate transaction status
    status_options = ['approved', 'pending', 'declined']
    if is_fraud and random.random() < 0.4:
        # Some fraudulent transactions might be declined or pending
        status = random.choices(status_options, weights=[0.6, 0.15, 0.25])[0]
    else:
        # Most transactions are approved
        status = random.choices(status_options, weights=[0.95, 0.03, 0.02])[0]
    
    # Create transaction object
    transaction = {
        'transaction_id': transaction_id,
        'user_id': user_id,
        'timestamp': timestamp_str,
        'amount': amount,
        'device_type': device_type,
        'location': location,
        'is_vpn': is_vpn,
        'card_type': card_type,
        'status': status,
        # Add a label for training purposes (not in the original schema but useful)
        'is_fraud': is_fraud
    }
    
    return transaction

def generate_historical_data(num_transactions=50000, fraud_ratio=0.1):
    """Generate historical transaction data with realistic fraud patterns"""
    logger.info(f"Generating {num_transactions} transactions with {fraud_ratio*100}% fraud ratio")
    
    transactions = []
    num_fraud = int(num_transactions * fraud_ratio)
    num_legitimate = num_transactions - num_fraud
    
    # Generate legitimate transactions
    for _ in range(num_legitimate):
        transactions.append(generate_transaction(is_fraud=False))
    
    # Generate fraudulent transactions
    for _ in range(num_fraud):
        transactions.append(generate_transaction(is_fraud=True))
    
    # Shuffle the transactions to mix fraud and legitimate
    random.shuffle(transactions)
    
    # Time-sort transactions to simulate a realistic timeline
    transactions.sort(key=lambda x: x['timestamp'])
    
    # Create consistent user behavior patterns
    users = {}
    for transaction in transactions:
        user_id = transaction['user_id']
        if user_id not in users:
            users[user_id] = {
                'preferred_device': random.choice(['mobile', 'desktop', 'tablet']),
                'preferred_location': random.choice(['California, USA', 'New York, USA', 'Texas, USA', 'Florida, USA', 'Illinois, USA']),
                'preferred_card': random.choice(['credit', 'debit']),
                'transaction_count': 0,
                'avg_amount': 0
            }
        
        users[user_id]['transaction_count'] += 1
        users[user_id]['avg_amount'] = ((users[user_id]['avg_amount'] * (users[user_id]['transaction_count'] - 1)) + 
                                     transaction['amount']) / users[user_id]['transaction_count']
    
    # Add user behavior consistency to non-fraud transactions (with some randomness)
    for i, transaction in enumerate(transactions):
        if not transaction['is_fraud'] and random.random() < 0.85:  # 85% of legitimate transactions follow patterns
            user_id = transaction['user_id']
            user = users[user_id]
            
            # Apply user preferences with some variability
            if random.random() < 0.9:  # 90% chance to use preferred device
                transaction['device_type'] = user['preferred_device']
            
            if random.random() < 0.85:  # 85% chance to use preferred location
                transaction['location'] = user['preferred_location']
            
            if random.random() < 0.95:  # 95% chance to use preferred card
                transaction['card_type'] = user['preferred_card']
            
            # Adjust amount to be closer to user's average (with variation)
            if random.random() < 0.7:  # 70% of transactions are near average
                avg = user['avg_amount']
                variation = avg * 0.3  # 30% variation
                transaction['amount'] = round(random.uniform(avg - variation, avg + variation), 2)
                transaction['amount'] = max(transaction['amount'], 1.0)  # Ensure positive amount
            
            transactions[i] = transaction
    
    return transactions

def upload_to_s3(transactions, bucket, key):
    """Upload transactions to S3 bucket"""
    try:
        # Convert to JSON
        transactions_json = json.dumps(transactions, indent=2)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=transactions_json
        )
        
        logger.info(f"Uploaded {len(transactions)} transactions to s3://{bucket}/{key}")
        return True
    except Exception as e:
        logger.error(f"Error uploading data to S3: {str(e)}")
        return False

def lambda_handler(event, context):
    """Lambda handler for generating historical transaction data"""
    try:
        # Check if data already exists
        if check_data_exists():
            return {
                'statusCode': 200,
                'body': json.dumps('Historical data already exists. No action taken.')
            }
        
        # Generate historical data
        transactions = generate_historical_data(NUM_TRANSACTIONS, FRAUD_RATIO)
        
        # Upload to S3
        upload_success = upload_to_s3(
            transactions, 
            S3_BUCKET, 
            f"{DATA_PREFIX}/transactions.json"
        )
        
        if upload_success:
            return {
                'statusCode': 200,
                'body': json.dumps(f'Successfully generated and uploaded {NUM_TRANSACTIONS} transactions')
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps('Failed to upload transactions to S3')
            }
            
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }

# For local testing
if __name__ == "__main__":
    lambda_handler({}, None)