#!/usr/bin/env python3
"""
Transaction Data Generator for Fraud Detection System

This script generates simulated transaction data for testing the fraud detection system.
It can output to a file or directly to a Kinesis stream.
"""

import json
import random
import time
import datetime
import uuid
import boto3
import argparse
import logging
from faker import Faker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('transaction-generator')

# Initialize Faker for generating realistic data
fake = Faker()

def generate_transaction(fraud_probability=0.05):
    """
    Generate a random transaction with optional fraud characteristics
    
    Args:
        fraud_probability: Probability of generating a fraudulent transaction
        
    Returns:
        Dictionary containing transaction data
    """
    
    # Determine if this transaction should be fraudulent
    is_fraud = random.random() < fraud_probability
    
    # Generate a transaction ID
    transaction_id = f"T{uuid.uuid4().hex[:8].upper()}"
    
    # Generate a user ID
    user_id = f"U{random.randint(1000, 9999)}"
    
    # Generate timestamp (within the last hour)
    timestamp = datetime.datetime.now() - datetime.timedelta(
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Generate transaction amount
    if is_fraud:
        # Fraudulent transactions tend to be larger
        amount = round(random.uniform(500, 3000), 2)
    else:
        # Normal transactions
        amount = round(random.uniform(10, 500), 2)
    
    # Generate device type
    device_options = ['mobile', 'desktop', 'tablet']
    if is_fraud:
        # Fraudulent transactions more likely from mobile
        device_type = random.choices(device_options, weights=[0.7, 0.2, 0.1])[0]
    else:
        # Normal distribution
        device_type = random.choices(device_options, weights=[0.4, 0.5, 0.1])[0]
    
    # Generate location
    locations = [
        'California, USA', 'New York, USA', 'Texas, USA', 'Florida, USA', 
        'Illinois, USA', 'London, UK', 'Paris, France', 'Berlin, Germany', 
        'Tokyo, Japan', 'Sydney, Australia'
    ]
    
    if is_fraud:
        # Fraudulent transactions more likely from unusual locations
        location = random.choice(locations[5:])  # Non-US locations
    else:
        # Normal transactions more likely from US
        location = random.choice(locations[:5])  # US locations
    
    # Generate VPN usage
    if is_fraud:
        # Fraudulent transactions more likely to use VPN
        is_vpn = random.choices([True, False], weights=[0.7, 0.3])[0]
    else:
        # Normal transactions less likely to use VPN
        is_vpn = random.choices([True, False], weights=[0.1, 0.9])[0]
    
    # Generate card type
    card_options = ['credit', 'debit', 'gift']
    if is_fraud:
        # Fraudulent transactions more likely with gift cards
        card_type = random.choices(card_options, weights=[0.5, 0.2, 0.3])[0]
    else:
        # Normal distribution
        card_type = random.choices(card_options, weights=[0.6, 0.35, 0.05])[0]
    
    # Generate transaction status
    status_options = ['approved', 'pending', 'declined']
    if is_fraud and random.random() < 0.3:
        # Some fraudulent transactions might be declined
        status = 'declined'
    else:
        # Most transactions are approved
        status = 'approved'
    
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
        'status': status
    }
    
    return transaction

def send_to_kinesis(transaction, stream_name):
    """
    Send transaction to Kinesis Data Stream
    
    Args:
        transaction: Transaction data dictionary
        stream_name: Name of the Kinesis stream
        
    Returns:
        Response from Kinesis or None if error
    """
    try:
        kinesis = boto3.client('kinesis')
        
        # Convert transaction to JSON
        transaction_json = json.dumps(transaction)
        
        # Send to Kinesis
        response = kinesis.put_record(
            StreamName=stream_name,
            Data=transaction_json,
            PartitionKey=transaction['transaction_id']
        )
        
        return response
    except Exception as e:
        logger.error(f"Error sending to Kinesis: {str(e)}")
        return None

def write_to_file(transaction, filename):
    """
    Write transaction to a file
    
    Args:
        transaction: Transaction data dictionary
        filename: Name of the output file
        
    Returns:
        None
    """
    try:
        with open(filename, 'a') as f:
            f.write(json.dumps(transaction) + '\n')
    except Exception as e:
        logger.error(f"Error writing to file: {str(e)}")

def main():
    """Main function to generate transactions"""
    
    parser = argparse.ArgumentParser(description='Generate simulated transaction data')
    parser.add_argument('--mode', choices=['kinesis', 'file'], default='file', 
                      help='Output mode: send to Kinesis or write to file')
    parser.add_argument('--stream', default='fraud-detection-stream',
                      help='Kinesis stream name (for kinesis mode)')
    parser.add_argument('--file', default='transactions.jsonl',
                      help='Output filename (for file mode)')
    parser.add_argument('--count', type=int, default=100,
                      help='Number of transactions to generate')
    parser.add_argument('--rate', type=float, default=1.0,
                      help='Transactions per second')
    parser.add_argument('--fraud-rate', type=float, default=0.05,
                      help='Probability of generating fraudulent transactions')
    
    args = parser.parse_args()
    
    logger.info(f"Generating {args.count} transactions at {args.rate} per second")
    logger.info(f"Fraud probability: {args.fraud_rate}")
    
    # Generate and output transactions
    for i in range(args.count):
        # Generate a transaction
        transaction = generate_transaction(fraud_probability=args.fraud_rate)
        
        # Output transaction
        if args.mode == 'kinesis':
            response = send_to_kinesis(transaction, args.stream)
            if response:
                logger.info(f"Sent transaction {transaction['transaction_id']} to Kinesis - Shard: {response['ShardId']}")
            else:
                logger.info(f"Failed to send transaction {transaction['transaction_id']} to Kinesis")
        else:
            write_to_file(transaction, args.file)
            logger.info(f"Wrote transaction {transaction['transaction_id']} to {args.file}")
        
        # Calculate delay to maintain the rate
        if i < args.count - 1:  # No need to wait after the last transaction
            time.sleep(1.0 / args.rate)
    
    logger.info(f"Generated {args.count} transactions")

if __name__ == "__main__":
    main()