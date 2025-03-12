import json
import random
import time
import datetime
import uuid
import boto3
import argparse
from faker import Faker

# Initialize Faker
fake = Faker()

def generate_transaction(fraud_probability=0.05):
    """Generate a random transaction with optional fraud characteristics"""
    
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

def send_to_kinesis(transaction, stream_name, region="us-east-1"):
    """Send transaction to Kinesis Data Stream"""
    try:
        kinesis = boto3.client('kinesis', region_name=region)
        
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
        print(f"Error sending to Kinesis: {str(e)}")
        return None

def send_batch_to_kinesis(transactions, stream_name, region="us-east-1"):
    """Send a batch of transactions to Kinesis Data Stream"""
    try:
        kinesis = boto3.client('kinesis', region_name=region)
        
        # Prepare records
        records = []
        for transaction in transactions:
            # Convert transaction to JSON
            transaction_json = json.dumps(transaction)
            
            # Add to records list
            records.append({
                'Data': transaction_json,
                'PartitionKey': transaction['transaction_id']
            })
        
        # Send batch to Kinesis
        response = kinesis.put_records(
            StreamName=stream_name,
            Records=records
        )
        
        # Check for failures
        failed_count = response.get('FailedRecordCount', 0)
        if failed_count > 0:
            print(f"Warning: {failed_count} records failed to be sent to Kinesis")
        
        return response
    except Exception as e:
        print(f"Error sending batch to Kinesis: {str(e)}")
        return None

def write_to_file(transaction, filename):
    """Write transaction to a file"""
    try:
        with open(filename, 'a') as f:
            f.write(json.dumps(transaction) + '\n')
    except Exception as e:
        print(f"Error writing to file: {str(e)}")

def main():
    """Main function to generate transactions"""
    
    parser = argparse.ArgumentParser(description='Generate simulated transaction data')
    parser.add_argument('--mode', choices=['kinesis', 'file', 'both'], default='both', 
                      help='Output mode: send to Kinesis, write to file, or both')
    parser.add_argument('--stream', default='ssense-transaction-stream',
                      help='Kinesis stream name (for kinesis mode)')
    parser.add_argument('--region', default='us-east-1',
                      help='AWS region for Kinesis stream')
    parser.add_argument('--file', default='transactions.jsonl',
                      help='Output filename (for file mode)')
    parser.add_argument('--count', type=int, default=100,
                      help='Number of transactions to generate')
    parser.add_argument('--rate', type=float, default=1.0,
                      help='Transactions per second')
    parser.add_argument('--batch-size', type=int, default=25,
                      help='Batch size for Kinesis (max 500)')
    parser.add_argument('--fraud-rate', type=float, default=0.05,
                      help='Probability of generating fraudulent transactions')
    parser.add_argument('--continuous', action='store_true',
                      help='Run continuously until interrupted')
    
    args = parser.parse_args()
    
    print(f"Generating transactions at {args.rate} per second")
    print(f"Fraud probability: {args.fraud_rate}")
    
    # Batch mode or continuous mode
    if args.continuous:
        try:
            transaction_count = 0
            batch_transactions = []
            batch_start_time = time.time()
            
            print("Generating transactions continuously. Press Ctrl+C to stop.")
            while True:
                # Generate a transaction
                transaction = generate_transaction(fraud_probability=args.fraud_rate)
                transaction_count += 1
                
                # Add to batch if using Kinesis
                if args.mode in ['kinesis', 'both']:
                    batch_transactions.append(transaction)
                
                # Write to file if needed
                if args.mode in ['file', 'both']:
                    write_to_file(transaction, args.file)
                    
                # Send batch to Kinesis if batch size reached
                if args.mode in ['kinesis', 'both'] and len(batch_transactions) >= args.batch_size:
                    send_batch_to_kinesis(batch_transactions, args.stream, args.region)
                    print(f"Sent batch of {len(batch_transactions)} transactions to Kinesis")
                    batch_transactions = []
                    batch_start_time = time.time()
                
                # Print progress every 100 transactions
                if transaction_count % 100 == 0:
                    print(f"Generated {transaction_count} transactions so far")
                
                # Calculate delay to maintain the rate
                next_transaction_time = batch_start_time + (1.0 / args.rate)
                sleep_time = max(0, next_transaction_time - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            # Send remaining transactions in batch
            if args.mode in ['kinesis', 'both'] and batch_transactions:
                send_batch_to_kinesis(batch_transactions, args.stream, args.region)
                print(f"Sent final batch of {len(batch_transactions)} transactions to Kinesis")
            
            print(f"\nStopped after generating {transaction_count} transactions")
    else:
        # Generate fixed count of transactions
        batch_transactions = []
        for i in range(args.count):
            # Generate a transaction
            transaction = generate_transaction(fraud_probability=args.fraud_rate)
            
            # Output transaction
            if args.mode in ['kinesis', 'both']:
                batch_transactions.append(transaction)
                
                # Send batch if reached batch size or last transaction
                if len(batch_transactions) >= args.batch_size or i == args.count - 1:
                    response = send_batch_to_kinesis(batch_transactions, args.stream, args.region)
                    if response:
                        print(f"Sent batch of {len(batch_transactions)} transactions to Kinesis")
                    else:
                        print(f"Failed to send batch to Kinesis")
                    batch_transactions = []
            
            if args.mode in ['file', 'both']:
                write_to_file(transaction, args.file)
                print(f"Wrote transaction {transaction['transaction_id']} to {args.file}")
            
            # Calculate delay to maintain the rate
            if i < args.count - 1:  # No need to wait after the last transaction
                time.sleep(1.0 / args.rate)
        
        print(f"Generated {args.count} transactions")

if __name__ == "__main__":
    main()