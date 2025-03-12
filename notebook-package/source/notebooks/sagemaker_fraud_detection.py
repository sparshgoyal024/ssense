#!/usr/bin/env python3

# E-commerce Fraud Detection SageMaker Processing Job
# This script processes transaction data, trains a fraud detection model
# and deploys it as a SageMaker endpoint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import sagemaker
from sagemaker import Session
from sagemaker.serializers import CSVSerializer
import io
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import sys
import time
import json
import logging
import datetime
from sagemaker import image_uris
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner
import argparse
import random
import uuid
from tqdm import tqdm

# Configure logging
logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Sagemaker specific arguments
    parser.add_argument('--region', type=str, default='us-west-2')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/processing/model')
    parser.add_argument('--output-data-dir', type=str, default='/opt/ml/processing/output')
    
    # Custom arguments
    parser.add_argument('--stack-name', type=str, default='fraudssenseassessment')
    parser.add_argument('--s3-bucket', type=str, required=False)
    parser.add_argument('--role-arn', type=str, required=False)
    parser.add_argument('--solution-prefix', type=str, default='ssense-fraud')
    parser.add_argument('--auto-deploy', type=bool, default=True)
    
    return parser.parse_args()

def get_stack_outputs(stack_name):
    """Get CloudFormation stack outputs"""
    try:
        # First check if outputs file exists
        outputs_file = '/home/ec2-user/SageMaker/stack_outputs_processed.json'
        if os.path.exists(outputs_file):
            try:
                with open(outputs_file, 'r') as f:
                    stack_outputs = json.load(f)
                logger.info(f"Loaded stack outputs from file: {outputs_file}")
                return stack_outputs
            except Exception as file_error:
                logger.warning(f"Could not read stack outputs file: {str(file_error)}")
                
        # If file doesn't exist or can't be read, query CloudFormation
        cfn = boto3.client('cloudformation')
        response = cfn.describe_stacks(StackName=stack_name)
        
        # Convert list of outputs to dictionary
        if 'Stacks' in response and len(response['Stacks']) > 0:
            outputs = {}
            for output in response['Stacks'][0]['Outputs']:
                outputs[output['OutputKey']] = output['OutputValue']
            
            # Set the values from stack outputs
            result = {
                "FraudStackName": stack_name,
                "SolutionPrefix": "ssense-fraud",
                "AwsAccountId": boto3.client('sts').get_caller_identity().get('Account'),
                "AwsRegion": boto3.session.Session().region_name,
                "IamRole": outputs.get('NotebookRoleArn', ''),
                "ModelDataBucket": outputs.get('ModelDataBucket', ''),
                "SolutionsS3Bucket": os.environ.get('NotebookCodeS3Bucket', '767089282839-notebook-code'),
                "DynamoDBTableName": outputs.get('DynamoDBTableName', ''),
                "DynamoDBTableArn": outputs.get('DynamoDBTableArn', ''),
                "KinesisStreamName": outputs.get('KinesisStreamName', ''),
                "KinesisStreamArn": outputs.get('KinesisStreamArn', ''),
                "SNSTopicArn": outputs.get('SNSTopicArn', '')
            }
            
            return result
    except Exception as e:
        logger.warning(f"Could not retrieve CloudFormation outputs: {str(e)}")
    
    # Return None so we know values aren't from stack
    return None

def load_transaction_data_from_s3(bucket_name, prefix='historical-data', filename='transactions.json'):
    """Load transaction data from S3"""
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # Full S3 key
    s3_key = f"{prefix}/{filename}"
    
    try:
        logger.info(f"Loading transaction data from s3://{bucket_name}/{s3_key}")
        
        # Get object from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        
        # Read JSON content
        json_content = response['Body'].read().decode('utf-8')
        transactions = json.loads(json_content)
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        logger.info(f"Successfully loaded {len(df)} transactions from S3")
        logger.info(f"Fraud percentage: {df['is_fraud'].mean() * 100:.2f}%")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading transaction data from S3: {str(e)}")
        raise

def engineer_features_for_ml(df):
    """Feature engineering for ML-based fraud detection"""
    # Create a copy to avoid modifying the original
    df_features = df.copy()
    
    # Ensure we have a fraud label column from the data
    if 'is_fraud' not in df_features.columns:
        raise ValueError("Data must contain an 'is_fraud' column with ground truth labels")
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(df_features['timestamp']):
        df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
    
    # Extract time-based features
    df_features['hour_of_day'] = df_features['timestamp'].dt.hour
    df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
    df_features['month'] = df_features['timestamp'].dt.month
    df_features['day_of_month'] = df_features['timestamp'].dt.day
    df_features['is_weekend'] = df_features['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df_features['is_night'] = df_features['hour_of_day'].apply(lambda x: 1 if (x < 6 or x >= 22) else 0)
    
    # User behavior features - calculated properly for both training and prediction
    # Group by user_id to get transaction counts and statistics
    user_stats = df_features.groupby('user_id').agg({
        'transaction_id': 'count',
        'amount': ['mean', 'std', 'max'],
        'is_vpn': 'mean'
    })
    
    # Flatten multi-level columns
    user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
    user_stats.rename(columns={
        'transaction_id_count': 'user_transaction_count',
        'amount_mean': 'user_avg_amount',
        'amount_std': 'user_amount_std',
        'amount_max': 'user_max_amount',
        'is_vpn_mean': 'user_vpn_ratio'
    }, inplace=True)
    
    # Handle users with only one transaction (no std)
    user_stats['user_amount_std'].fillna(0, inplace=True)
    
    # Reset index to make user_id a column again
    user_stats.reset_index(inplace=True)
    
    # Merge user statistics back to the main dataframe
    df_features = df_features.merge(user_stats, on='user_id', how='left')
    
    # Calculate transaction amount z-score relative to user's history
    # Use a safe calculation that handles division by zero
    df_features['amount_zscore'] = df_features.apply(
        lambda row: (row['amount'] - row['user_avg_amount']) / (row['user_amount_std'] if row['user_amount_std'] > 0 else 1),
        axis=1
    )
    
    # Create features for relative transaction size
    df_features['amount_to_max_ratio'] = df_features.apply(
        lambda row: row['amount'] / row['user_max_amount'] if row['user_max_amount'] > 0 else 0,
        axis=1
    )
    
    # Location features - use past fraud rates instead of hardcoded risk
    location_fraud_rates = df_features.groupby('location')['is_fraud'].mean().to_dict()
    df_features['location_fraud_rate'] = df_features['location'].map(location_fraud_rates)
    
    # Handle NaN values for new locations
    df_features['location_fraud_rate'].fillna(df_features['is_fraud'].mean(), inplace=True)
    
    # Device and card type features
    device_fraud_rates = df_features.groupby('device_type')['is_fraud'].mean().to_dict()
    df_features['device_fraud_rate'] = df_features['device_type'].map(device_fraud_rates)
    
    card_fraud_rates = df_features.groupby('card_type')['is_fraud'].mean().to_dict()
    df_features['card_fraud_rate'] = df_features['card_type'].map(card_fraud_rates)
    
    # Convert boolean to integer if needed
    if pd.api.types.is_bool_dtype(df_features['is_vpn']):
        df_features['is_vpn'] = df_features['is_vpn'].astype(int)
    
    # Create dummy variables for categorical features
    categorical_features = ['device_type', 'card_type', 'status', 'location']
    df_features = pd.get_dummies(df_features, columns=categorical_features, drop_first=False)
    
    # New feature: is this amount unusual for this user?
    df_features['is_unusual_amount'] = (
        (df_features['amount'] > (df_features['user_avg_amount'] + 2 * df_features['user_amount_std'])) |
        (df_features['amount'] < (df_features['user_avg_amount'] - 2 * df_features['user_amount_std']))
    ).astype(int)
    
    # New feature: is this a new user (fewer than N transactions)?
    df_features['is_new_user'] = (df_features['user_transaction_count'] <= 3).astype(int)
    
    # New feature: is this the user's largest transaction?
    df_features['is_largest_tx'] = (df_features['amount'] >= df_features['user_max_amount'] * 0.95).astype(int)
    
    # Check for duplicate columns
    if df_features.columns.duplicated().any():
        logger.warning("Found duplicate columns after feature engineering.")
        # Get the duplicated column names
        duplicated_cols = df_features.columns[df_features.columns.duplicated()].tolist()
        logger.warning(f"Duplicate columns: {duplicated_cols}")
        
        # Drop duplicates
        df_features = df_features.loc[:, ~df_features.columns.duplicated()]
        logger.info(f"Removed duplicate columns. New shape: {df_features.shape}")
    
    return df_features

def create_ml_preprocessing_pipeline(df_features, X=None):
    """Create a scikit-learn preprocessing pipeline for ML model training"""
    # Define numeric features to use
    numeric_features = [
        'amount', 'hour_of_day', 'day_of_week', 'month', 'day_of_month', 
        'is_weekend', 'is_night', 'is_vpn', 'user_transaction_count',
        'user_avg_amount', 'user_amount_std', 'amount_zscore', 
        'amount_to_max_ratio', 'location_fraud_rate', 'device_fraud_rate', 
        'card_fraud_rate', 'is_unusual_amount', 'is_new_user', 'is_largest_tx'
    ]
    
    # If X is provided, check for duplicates in X
    if X is not None:
        # Check for duplicate columns in X
        if X.columns.duplicated().any():
            dupe_cols = X.columns[X.columns.duplicated()].tolist()
            logger.warning(f"Found duplicate columns in X: {dupe_cols}")
            
            # Find first occurrence of each column name
            keep_cols = ~X.columns.duplicated(keep='first')
            X_unique = X.loc[:, keep_cols]
            logger.info(f"Removed duplicate columns. X shape before: {X.shape}, after: {X_unique.shape}")
            
            # Update the provided X dataframe in-place with unique columns
            X = X_unique
    
    # Ensure we only use features that exist in the dataframe
    available_numeric_features = [col for col in numeric_features if col in df_features.columns]
    
    logger.info(f"Using these numeric features for preprocessing: {available_numeric_features}")
    
    # Create column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), available_numeric_features),
        ],
        remainder='passthrough'  # This keeps the dummy variables without scaling
    )
    
    return preprocessor

def train_model_and_create_endpoint(args):
    """Main function to train the model and deploy endpoint"""
    logger.info("Starting fraud detection model training and deployment")
    
    # Get stack outputs for configuration
    stack_outputs = get_stack_outputs(args.stack_name)
    
    if stack_outputs:
        logger.info("Successfully loaded stack outputs")
        SOLUTION_PREFIX = args.solution_prefix
        SAGEMAKER_IAM_ROLE = args.role_arn or stack_outputs.get("IamRole")
        bucket = args.s3_bucket or stack_outputs.get("ModelDataBucket")
        KINESIS_STREAM_NAME = stack_outputs.get("KinesisStreamName")
    else:
        logger.warning("Using default values because CloudFormation outputs could not be retrieved")
        SOLUTION_PREFIX = args.solution_prefix
        SAGEMAKER_IAM_ROLE = args.role_arn
        bucket = args.s3_bucket
        KINESIS_STREAM_NAME = "ssense-transaction-stream"
    
    if not bucket or not SAGEMAKER_IAM_ROLE:
        error_message = "Cannot proceed without S3 bucket and IAM role"
        logger.error(error_message)
        raise ValueError(error_message)
    
    logger.info(f"Using bucket: {bucket}")
    logger.info(f"Using IAM role: {SAGEMAKER_IAM_ROLE}")
    
    # Initialize AWS clients
    session = sagemaker.Session()
    s3 = boto3.resource('s3')
    sm_client = boto3.client('sagemaker')

    # Load data
    df = load_transaction_data_from_s3(bucket)
    
    # Apply feature engineering
    df_engineered = engineer_features_for_ml(df)
    
    # Define features
    numeric_features = [
        'amount', 'hour_of_day', 'day_of_week', 'month', 'day_of_month',
        'is_weekend', 'is_night', 'is_vpn', 'user_transaction_count',
        'user_avg_amount', 'user_amount_std', 'amount_zscore', 
        'amount_to_max_ratio', 'location_fraud_rate', 'device_fraud_rate', 
        'card_fraud_rate', 'is_unusual_amount', 'is_new_user', 'is_largest_tx'
    ]

    # Get categorical columns
    categorical_columns = [col for col in df_engineered.columns if 
                          col.startswith('device_type_') or 
                          col.startswith('card_type_') or 
                          col.startswith('status_') or
                          col.startswith('location_')]

    # Combine all features
    features = numeric_features + categorical_columns

    # Define target
    target = 'is_fraud'

    # Check for duplicated columns in the engineered dataframe
    if df_engineered.columns.duplicated().any():
        logger.warning("Found duplicate columns in engineered dataframe.")
        dupe_cols = df_engineered.columns[df_engineered.columns.duplicated()].tolist()
        logger.warning(f"Duplicate columns: {dupe_cols}")
        
        # Remove duplicates, keeping only the first occurrence
        df_engineered = df_engineered.loc[:, ~df_engineered.columns.duplicated()]
        logger.info(f"Removed duplicate columns. Shape: {df_engineered.shape}")

    # Remove features that don't exist
    features = [f for f in features if f in df_engineered.columns]

    # Split the data
    X = df_engineered[features]
    y = df_engineered[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Testing set shape: {X_test.shape}")
    logger.info(f"Fraud ratio in training: {y_train.mean():.2f}")
    logger.info(f"Fraud ratio in testing: {y_test.mean():.2f}")

    # Create preprocessing pipeline, passing both the feature dataframe and X_train
    preprocessor = create_ml_preprocessing_pipeline(df_engineered, X_train)

    # Remove duplicate columns from X_train and X_test if they exist
    if X_train.columns.duplicated().any():
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        logger.info(f"Removed duplicate columns from X_train. New shape: {X_train.shape}")

    if X_test.columns.duplicated().any():
        X_test = X_test.loc[:, ~X_test.columns.duplicated()]
        logger.info(f"Removed duplicate columns from X_test. New shape: {X_test.shape}")

    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logger.info(f"Processed training data shape: {X_train_processed.shape}")

    # Create a further split of training data to get a validation set
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create train file in SVMlight format
    train_file = io.BytesIO()
    dump_svmlight_file(X_train_split, y_train_split, train_file)
    train_file.seek(0)

    # Create validation file in SVMlight format
    validation_file = io.BytesIO()
    dump_svmlight_file(X_val, y_val, validation_file)
    validation_file.seek(0)

    # Define S3 paths
    prefix = 'fraud-classifier'

    # Upload train data
    train_key = f'{prefix}/train/train.libsvm'
    s3.Bucket(bucket).Object(train_key).upload_fileobj(train_file)
    train_data_s3_uri = f's3://{bucket}/{train_key}'

    # Upload validation data
    val_key = f'{prefix}/validation/validation.libsvm'
    s3.Bucket(bucket).Object(val_key).upload_fileobj(validation_file)
    validation_data_s3_uri = f's3://{bucket}/{val_key}'

    # Set output location
    output_s3_uri = f's3://{bucket}/{prefix}/output'

    logger.info(f"Uploaded training data to {train_data_s3_uri}")
    logger.info(f"Uploaded validation data to {validation_data_s3_uri}")

    # Get the XGBoost image - use the newer SageMaker API
    container = image_uris.retrieve("xgboost", boto3.Session().region_name, version="1.0-1")

    # Define hyperparameter ranges
    hyperparameter_ranges = {
        'max_depth': IntegerParameter(3, 10),
        'eta': ContinuousParameter(0.01, 0.3),
        'gamma': ContinuousParameter(0, 5),
        'min_child_weight': IntegerParameter(1, 10),
        'subsample': ContinuousParameter(0.5, 1.0),
        'colsample_bytree': ContinuousParameter(0.5, 1.0)
    }

    # Create an estimator with both train and validation channels
    xgb = sagemaker.estimator.Estimator(
        container,
        role=SAGEMAKER_IAM_ROLE,
        train_instance_count=1,
        train_instance_type='ml.m5.xlarge',
        output_path=output_s3_uri,
        sagemaker_session=session,
        base_job_name='fraud-detection-xgb'
    )

    # Set static hyperparameters
    xgb.set_hyperparameters(
        objective='binary:logistic',
        eval_metric='auc',
        num_round=100,
        rate_drop=0.1,
        scale_pos_weight=10,  # Helpful for imbalanced datasets
        # Add the following for better handling of missing values and numerical stability
        tree_method='auto',
        max_delta_step=3,    # Helpful for unbalanced classes
        early_stopping_rounds=10
    )

    # Create the tuner with the correct metric name
    tuner = HyperparameterTuner(
        xgb,
        'validation:auc',  # Make sure this matches eval_metric
        hyperparameter_ranges,
        max_jobs=5,
        max_parallel_jobs=2,
        objective_type='Maximize'
    )

    # Start the hyperparameter tuning job with both train and validation
    tuner.fit(
        {
            'train': train_data_s3_uri,
            'validation': validation_data_s3_uri
        },
        wait=True  # Wait for completion since we're in a processing job
    )
    logger.info("Hyperparameter tuning job completed")

    # Save feature information for later use with the model
    feature_info = {
        'feature_names': list(X_train.columns),
        'categorical_features': [col for col in X_train.columns if 
                                col.startswith('device_type_') or 
                                col.startswith('card_type_') or 
                                col.startswith('status_') or
                                col.startswith('location_')],
        'numeric_features': [col for col in X_train.columns if 
                            not (col.startswith('device_type_') or 
                                col.startswith('card_type_') or 
                                col.startswith('status_') or
                                col.startswith('location_'))]
    }

    # Store feature information in S3 for deployment
    feature_info_key = f'{prefix}/model/feature_info.json'
    s3.Bucket(bucket).Object(feature_info_key).put(
        Body=json.dumps(feature_info, indent=2)
    )
    logger.info(f"Saved feature information to s3://{bucket}/{feature_info_key}")

    # Get the best model from hyperparameter tuning
    tuning_job_name = tuner.latest_tuning_job.job_name
    best_job_name = sm_client.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name
    )['BestTrainingJob']['TrainingJobName']

    logger.info(f"Best training job: {best_job_name}")

    # Get the best hyperparameters
    best_hyperparameters = sm_client.describe_training_job(
        TrainingJobName=best_job_name
    )['HyperParameters']

    logger.info("Best hyperparameters:")
    for param, value in best_hyperparameters.items():
        logger.info(f"  {param}: {value}")

    # Create model
    model_name = f"fraud-detection-model-{int(time.time())}"
    model_info = sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': container,
            'ModelDataUrl': f"{output_s3_uri}/{best_job_name}/output/model.tar.gz"
        },
        ExecutionRoleArn=SAGEMAKER_IAM_ROLE
    )

    logger.info(f"Created model: {model_name}")

    # Store model metadata including the feature information
    model_metadata = {
        'model_name': model_name,
        'training_job': best_job_name,
        'hyperparameters': {k: v for k, v in best_hyperparameters.items() if not k.startswith('_')},
        'creation_time': time.time(),
        'feature_info': feature_info  # This comes from the previous step
    }

    # Save model metadata to S3
    metadata_key = f'{prefix}/models/{model_name}/metadata.json'
    s3.Bucket(bucket).Object(metadata_key).put(
        Body=json.dumps(model_metadata, indent=2)
    )

    logger.info(f"Saved model metadata to s3://{bucket}/{metadata_key}")
    
    # Deploy endpoint if auto-deploy is enabled
    if args.auto_deploy:
        # Create endpoint configuration
        endpoint_config_name = f"fraud-detection-config-{int(time.time())}"
        endpoint_config = sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'VariantName': 'default',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.t2.medium',
                'InitialVariantWeight': 1
            }]
        )

        logger.info(f"Created endpoint configuration: {endpoint_config_name}")

        # Create endpoint
        endpoint_name = f"{SOLUTION_PREFIX}-xgb"
        endpoint = sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        logger.info(f"Endpoint {endpoint_name} creation initiated")

        # Wait for endpoint to become available
        logger.info("Waiting for endpoint to be in service...")
        waiter = sm_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        logger.info(f"Endpoint {endpoint_name} is now in service")

        # Store endpoint information in a central registry
        endpoint_info = {
            'endpoint_name': endpoint_name,
            'model_name': model_name,
            'creation_time': time.time(),
            'instance_type': 'ml.t2.medium',
            'instance_count': 1,
            'endpoint_config': endpoint_config_name
        }

        # Save endpoint info to S3
        endpoint_info_key = f'{prefix}/endpoints/{endpoint_name}/info.json'
        s3.Bucket(bucket).Object(endpoint_info_key).put(
            Body=json.dumps(endpoint_info, indent=2)
        )

        logger.info(f"Endpoint {endpoint_name} is ready for inference")
        
        # Test the endpoint with some sample data
        test_sample_size = min(1000, len(X_test))
        test_sample = X_test.iloc[:test_sample_size]
        y_test_sample = y_test.iloc[:test_sample_size]
        
        # Create a predictor for the endpoint
        predictor = sagemaker.Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=session,
            serializer=CSVSerializer()
        )
        
        # Format features for prediction
        def format_features_for_prediction(row, feature_count=len(X_train.columns)):
            """Format a row of features for prediction"""
            features_list = []
            feature_keys = list(row.index)
            
            for i in range(feature_count):
                if i < len(feature_keys):
                    feature = feature_keys[i]
                    features_list.append(str(row[feature]))
                else:
                    features_list.append('0')
            
            return ','.join(features_list)
        
        # Get predictions for test sample
        logger.info(f"Testing endpoint with {test_sample_size} samples")
        y_pred_proba = []
        batch_size = 50
        
        for i in range(0, len(test_sample), batch_size):
            batch = test_sample.iloc[i:i+batch_size]
            batch_features = [format_features_for_prediction(row) for _, row in batch.iterrows()]
            
            # Send each row separately to avoid CSV parsing issues
            batch_predictions = []
            for features_str in batch_features:
                try:
                    response = predictor.predict(features_str)
                    pred = float(response.decode('utf-8'))
                    batch_predictions.append(pred)
                except Exception as e:
                    logger.error(f"Error with prediction: {str(e)}")
                    batch_predictions.append(0.5)  # Default fallback value
            
            y_pred_proba.extend(batch_predictions)
        
        # Convert probabilities to binary predictions
        y_pred = [1 if p >= 0.5 else 0 for p in y_pred_proba]
        
        # Calculate metrics
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        # Print classification report
        logger.info("\nClassification Report:")
        report = classification_report(y_test_sample, y_pred)
        logger.info(report)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test_sample, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(cm)
        
        # Calculate ROC AUC
        roc_auc = roc_auc_score(y_test_sample, y_pred_proba)
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        
# Save test results to S3
        test_results = {
            'endpoint_name': endpoint_name,
            'test_time': time.time(),
            'metrics': {
                'roc_auc': float(roc_auc),
                'confusion_matrix': cm.tolist(),
                'classification_report': classification_report(y_test_sample, y_pred, output_dict=True)
            },
            'test_size': len(test_sample),
            'positive_rate': float(sum(y_pred) / len(y_pred)),
            'true_positive_rate': float(sum([1 if y_pred[i] == 1 and y_test_sample.iloc[i] == 1 else 0 for i in range(len(y_pred))]) / sum(y_test_sample)) if sum(y_test_sample) > 0 else 0
        }
       
       # Save test results to S3
        test_results_key = f'{prefix}/endpoints/{endpoint_name}/test_results.json'
        s3.Bucket(bucket).Object(test_results_key).put(
            Body=json.dumps(test_results, indent=2, default=str)
        )
        logger.info(f"Saved test results to s3://{bucket}/{test_results_key}")
        
        # Calculate business metrics
        fraud_detected = cm[1, 1]  # True positives
        fraud_missed = cm[1, 0]    # False negatives
        false_alarms = cm[0, 1]    # False positives
        correct_negatives = cm[0, 0]  # True negatives
        
        # Calculate important business metrics
        detection_rate = fraud_detected / (fraud_detected + fraud_missed) if (fraud_detected + fraud_missed) > 0 else 0
        false_positive_rate = false_alarms / (false_alarms + correct_negatives) if (false_alarms + correct_negatives) > 0 else 0
        precision = fraud_detected / (fraud_detected + false_alarms) if (fraud_detected + false_alarms) > 0 else 0
        
        logger.info("\nBusiness Impact Metrics:")
        logger.info(f"Fraud Detection Rate: {detection_rate:.2%}")
        logger.info(f"False Alarm Rate: {false_positive_rate:.2%}")
        logger.info(f"Precision (% of flagged transactions that are actual fraud): {precision:.2%}")
        
        # Save detailed model details for reference
        model_details = {
            'endpoint_name': endpoint_name,
            'model_name': model_name,
            'features': features,
            'evaluation_timestamp': time.time(),
            'performance': {
                'roc_auc': float(roc_auc),
                'confusion_matrix': cm.tolist(),
                'classification_report': classification_report(y_test_sample, y_pred, output_dict=True),
                'business_metrics': {
                    'fraud_detection_rate': float(detection_rate),
                    'false_alarm_rate': float(false_positive_rate),
                    'precision': float(precision),
                    'fraud_cases_detected': int(fraud_detected),
                    'fraud_cases_missed': int(fraud_missed),
                    'false_alarms': int(false_alarms)
                }
            },
            'model_parameters': {
                'threshold': 0.5,  # Default classification threshold
            }
        }
        
        # Save to S3
        model_details_key = f'{prefix}/model-details.json'
        s3.Bucket(bucket).Object(model_details_key).put(
            Body=json.dumps(model_details, indent=2, default=str)
        )
        logger.info(f"Model details saved to s3://{bucket}/{model_details_key}")
    
    logger.info("Fraud detection model training and deployment completed successfully")
    return {
        'model_name': model_name,
        'model_data': f"{output_s3_uri}/{best_job_name}/output/model.tar.gz",
        'endpoint_name': endpoint_name if args.auto_deploy else None,
        'feature_info_path': f's3://{bucket}/{feature_info_key}',
        'model_metrics': {
            'roc_auc': float(roc_auc) if args.auto_deploy else None,
            'detection_rate': float(detection_rate) if args.auto_deploy else None,
            'false_positive_rate': float(false_positive_rate) if args.auto_deploy else None,
            'precision': float(precision) if args.auto_deploy else None
        } if args.auto_deploy else None
    }

# Generate test transactions for Kinesis
def generate_test_transactions(num_transactions=10, fraud_ratio=0.3):
   """Generate test transactions for Kinesis"""
   test_transactions = []
   
   # Calculate counts
   num_fraud = int(num_transactions * fraud_ratio)
   num_legitimate = num_transactions - num_fraud
   
   # Generate legitimate transactions
   for _ in range(num_legitimate):
       transaction = generate_transaction(is_fraud=False)
       test_transactions.append(transaction)
   
   # Generate fraudulent transactions
   for _ in range(num_fraud):
       transaction = generate_transaction(is_fraud=True)
       test_transactions.append(transaction)
   
   # Shuffle to mix fraud and legitimate
   random.shuffle(test_transactions)
   
   return test_transactions

def generate_transaction(is_fraud=False):
   """Generate a single transaction with realistic properties"""
   # Generate a transaction ID
   transaction_id = f"T{uuid.uuid4().hex[:8].upper()}"
   
   # Generate a user ID
   user_id = f"U{random.randint(10000, 99999)}"
   
   # Generate timestamp (current time with small random offset)
   timestamp = datetime.datetime.now() - datetime.timedelta(
       minutes=random.randint(0, 59),
       seconds=random.randint(0, 59)
   )
   timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')
   
   # Generate amount based on fraud flag
   if is_fraud:
       amount = round(random.uniform(800, 3000), 2) if random.random() < 0.7 else round(random.uniform(0.5, 20), 2)
   else:
       amount = round(random.uniform(50, 500), 2)
   
   # Generate device type
   device_options = ['mobile', 'desktop', 'tablet']
   if is_fraud:
       device_type = random.choices(device_options, weights=[0.7, 0.2, 0.1])[0]
   else:
       device_type = random.choices(device_options, weights=[0.4, 0.5, 0.1])[0]
   
   # Generate location
   locations = [
       'California, USA', 'New York, USA', 'Texas, USA', 'Florida, USA', 
       'Illinois, USA', 'London, UK', 'Paris, France', 'Berlin, Germany', 
       'Tokyo, Japan', 'Sydney, Australia'
   ]
   
   if is_fraud:
       location = random.choice(locations[5:])  # Foreign locations
   else:
       location = random.choice(locations[:5])  # US locations
   
   # Generate VPN usage
   if is_fraud:
       is_vpn = random.choices([True, False], weights=[0.7, 0.3])[0]
   else:
       is_vpn = random.choices([True, False], weights=[0.1, 0.9])[0]
   
   # Generate card type
   card_options = ['credit', 'debit', 'gift']
   if is_fraud:
       card_type = random.choices(card_options, weights=[0.5, 0.2, 0.3])[0]
   else:
       card_type = random.choices(card_options, weights=[0.4, 0.55, 0.05])[0]
   
   # Generate status
   status_options = ['approved', 'pending', 'declined']
   if is_fraud and random.random() < 0.3:
       status = random.choices(status_options, weights=[0.6, 0.2, 0.2])[0]
   else:
       status = random.choices(status_options, weights=[0.95, 0.03, 0.02])[0]
   
   # Create transaction
   return {
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

def send_test_transactions_to_kinesis(kinesis_stream_name, num_transactions=10):
   """Send test transactions to Kinesis for demonstration"""
   if not kinesis_stream_name:
       logger.warning("Kinesis stream name not provided, skipping transaction generation")
       return
   
   try:
       kinesis = boto3.client('kinesis')
       test_transactions = generate_test_transactions(num_transactions, fraud_ratio=0.2)
       
       for transaction in test_transactions:
           # Convert transaction to JSON
           transaction_json = json.dumps(transaction)
           
           # Send to Kinesis
           response = kinesis.put_record(
               StreamName=kinesis_stream_name,
               Data=transaction_json,
               PartitionKey=transaction['transaction_id']
           )
           
           logger.info(f"Sent transaction {transaction['transaction_id']} to Kinesis: Shard {response['ShardId']}")
       
       logger.info(f"Sent {len(test_transactions)} test transactions to Kinesis stream {kinesis_stream_name}")
       return test_transactions
   except Exception as e:
       logger.error(f"Error sending test transactions to Kinesis: {str(e)}")
       return None
   
def check_dynamodb_for_results(transaction_ids, table_name):
    """Check DynamoDB for processing results of test transactions"""
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    results = []
    
    for tx_id in transaction_ids:
        try:
            response = table.get_item(Key={'transaction_id': tx_id})
            if 'Item' in response:
                results.append(response['Item'])
                logger.info(f"Found transaction {tx_id} in DynamoDB")
            else:
                logger.info(f"Transaction {tx_id} not found in DynamoDB yet.")
        except Exception as e:
            logger.error(f"Error retrieving transaction {tx_id}: {str(e)}")
    
    return results

def visualize_and_save_metrics(y_test, y_pred, y_pred_proba, output_dir=None):
    """Create and save visualizations of model performance metrics"""
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Print classification report
    report = classification_report(y_test, y_pred)
    logger.info("\nClassification Report:")
    logger.info(report)
    
    # Create and save confusion matrix
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logger.info(str(cm))
    
    # If output directory is specified, save visualizations
    if output_dir:
        # Confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Random prediction line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        
        # Threshold optimization plot
        thresholds_to_try = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        threshold_metrics = []
        
        for threshold in thresholds_to_try:
            # Convert probabilities to predictions using this threshold
            y_pred_at_threshold = [1 if p >= threshold else 0 for p in y_pred_proba]
            
            # Calculate confusion matrix at this threshold
            cm_at_threshold = confusion_matrix(y_test, y_pred_at_threshold)
            
            # Extract metrics
            tp = cm_at_threshold[1, 1]  # True positives
            fn = cm_at_threshold[1, 0]  # False negatives
            fp = cm_at_threshold[0, 1]  # False positives
            tn = cm_at_threshold[0, 0]  # True negatives
            
            # Calculate rates
            detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
            false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Store metrics
            threshold_metrics.append({
                'threshold': threshold,
                'detection_rate': detection_rate,
                'false_alarm_rate': false_alarm_rate,
                'precision': precision,
                'true_positives': int(tp),
                'false_negatives': int(fn),
                'false_positives': int(fp),
                'true_negatives': int(tn)
            })
        
        # Plot metrics across different thresholds
        plt.figure(figsize=(10, 6))
        thresholds = [m['threshold'] for m in threshold_metrics]
        detection_rates = [m['detection_rate'] for m in threshold_metrics]
        false_alarm_rates = [m['false_alarm_rate'] for m in threshold_metrics]
        precisions = [m['precision'] for m in threshold_metrics]
        
        plt.plot(thresholds, detection_rates, 'bo-', label='Detection Rate')
        plt.plot(thresholds, false_alarm_rates, 'ro-', label='False Alarm Rate')
        plt.plot(thresholds, precisions, 'go-', label='Precision')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title('Performance Metrics at Different Thresholds')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'threshold_metrics.png'))
        
        # Find optimal threshold
        good_thresholds = [m for m in threshold_metrics if m['detection_rate'] >= 0.8]
        if good_thresholds:
            recommended = min(good_thresholds, key=lambda x: x['false_alarm_rate'])
            logger.info("\nRecommended threshold based on business needs:")
            logger.info(f"Threshold: {recommended['threshold']}")
            logger.info(f"Detection Rate: {recommended['detection_rate']:.2%}")
            logger.info(f"False Alarm Rate: {recommended['false_alarm_rate']:.2%}")
            logger.info(f"Precision: {recommended['precision']:.2%}")
        else:
            logger.info("No threshold meets minimum detection rate requirement of 80%")
            
        # Save threshold metrics
        with open(os.path.join(output_dir, 'threshold_metrics.json'), 'w') as f:
            json.dump(threshold_metrics, f, indent=2, default=str)
    
    # Calculate business metrics
    fraud_detected = cm[1, 1]  # True positives
    fraud_missed = cm[1, 0]    # False negatives
    false_alarms = cm[0, 1]    # False positives
    correct_negatives = cm[0, 0]  # True negatives
    
    detection_rate = fraud_detected / (fraud_detected + fraud_missed) if (fraud_detected + fraud_missed) > 0 else 0
    false_positive_rate = false_alarms / (false_alarms + correct_negatives) if (false_alarms + correct_negatives) > 0 else 0
    precision = fraud_detected / (fraud_detected + false_alarms) if (fraud_detected + false_alarms) > 0 else 0
    
    logger.info("\nBusiness Impact Metrics:")
    logger.info(f"Fraud Detection Rate: {detection_rate:.2%}")
    logger.info(f"False Alarm Rate: {false_positive_rate:.2%}")
    logger.info(f"Precision: {precision:.2%}")
    
    return {
        'roc_auc': roc_auc,
        'detection_rate': detection_rate,
        'false_positive_rate': false_positive_rate,
        'precision': precision,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'threshold_metrics': threshold_metrics if 'threshold_metrics' in locals() else None
    }

def monitor_kinesis_transactions(stack_outputs, num_test_transactions=10):
    """Send test transactions to Kinesis and monitor results in DynamoDB"""
    if not stack_outputs or not stack_outputs.get('KinesisStreamName') or not stack_outputs.get('DynamoDBTableName'):
        logger.warning("Required configuration missing for Kinesis testing")
        return None
    
    kinesis_stream_name = stack_outputs.get('KinesisStreamName')
    dynamodb_table = stack_outputs.get('DynamoDBTableName')
    
    logger.info(f"Sending {num_test_transactions} test transactions to Kinesis stream: {kinesis_stream_name}")
    transactions = send_test_transactions_to_kinesis(kinesis_stream_name, num_test_transactions)
    
    if not transactions:
        logger.error("Failed to send test transactions to Kinesis")
        return None
    
    # Get transaction IDs
    tx_ids = [tx['transaction_id'] for tx in transactions]
    
    # Wait for Lambda to process (with a timeout)
    max_wait_time = 60  # seconds
    max_retries = 6
    wait_time = max_wait_time / max_retries
    
    for retry in range(max_retries):
        logger.info(f"Waiting for Lambda processing, retry {retry+1}/{max_retries}...")
        time.sleep(wait_time)
        
        # Check if transactions appear in DynamoDB
        results = check_dynamodb_for_results(tx_ids, dynamodb_table)
        
        if results:
            logger.info(f"Found {len(results)} out of {len(tx_ids)} transactions in DynamoDB")
            if len(results) >= len(tx_ids) * 0.5:  # If at least half are processed, consider it successful
                break
    
    if not results:
        logger.warning("No results found in DynamoDB after waiting")
    else:
        # Analyze results
        df_results = pd.DataFrame(results)
        
        # Count fraud predictions
        fraud_count = sum(1 for item in results if item.get('is_fraud', False))
        logger.info(f"Detected {fraud_count} potential fraudulent transactions out of {len(results)}")
        
        # Get average fraud probability
        avg_fraud_prob = sum(float(item.get('fraud_probability', 0)) for item in results) / len(results)
        logger.info(f"Average fraud probability: {avg_fraud_prob:.2%}")
    
    return {
        'sent_transactions': len(transactions),
        'processed_transactions': len(results) if results else 0,
        'fraud_detected': fraud_count if 'fraud_count' in locals() else 0,
        'avg_fraud_probability': avg_fraud_prob if 'avg_fraud_prob' in locals() else None
    }

if __name__ == "__main__":
   args = parse_args()
   
   # Train model and create endpoint
   result = train_model_and_create_endpoint(args)
   
   # Print result summary
   logger.info("\n===== Result Summary =====")
   logger.info(f"Model Name: {result['model_name']}")
   logger.info(f"Model Data: {result['model_data']}")
   
   if result['endpoint_name']:
       logger.info(f"Endpoint Name: {result['endpoint_name']}")
       logger.info("\nModel Metrics:")
       logger.info(f"ROC AUC: {result['model_metrics']['roc_auc']:.4f}")
       logger.info(f"Detection Rate: {result['model_metrics']['detection_rate']:.2%}")
       logger.info(f"False Positive Rate: {result['model_metrics']['false_positive_rate']:.2%}")
       logger.info(f"Precision: {result['model_metrics']['precision']:.2%}")
   
   # Optional: Send test transactions to Kinesis
   stack_outputs = get_stack_outputs(args.stack_name)
   if stack_outputs and stack_outputs.get("KinesisStreamName"):
       kinesis_stream_name = stack_outputs.get("KinesisStreamName")
       logger.info(f"\nSending test transactions to Kinesis stream {kinesis_stream_name}")
       send_test_transactions_to_kinesis(kinesis_stream_name, num_transactions=5)

   # Write results to output directory if specified
   if args.output_data_dir:
       os.makedirs(args.output_data_dir, exist_ok=True)
       with open(os.path.join(args.output_data_dir, 'result.json'), 'w') as f:
           json.dump(result, f, indent=2, default=str)

# In the main function, add this after the model deployment:
if args.auto_deploy:
    # After model deployment is complete, test the entire pipeline
    logger.info("\n===== Testing End-to-End Pipeline =====")
    
    # Save visualizations to output directory
    if args.output_data_dir:
        viz_output_dir = os.path.join(args.output_data_dir, 'visualizations')
        metrics = visualize_and_save_metrics(y_test, y_pred, y_pred_proba, output_dir=viz_output_dir)
        logger.info(f"Saved model performance visualizations to {viz_output_dir}")
    
    # Test Kinesis pipeline if configuration is available
    if stack_outputs and stack_outputs.get('KinesisStreamName') and stack_outputs.get('DynamoDBTableName'):
        logger.info("\n===== Testing Kinesis Pipeline =====")
        kinesis_results = monitor_kinesis_transactions(stack_outputs, num_test_transactions=5)
        
        if kinesis_results and args.output_data_dir:
            # Save Kinesis test results
            with open(os.path.join(args.output_data_dir, 'kinesis_test_results.json'), 'w') as f:
                json.dump(kinesis_results, f, indent=2)