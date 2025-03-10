#!/usr/bin/env python3
"""
Fraud Detection Model Training Script

This script trains a fraud detection model using historical transaction data.
It uses scikit-learn to build a simple model that can be deployed to SageMaker.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve
)
import joblib
import boto3
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fraud-model-training')

def load_data(data_path=None):
    """
    Load transaction data for training
    
    Args:
        data_path: Path to CSV data file (will generate mock data if None)
        
    Returns:
        DataFrame with transaction data
    """
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        return pd.read_csv(data_path)
    
    # Generate mock data
    logger.info("Generating mock fraud detection data")
    np.random.seed(42)
    
    # Number of transactions
    n_samples = 10000
    
    # Generate transaction IDs
    transaction_ids = [f"T{i}" for i in range(1, n_samples + 1)]
    
    # Generate user IDs
    user_ids = [f"U{np.random.randint(1, 1000)}" for _ in range(n_samples)]
    
    # Generate fraud labels (imbalanced dataset - 5% fraud)
    is_fraud = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    # Generate transaction amounts
    amounts = []
    for fraud in is_fraud:
        if fraud:
            amounts.append(abs(np.random.normal(500, 300)))  # Higher amounts for fraud
        else:
            amounts.append(abs(np.random.normal(100, 50)))   # Lower amounts for normal
    
    # Generate device types
    device_options = ['mobile', 'desktop', 'tablet']
    device_types = []
    for fraud in is_fraud:
        if fraud:
            device_types.append(np.random.choice(device_options, p=[0.7, 0.2, 0.1]))
        else:
            device_types.append(np.random.choice(device_options, p=[0.4, 0.5, 0.1]))
    
    # Generate locations
    locations = [
        'California, USA', 'New York, USA', 'Texas, USA', 'Florida, USA', 
        'Illinois, USA', 'London, UK', 'Paris, France', 'Berlin, Germany', 
        'Tokyo, Japan', 'Sydney, Australia'
    ]
    
    transaction_locations = []
    for fraud in is_fraud:
        if fraud:
            transaction_locations.append(np.random.choice(locations[5:], p=[0.25, 0.25, 0.25, 0.25]))
        else:
            transaction_locations.append(np.random.choice(locations[:5], p=[0.3, 0.3, 0.2, 0.1, 0.1]))
    
    # Generate VPN usage
    is_vpn = []
    for fraud in is_fraud:
        if fraud:
            is_vpn.append(np.random.choice([True, False], p=[0.7, 0.3]))
        else:
            is_vpn.append(np.random.choice([True, False], p=[0.1, 0.9]))
    
    # Generate card types
    card_options = ['credit', 'debit', 'gift']
    card_types = []
    for fraud in is_fraud:
        if fraud:
            card_types.append(np.random.choice(card_options, p=[0.5, 0.2, 0.3]))
        else:
            card_types.append(np.random.choice(card_options, p=[0.6, 0.35, 0.05]))
    
    # Generate transaction status
    status_options = ['approved', 'pending', 'declined']
    statuses = []
    for fraud in is_fraud:
        if fraud:
            statuses.append(np.random.choice(status_options, p=[0.6, 0.1, 0.3]))
        else:
            statuses.append(np.random.choice(status_options, p=[0.9, 0.05, 0.05]))
    
    # Create DataFrame
    df = pd.DataFrame({
        'transaction_id': transaction_ids,
        'user_id': user_ids,
        'amount': amounts,
        'device_type': device_types,
        'location': transaction_locations,
        'is_vpn': is_vpn,
        'card_type': card_types,
        'status': statuses,
        'is_fraud': is_fraud
    })
    
    return df

def engineer_features(df):
    """
    Extract and engineer features from transaction data
    
    Args:
        df: DataFrame with raw transaction data
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering features")
    
    # Create a copy to avoid modifying the original
    df_features = df.copy()
    
    # Location risk score (simplified)
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
    df_features['location_risk'] = df_features['location'].map(location_risk)
    df_features['location_risk'].fillna(0.5, inplace=True)
    
    # Amount bucketing
    df_features['amount_bucket'] = pd.cut(
        df_features['amount'],
        bins=[0, 50, 100, 200, 500, 1000, float('inf')],
        labels=['very_small', 'small', 'medium', 'large', 'very_large', 'huge']
    )
    
    # Create a feature for high-risk combination
    df_features['high_risk_combo'] = (
        (df_features['amount'] > 500) & 
        (df_features['is_vpn'] == True) & 
        (df_features['location_risk'] > 0.3)
    ).astype(int)
    
    return df_features

def train_model(df, model_type='random_forest'):
    """
    Train fraud detection model
    
    Args:
        df: DataFrame with transaction data
        model_type: Type of model to train ('random_forest' or 'logistic_regression')
        
    Returns:
        Trained model and evaluation metrics
    """
    logger.info(f"Training {model_type} model")
    
    # Engineer features
    df_features = engineer_features(df)
    
    # Select features and target
    feature_cols = ['amount', 'location_risk', 'is_vpn', 'high_risk_combo']
    categorical_cols = ['device_type', 'card_type', 'status', 'amount_bucket']
    target_col = 'is_fraud'
    
    X = df_features[feature_cols + categorical_cols]
    y = df_features[target_col]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples")
    
    # Create column transformer for preprocessing
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, feature_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Choose model type
    if model_type == 'logistic_regression':
        classifier = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
    else:  # Default to random forest
        classifier = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
    
    # Create and train pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob)
    }
    
    # Print evaluation results
    logger.info("Model Evaluation:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(conf_matrix)
    
    # Save feature importances if random forest
    if model_type == 'random_forest':
        # Get preprocessor feature names
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                for col in cols:
                    for cat in trans.categories_[0]:
                        feature_names.append(f"{col}_{cat}")
        
        # Extract feature importances
        importances = pipeline.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        logger.info("\nFeature Importances:")
        for i in range(min(20, len(feature_names))):
            if i < len(indices):
                logger.info(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    return pipeline, metrics, (X_test, y_test, y_pred, y_prob)

def plot_results(model_results, output_dir='.'):
    """
    Generate and save evaluation plots
    
    Args:
        model_results: Tuple with (X_test, y_test, y_pred, y_prob)
        output_dir: Directory to save plots
        
    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    X_test, y_test, y_pred, y_prob = model_results
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))

def save_model(model, metrics, output_dir='.', s3_bucket=None):
    """
    Save model and metrics
    
    Args:
        model: Trained model pipeline
        metrics: Dictionary with evaluation metrics
        output_dir: Directory to save model
        s3_bucket: S3 bucket to upload model (optional)
        
    Returns:
        Path to saved model
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model
    model_path = os.path.join(output_dir, 'fraud_detection_model.joblib')
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        # Convert float values for JSON serialization
        serializable_metrics = {k: float(v) for k, v in metrics.items()}
        json.dump(serializable_metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Upload to S3 if bucket specified
    if s3_bucket:
        try:
            s3 = boto3.client('s3')
            
            # Upload model
            s3.upload_file(model_path, s3_bucket, 'models/fraud_detection_model.joblib')
            logger.info(f"Model uploaded to s3://{s3_bucket}/models/fraud_detection_model.joblib")
            
            # Upload metrics
            s3.upload_file(metrics_path, s3_bucket, 'models/model_metrics.json')
            logger.info(f"Metrics uploaded to s3://{s3_bucket}/models/model_metrics.json")
            
            # Upload model as tar.gz for SageMaker
            sagemaker_dir = os.path.join(output_dir, 'sagemaker')
            os.makedirs(sagemaker_dir, exist_ok=True)
            
            # Copy model
            shutil.copy(model_path, os.path.join(sagemaker_dir, 'model.joblib'))
            
            # Copy inference script
            inference_script = """
import os
import json
import joblib
import numpy as np
import pandas as pd

# Load model once when the container starts
model_path = os.path.join('/opt/ml/model', 'model.joblib')
model = joblib.load(model_path)

def input_fn(request_body, request_content_type):
    """Parse input data payload"""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        
        # Convert to DataFrame with expected schema
        if isinstance(data, dict):
            # Single record
            df = pd.DataFrame([data])
        else:
            # Multiple records
            df = pd.DataFrame(data)
            
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make prediction with the model"""
    # Make prediction
    try:
        fraud_probability = float(model.predict_proba(input_data)[0, 1])
        is_fraud = bool(model.predict(input_data)[0])
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        # Fallback to a basic rule
        fraud_probability = 0.5 if input_data['amount'].values[0] > 500 else 0.1
        is_fraud = fraud_probability > 0.5
    
    return {
        'is_fraud': is_fraud,
        'fraud_probability': fraud_probability,
        'risk_score': fraud_probability
    }

def output_fn(prediction, response_content_type):
    """Format prediction output"""
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
"""
            
            with open(os.path.join(sagemaker_dir, 'inference.py'), 'w') as f:
                f.write(inference_script)
            
            # Create tar.gz
            import tarfile
            
            tar_path = os.path.join(output_dir, 'fraud_detection_model.tar.gz')
            with tarfile.open(tar_path, 'w:gz') as tar:
                tar.add(sagemaker_dir, arcname='.')
            
            # Upload to S3
            s3.upload_file(tar_path, s3_bucket, 'models/fraud_detection_model.tar.gz')
            logger.info(f"SageMaker model archive uploaded to s3://{s3_bucket}/models/fraud_detection_model.tar.gz")
            
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
    
    return model_path

def main():
    """Main function to train and evaluate model"""
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--data-path', type=str, default=None,
                      help='Path to training data CSV')
    parser.add_argument('--model-type', type=str, choices=['random_forest', 'logistic_regression'],
                      default='random_forest', help='Type of model to train')
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Directory to save model and results')
    parser.add_argument('--s3-bucket', type=str, default=None,
                      help='S3 bucket to upload model artifacts')
    parser.add_argument('--plot', action='store_true',
                      help='Generate evaluation plots')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.data_path)
    logger.info(f"Loaded {len(df)} transactions, {df['is_fraud'].sum()} fraudulent ({df['is_fraud'].mean()*100:.2f}%)")
    
    # Print data sample
    logger.info("\nData sample:")
    logger.info(df.head())
    
    # Train model
    model, metrics, model_results = train_model(df, model_type=args.model_type)
    
    # Save model and metrics
    save_model(model, metrics, output_dir=args.output_dir, s3_bucket=args.s3_bucket)
    
    # Generate plots if requested
    if args.plot:
        plot_results(model_results, output_dir=args.output_dir)

if __name__ == "__main__":
    main()