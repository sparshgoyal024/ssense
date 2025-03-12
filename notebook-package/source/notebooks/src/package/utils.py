import os
import sys
import boto3
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_current_folder(globals_dict):
    """Get the directory of the current file"""
    current_file = globals_dict.get('__file__')
    if current_file:
        return os.path.dirname(os.path.abspath(current_file))
    else:
        return os.getcwd()

def read_json_file(file_path):
    """Read a JSON file and return its contents"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {str(e)}")
        return None

def write_json_file(data, file_path):
    """Write data to a JSON file"""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error writing JSON file {file_path}: {str(e)}")
        return False

def upload_to_s3(file_path, bucket, key):
    """Upload a file to S3"""
    try:
        s3 = boto3.resource('s3')
        s3.Bucket(bucket).upload_file(file_path, key)
        logger.info(f"Uploaded {file_path} to s3://{bucket}/{key}")
        return True
    except Exception as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        return False

def download_from_s3(bucket, key, file_path):
    """Download a file from S3"""
    try:
        s3 = boto3.resource('s3')
        s3.Bucket(bucket).download_file(key, file_path)
        logger.info(f"Downloaded s3://{bucket}/{key} to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading from S3: {str(e)}")
        return False

def get_location_risk(location):
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