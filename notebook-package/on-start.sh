#!/bin/bash
set -e

# This script runs after the notebook lifecycle configuration completes

echo "Running on-start.sh script..."

# Create package directory if it doesn't exist
mkdir -p /home/ec2-user/SageMaker/src/package

# Copy the config.py to the correct location (assuming it was downloaded)
if [ -f "/home/ec2-user/SageMaker/config.py" ]; then
  cp /home/ec2-user/SageMaker/config.py /home/ec2-user/SageMaker/src/package/config.py
  echo "Copied config.py to package directory"
fi

# Download the config.py from S3 if needed
if [ ! -f "/home/ec2-user/SageMaker/src/package/config.py" ]; then
  aws s3 cp s3://767089282839-notebook-code/fraud-detection/notebooks/config.py /home/ec2-user/SageMaker/src/package/config.py
  echo "Downloaded config.py from S3"
fi

# Ensure proper permissions for files and directories
chmod -R 755 /home/ec2-user/SageMaker/src
chown -R ec2-user:ec2-user /home/ec2-user/SageMaker/src

echo "on-start.sh completed successfully"