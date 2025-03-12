#!/bin/bash
# set -e

# # This script runs after the notebook lifecycle configuration completes

# echo "Running on-start.sh script..."

# # Create package directory if it doesn't exist
# mkdir -p /home/ec2-user/SageMaker/src/package

# # Copy the config.py to the correct location (assuming it was downloaded)
# if [ -f "/home/ec2-user/SageMaker/config.py" ]; then
#   cp /home/ec2-user/SageMaker/config.py /home/ec2-user/SageMaker/src/package/config.py
#   echo "Copied config.py to package directory"
# fi

# # Download the config.py from S3 if needed
# if [ ! -f "/home/ec2-user/SageMaker/src/package/config.py" ]; then
#   aws s3 cp s3://767089282839-notebook-code/fraud-detection/notebooks/config.py /home/ec2-user/SageMaker/src/package/config.py
#   echo "Downloaded config.py from S3"
# fi

# # Ensure proper permissions for files and directories
# chmod -R 755 /home/ec2-user/SageMaker/src
# chown -R ec2-user:ec2-user /home/ec2-user/SageMaker/src

# echo "on-start.sh completed successfully"



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

# get stack outputs for use in creating the processing job
STACK_NAME=$(aws cloudformation describe-stacks --query 'Stacks[?StackStatus==`CREATE_COMPLETE`].StackName' --output text | grep ssense-fraud)
NOTEBOOK_ROLE=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`NotebookRoleArn`].OutputValue' --output text)
S3_BUCKET=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`ModelDataBucket`].OutputValue' --output text)

# create a timestamp for unique job name
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
JOB_NAME="FraudDetectionProcessing-${TIMESTAMP}"

# create the processing job
echo "Creating SageMaker processing job: $JOB_NAME"
aws sagemaker create-processing-job \
    --processing-job-name $JOB_NAME \
    --processing-resources '{"ClusterConfig": {"InstanceCount": 1, "InstanceType": "ml.m5.xlarge", "VolumeSizeInGB": 30}}' \
    --stopping-condition '{"MaxRuntimeInSeconds": 7200}' \
    --app-specification '{"ImageUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3", "ContainerEntrypoint": ["python3", "fraud_detection_processor.py"], "ContainerArguments": ["--stack-name", "'$STACK_NAME'", "--s3-bucket", "'$S3_BUCKET'", "--solution-prefix", "ssense-fraud", "--auto-deploy", "True"]}' \
    --role-arn $NOTEBOOK_ROLE \
    --processing-inputs '[{"InputName": "code", "S3Input": {"S3Uri": "s3://767089282839-notebook-code/fraud-detection/notebooks/fraud_detection_processor.py", "LocalPath": "/opt/ml/processing/input/code", "S3DataType": "S3Prefix", "S3InputMode": "File", "S3DataDistributionType": "FullyReplicated"}}]' \
    --processing-output-config '{"Outputs": [{"OutputName": "results", "S3Output": {"S3Uri": "s3://'$S3_BUCKET'/processing-outputs/", "LocalPath": "/opt/ml/processing/output", "S3UploadMode": "EndOfJob"}}]}'

echo "SageMaker processing job created: $JOB_NAME"

# Ensure proper permissions for files and directories
chmod -R 755 /home/ec2-user/SageMaker/src
chown -R ec2-user:ec2-user /home/ec2-user/SageMaker/src

echo "on-start.sh completed successfully"