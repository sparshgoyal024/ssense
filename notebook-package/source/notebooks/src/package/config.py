import json
import boto3
import os
from pathlib import Path

# Try to get stack name from environment or use a default
STACK_NAME = os.environ.get('STACK_NAME', 'fraudssenseassessment')
AWS_REGION = "us-west-2"  # Fixed region as per requirement

# Initialize CloudFormation client
cfn = boto3.client('cloudformation', region_name=AWS_REGION)

# Function to get all CloudFormation outputs
def get_stack_outputs():
    try:
        # First check if outputs file exists
        outputs_file = Path('/home/ec2-user/SageMaker/stack_outputs.json')
        if outputs_file.exists():
            try:
                with open(outputs_file, 'r') as f:
                    outputs_json = json.load(f)
                    outputs = {}
                    for output in outputs_json:
                        outputs[output['OutputKey']] = output['OutputValue']
                    return outputs
            except Exception as file_error:
                print(f"Warning: Could not read stack outputs file: {str(file_error)}")
                
        # If file doesn't exist or can't be read, query CloudFormation
        response = cfn.describe_stacks(StackName=STACK_NAME)
        
        # Convert list of outputs to dictionary
        if 'Stacks' in response and len(response['Stacks']) > 0:
            outputs = {}
            for output in response['Stacks'][0]['Outputs']:
                outputs[output['OutputKey']] = output['OutputValue']
            
            # Set the values from stack outputs or use defaults if not found
            result = {
                "FraudStackName": STACK_NAME,
                "SolutionPrefix": "ssense-fraud",
                "AwsAccountId": boto3.client('sts').get_caller_identity().get('Account'),
                "AwsRegion": AWS_REGION,
                "IamRole": outputs.get('NotebookRoleArn', ''),
                "ModelDataBucket": outputs.get('ModelDataBucket', ''),
                "SolutionsS3Bucket": os.environ.get('NotebookCodeS3Bucket', '767089282839-notebook-code'),
                "RESTAPIGateway": outputs.get('RestApiId', ''),
                "SolutionName": "fraud-detection",
                "TestOutputsS3Bucket": outputs.get('ResultsBucket', ''),
                "DynamoDBTableName": outputs.get('DynamoDBTableName', ''),
                "DynamoDBTableArn": outputs.get('DynamoDBTableArn', ''),
                "KinesisStreamName": outputs.get('KinesisStreamName', ''),
                "KinesisStreamArn": outputs.get('KinesisStreamArn', ''),
                "SNSTopicArn": outputs.get('SNSTopicArn', ''),
                "ApiEndpoint": outputs.get('ApiEndpoint', '')
            }
            
            # Write the result to a file for debugging
            try:
                with open('/home/ec2-user/SageMaker/stack_outputs_processed.json', 'w') as f:
                    json.dump(result, f, indent=2)
            except Exception as write_error:
                print(f"Warning: Could not write processed outputs file: {str(write_error)}")
                
            return result
    except Exception as e:
        print(f"Warning: Could not retrieve CloudFormation outputs: {str(e)}")
        # Return None so we know values aren't from stack
        return None

# Get stack outputs
stack_outputs = get_stack_outputs()

# Set configuration values - prefer stack outputs but fall back to defaults if needed
if stack_outputs:
    SOLUTION_PREFIX = "ssense-fraud"
    AWS_ACCOUNT_ID = stack_outputs.get("AwsAccountId")
    SAGEMAKER_IAM_ROLE = stack_outputs.get("IamRole")
    MODEL_DATA_S3_BUCKET = stack_outputs.get("ModelDataBucket")
    SOLUTIONS_S3_BUCKET = stack_outputs.get("SolutionsS3Bucket")
    REST_API_GATEWAY = stack_outputs.get("RESTAPIGateway")
    SOLUTION_NAME = "fraud-detection"
    TEST_OUTPUTS_S3_BUCKET = stack_outputs.get("TestOutputsS3Bucket")
    DYNAMODB_TABLE = stack_outputs.get("DynamoDBTableName")
    DYNAMODB_TABLE_ARN = stack_outputs.get("DynamoDBTableArn")
    KINESIS_STREAM_NAME = stack_outputs.get("KinesisStreamName")
    KINESIS_STREAM_ARN = stack_outputs.get("KinesisStreamArn")
    SNS_TOPIC_ARN = stack_outputs.get("SNSTopicArn")
    API_ENDPOINT = stack_outputs.get("ApiEndpoint")
else:
    # We couldn't get values from CloudFormation, so we have to use defaults
    # This is not ideal but allows code to run for testing
    print("WARNING: Using default values because CloudFormation outputs could not be retrieved")
    SOLUTION_PREFIX = "ssense-fraud"
    AWS_ACCOUNT_ID = "767089282839"  # This should be fetched dynamically in production
    SAGEMAKER_IAM_ROLE = ""  # This should come from CloudFormation
    MODEL_DATA_S3_BUCKET = ""  # This should come from CloudFormation
    SOLUTIONS_S3_BUCKET = "767089282839-notebook-code"
    REST_API_GATEWAY = ""  # This should come from CloudFormation
    SOLUTION_NAME = "fraud-detection"
    TEST_OUTPUTS_S3_BUCKET = ""  # This should come from CloudFormation
    DYNAMODB_TABLE = "ssense-fraud-transactions"
    DYNAMODB_TABLE_ARN = ""  # This should come from CloudFormation
    KINESIS_STREAM_NAME = "ssense-transaction-stream" 
    KINESIS_STREAM_ARN = ""  # This should come from CloudFormation
    SNS_TOPIC_ARN = ""  # This should come from CloudFormation
    API_ENDPOINT = ""  # This should come from CloudFormation

# Model configuration
MODEL_PATH = "models/fraud_detection_model.joblib"

# Feature configuration
CATEGORICAL_FEATURES = ['device_type', 'card_type']
NUMERICAL_FEATURES = ['amount', 'is_vpn', 'hour_of_day', 'is_weekend', 'location_risk']

# Print confirmation
print("Config values successfully loaded!")