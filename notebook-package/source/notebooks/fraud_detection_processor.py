import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
import time

# Initialize SageMaker session and role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Configure the script processor
script_processor = ScriptProcessor(
    command=['python3'],
    image_uri='763104351884.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    sagemaker_session=sagemaker_session,
    max_runtime_in_seconds=7200  # 2 hours maximum runtime
)

# Stack info
stack_name = 'fraudssenseassessment'
s3_bucket = 'fraudassessmentssensemodel'  # Replace with your bucket name

# Start the processing job
processing_job = script_processor.run(
    code='sagemaker_fraud_detection.py',
    arguments=[
        '--stack-name', stack_name,
        '--s3-bucket', s3_bucket,
        '--auto-deploy', 'True',
        '--solution-prefix', 'ssense-fraud'
    ],
    outputs=[
        ProcessingOutput(
            output_name='results',
            source='/opt/ml/processing/output/',
            destination=f's3://{s3_bucket}/processing-outputs/'
        )
    ],
    wait=False
)

# Get the processing job name
processing_job_name = processing_job.job_name
print(f"Started processing job: {processing_job_name}")