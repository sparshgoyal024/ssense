/**
 * Fraud Detection System - Infrastructure as Code
 *
 * This Terraform configuration defines the AWS resources needed for
 * the fraud detection pipeline.
 */

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Deployment environment (dev, staging, production)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "fraud-detection"
}

# Random suffix for unique resource names
resource "random_id" "suffix" {
  byte_length = 4
}

# S3 bucket for storing transaction data and model artifacts
resource "aws_s3_bucket" "data_bucket" {
  bucket = "${var.project_name}-data-${random_id.suffix.hex}"
  force_destroy = true
  
  tags = {
    Name        = "Fraud Detection Data"
    Environment = var.environment
    Project     = var.project_name
  }
}

# S3 bucket versioning
resource "aws_s3_bucket_versioning" "data_bucket_versioning" {
  bucket = aws_s3_bucket.data_bucket.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# Kinesis Data Stream for real-time transaction data
resource "aws_kinesis_stream" "transaction_stream" {
  name             = "${var.project_name}-stream"
  shard_count      = 1
  retention_period = 24
  
  tags = {
    Name        = "Transaction Data Stream"
    Environment = var.environment
    Project     = var.project_name
  }
}

# DynamoDB table for storing fraud detection results
resource "aws_dynamodb_table" "results_table" {
  name           = "${var.project_name}-results"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "transaction_id"
  
  attribute {
    name = "transaction_id"
    type = "S"
  }
  
  attribute {
    name = "user_id"
    type = "S"
  }
  
  global_secondary_index {
    name               = "UserIDIndex"
    hash_key           = "user_id"
    projection_type    = "ALL"
    write_capacity     = 0
    read_capacity      = 0
  }
  
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
  
  tags = {
    Name        = "Fraud Detection Results"
    Environment = var.environment
    Project     = var.project_name
  }
}

# SNS Topic for fraud alerts - modified to handle pre-existing topic
resource "aws_sns_topic" "fraud_alerts" {
  name = "${var.project_name}-alerts"
  
  # Remove tags to avoid conflicts with existing resources
  # tags = {
  #   Name        = "Fraud Detection Alerts"
  #   Environment = var.environment
  #   Project     = var.project_name
  # }
  
  # Add lifecycle block to prevent destruction
  lifecycle {
    prevent_destroy = true
    ignore_changes = [
      tags
    ]
  }
}

# IAM role for Lambda function
resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name        = "Fraud Detection Lambda Role"
    Environment = var.environment
    Project     = var.project_name
  }
}

# IAM policy for Lambda function
resource "aws_iam_policy" "lambda_policy" {
  name        = "${var.project_name}-lambda-policy"
  description = "Policy for fraud detection Lambda function"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "kinesis:GetRecords",
          "kinesis:GetShardIterator",
          "kinesis:DescribeStream",
          "kinesis:ListShards",
        ]
        Effect   = "Allow"
        Resource = aws_kinesis_stream.transaction_stream.arn
      },
      {
        Action = [
          "s3:PutObject",
          "s3:GetObject"
        ]
        Effect   = "Allow"
        Resource = [
          "${aws_s3_bucket.data_bucket.arn}/*"
        ]
      },
      {
        Action = [
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:UpdateItem",
          "dynamodb:Query"
        ]
        Effect   = "Allow"
        Resource = aws_dynamodb_table.results_table.arn
      },
      {
        Action = [
          "sns:Publish"
        ]
        Effect   = "Allow"
        Resource = aws_sns_topic.fraud_alerts.arn
      },
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Effect   = "Allow"
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Action = [
          "sagemaker:InvokeEndpoint"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

# Attach policy to IAM role
resource "aws_iam_role_policy_attachment" "lambda_policy_attachment" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.lambda_policy.arn
}

  # Lambda function for processing transactions
resource "aws_lambda_function" "fraud_detection_lambda" {
  function_name    = "${var.project_name}-processor"
  role             = aws_iam_role.lambda_role.arn
  handler          = "fraud_detector.handler"
  runtime          = "python3.9"
  timeout          = 60
  memory_size      = 256
  
  # Use placeholder zip code content
  filename         = "${path.module}/lambda_function.zip"
  
  # Generate a dummy lambda zip if it doesn't exist
  provisioner "local-exec" {
    command = "echo 'placeholder' > temp.txt && zip ${path.module}/lambda_function.zip temp.txt && rm temp.txt"
    interpreter = ["bash", "-c"]
  }
  
  environment {
    variables = {
      DYNAMODB_TABLE     = aws_dynamodb_table.results_table.name
      ALERT_TOPIC_ARN    = aws_sns_topic.fraud_alerts.arn
      ENVIRONMENT        = var.environment
      SAGEMAKER_ENDPOINT = var.sagemaker_endpoint
      RISK_THRESHOLD     = "0.7"
    }
  }
  
  tags = {
    Name        = "Fraud Detection Processor"
    Environment = var.environment
    Project     = var.project_name
  }

  depends_on = [
    aws_iam_role_policy_attachment.lambda_policy_attachment
  ]
}

# Lambda event source mapping for Kinesis
resource "aws_lambda_event_source_mapping" "kinesis_lambda_mapping" {
  event_source_arn  = aws_kinesis_stream.transaction_stream.arn
  function_name     = aws_lambda_function.fraud_detection_lambda.function_name
  starting_position = "LATEST"
  batch_size        = 100
}

# CloudWatch Alarm for monitoring fraud detection
resource "aws_cloudwatch_metric_alarm" "fraud_detection_alarm" {
  alarm_name          = "${var.project_name}-high-fraud-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "FraudDetectionCount"
  namespace           = "CustomMetrics/FraudDetection"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "This alarm monitors for high fraud detection rates"
  alarm_actions       = [aws_sns_topic.fraud_alerts.arn]
  
  dimensions = {
    Environment = var.environment
  }
  
  tags = {
    Name        = "Fraud Detection Alarm"
    Environment = var.environment
    Project     = var.project_name
  }
}

# SageMaker Resources for Prototyping

# IAM role for SageMaker
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_name}-sagemaker-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name        = "Fraud Detection SageMaker Role"
    Environment = var.environment
    Project     = var.project_name
  }
}

# IAM policy for SageMaker execution
resource "aws_iam_policy" "sagemaker_policy" {
  name        = "${var.project_name}_sagemaker_exec_policy"
  description = "Policy for SageMaker execution role"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "${aws_s3_bucket.data_bucket.arn}",
          "${aws_s3_bucket.data_bucket.arn}/*",
          "${aws_s3_bucket.training_data_bucket.arn}",
          "${aws_s3_bucket.training_data_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability"
        ]
        Resource = "arn:aws:ecr:${var.aws_region}:121021644041:repository/sagemaker-scikit-learn"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      }
    ]
  })
}

# Attach policies to SageMaker role
resource "aws_iam_role_policy_attachment" "sagemaker_policy_attachment" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = aws_iam_policy.sagemaker_policy.arn
}

# Also attach the AWS managed SageMaker policy for good measure
resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Create SageMaker Model - Using fully-qualified ARN for the image
resource "aws_sagemaker_model" "fraud_model" {
  name               = "${var.project_name}-model"
  execution_role_arn = aws_iam_role.sagemaker_role.arn
  
  primary_container {
    # Use a BYOC (bring your own container) approach with a simple model
    # image = "121021644041.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3"
    
    # Use a built-in algorithm instead (no ECR permissions needed)
    image = "174872318107.dkr.ecr.${var.aws_region}.amazonaws.com/kmeans:1"
    
    model_data_url = "s3://${aws_s3_bucket.data_bucket.id}/models/fraud_detection_model.tar.gz"
  }
  
  tags = {
    Name        = "Fraud Detection Model"
    Environment = var.environment
    Project     = var.project_name
  }

  # This is a prototype so we'll create a placeholder model config
  provisioner "local-exec" {
    command = <<EOT
      # Create a simple model artifact for prototyping
      mkdir -p tmp_model
      echo '{
        "model_type": "rule_based",
        "threshold": 0.7,
        "version": "0.1"
      }' > tmp_model/model_config.json
      touch tmp_model/inference.py
      tar -czf model.tar.gz -C tmp_model .
      aws s3 cp model.tar.gz s3://${aws_s3_bucket.data_bucket.id}/models/fraud_detection_model.tar.gz || echo "Failed to upload model, continuing anyway"
      rm -rf tmp_model model.tar.gz
    EOT
  }
}

# Create SageMaker Endpoint Configuration
resource "aws_sagemaker_endpoint_configuration" "fraud_endpoint_config" {
  name = "${var.project_name}-endpoint-config"
  
  production_variants {
    variant_name           = "default"
    model_name             = aws_sagemaker_model.fraud_model.name
    instance_type          = "ml.t2.medium"  # Smallest instance for prototyping
    initial_instance_count = 1
  }
  
  tags = {
    Name        = "Fraud Detection Endpoint Config"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Create SageMaker Endpoint
resource "aws_sagemaker_endpoint" "fraud_endpoint" {
  name                 = "${var.project_name}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.fraud_endpoint_config.name
  
  tags = {
    Name        = "Fraud Detection Endpoint"
    Environment = var.environment
    Project     = var.project_name
  }
}

# SageMaker Variables
variable "sagemaker_endpoint" {
  description = "SageMaker endpoint name for fraud detection"
  type        = string
  default     = "fraud-detection-endpoint"
}

# S3 bucket for historical training data
resource "aws_s3_bucket" "training_data_bucket" {
  bucket = "${var.project_name}-training-${random_id.suffix.hex}"
  force_destroy = true
  
  tags = {
    Name        = "Fraud Detection Training Data"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Upload sample historical data for training (for prototype)
resource "null_resource" "upload_sample_data" {
  provisioner "local-exec" {
    command = <<EOT
      # Create sample training data
      echo 'transaction_id,user_id,amount,device_type,location,is_vpn,card_type,status,is_fraud
T1,U1234,120.50,mobile,"California, USA",false,credit,approved,0
T2,U2345,1500.75,mobile,"Berlin, Germany",true,gift,approved,1
T3,U3456,50.25,desktop,"New York, USA",false,debit,approved,0
T4,U4567,2000.00,mobile,"Paris, France",true,credit,declined,1
T5,U5678,75.30,desktop,"Texas, USA",false,credit,approved,0' > sample_transactions.csv
      
      # Upload to S3
      aws s3 cp sample_transactions.csv s3://${aws_s3_bucket.training_data_bucket.id}/historical/sample_transactions.csv
      
      # Clean up
      rm sample_transactions.csv
    EOT
  }

  depends_on = [
    aws_s3_bucket.training_data_bucket
  ]
}

# Comment out the problematic Firehose resource for now to get the rest working
# We can fix and re-add it later
/*
resource "aws_kinesis_firehose_delivery_stream" "transaction_delivery_stream" {
  name        = "${var.project_name}-delivery-stream"
  destination = "extended_s3"
  
  kinesis_source_configuration {
    kinesis_stream_arn = aws_kinesis_stream.transaction_stream.arn
    role_arn           = aws_iam_role.firehose_role.arn
  }
  
  extended_s3_configuration {
    role_arn           = aws_iam_role.firehose_role.arn
    bucket_arn         = aws_s3_bucket.data_bucket.arn
    prefix             = "transactions/"
    buffer_size        = 5
    buffer_interval    = 300
    compression_format = "GZIP"
  }
}
*/

# IAM role for Firehose - commenting out since we disabled Firehose
/*
resource "aws_iam_role" "firehose_role" {
  name = "${var.project_name}-firehose-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "firehose.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name        = "Firehose Role"
    Environment = var.environment
    Project     = var.project_name
  }
}

# IAM policy for Firehose
resource "aws_iam_policy" "firehose_policy" {
  name        = "${var.project_name}-firehose-policy"
  description = "Policy for Firehose to read from Kinesis and write to S3"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "kinesis:DescribeStream",
          "kinesis:GetShardIterator",
          "kinesis:GetRecords"
        ]
        Effect   = "Allow"
        Resource = aws_kinesis_stream.transaction_stream.arn
      },
      {
        Action = [
          "s3:AbortMultipartUpload",
          "s3:GetBucketLocation",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:ListBucketMultipartUploads",
          "s3:PutObject"
        ]
        Effect   = "Allow"
        Resource = [
          aws_s3_bucket.data_bucket.arn,
          "${aws_s3_bucket.data_bucket.arn}/*"
        ]
      }
    ]
  })
}

# Attach policy to Firehose role
resource "aws_iam_role_policy_attachment" "firehose_policy_attachment" {
  role       = aws_iam_role.firehose_role.name
  policy_arn = aws_iam_policy.firehose_policy.arn
}
*/

# Output values - corrected format
output "kinesis_stream_name" {
  value       = aws_kinesis_stream.transaction_stream.name
  description = "Kinesis Stream Name"
}

output "s3_bucket_name" {
  value       = aws_s3_bucket.data_bucket.id
  description = "S3 Bucket Name"
}

output "training_bucket_name" {
  value       = aws_s3_bucket.training_data_bucket.id
  description = "Training Data Bucket Name"
}

output "dynamodb_table_name" {
  value       = aws_dynamodb_table.results_table.name
  description = "DynamoDB Table Name"
}

output "lambda_function_name" {
  value       = aws_lambda_function.fraud_detection_lambda.function_name
  description = "Lambda Function Name"
}

output "alert_topic_arn" {
  value       = aws_sns_topic.fraud_alerts.arn
  description = "SNS Alert Topic ARN"
}

output "sagemaker_endpoint_name" {
  value       = aws_sagemaker_endpoint.fraud_endpoint.name
  description = "SageMaker Endpoint Name"
}