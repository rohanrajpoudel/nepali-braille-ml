#!/bin/bash
# AWS App Runner Deployment Script (Simpler Alternative)
# This script automates the deployment to AWS App Runner

set -e

# Configuration
export AWS_REGION=${AWS_REGION:-us-east-1}
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPOSITORY_NAME=nepali-braille-api
export SERVICE_NAME=nepali-braille-api

echo "=========================================="
echo "AWS App Runner Deployment Script"
echo "=========================================="
echo "Region: $AWS_REGION"
echo "Account ID: $AWS_ACCOUNT_ID"
echo "Repository: $REPOSITORY_NAME"
echo ""

# Step 1: Create ECR Repository
echo "Step 1: Creating ECR repository..."
aws ecr create-repository \
    --repository-name $REPOSITORY_NAME \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true 2>/dev/null || echo "Repository already exists"

# Step 2: Login to ECR
echo "Step 2: Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Step 3: Build and Push Image
echo "Step 3: Building Docker image..."
cd "$(dirname "$0")"
docker build -t $REPOSITORY_NAME:latest .

echo "Step 4: Tagging image..."
docker tag $REPOSITORY_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:latest

echo "Step 5: Pushing image to ECR..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:latest

# Step 6: Create App Runner Service
echo "Step 6: Creating App Runner service..."
IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:latest"

cat > apprunner-service.json << EOF
{
  "ServiceName": "$SERVICE_NAME",
  "SourceConfiguration": {
    "ImageRepository": {
      "ImageIdentifier": "$IMAGE_URI",
      "ImageConfiguration": {
        "Port": "8080",
        "RuntimeEnvironmentVariables": {}
      },
      "ImageRepositoryType": "ECR"
    },
    "AutoDeploymentsEnabled": true
  },
  "InstanceConfiguration": {
    "Cpu": "2 vCPU",
    "Memory": "4 GB"
  }
}
EOF

# Check if service exists
SERVICE_ARN=$(aws apprunner list-services --region $AWS_REGION --query "ServiceSummaryList[?ServiceName=='$SERVICE_NAME'].ServiceArn" --output text 2>/dev/null || echo "")

if [ -z "$SERVICE_ARN" ]; then
    echo "Creating new App Runner service..."
    aws apprunner create-service \
        --cli-input-json file://apprunner-service.json \
        --region $AWS_REGION > service-output.json
    
    SERVICE_ARN=$(cat service-output.json | grep -o '"ServiceArn": "[^"]*' | cut -d'"' -f4)
    echo "Service created: $SERVICE_ARN"
else
    echo "Service already exists, updating..."
    aws apprunner update-service \
        --service-arn $SERVICE_ARN \
        --source-configuration ImageRepository={ImageIdentifier=$IMAGE_URI,ImageConfiguration={Port=8080},ImageRepositoryType=ECR} \
        --region $AWS_REGION > service-output.json
fi

# Cleanup
rm -f apprunner-service.json service-output.json

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "To get your service URL, run:"
echo "  aws apprunner describe-service --service-arn $SERVICE_ARN --region $AWS_REGION --query 'Service.ServiceUrl' --output text"
echo ""
echo "Or check the AWS Console:"
echo "  https://console.aws.amazon.com/apprunner/home?region=$AWS_REGION"
