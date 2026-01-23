#!/bin/bash
# AWS Deployment Script for Nepali Braille OBR API
# This script automates the deployment to AWS ECS Fargate

set -e

# Configuration
export AWS_REGION=${AWS_REGION:-us-east-1}
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPOSITORY_NAME=nepali-braille-api
export CLUSTER_NAME=braille-api-cluster
export SERVICE_NAME=braille-api-service
export TASK_FAMILY=nepali-braille-api

echo "=========================================="
echo "AWS Deployment Script"
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

# Step 6: Create CloudWatch Log Group
echo "Step 6: Creating CloudWatch log group..."
aws logs create-log-group --log-group-name /ecs/$REPOSITORY_NAME --region $AWS_REGION 2>/dev/null || echo "Log group already exists"

# Step 7: Create ECS Cluster
echo "Step 7: Creating ECS cluster..."
aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $AWS_REGION 2>/dev/null || echo "Cluster already exists"

# Step 8: Register Task Definition
echo "Step 8: Registering task definition..."
IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:latest"

cat > task-definition.json << EOF
{
  "family": "$TASK_FAMILY",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "braille-api",
      "image": "$IMAGE_URI",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/$REPOSITORY_NAME",
          "awslogs-region": "$AWS_REGION",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
EOF

aws ecs register-task-definition --cli-input-json file://task-definition.json --region $AWS_REGION

# Step 9: Get VPC and Subnet
echo "Step 9: Getting VPC configuration..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region $AWS_REGION)
SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[0].SubnetId" --output text --region $AWS_REGION)

# Step 10: Create Security Group
echo "Step 10: Creating security group..."
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name braille-api-sg \
    --description "Security group for Braille API" \
    --vpc-id $VPC_ID \
    --region $AWS_REGION \
    --query 'GroupId' \
    --output text 2>/dev/null || \
    aws ec2 describe-security-groups --filters "Name=group-name,Values=braille-api-sg" --query "SecurityGroups[0].GroupId" --output text --region $AWS_REGION)

# Allow inbound traffic
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 8080 \
    --cidr 0.0.0.0/0 \
    --region $AWS_REGION 2>/dev/null || echo "Security group rule already exists"

# Step 11: Create or Update Service
echo "Step 11: Creating/updating ECS service..."
aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $AWS_REGION --query 'services[0].status' --output text 2>/dev/null | grep -q "ACTIVE" && \
    SERVICE_EXISTS=true || SERVICE_EXISTS=false

if [ "$SERVICE_EXISTS" = true ]; then
    echo "Service exists, updating..."
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service $SERVICE_NAME \
        --task-definition $TASK_FAMILY \
        --region $AWS_REGION > /dev/null
else
    echo "Creating new service..."
    aws ecs create-service \
        --cluster $CLUSTER_NAME \
        --service-name $SERVICE_NAME \
        --task-definition $TASK_FAMILY \
        --desired-count 1 \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
        --region $AWS_REGION > /dev/null
fi

# Cleanup
rm -f task-definition.json

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "To get your service URL, run:"
echo "  aws ecs list-tasks --cluster $CLUSTER_NAME --service-name $SERVICE_NAME --region $AWS_REGION"
echo ""
echo "To view logs:"
echo "  aws logs tail /ecs/$REPOSITORY_NAME --follow --region $AWS_REGION"
echo ""
echo "To check service status:"
echo "  aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $AWS_REGION"
