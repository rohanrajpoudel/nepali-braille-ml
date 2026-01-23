# AWS Deployment Guide

This guide explains how to deploy the Nepali Braille OBR API to AWS for use with a Flutter app.

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **Docker** installed (for local testing)
4. **AWS credentials** configured (`aws configure`)

## Option 1: AWS ECS Fargate (Recommended)

ECS Fargate is AWS's serverless container platform, similar to Google Cloud Run.

### Step 1: Install Prerequisites

```bash
# Install AWS CLI (if not already installed)
# macOS:
brew install awscli

# Linux:
sudo apt-get install awscli

# Windows: Download from https://aws.amazon.com/cli/

# Configure AWS credentials
aws configure
# Enter your Access Key ID, Secret Access Key, region (e.g., us-east-1), and output format (json)
```

### Step 2: Create ECR Repository

ECR (Elastic Container Registry) is AWS's Docker registry.

```bash
# Set variables
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPOSITORY_NAME=nepali-braille-api

# Create ECR repository
aws ecr create-repository \
    --repository-name $REPOSITORY_NAME \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true

# Get login token and login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

### Step 3: Build and Push Docker Image

```bash
# Navigate to app directory
cd app

# Build the image
docker build -t $REPOSITORY_NAME:latest .

# Tag the image
docker tag $REPOSITORY_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:latest

# Push to ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:latest
```

### Step 4: Create ECS Cluster

```bash
# Create cluster
aws ecs create-cluster --cluster-name braille-api-cluster --region $AWS_REGION
```

### Step 5: Create Task Definition

Create a file `task-definition.json`:

```json
{
  "family": "nepali-braille-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "braille-api",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/nepali-braille-api:latest",
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
          "awslogs-group": "/ecs/nepali-braille-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

Replace `YOUR_ACCOUNT_ID` with your actual AWS account ID.

```bash
# Create CloudWatch log group
aws logs create-log-group --log-group-name /ecs/nepali-braille-api --region $AWS_REGION

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json --region $AWS_REGION
```

### Step 6: Create VPC and Security Group

```bash
# Get default VPC ID
export VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region $AWS_REGION)

# Get default subnet IDs
export SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[*].SubnetId" --output text --region $AWS_REGION | awk '{print $1}')

# Create security group
export SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name braille-api-sg \
    --description "Security group for Braille API" \
    --vpc-id $VPC_ID \
    --region $AWS_REGION \
    --query 'GroupId' \
    --output text)

# Allow inbound traffic on port 8080
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 8080 \
    --cidr 0.0.0.0/0 \
    --region $AWS_REGION
```

### Step 7: Create Application Load Balancer (Optional but Recommended)

For production, use an ALB to handle traffic:

```bash
# Create target group
export TARGET_GROUP_ARN=$(aws elbv2 create-target-group \
    --name braille-api-tg \
    --protocol HTTP \
    --port 8080 \
    --vpc-id $VPC_ID \
    --target-type ip \
    --health-check-path / \
    --region $AWS_REGION \
    --query 'TargetGroups[0].TargetGroupArn' \
    --output text)

# Create load balancer
export ALB_ARN=$(aws elbv2 create-load-balancer \
    --name braille-api-alb \
    --subnets $(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[*].SubnetId" --output text --region $AWS_REGION) \
    --security-groups $SECURITY_GROUP_ID \
    --region $AWS_REGION \
    --query 'LoadBalancers[0].LoadBalancerArn' \
    --output text)

# Create listener
aws elbv2 create-listener \
    --load-balancer-arn $ALB_ARN \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=$TARGET_GROUP_ARN \
    --region $AWS_REGION

# Get ALB DNS name
aws elbv2 describe-load-balancers \
    --load-balancer-arns $ALB_ARN \
    --region $AWS_REGION \
    --query 'LoadBalancers[0].DNSName' \
    --output text
```

### Step 8: Create ECS Service

```bash
# Create service (with ALB)
aws ecs create-service \
    --cluster braille-api-cluster \
    --service-name braille-api-service \
    --task-definition nepali-braille-api \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
    --load-balancers "targetGroupArn=$TARGET_GROUP_ARN,containerName=braille-api,containerPort=8080" \
    --region $AWS_REGION

# Or without ALB (direct access via public IP)
aws ecs create-service \
    --cluster braille-api-cluster \
    --service-name braille-api-service \
    --task-definition nepali-braille-api \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
    --region $AWS_REGION
```

### Step 9: Get Service URL

```bash
# If using ALB, get the DNS name (from Step 7)
# Your API URL will be: http://YOUR-ALB-DNS-NAME

# If not using ALB, get the public IP of the task
aws ecs list-tasks --cluster braille-api-cluster --service-name braille-api-service --region $AWS_REGION
aws ecs describe-tasks --cluster braille-api-cluster --tasks TASK_ID --region $AWS_REGION --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text
# Then get the public IP from the network interface
```

## Option 2: AWS App Runner (Simpler Alternative)

App Runner is AWS's simplest container deployment service.

### Step 1: Push to ECR (Same as ECS Step 2-3)

### Step 2: Create App Runner Service

```bash
# Create apprunner-service.json
cat > apprunner-service.json << EOF
{
  "ServiceName": "nepali-braille-api",
  "SourceConfiguration": {
    "ImageRepository": {
      "ImageIdentifier": "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:latest",
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

# Create service
aws apprunner create-service \
    --cli-input-json file://apprunner-service.json \
    --region $AWS_REGION
```

### Step 3: Get Service URL

```bash
aws apprunner describe-service \
    --service-arn <SERVICE_ARN> \
    --region $AWS_REGION \
    --query 'Service.ServiceUrl' \
    --output text
```

## Option 3: AWS Lambda with Container (Serverless)

For serverless deployment with automatic scaling.

### Step 1: Create Lambda Dockerfile

Create `Dockerfile.lambda`:

```dockerfile
FROM public.ecr.aws/lambda/python:3.10

WORKDIR ${LAMBDA_TASK_ROOT}

# Install system dependencies
RUN yum install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && yum clean all

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Lambda handler
CMD [ "main.handler" ]
```

### Step 2: Create Lambda Handler

Add to `main.py`:

```python
def handler(event, context):
    # Parse API Gateway event
    from mangum import Mangum
    handler = Mangum(app)
    return handler(event, context)
```

### Step 3: Build and Deploy

```bash
# Build image
docker build -f Dockerfile.lambda -t braille-api-lambda .

# Tag and push to ECR
docker tag braille-api-lambda:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/braille-api-lambda:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/braille-api-lambda:latest

# Create Lambda function
aws lambda create-function \
    --function-name nepali-braille-api \
    --package-type Image \
    --code ImageUri=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/braille-api-lambda:latest \
    --role arn:aws:iam::$AWS_ACCOUNT_ID:role/lambda-execution-role \
    --timeout 300 \
    --memory-size 3008 \
    --region $AWS_REGION
```

## Testing Your Deployment

```bash
# Test health endpoint
curl http://YOUR-SERVICE-URL/

# Test detection endpoint
curl -X POST http://YOUR-SERVICE-URL/detect \
  -F "file=@path/to/your/image.jpg"
```

## Flutter Integration

Update your Flutter API service to use the AWS endpoint:

```dart
class BrailleAPI {
  // For ECS/App Runner
  final String baseUrl = 'http://your-alb-dns-name.us-east-1.elb.amazonaws.com';
  
  // Or for App Runner
  // final String baseUrl = 'https://your-service-id.us-east-1.awsapprunner.com';
  
  // Rest of the code remains the same...
}
```

## Cost Optimization

### ECS Fargate
- **Pay per use**: ~$0.04/vCPU-hour, ~$0.004/GB-hour
- **Estimated**: ~$30-50/month for light usage
- Use auto-scaling to scale down to 0 when not in use

### App Runner
- **Pay per use**: Similar pricing to Fargate
- **Simpler**: No VPC/ALB setup needed
- **Auto-scaling**: Built-in

### Lambda
- **Pay per request**: $0.20 per 1M requests
- **Compute**: $0.0000166667 per GB-second
- **Best for**: Low/irregular traffic

## Monitoring

### CloudWatch Logs
```bash
# View logs
aws logs tail /ecs/nepali-braille-api --follow --region $AWS_REGION
```

### CloudWatch Metrics
- Monitor CPU, Memory, Request count
- Set up alarms for errors

## Troubleshooting

### Issue: Cannot pull image from ECR
- Ensure IAM role has ECR permissions
- Check that you're logged into ECR

### Issue: Task fails to start
- Check CloudWatch logs
- Verify security group allows traffic
- Ensure task has enough memory (4GB recommended)

### Issue: Timeout errors
- Increase task timeout in task definition
- Check if image is too large (optimize Dockerfile)

### Issue: Out of memory
- Increase memory allocation (4GB minimum recommended)
- Check CloudWatch metrics

## Cleanup

To remove all resources:

```bash
# Stop ECS service
aws ecs update-service --cluster braille-api-cluster --service braille-api-service --desired-count 0 --region $AWS_REGION
aws ecs delete-service --cluster braille-api-cluster --service braille-api-service --region $AWS_REGION

# Delete cluster
aws ecs delete-cluster --cluster braille-api-cluster --region $AWS_REGION

# Delete ECR repository
aws ecr delete-repository --repository-name $REPOSITORY_NAME --force --region $AWS_REGION

# Delete security group
aws ec2 delete-security-group --group-id $SECURITY_GROUP_ID --region $AWS_REGION
```

## Quick Start Script

Save this as `deploy-aws.sh`:

```bash
#!/bin/bash
set -e

export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPOSITORY_NAME=nepali-braille-api

echo "Creating ECR repository..."
aws ecr create-repository --repository-name $REPOSITORY_NAME --region $AWS_REGION || true

echo "Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

echo "Building and pushing image..."
cd app
docker build -t $REPOSITORY_NAME:latest .
docker tag $REPOSITORY_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:latest

echo "Deployment complete!"
echo "Image URI: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:latest"
```

Make it executable and run:
```bash
chmod +x deploy-aws.sh
./deploy-aws.sh
```
