# AWS Quick Start Guide

## Prerequisites

1. **Install AWS CLI:**
   ```bash
   # macOS
   brew install awscli
   
   # Linux
   sudo apt-get install awscli
   
   # Windows: Download from https://aws.amazon.com/cli/
   ```

2. **Configure AWS credentials:**
   ```bash
   aws configure
   # Enter your Access Key ID, Secret Access Key, region (e.g., us-east-1), and output format (json)
   ```

## Option 1: AWS App Runner (Easiest - Recommended)

App Runner is the simplest option, similar to Google Cloud Run.

### Steps:

1. **Navigate to app directory:**
   ```bash
   cd app
   ```

2. **Make script executable and run:**
   ```bash
   chmod +x deploy-aws-apprunner.sh
   ./deploy-aws-apprunner.sh
   ```

3. **Get your service URL:**
   ```bash
   # Get the service ARN first
   SERVICE_ARN=$(aws apprunner list-services --region us-east-1 --query "ServiceSummaryList[?ServiceName=='nepali-braille-api'].ServiceArn" --output text)
   
   # Get the URL
   aws apprunner describe-service --service-arn $SERVICE_ARN --region us-east-1 --query 'Service.ServiceUrl' --output text
   ```

4. **Test your API:**
   ```bash
   curl http://YOUR-SERVICE-URL/
   ```

**That's it!** Your API is now live on AWS.

## Option 2: AWS ECS Fargate (More Control)

For more control over networking and load balancing.

### Steps:

1. **Navigate to app directory:**
   ```bash
   cd app
   ```

2. **Make script executable and run:**
   ```bash
   chmod +x deploy-aws.sh
   ./deploy-aws.sh
   ```

3. **Get your service URL:**
   ```bash
   # Get task public IP
   TASK_ARN=$(aws ecs list-tasks --cluster braille-api-cluster --service-name braille-api-service --region us-east-1 --query 'taskArns[0]' --output text)
   ENI_ID=$(aws ecs describe-tasks --cluster braille-api-cluster --tasks $TASK_ARN --region us-east-1 --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)
   aws ec2 describe-network-interfaces --network-interface-ids $ENI_ID --region us-east-1 --query 'NetworkInterfaces[0].Association.PublicIp' --output text
   ```

4. **Access your API:**
   ```bash
   curl http://YOUR-PUBLIC-IP:8080/
   ```

## Manual Deployment (If Scripts Don't Work)

### Step 1: Create ECR Repository

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPOSITORY_NAME=nepali-braille-api

aws ecr create-repository \
    --repository-name $REPOSITORY_NAME \
    --region $AWS_REGION
```

### Step 2: Build and Push Image

```bash
# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build image
cd app
docker build -t $REPOSITORY_NAME:latest .

# Tag image
docker tag $REPOSITORY_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:latest

# Push image
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:latest
```

### Step 3: Deploy to App Runner (Easiest)

1. Go to AWS Console → App Runner
2. Click "Create service"
3. Choose "Container registry" → "Amazon ECR"
4. Select your repository and image
5. Configure:
   - Port: 8080
   - CPU: 2 vCPU
   - Memory: 4 GB
6. Click "Create & deploy"

### Step 4: Get Service URL

After deployment, you'll see the service URL in the App Runner console.

## Flutter Integration

Update your Flutter app with the AWS service URL:

```dart
class BrailleAPI {
  // Replace with your actual AWS service URL
  final String baseUrl = 'https://your-service-id.us-east-1.awsapprunner.com';
  // Or for ECS: 'http://your-public-ip:8080'
  
  // Rest of your code...
}
```

## Troubleshooting

### Script fails with "command not found"
- Make sure you're in the `app` directory
- Make script executable: `chmod +x deploy-aws.sh`

### "Cannot connect to Docker daemon"
- Make sure Docker is running
- On Linux, you might need: `sudo usermod -aG docker $USER` and logout/login

### "Access Denied" errors
- Check your AWS credentials: `aws sts get-caller-identity`
- Ensure your IAM user has permissions for ECR, ECS/App Runner

### Service not accessible
- Check security groups allow traffic on port 8080
- For App Runner, it should work automatically
- For ECS, ensure task has public IP and security group allows inbound traffic

## Viewing Logs

### App Runner:
```bash
SERVICE_ARN=$(aws apprunner list-services --region us-east-1 --query "ServiceSummaryList[?ServiceName=='nepali-braille-api'].ServiceArn" --output text)
# View logs in CloudWatch Console or:
aws logs tail /aws/apprunner/nepali-braille-api --follow
```

### ECS:
```bash
aws logs tail /ecs/nepali-braille-api --follow --region us-east-1
```

## Cost Estimate

- **App Runner**: ~$0.007/vCPU-hour + $0.0008/GB-hour
- **Estimated monthly**: $30-50 for light usage (similar to Cloud Run)
- **Free tier**: 750 hours/month of App Runner (first 3 months)

## Next Steps

1. Test your API endpoint
2. Update Flutter app with the service URL
3. Monitor usage in AWS Console
4. Set up CloudWatch alarms for errors

For detailed information, see [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md).
