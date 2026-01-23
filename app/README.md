# Nepali Braille OBR API

FastAPI service for converting Braille images to text using Object to Braille Recognition (OBR).

## Features

- Accepts images of size 1500x2000px
- Uses YOLO model (`best.pt`) for Braille dot detection
- Converts detected dots to text using Bharati Braille mapping
- RESTful API endpoint for easy integration with Flutter apps

## API Endpoints

### POST `/detect`
Converts a Braille image to text.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Image file (JPEG, PNG, etc.)

**Response:**
```json
{
  "success": true,
  "dot_count": 150,
  "text": "अक्षर...",
  "lines_detected": 5
}
```

### GET `/`
Health check endpoint.

### GET `/health`
Health check endpoint for monitoring.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure `models/best.pt` exists in the `app/models/` directory.

3. Run the server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

## AWS Deployment

### Quick Start (Recommended)

**For the easiest deployment, use AWS App Runner:**

1. **Install and configure AWS CLI:**
```bash
# Install AWS CLI (if not installed)
# macOS: brew install awscli
# Linux: sudo apt-get install awscli
# Windows: Download from https://aws.amazon.com/cli/

# Configure credentials
aws configure
```

2. **Deploy:**
```bash
cd app
chmod +x deploy-aws-apprunner.sh
./deploy-aws-apprunner.sh
```

3. **Get your service URL:**
```bash
SERVICE_ARN=$(aws apprunner list-services --region us-east-1 --query "ServiceSummaryList[?ServiceName=='nepali-braille-api'].ServiceArn" --output text)
aws apprunner describe-service --service-arn $SERVICE_ARN --region us-east-1 --query 'Service.ServiceUrl' --output text
```

**That's it!** Your API is now live.

### Alternative: ECS Fargate (More Control)

For more control over networking:

```bash
cd app
chmod +x deploy-aws.sh
./deploy-aws.sh
```

### Documentation

- **[AWS_QUICK_START.md](AWS_QUICK_START.md)** - Quick start guide with step-by-step instructions
- **[AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md)** - Detailed deployment guide with all options

### Resource Recommendations
- **Memory**: 4GB (for YOLO model loading)
- **CPU**: 2 vCPU (for faster inference)
- **Timeout**: 300s (for processing large images)

## Flutter Integration Example

```dart
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<Map<String, dynamic>> convertBrailleImage(File imageFile) async {
  // Replace with your AWS service URL
  // For ECS: http://your-alb-dns-name.us-east-1.elb.amazonaws.com
  // For App Runner: https://your-service-id.us-east-1.awsapprunner.com
  var request = http.MultipartRequest(
    'POST',
    Uri.parse('http://your-aws-service-url/detect'),
  );
  
  request.files.add(
    await http.MultipartFile.fromPath('file', imageFile.path),
  );
  
  var response = await request.send();
  var responseData = await response.stream.bytesToString();
  
  return json.decode(responseData);
}
```

## Project Structure

```
app/
├── main.py           # FastAPI application
├── detector.py       # YOLO dot detection
├── grid.py           # Grid detection and binary conversion
├── decoder.py        # Braille to text decoding
├── models/
│   └── best.pt      # YOLO model weights
├── requirements.txt  # Python dependencies
├── Dockerfile        # Docker configuration
└── README.md        # This file
```

## Notes

- Images are automatically resized to 1500x2000px for processing
- The model expects images in BGR format (OpenCV default)
- Processing includes 180° rotation for better detection accuracy
- The API uses tile-based detection for handling large images efficiently
