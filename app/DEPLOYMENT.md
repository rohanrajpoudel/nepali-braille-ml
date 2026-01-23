# Google Cloud Deployment Guide

This guide explains how to deploy the Nepali Braille OBR API to Google Cloud Run for use with a Flutter app.

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud SDK** installed and configured
3. **Docker** installed (for local testing)
4. **Project ID** created in Google Cloud Console

## Step 1: Set Up Google Cloud Project

```bash
# Login to Google Cloud
gcloud auth login

# Set your project ID
export PROJECT_ID=your-project-id
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## Step 2: Prepare Your Model

Ensure `models/best.pt` is in the `app/models/` directory. The Dockerfile will copy this during build.

## Step 3: Build and Deploy

### Option A: Deploy from Source (Recommended)

This uses Cloud Build to build and deploy automatically:

```bash
# Deploy directly from source
gcloud run deploy nepali-braille-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0
```

### Option B: Build Docker Image Locally

```bash
# Build the image
docker build -t gcr.io/${PROJECT_ID}/nepali-braille-api .

# Push to Google Container Registry
docker push gcr.io/${PROJECT_ID}/nepali-braille-api

# Deploy to Cloud Run
gcloud run deploy nepali-braille-api \
  --image gcr.io/${PROJECT_ID}/nepali-braille-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10
```

## Step 4: Get Your API URL

After deployment, you'll receive a URL like:
```
https://nepali-braille-api-xxxxx-uc.a.run.app
```

Save this URL for your Flutter app.

## Step 5: Test the API

```bash
# Test health endpoint
curl https://your-service-url.run.app/

# Test detection endpoint
curl -X POST https://your-service-url.run.app/detect \
  -F "file=@path/to/your/image.jpg"
```

## Flutter Integration

### 1. Add HTTP dependency to `pubspec.yaml`:

```yaml
dependencies:
  http: ^1.1.0
```

### 2. Create API service:

```dart
// lib/services/braille_api.dart
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

class BrailleAPI {
  final String baseUrl;
  
  BrailleAPI({required this.baseUrl});
  
  Future<BrailleResponse> convertImage(File imageFile) async {
    var request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/detect'),
    );
    
    // Add image file
    request.files.add(
      await http.MultipartFile.fromPath('file', imageFile.path),
    );
    
    // Send request
    var streamedResponse = await request.send();
    var response = await http.Response.fromStream(streamedResponse);
    
    if (response.statusCode == 200) {
      var data = json.decode(response.body);
      return BrailleResponse.fromJson(data);
    } else {
      throw Exception('Failed to convert image: ${response.body}');
    }
  }
}

class BrailleResponse {
  final bool success;
  final int dotCount;
  final String text;
  final int linesDetected;
  
  BrailleResponse({
    required this.success,
    required this.dotCount,
    required this.text,
    required this.linesDetected,
  });
  
  factory BrailleResponse.fromJson(Map<String, dynamic> json) {
    return BrailleResponse(
      success: json['success'] ?? false,
      dotCount: json['dot_count'] ?? 0,
      text: json['text'] ?? '',
      linesDetected: json['lines_detected'] ?? 0,
    );
  }
}
```

### 3. Use in your Flutter app:

```dart
import 'package:image_picker/image_picker.dart';
import 'services/braille_api.dart';

class BrailleConverter extends StatefulWidget {
  @override
  _BrailleConverterState createState() => _BrailleConverterState();
}

class _BrailleConverterState extends State<BrailleConverter> {
  final BrailleAPI api = BrailleAPI(
    baseUrl: 'https://your-service-url.run.app',
  );
  
  File? _image;
  String? _convertedText;
  bool _isLoading = false;
  
  Future<void> _pickAndConvert() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(
      source: ImageSource.camera,
      maxWidth: 1500,
      maxHeight: 2000,
    );
    
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _isLoading = true;
      });
      
      try {
        final response = await api.convertImage(_image!);
        setState(() {
          _convertedText = response.text;
          _isLoading = false;
        });
      } catch (e) {
        setState(() {
          _isLoading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Braille Converter')),
      body: Column(
        children: [
          ElevatedButton(
            onPressed: _pickAndConvert,
            child: Text('Pick Image & Convert'),
          ),
          if (_isLoading)
            CircularProgressIndicator(),
          if (_convertedText != null)
            Expanded(
              child: SingleChildScrollView(
                child: Text(_convertedText!),
              ),
            ),
        ],
      ),
    );
  }
}
```

## Monitoring and Logs

View logs in Google Cloud Console:
```bash
gcloud run services logs read nepali-braille-api --region us-central1
```

Or in the Cloud Console:
1. Go to Cloud Run
2. Click on your service
3. Click on "Logs" tab

## Cost Optimization

- **Min instances**: Set to 0 to scale down when not in use
- **Max instances**: Adjust based on expected traffic
- **Memory/CPU**: Start with 2Gi/2 CPU, adjust based on performance
- **Timeout**: 300s should be sufficient for most images

## Troubleshooting

### Issue: Model not found
- Ensure `models/best.pt` exists in `app/models/` directory
- Check Dockerfile copies the models directory correctly

### Issue: Out of memory
- Increase memory allocation: `--memory 4Gi`
- Consider using a smaller model or optimizing image size

### Issue: Timeout errors
- Increase timeout: `--timeout 600`
- Optimize image preprocessing

### Issue: CORS errors (if accessing from web)
Add CORS middleware to `main.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Security Considerations

1. **Authentication**: Consider adding authentication for production:
```bash
gcloud run deploy nepali-braille-api \
  --no-allow-unauthenticated
```

2. **API Keys**: Use API keys or service accounts for Flutter app
3. **Rate Limiting**: Consider adding rate limiting for production use
