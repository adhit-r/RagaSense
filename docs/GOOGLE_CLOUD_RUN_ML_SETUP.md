# Google Cloud Run ML Setup - NOT CURRENTLY USED

## Status: Not Implemented

**Note**: Google Cloud Run is not currently being used for ML model hosting in RagaSense. The application is using a local FastAPI backend for ML services.

## Current ML Architecture

### Local FastAPI Backend
- **Location**: `backend/` directory
- **ML Service**: `ml/working_raga_detector.py`
- **API Endpoints**: `backend/api/endpoints/raga_detect.py`
- **Deployment**: Local development server

### Why Not Google Cloud Run?

The decision was made to use a local FastAPI backend instead of Google Cloud Run for the following reasons:

1. **Simplified Development**: Easier to develop and debug locally
2. **Cost Considerations**: No cloud infrastructure costs during development
3. **Dependency Management**: Simpler dependency management with local Python environment
4. **Integration**: Better integration with the existing FastAPI backend

## Future Considerations

If you decide to migrate to Google Cloud Run in the future, the following setup would be required:

### 1. Google Cloud Project Setup
- Create a new project or use existing project
- Enable Cloud Run API
- Set up service account with appropriate permissions

### 2. Container Configuration
- Create Dockerfile for ML service
- Configure Cloud Run service
- Set up environment variables

### 3. Deployment
- Build and push container to Google Container Registry
- Deploy to Cloud Run
- Configure custom domain (optional)

### 4. Integration
- Update frontend to use Cloud Run endpoints
- Configure authentication and security
- Set up monitoring and logging

## Current ML Setup

### Local Development
```bash
# Start the backend server
python -m backend.main

# Test ML functionality
python scripts/test_raga_detection.py
```

### Production Deployment
```bash
# Using Docker Compose
docker-compose up -d

# Or direct deployment
python -m backend.main --host 0.0.0.0 --port 8000
```

## Environment Variables

The current setup uses these environment variables:

```env
# Backend API
NEXT_PUBLIC_API_URL=http://localhost:8000

# Database (if using external database)
DATABASE_URL=your_database_url_here
```

## Migration Path

If you want to migrate to Google Cloud Run in the future:

1. **Containerize ML Service**: Create Dockerfile for ML components
2. **Set up Google Cloud**: Configure project and services
3. **Deploy to Cloud Run**: Build and deploy container
4. **Update Frontend**: Point to Cloud Run endpoints
5. **Configure Monitoring**: Set up logging and monitoring

## Conclusion

The current local FastAPI setup provides a robust, cost-effective solution for ML services. Google Cloud Run can be considered for future scaling needs or if cloud deployment becomes necessary.

For now, focus on:
- Improving ML model accuracy
- Enhancing the local FastAPI backend
- Optimizing the Convex database integration
- Building the Lynx frontend features
