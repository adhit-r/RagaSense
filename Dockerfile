# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_ml.txt .
RUN pip install --no-cache-dir -r requirements_ml.txt

# Copy application code
COPY . .

# Create model directory
RUN mkdir -p /app/models

# Download models from Cloud Storage (will be done at runtime)
RUN echo "Models will be downloaded from Cloud Storage"

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "ml/cloud_run_app.py"]