# Use Python 3.12 slim image (MediaPipe compatible)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and RetinaFace/TensorFlow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-glog-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY crop-and-slideshow.py .

# Create output directory
RUN mkdir -p /app/output

# Create input directory for mounting photos
RUN mkdir -p /app/input

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port for serving the HTML slideshow (optional)
EXPOSE 8000

# Default command
CMD ["python", "crop-and-slideshow.py"]