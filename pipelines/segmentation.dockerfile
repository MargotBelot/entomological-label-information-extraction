FROM python:3.10-slim

WORKDIR /app

# Copy everything from the build context into the container
COPY . /app

# Ensure Python finds modules like `label_processing`
ENV PYTHONPATH=/app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libstdc++6 \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    libomp5 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir -r pipelines/requirements/segmentation.txt

# Set environment variables for threading
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Run the detection script
CMD ["python", "scripts/processing/detection.py"]