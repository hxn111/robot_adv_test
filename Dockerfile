# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    ripgrep \
    unzip \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set HF Cache directory
RUN mkdir -p /app/.cache/huggingface && \
    chmod -R 777 /app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# Copy project files
COPY . .
