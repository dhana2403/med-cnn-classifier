# Use official PyTorch image with CUDA (you can switch to CPU-only if needed)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (e.g., unzip)
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

# Copy Python dependency list and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Run dataset setup if zip files are available
RUN python data/data_download.py || echo "Dataset not extracted — ensure zip files are in data/ locally before building if needed."

# Default command to train the model
CMD ["python", "train.py"]

#If you're only testing/predicting, change the last line to:
CMD ["python", "test.py"]





