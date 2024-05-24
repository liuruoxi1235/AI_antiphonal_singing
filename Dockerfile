FROM python:3.10-slim

# Install required system packages
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    pkg-config \
    libhdf5-dev \
    gcc \
    g++ \
    make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /app/song1

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt requirements.txt

# Upgrade pip and install requirements
RUN /app/song1/bin/pip install --upgrade pip \
    && /app/song1/bin/pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Command to run the application
CMD ["./song1/bin/python", "app.py"]