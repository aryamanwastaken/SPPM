FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create and set working directory
WORKDIR /app

# Install Java for PySpark
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-11-jre-headless \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directory for config and logs
RUN mkdir -p /app/config /app/logs /app/models

# Set up entry point
ENTRYPOINT ["python", "stock_propensity_model/main.py"]