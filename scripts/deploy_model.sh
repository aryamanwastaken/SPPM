#!/bin/bash
set -e

# Script to deploy the Stock Purchase Propensity Model API

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Make sure all required environment variables are set."
fi

# Check if model exists
if [ ! -f "models/xgboost_model.json" ]; then
    echo "Model file not found. Please train the model first."
    exit 1
fi

# Create necessary directories
mkdir -p logs

# Start the API server
echo "Starting API server..."
uvicorn stock_propensity_model.api:app --host 0.0.0.0 --port 8000 --workers 4

echo "API server started on port 8000."