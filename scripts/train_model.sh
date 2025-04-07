#!/bin/bash
set -e

# Script to train the Stock Purchase Propensity Model

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Make sure all required environment variables are set."
fi

# Create necessary directories
mkdir -p logs models

# Run the training pipeline
echo "Starting model training pipeline..."
python -m stock_propensity_model.main

echo "Training completed."