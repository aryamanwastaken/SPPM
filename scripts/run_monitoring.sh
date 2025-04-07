#!/bin/bash
set -e

# Script to run monitoring for the Stock Purchase Propensity Model

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Make sure all required environment variables are set."
fi

# Create necessary directories
mkdir -p logs reports

# Run the monitoring
echo "Starting model monitoring..."
python -m stock_propensity_model.monitoring

echo "Monitoring completed."

# Check for alerts in log file
if grep -q "WARNING" logs/monitoring.log; then
    echo "ALERT: Warnings detected in monitoring. Please check logs/monitoring.log for details."
    grep "WARNING" logs/monitoring.log
fi
