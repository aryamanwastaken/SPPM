# Stock Purchase Propensity Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.3.0-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.6.2-green)
![AWS](https://img.shields.io/badge/AWS-S3-yellow)


## Overview

A production-grade machine learning pipeline that predicts a user's likelihood to purchase specific stocks based on real-time market data and historical user behavior patterns. The model ingests live feeds via API, processes them using Databricks, and outputs predictions to AWS S3.

## Key Features

- **Real-time Prediction Pipeline**: Processes live market data feeds and user activity to generate immediate purchase propensity predictions
- **Advanced Feature Engineering**: Creates sophisticated features like price momentum, volume ratio analysis, and user-specific investment pattern indicators
- **Gradient Boosting Model**: Uses XGBoost with hyperparameter optimization for high-accuracy predictions
- **Production-ready Architecture**: Complete with data validation, error handling, monitoring, and scaling capabilities
- **Containerized Deployment**: Docker and Docker Compose support for easy deployment in any environment
- **Performance Monitoring**: Automated drift detection and model performance tracking with MLflow integration
- **REST API Service**: FastAPI endpoint for real-time and batch predictions with automatic documentation

## Architecture

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Market Data  │───▶│   Databricks  │───▶│    XGBoost    │───▶│   S3 Output   │
│  Live Feed    │    │   Processing  │    │     Model     │    │  Predictions  │
└───────────────┘    └───────┬───────┘    └───────────────┘    └───────────────┘
                            │
┌───────────────┐           │                                  ┌───────────────┐
│  User Activity │──────────┘                                  │    MLflow     │
│     Data       │                                             │   Tracking    │
└───────────────┘                                              └───────────────┘
```

## Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerized deployment)
- AWS account with S3 access
- PySpark/Databricks environment (for distributed processing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-propensity-model.git
cd stock-propensity-model
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your AWS credentials and other configuration
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Running with Docker

The easiest way to run the complete system is using Docker Compose:

```bash
docker-compose up -d
```

This will start:
- The model training service
- The API service on port 8000
- MLflow tracking server on port 5000
- MinIO (S3-compatible storage) on port 9000

#### Running Directly

**Train the model:**
```bash
./scripts/train_model.sh
```

**Start the API server:**
```bash
./scripts/deploy_model.sh
```

**Run monitoring:**
```bash
./scripts/run_monitoring.sh
```

#### Making Predictions

Once the API is running, you can make predictions:

```python
import requests
import json

# For single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "user_id": "user123",
        "stock_id": "AAPL",
        "close_price": 150.25,
        "open_price": 148.50,
        "high_price": 151.10,
        "low_price": 147.90,
        "volume": 75000000,
        "avg_volume_30d": 80000000,
        "sector": "Technology",
        "market_cap_category": "Large",
        "day_of_week": "Monday",
        "days_since_last_purchase": 5,
        "portfolio_diversity_score": 0.75,
        "risk_tolerance": 0.8
    }
)

print(json.dumps(response.json(), indent=2))
```

## Model Performance

The model achieves the following metrics on recent data:

| Metric | Value |
|--------|-------|
| AUC | 0.88 |
| F1 Score | 0.81 |
| Precision | 0.79 |
| Recall | 0.83 |

## Project Structure

```
stock-propensity-model/
├── stock_propensity_model/        # Main package directory
│   ├── main.py                    # Main model implementation
│   ├── data_utils.py              # Data processing utilities
│   ├── monitoring.py              # Model monitoring utilities
│   └── api.py                     # REST API service
├── config/                        # Configuration files
├── models/                        # Saved models
├── logs/                          # Log files
├── tests/                         # Test directory
└── scripts/                       # Utility scripts
```

For more details, see the [Project Structure document](PROJECT_STRUCTURE.md).

## Configuration

The model is configured via `config/model_config.json`, which includes:

- Data source paths
- Model hyperparameters
- Feature definitions
- Output settings
- Monitoring thresholds

## API Documentation

When running, API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Monitoring and Maintenance

The system includes automated monitoring that:

1. Tracks model performance metrics
2. Detects data drift in feature distributions
3. Generates alerting and reporting
4. Logs all metrics to MLflow

To view metrics:
```bash
# Visit MLflow UI
open http://localhost:5000
```

## Development

### Running Tests

```bash
pytest
```

### Code Style

```bash
# Format code
black stock_propensity_model

# Lint code
flake8 stock_propensity_model
```

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgements

- XGBoost team for their excellent gradient boosting implementation
- Databricks for their distributed computing platform
- The open-source community for the amazing tools that made this possible
