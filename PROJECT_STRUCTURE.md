# Stock Purchase Propensity Model - Project Structure

Here's the complete structure of the project:

```
stock-propensity-model/
│
├── stock_propensity_model/        # Main package directory
│   ├── __init__.py                # Package initialization
│   ├── main.py                    # Main model implementation
│   ├── data_utils.py              # Data processing utilities
│   ├── monitoring.py              # Model monitoring utilities
│   └── api.py                     # REST API service
│
├── config/                        # Configuration files
│   └── model_config.json          # Model parameters and settings
│
├── models/                        # Directory for saved models
│   └── .gitkeep                   # Placeholder to ensure directory is tracked
│
├── logs/                          # Log files
│   └── .gitkeep                   # Placeholder to ensure directory is tracked
│
├── tests/                         # Test directory
│   ├── __init__.py                # Test package initialization
│   ├── test_model.py              # Model unit tests
│   └── test_api.py                # API unit tests
│
├── scripts/                       # Utility scripts
│   ├── train_model.sh             # Script to train the model
│   ├── deploy_model.sh            # Script to deploy the model
│   └── run_monitoring.sh          # Script to run monitoring
│
├── notebooks/                     # Jupyter notebooks for exploration
│   └── feature_analysis.ipynb     # Example notebook
│
├── .env.example                   # Example environment variables
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose configuration
├── .gitignore                     # Git ignore file
├── setup.py                       # Package setup script
└── README.md                      # Project documentation
```

## Key Components

1. **Main Model (`main.py`)**
   - Complete implementation of the Stock Propensity Model pipeline
   - Handles data ingestion, preprocessing, feature engineering, model training, and prediction

2. **Data Utilities (`data_utils.py`)**
   - Utilities for loading and processing data from S3
   - Feature engineering functions
   - Dataset splitting and preparation

3. **Monitoring (`monitoring.py`)**
   - Performance monitoring
   - Data drift detection
   - Reporting and alerting

4. **API Service (`api.py`)**
   - REST API for real-time predictions
   - Batch prediction endpoints
   - Model information and health checks

5. **Configuration (`config/model_config.json`)**
   - Model hyperparameters
   - Feature definitions
   - S3 paths and other settings

6. **Docker Setup**
   - Dockerfile for containerization
   - docker-compose.yml for orchestrating services

7. **Tests**
   - Unit tests for model and API
   - Test fixtures and mocks

## How to Use

1. Clone the repository
2. Set up environment variables (copy from `.env.example`)
3. Install dependencies: `pip install -r requirements.txt`
4. Run the model: `python -m stock_propensity_model.main`
5. Start the API: `python -m stock_propensity_model.api`
6. Run monitoring: `python -m stock_propensity_model.monitoring`

## Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# Run specific service
docker-compose up stock-propensity-api

# View logs
docker-compose logs -f
```