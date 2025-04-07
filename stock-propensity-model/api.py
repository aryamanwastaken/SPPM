"""API service for stock purchase propensity model predictions."""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import logging
from datetime import datetime
import boto3
import mlflow
import mlflow.xgboost
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.environ.get("LOG_FILE_PATH", "logs/api.log"),
    filemode='a'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Purchase Propensity API",
    description="API for predicting user propensity to purchase stocks",
    version="1.0.0"
)

# Load configuration
try:
    with open(os.environ.get("CONFIG_PATH", "config/model_config.json"), 'r') as f:
        config = json.load(f)
except Exception as e:
    logger.error(f"Failed to load configuration: {str(e)}")
    raise

# Initialize AWS clients
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_REGION', 'us-east-1')
)

# Initialize MLflow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "stock_propensity_model"))

# Load the model
model_path = os.environ.get("MODEL_SAVE_PATH", "models/xgboost_model.json")

# Check if model needs to be downloaded from S3
if not os.path.exists(model_path):
    try:
        bucket = config.get("model_bucket", "model-artifacts")
        key = f"{config.get('model_prefix', 'models')}/xgboost_model.json"
        s3_client.download_file(bucket, key, model_path)
        logger.info(f"Downloaded model from s3://{bucket}/{key} to {model_path}")
    except Exception as e:
        logger.error(f"Failed to download model from S3: {str(e)}")
        # Try to load from MLflow
        try:
            logger.info("Attempting to load model from MLflow")
            run_id = config.get("run_id")
            model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
            logger.info(f"Loaded model from MLflow run {run_id}")
        except Exception as mlflow_error:
            logger.error(f"Failed to load model from MLflow: {str(mlflow_error)}")
            raise RuntimeError("Failed to load model from both S3 and MLflow")
else:
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise

# Input schema validation
class StockFeatures(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    stock_id: str = Field(..., description="Unique identifier for the stock")
    
    # Market data features
    close_price: float = Field(..., description="Latest closing price")
    open_price: float = Field(..., description="Opening price")
    high_price: float = Field(..., description="High price of the day")
    low_price: float = Field(..., description="Low price of the day")
    volume: int = Field(..., description="Trading volume")
    avg_volume_30d: int = Field(..., description="Average 30-day volume")
    sector: str = Field(..., description="Stock sector (e.g., Technology, Finance)")
    market_cap_category: str = Field(..., description="Market cap category (Large, Mid, Small)")
    day_of_week: str = Field(..., description="Day of the week")
    
    # User behavior features
    days_since_last_purchase: int = Field(..., description="Days since user's last stock purchase")
    portfolio_diversity_score: float = Field(..., description="Measure of portfolio diversity (0-1)")
    risk_tolerance: float = Field(..., description="User's risk tolerance score (0-1)")
    
    class Config:
        schema_extra = {
            "example": {
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
        }

class BatchPredictionRequest(BaseModel):
    features: List[StockFeatures]

class PredictionResponse(BaseModel):
    user_id: str
    stock_id: str
    probability: float
    prediction: int
    prediction_time: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    model_version: str

def preprocess_features(features_data):
    """Preprocess input features similar to the training pipeline."""
    try:
        # Convert to DataFrame
        if isinstance(features_data, list):
            df = pd.DataFrame([item.dict() for item in features_data])
        else:
            df = pd.DataFrame([features_data.dict()])
        
        # Feature engineering (same as in training)
        df['price_momentum'] = (df['close_price'] - df['open_price']) / df['open_price']
        df['volume_ratio'] = df['volume'] / df['avg_volume_30d']
        df['volatility'] = df['high_price'] - df['low_price']
        
        # One-hot encode categorical variables
        # Sector
        sectors = ['Technology', 'Finance', 'Healthcare', 'Consumer', 'Energy', 
                  'Utilities', 'Materials', 'RealEstate', 'Other']
        for sector in sectors:
            df[f'sector_{sector}'] = (df['sector'] == sector).astype(int)
        
        # Market cap
        market_caps = ['Large', 'Mid', 'Small']
        for cap in market_caps:
            df[f'market_cap_category_{cap}'] = (df['market_cap_category'] == cap).astype(int)
        
        # Day of week
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        for day in days:
            df[f'day_of_week_{day}'] = (df['day_of_week'] == day).astype(int)
        
        # Select features in the same order as training
        feature_cols = config.get("feature_columns", [])
        
        # Check for missing columns and add them with zeros
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Final feature set
        X = df[feature_cols]
        
        # Track original IDs for response
        ids = df[['user_id', 'stock_id']]
        
        return X, ids
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature preprocessing failed: {str(e)}")

def generate_predictions(features_df, ids_df):
    """Generate predictions using the loaded model."""
    try:
        # Convert to DMatrix for XGBoost
        dmatrix = xgb.DMatrix(features_df.values)
        
        # Make predictions
        probabilities = model.predict(dmatrix)
        threshold = config.get("prediction_threshold", 0.5)
        predictions = (probabilities >= threshold).astype(int)
        
        # Create response objects
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        results = []
        for i in range(len(ids_df)):
            results.append({
                "user_id": ids_df.iloc[i]['user_id'],
                "stock_id": ids_df.iloc[i]['stock_id'],
                "probability": float(probabilities[i]),
                "prediction": int(predictions[i]),
                "prediction_time": current_time
            })
        
        return results
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction generation failed: {str(e)}")

def log_predictions_async(predictions):
    """Asynchronously log predictions to S3."""
    try:
        # Create DataFrame from predictions
        df = pd.DataFrame(predictions)
        
        # Add timestamp partition
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Local path for temporary save
        local_path = f"logs/predictions_{timestamp}.csv"
        df.to_csv(local_path, index=False)
        
        # Upload to S3
        bucket = config.get("output_bucket")
        prefix = config.get("output_prefix")
        s3_path = f"{prefix}/prediction_date={datetime.now().strftime('%Y-%m-%d')}/predictions_{timestamp}.csv"
        
        s3_client.upload_file(local_path, bucket, s3_path)
        logger.info(f"Logged {len(predictions)} predictions to s3://{bucket}/{s3_path}")
        
        # Clean up local file
        os.remove(local_path)
        
    except Exception as e:
        logger.error(f"Failed to log predictions: {str(e)}")
        # Don't raise exception, just log error to prevent API failure

@app.get("/")
async def root():
    return {"message": "Stock Purchase Propensity Model API", "status": "active"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: StockFeatures, background_tasks: BackgroundTasks):
    """Generate prediction for a single stock-user pair."""
    logger.info(f"Received prediction request for user {features.user_id} and stock {features.stock_id}")
    
    # Preprocess features
    X, ids = preprocess_features(features)
    
    # Generate predictions
    predictions = generate_predictions(X, ids)
    
    # Log prediction asynchronously
    background_tasks.add_task(log_predictions_async, predictions)
    
    return predictions[0]

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """Generate predictions for multiple stock-user pairs."""
    logger.info(f"Received batch prediction request with {len(request.features)} items")
    
    # Preprocess features
    X, ids = preprocess_features(request.features)
    
    # Generate predictions
    predictions = generate_predictions(X, ids)
    
    # Log predictions asynchronously
    background_tasks.add_task(log_predictions_async, predictions)
    
    return {
        "predictions": predictions,
        "count": len(predictions),
        "model_version": os.environ.get("MODEL_VERSION", "1.0.0")
    }

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    try:
        # Extract model attributes
        attributes = {
            "model_type": "XGBoost",
            "features": config.get("feature_columns", []),
            "threshold": config.get("prediction_threshold", 0.5),
            "version": os.environ.get("MODEL_VERSION", "1.0.0"),
            "last_updated": datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S')
                if os.path.exists(model_path) else "Unknown"
        }
        
        return attributes
    
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)