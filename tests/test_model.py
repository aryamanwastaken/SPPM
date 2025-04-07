"""Unit tests for the Stock Propensity Model."""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from unittest.mock import patch, MagicMock, mock_open

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model class
from stock_propensity_model.main import StockPropensityModel

# Mock config for testing
@pytest.fixture
def mock_config():
    return {
        "market_data_path": "test_data/market_data.parquet",
        "user_data_path": "test_data/user_data.parquet",
        "output_bucket": "test-bucket",
        "output_prefix": "test-output",
        "mlflow_tracking_uri": "sqlite:///test_mlflow.db",
        "experiment_name": "test_experiment",
        "categorical_columns": ["sector", "market_cap_category"],
        "numerical_columns": ["close_price", "volume", "price_momentum", "volume_ratio", "volatility"],
        "feature_columns": ["price_momentum", "volume_ratio", "volatility", "days_since_last_purchase"],
        "label_column": "purchased_stock",
        "random_seed": 42
    }

# Create test data
@pytest.fixture
def test_data():
    # Create market data
    market_data = pd.DataFrame({
        "user_id": ["user1", "user2", "user1", "user2"],
        "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
        "stock_id": ["AAPL", "GOOGL", "AAPL", "GOOGL"],
        "open_price": [150.0, 2500.0, 151.0, 2510.0],
        "close_price": [153.0, 2520.0, 149.0, 2530.0],
        "high_price": [155.0, 2530.0, 152.0, 2540.0],
        "low_price": [149.0, 2490.0, 148.0, 2500.0],
        "volume": [1000000, 500000, 1200000, 600000],
        "avg_volume_30d": [1100000, 550000, 1100000, 550000],
        "sector": ["Technology", "Technology", "Technology", "Technology"],
        "market_cap_category": ["Large", "Large", "Large", "Large"]
    })
    
    # Create user data
    user_data = pd.DataFrame({
        "user_id": ["user1", "user2", "user1", "user2"],
        "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
        "last_purchase_date": ["2022-12-25", "2022-12-28", "2022-12-25", "2022-12-28"],
        "current_date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
        "purchased_stock": [1, 0, 0, 1]
    })
    
    return {
        "market_data": market_data,
        "user_data": user_data
    }

# Mock Spark session
@pytest.fixture
def mock_spark_session():
    mock_spark = MagicMock()
    mock_read = MagicMock()
    mock_spark.read.return_value = mock_read
    
    def mock_read_parquet(path):
        if "market_data" in path:
            return MagicMock(toPandas=lambda: test_data()["market_data"])
        elif "user_data" in path:
            return MagicMock(toPandas=lambda: test_data()["user_data"])
        return MagicMock()
    
    mock_read.parquet.side_effect = mock_read_parquet
    return mock_spark

# Mock S3 client
@pytest.fixture
def mock_s3_client():
    return MagicMock()

# Mock MLflow
@pytest.fixture
def mock_mlflow():
    mlflow = MagicMock()
    mlflow.start_run.return_value.__enter__.return_value.info.run_id = "test_run_id"
    return mlflow

# Test model initialization
def test_init(mock_config):
    # Mock config loading
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))):
        with patch("json.load", return_value=mock_config):
            with patch("pyspark.sql.SparkSession.builder.appName") as mock_app:
                mock_app.return_value.config.return_value.config.return_value.config.return_value.config.return_value.getOrCreate.return_value = MagicMock()
                with patch("boto3.client", return_value=MagicMock()):
                    model = StockPropensityModel(config_path="fake_path")
                    
                    assert model.config == mock_config
                    assert model.model is None
                    assert model.scaler is not None

# Test data ingestion
@patch("stock_propensity_model.main.StockPropensityModel.__init__", return_value=None)
def test_ingest_data(mock_init, mock_spark_session, mock_config, test_data):
    model = StockPropensityModel()
    model.spark = mock_spark_session
    model.config = mock_config
    
    # Create DataFrame return values
    market_df = mock_spark_session.read.parquet("market_data_path")
    user_df = mock_spark_session.read.parquet("user_data_path")
    
    # Mock joined DataFrame
    joined_df = MagicMock()
    market_df.join.return_value = joined_df
    joined_df.count.return_value = 4
    
    result = model.ingest_data()
    
    # Check if both data sources were loaded
    mock_spark_session.read.parquet.assert_any_call(mock_config["market_data_path"])
    mock_spark_session.read.parquet.assert_any_call(mock_config["user_data_path"])
    
    # Check if join was called
    market_df.join.assert_called_once()
    
    # Check result
    assert result == joined_df

# Test data preprocessing
@patch("stock_propensity_model.main.StockPropensityModel.__init__", return_value=None)
def test_preprocess_data(mock_init, mock_config, mock_spark_session):
    model = StockPropensityModel()
    model.config = mock_config
    model.spark = mock_spark_session
    model.scaler = MagicMock()
    model.scaler.fit_transform.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    # Create mock DataFrame
    mock_df = MagicMock()
    
    # Mock withColumn to return a new mock each time
    def mock_with_column(col_name, expr):
        new_mock = MagicMock()
        new_mock.withColumn = mock_with_column
        new_mock.select = MagicMock(return_value=new_mock)
        new_mock.toPandas = MagicMock(return_value=pd.DataFrame({
            "price_momentum": [0.1, -0.05],
            "volume_ratio": [0.9, 1.1],
            "volatility": [6.0, 10.0],
            "days_since_last_purchase": [7, 4],
            "purchased_stock": [1, 0]
        }))
        return new_mock
    
    mock_df.withColumn = mock_with_column
    
    X, y = model.preprocess_data(mock_df)
    
    # Check that the result contains features and labels
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == 2
    assert len(y) == 2