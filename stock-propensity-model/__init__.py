"""
Stock Purchase Propensity Model
===============================

A machine learning pipeline that predicts user propensity to purchase specific stocks
based on market data and user behavior patterns.

This package contains a complete machine learning pipeline with:
- Data processing and feature engineering
- Model training with XGBoost
- Model deployment and real-time prediction
- Performance monitoring and data drift detection

Author: Aryaman Patel
Version: 1.0.0
"""

__version__ = '1.0.0'

from stock_propensity_model.main import StockPropensityModel
from stock_propensity_model.data_utils import DataProcessor, get_yesterday_data, get_training_dataset
from stock_propensity_model.monitoring import ModelMonitor, run_daily_monitoring