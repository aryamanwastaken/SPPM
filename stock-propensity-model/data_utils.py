"""Data utilities for the Stock Purchase Propensity Model."""

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf, lit, datediff, to_date
from pyspark.sql.types import DoubleType, StringType, IntegerType
import boto3
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing utilities for stock propensity model."""
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        """Initialize with Spark session and configuration.
        
        Args:
            spark: Active SparkSession
            config: Configuration dictionary
        """
        self.spark = spark
        self.config = config
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('AWS_REGION', 'us-east-1')
        )
    
    def fetch_market_data(self, start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> DataFrame:
        """Fetch market data from S3 for given date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format, defaults to yesterday
            end_date: End date in YYYY-MM-DD format, defaults to today
            
        Returns:
            Spark DataFrame with market data
        """
        try:
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            if not start_date:
                start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Construct S3 path
            s3_path = self.config.get("market_data_path")
            
            # Load data from S3
            market_df = self.spark.read.parquet(s3_path)
            
            # Filter by date range
            market_df = market_df.filter(
                (col("date") >= start_date) & (col("date") <= end_date)
            )
            
            logger.info(f"Loaded market data from {start_date} to {end_date}: {market_df.count()} rows")
            return market_df
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            raise
    
    def fetch_user_data(self, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> DataFrame:
        """Fetch user transaction data from S3 for given date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format, defaults to yesterday
            end_date: End date in YYYY-MM-DD format, defaults to today
            
        Returns:
            Spark DataFrame with user data
        """
        try:
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            if not start_date:
                start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Construct S3 path
            s3_path = self.config.get("user_data_path")
            
            # Load data from S3
            user_df = self.spark.read.parquet(s3_path)
            
            # Filter by date range
            user_df = user_df.filter(
                (col("date") >= start_date) & (col("date") <= end_date)
            )
            
            logger.info(f"Loaded user data from {start_date} to {end_date}: {user_df.count()} rows")
            return user_df
            
        except Exception as e:
            logger.error(f"Error fetching user data: {str(e)}")
            raise
    
    def engineer_features(self, market_df: DataFrame, user_df: DataFrame) -> DataFrame:
        """Create engineered features from raw data.
        
        Args:
            market_df: Market data DataFrame
            user_df: User data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        try:
            # Join datasets
            joined_df = market_df.join(
                user_df,
                on=["user_id", "date"],
                how="inner"
            )
            
            # Calculate market features
            joined_df = joined_df.withColumn(
                "price_momentum", 
                (col("close_price") - col("open_price")) / col("open_price")
            )
            
            joined_df = joined_df.withColumn(
                "volume_ratio", 
                col("volume") / col("avg_volume_30d")
            )
            
            joined_df = joined_df.withColumn(
                "volatility", 
                col("high_price") - col("low_price")
            )
            
            # Calculate user features
            joined_df = joined_df.withColumn(
                "days_since_last_purchase",
                datediff(to_date(col("current_date")), to_date(col("last_purchase_date")))
            )
            
            # One-hot encode categorical variables
            categorical_cols = self.config.get("categorical_columns", [])
            for col_name in categorical_cols:
                joined_df = self._one_hot_encode(joined_df, col_name)
            
            logger.info(f"Feature engineering completed, resulting in {joined_df.count()} rows")
            return joined_df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def _one_hot_encode(self, df: DataFrame, column_name: str) -> DataFrame:
        """One-hot encode a categorical column.
        
        Args:
            df: Input DataFrame
            column_name: Column to encode
            
        Returns:
            DataFrame with one-hot encoded columns
        """
        # Get distinct values
        categories = df.select(column_name).distinct().rdd.flatMap(lambda x: x).collect()
        
        # Create a binary column for each category
        for category in categories:
            df = df.withColumn(
                f"{column_name}_{category}", 
                (col(column_name) == category).cast("int")
            )
        
        return df
    
    def get_train_test_split(self, df: DataFrame, test_size: float = 0.2, 
                           seed: int = 42) -> Tuple[DataFrame, DataFrame]:
        """Split data into training and testing sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion for test set
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Create a random split
        train_df, test_df = df.randomSplit([1.0 - test_size, test_size], seed=seed)
        
        logger.info(f"Data split into {train_df.count()} training and {test_df.count()} testing rows")
        return train_df, test_df
    
    def save_processed_data(self, df: DataFrame, bucket: str, prefix: str) -> str:
        """Save processed data to S3.
        
        Args:
            df: DataFrame to save
            bucket: S3 bucket name
            prefix: S3 prefix (folder)
            
        Returns:
            S3 URI of saved data
        """
        try:
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Construct S3 path
            s3_path = f"s3://{bucket}/{prefix}/processed_data_{timestamp}.parquet"
            
            # Save to S3
            df.write.parquet(s3_path, mode="overwrite")
            
            logger.info(f"Saved processed data to {s3_path}")
            return s3_path
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def load_latest_processed_data(self, bucket: str, prefix: str) -> DataFrame:
        """Load the most recent processed data from S3.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 prefix (folder)
            
        Returns:
            DataFrame with processed data
        """
        try:
            # List objects in the prefix
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=f"{prefix}/processed_data_"
            )
            
            if 'Contents' not in response:
                raise ValueError(f"No processed data found in s3://{bucket}/{prefix}/")
            
            # Sort by last modified and get the latest
            contents = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
            latest_key = contents[0]['Key']
            
            # Load the latest file
            s3_path = f"s3://{bucket}/{latest_key}"
            df = self.spark.read.parquet(s3_path)
            
            logger.info(f"Loaded latest processed data from {s3_path}: {df.count()} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading latest processed data: {str(e)}")
            raise


def get_yesterday_data(spark: SparkSession, config: Dict[str, Any]) -> DataFrame:
    """Convenience function to get yesterday's data.
    
    Args:
        spark: Active SparkSession
        config: Configuration dictionary
        
    Returns:
        DataFrame with yesterday's data
    """
    # Get yesterday's date
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Initialize processor
    processor = DataProcessor(spark, config)
    
    # Fetch data
    market_df = processor.fetch_market_data(start_date=yesterday, end_date=yesterday)
    user_df = processor.fetch_user_data(start_date=yesterday, end_date=yesterday)
    
    # Engineer features
    return processor.engineer_features(market_df, user_df)


def get_training_dataset(spark: SparkSession, config: Dict[str, Any], 
                        days: int = 30) -> DataFrame:
    """Get training dataset for the last N days.
    
    Args:
        spark: Active SparkSession
        config: Configuration dictionary
        days: Number of days to include
        
    Returns:
        DataFrame with training data
    """
    # Calculate start date
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Initialize processor
    processor = DataProcessor(spark, config)
    
    # Fetch data
    market_df = processor.fetch_market_data(start_date=start_date, end_date=end_date)
    user_df = processor.fetch_user_data(start_date=start_date, end_date=end_date)
    
    # Engineer features
    return processor.engineer_features(market_df, user_df)