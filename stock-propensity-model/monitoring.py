"""Monitoring utilities for the Stock Purchase Propensity Model."""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, avg, stddev, min, max, count, when, lit
from pyspark.sql.types import DoubleType
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import mlflow

# Configure logging
logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitoring class for the Stock Purchase Propensity Model."""
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        """Initialize with Spark session and configuration."""
        self.spark = spark
        self.config = config
        
        # Initialize AWS clients
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('AWS_REGION', 'us-east-1')
        )
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.config.get('mlflow_tracking_uri', 'http://localhost:5000'))
        mlflow.set_experiment(self.config.get('experiment_name', 'stock_propensity_model'))
    
    def get_predictions(self, days: int = 7) -> DataFrame:
        """Get recent predictions from S3.
        
        Args:
            days: Number of days of predictions to retrieve
            
        Returns:
            DataFrame containing predictions
        """
        try:
            # Calculate date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Get bucket and prefix
            bucket = self.config.get("output_bucket")
            prefix = self.config.get("output_prefix")
            
            # Create list of prefixes to check (one per day)
            prefixes = []
            current_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            while current_date <= end_date_dt:
                date_str = current_date.strftime('%Y-%m-%d')
                prefixes.append(f"{prefix}/prediction_date={date_str}")
                current_date += timedelta(days=1)
            
            # Initialize empty list for all keys
            all_keys = []
            
            # Get all prediction files
            for day_prefix in prefixes:
                response = self.s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=day_prefix
                )
                
                if 'Contents' in response:
                    keys = [item['Key'] for item in response['Contents']]
                    all_keys.extend(keys)
            
            if not all_keys:
                logger.warning(f"No prediction files found in the last {days} days")
                return self.spark.createDataFrame([], schema="user_id string, probability double, prediction int, prediction_date string")
            
            # Load all files as a single DataFrame
            predictions_df = self.spark.read.parquet(
                *[f"s3://{bucket}/{key}" for key in all_keys]
            )
            
            logger.info(f"Loaded {predictions_df.count()} predictions from the last {days} days")
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error retrieving predictions: {str(e)}")
            raise
    
    def get_actual_purchases(self, days: int = 7) -> DataFrame:
        """Get actual purchase data for comparison.
        
        Args:
            days: Number of days of data to retrieve
            
        Returns:
            DataFrame containing actual purchase data
        """
        try:
            # Calculate date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Construct S3 path
            s3_path = self.config.get("user_data_path")
            
            # Load data from S3
            user_df = self.spark.read.parquet(s3_path)
            
            # Filter by date range
            user_df = user_df.filter(
                (col("date") >= start_date) & (col("date") <= end_date)
            )
            
            # Select only the columns we need
            actual_df = user_df.select(
                "user_id", 
                "date", 
                col("purchased_stock").cast(DoubleType())
            )
            
            logger.info(f"Loaded {actual_df.count()} actual purchase records from the last {days} days")
            return actual_df
            
        except Exception as e:
            logger.error(f"Error retrieving actual purchases: {str(e)}")
            raise
    
    def calculate_metrics(self, predictions_df: DataFrame, 
                         actuals_df: DataFrame) -> Dict[str, float]:
        """Calculate performance metrics by comparing predictions to actuals.
        
        Args:
            predictions_df: DataFrame with predictions
            actuals_df: DataFrame with actual purchase data
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Join predictions with actuals
            combined_df = predictions_df.join(
                actuals_df,
                on=["user_id"],
                how="inner"
            )
            
            # Calculate metrics
            metrics_df = combined_df.select(
                (col("prediction") == col("purchased_stock")).cast("int").alias("correct"),
                when(col("prediction") == 1, 
                     when(col("purchased_stock") == 1, 1).otherwise(0)).alias("true_positive"),
                when(col("prediction") == 1, 
                     when(col("purchased_stock") == 0, 1).otherwise(0)).alias("false_positive"),
                when(col("prediction") == 0, 
                     when(col("purchased_stock") == 1, 1).otherwise(0)).alias("false_negative"),
                when(col("prediction") == 0, 
                     when(col("purchased_stock") == 0, 1).otherwise(0)).alias("true_negative")
            )
            
            # Aggregate metrics
            agg_metrics = metrics_df.agg(
                (count("correct") * 1.0).alias("total"),
                (sum("correct") * 1.0).alias("correct_count"),
                (sum("true_positive") * 1.0).alias("tp"),
                (sum("false_positive") * 1.0).alias("fp"),
                (sum("false_negative") * 1.0).alias("fn"),
                (sum("true_negative") * 1.0).alias("tn")
            ).collect()[0]
            
            # Convert to Python dict
            metrics = {
                "total_predictions": float(agg_metrics["total"]),
                "accuracy": float(agg_metrics["correct_count"] / agg_metrics["total"]) 
                    if agg_metrics["total"] > 0 else 0.0,
                "precision": float(agg_metrics["tp"] / (agg_metrics["tp"] + agg_metrics["fp"])) 
                    if (agg_metrics["tp"] + agg_metrics["fp"]) > 0 else 0.0,
                "recall": float(agg_metrics["tp"] / (agg_metrics["tp"] + agg_metrics["fn"])) 
                    if (agg_metrics["tp"] + agg_metrics["fn"]) > 0 else 0.0,
                "f1_score": float(2 * agg_metrics["tp"] / (2 * agg_metrics["tp"] + agg_metrics["fp"] + agg_metrics["fn"])) 
                    if (2 * agg_metrics["tp"] + agg_metrics["fp"] + agg_metrics["fn"]) > 0 else 0.0,
                "true_positive_rate": float(agg_metrics["tp"] / (agg_metrics["tp"] + agg_metrics["fn"])) 
                    if (agg_metrics["tp"] + agg_metrics["fn"]) > 0 else 0.0,
                "false_positive_rate": float(agg_metrics["fp"] / (agg_metrics["fp"] + agg_metrics["tn"])) 
                    if (agg_metrics["fp"] + agg_metrics["tn"]) > 0 else 0.0
            }
            
            logger.info(f"Calculated performance metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def check_data_drift(self, current_data: DataFrame, 
                        reference_data: Optional[DataFrame] = None) -> Dict[str, Any]:
        """Check for data drift by comparing feature distributions.
        
        Args:
            current_data: Current data to analyze
            reference_data: Historical reference data for comparison
            
        Returns:
            Dictionary of drift metrics
        """
        try:
            # If no reference data provided, get from S3
            if reference_data is None:
                # Use reference data path from config
                ref_path = self.config.get("reference_data_path")
                reference_data = self.spark.read.parquet(ref_path)
                logger.info(f"Loaded reference data from {ref_path}")
            
            # Get numerical columns to analyze
            numerical_cols = self.config.get("numerical_columns", [])
            
            # Initialize drift metrics
            drift_metrics = {}
            
            # Calculate statistics for each feature
            for col_name in numerical_cols:
                # Current data stats
                current_stats = current_data.select(
                    avg(col(col_name)).alias("mean"),
                    stddev(col(col_name)).alias("stddev"),
                    min(col(col_name)).alias("min"),
                    max(col(col_name)).alias("max")
                ).collect()[0]
                
                # Reference data stats
                ref_stats = reference_data.select(
                    avg(col(col_name)).alias("mean"),
                    stddev(col(col_name)).alias("stddev"),
                    min(col(col_name)).alias("min"),
                    max(col(col_name)).alias("max")
                ).collect()[0]
                
                # Calculate drift metrics
                mean_diff = abs(current_stats["mean"] - ref_stats["mean"])
                mean_diff_pct = mean_diff / abs(ref_stats["mean"]) if ref_stats["mean"] != 0 else float('inf')
                
                stddev_diff = abs(current_stats["stddev"] - ref_stats["stddev"])
                stddev_diff_pct = stddev_diff / abs(ref_stats["stddev"]) if ref_stats["stddev"] != 0 else float('inf')
                
                # Determine if drift detected
                is_drift = (mean_diff_pct > self.config.get("drift_threshold", 0.1) or 
                           stddev_diff_pct > self.config.get("drift_threshold", 0.1))
                
                # Store metrics
                drift_metrics[col_name] = {
                    "current_mean": float(current_stats["mean"]),
                    "reference_mean": float(ref_stats["mean"]),
                    "mean_diff_pct": float(mean_diff_pct),
                    "current_stddev": float(current_stats["stddev"]),
                    "reference_stddev": float(ref_stats["stddev"]),
                    "stddev_diff_pct": float(stddev_diff_pct),
                    "drift_detected": is_drift
                }
            
            # Overall drift status
            drift_metrics["any_drift_detected"] = any(
                metric["drift_detected"] for metric in drift_metrics.values() 
                if isinstance(metric, dict)
            )
            
            logger.info(f"Data drift analysis completed. Drift detected: {drift_metrics['any_drift_detected']}")
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Error checking data drift: {str(e)}")
            raise
    
    def log_metrics_to_mlflow(self, metrics: Dict[str, Any], run_id: Optional[str] = None):
        """Log performance metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            run_id: Optional MLflow run ID
        """
        try:
            # Start a new run if run_id not provided
            if run_id is None:
                with mlflow.start_run() as run:
                    # Log metrics
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)
                    
                    # Log run info
                    mlflow.set_tag("monitoring_timestamp", datetime.now().isoformat())
                    
                    logger.info(f"Logged metrics to MLflow run {run.info.run_id}")
            else:
                # Use existing run
                with mlflow.start_run(run_id=run_id):
                    # Log metrics
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)
                    
                    logger.info(f"Logged metrics to existing MLflow run {run_id}")
                    
        except Exception as e:
            logger.error(f"Error logging metrics to MLflow: {str(e)}")
            raise
    
    def generate_monitoring_report(self, metrics: Dict[str, Any], 
                                 drift_metrics: Dict[str, Any]) -> str:
        """Generate a monitoring report and save to S3.
        
        Args:
            metrics: Performance metrics
            drift_metrics: Data drift metrics
            
        Returns:
            S3 URI of the saved report
        """
        try:
            # Create timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create report data
            report = {
                "timestamp": datetime.now().isoformat(),
                "performance_metrics": metrics,
                "data_drift_metrics": drift_metrics,
                "model_version": self.config.get("model_version", "unknown")
            }
            
            # Convert to JSON
            report_json = json.dumps(report, indent=2)
            
            # Local path for temp save
            local_path = f"reports/monitoring_report_{timestamp}.json"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, 'w') as f:
                f.write(report_json)
            
            # Upload to S3
            bucket = self.config.get("output_bucket")
            prefix = self.config.get("reports_prefix", "reports")
            s3_path = f"{prefix}/monitoring_report_{timestamp}.json"
            
            self.s3_client.upload_file(
                local_path, 
                bucket, 
                s3_path
            )
            
            s3_uri = f"s3://{bucket}/{s3_path}"
            logger.info(f"Generated monitoring report and saved to {s3_uri}")
            
            return s3_uri
            
        except Exception as e:
            logger.error(f"Error generating monitoring report: {str(e)}")
            raise
    
    def run_monitoring(self, days: int = 7, 
                     save_report: bool = True, 
                     log_to_mlflow: bool = True) -> Dict[str, Any]:
        """Run complete monitoring process.
        
        Args:
            days: Number of days of data to analyze
            save_report: Whether to save report to S3
            log_to_mlflow: Whether to log metrics to MLflow
            
        Returns:
            Dictionary with all metrics
        """
        try:
            # Get predictions
            predictions_df = self.get_predictions(days=days)
            
            # Get actual purchases
            actuals_df = self.get_actual_purchases(days=days)
            
            # Calculate performance metrics
            performance_metrics = self.calculate_metrics(predictions_df, actuals_df)
            
            # Check for data drift
            # Get current data for drift analysis
            from stock_propensity_model.data_utils import get_yesterday_data
            current_data = get_yesterday_data(self.spark, self.config)
            
            # Reference data path
            ref_path = self.config.get("reference_data_path")
            reference_data = self.spark.read.parquet(ref_path)
            
            # Check drift
            drift_metrics = self.check_data_drift(current_data, reference_data)
            
            # Combine all metrics
            all_metrics = {
                "performance": performance_metrics,
                "drift": drift_metrics
            }
            
            # Save report if requested
            if save_report:
                report_path = self.generate_monitoring_report(performance_metrics, drift_metrics)
                all_metrics["report_path"] = report_path
            
            # Log to MLflow if requested
            if log_to_mlflow:
                # Flatten metrics for MLflow
                flat_metrics = {}
                for metric, value in performance_metrics.items():
                    flat_metrics[f"performance_{metric}"] = value
                
                # Add key drift metrics
                flat_metrics["drift_detected"] = int(drift_metrics.get("any_drift_detected", False))
                
                # Log to MLflow
                self.log_metrics_to_mlflow(flat_metrics)
            
            logger.info("Monitoring completed successfully")
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error running monitoring: {str(e)}")
            raise


def run_daily_monitoring(spark: SparkSession, config: Dict[str, Any]):
    """Run daily monitoring as a standalone script.
    
    Args:
        spark: Active SparkSession
        config: Configuration dictionary
    """
    try:
        # Initialize monitor
        monitor = ModelMonitor(spark, config)
        
        # Run monitoring for the last 7 days
        metrics = monitor.run_monitoring(days=7)
        
        # Check for alerts
        if metrics["drift"]["any_drift_detected"]:
            logger.warning("DATA DRIFT DETECTED! See report for details.")
        
        if metrics["performance"]["accuracy"] < config.get("alert_accuracy_threshold", 0.7):
            logger.warning(f"MODEL PERFORMANCE BELOW THRESHOLD! Accuracy: {metrics['performance']['accuracy']:.4f}")
        
        logger.info("Daily monitoring completed successfully")
        return metrics
        
    except Exception as e:
        logger.error(f"Error in daily monitoring: {str(e)}")
        raise


# Script entry point
if __name__ == "__main__":
    from pyspark.sql import SparkSession
    import json
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='logs/monitoring.log',
        filemode='a'
    )
    
    try:
        # Initialize Spark session
        spark = SparkSession.builder \
            .appName("StockPropensityMonitoring") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "8g") \
            .config("spark.dynamicAllocation.enabled", "true") \
            .getOrCreate()
            
        # Load configuration
        with open(os.environ.get("CONFIG_PATH", "config/model_config.json"), 'r') as f:
            config = json.load(f)
        
        # Run daily monitoring
        run_daily_monitoring(spark, config)
        
    except Exception as e:
        logger.error(f"Monitoring script failed: {str(e)}")
        raise