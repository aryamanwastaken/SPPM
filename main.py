from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode
from pyspark.sql.types import DoubleType, StringType
import pandas as pd
import numpy as np
import boto3
import json
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
import mlflow
import mlflow.xgboost
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockPropensityModel:
    def __init__(self, config_path='config/model_config.json'):
        """Initialize the Stock Propensity Model pipeline.
        
        Args:
            config_path (str): Path to configuration JSON
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("StockPropensityModel") \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.memory", "16g") \
            .config("spark.dynamicAllocation.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
            
        logger.info("Initialized Spark session")
        
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
        
        # Initialize model
        self.model = None
        self.scaler = StandardScaler()
        
        logger.info("StockPropensityModel initialized successfully")
    
    def ingest_data(self):
        """Ingest real-time stock data from API and user behavior data from S3."""
        logger.info("Starting data ingestion")
        
        # Load market data from API
        try:
            # This would typically call an external API
            # For this example, we'll load from S3
            market_data_path = self.config['market_data_path']
            market_data_df = self.spark.read.parquet(market_data_path)
            logger.info(f"Loaded market data from {market_data_path}")
            
            # Load user behavior data from S3
            user_data_path = self.config['user_data_path']
            user_data_df = self.spark.read.parquet(user_data_path)
            logger.info(f"Loaded user data from {user_data_path}")
            
            # Join datasets
            joined_df = market_data_df.join(
                user_data_df,
                on=['user_id', 'date'],
                how='inner'
            )
            
            logger.info(f"Joined data: {joined_df.count()} rows")
            return joined_df
            
        except Exception as e:
            logger.error(f"Error during data ingestion: {str(e)}")
            raise
    
    def preprocess_data(self, df):
        """Preprocess and feature engineer the data."""
        logger.info("Starting data preprocessing")
        
        try:
            # Feature engineering with Spark
            df = df.withColumn("price_momentum", 
                              (col("close_price") - col("open_price")) / col("open_price"))
            
            df = df.withColumn("volume_ratio", 
                              col("volume") / col("avg_volume_30d"))
            
            # Calculate volatility
            df = df.withColumn("volatility", 
                              col("high_price") - col("low_price")) 
            
            # User-specific features
            df = df.withColumn("days_since_last_purchase", 
                              col("current_date") - col("last_purchase_date"))
            
            # One-hot encode categorical variables
            categorical_cols = self.config.get("categorical_columns", ["sector", "market_cap_category"])
            for col_name in categorical_cols:
                df = self._one_hot_encode(df, col_name)
            
            # Select final feature set
            feature_cols = self.config.get("feature_columns", [])
            label_col = self.config.get("label_column", "purchased_stock")
            
            final_df = df.select(feature_cols + [label_col])
            
            # Convert to pandas for scikit-learn & XGBoost
            pdf = final_df.toPandas()
            
            # Handle missing values
            pdf = pdf.fillna(self.config.get("fill_value", 0))
            
            X = pdf.drop(columns=[label_col])
            y = pdf[label_col]
            
            # Scale numerical features
            numerical_cols = self.config.get("numerical_columns", [])
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            
            logger.info("Data preprocessing completed successfully")
            return X, y
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise
    
    def _one_hot_encode(self, df, column_name):
        """Helper function to one-hot encode categorical variables in Spark."""
        categories = df.select(column_name).distinct().rdd.flatMap(lambda x: x).collect()
        for category in categories:
            df = df.withColumn(f"{column_name}_{category}", 
                              (col(column_name) == category).cast("int"))
        return df
    
    def train_model(self, X, y):
        """Train XGBoost model with the processed data."""
        logger.info("Starting model training")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.get("test_size", 0.2),
                random_state=self.config.get("random_seed", 42)
            )
            
            # Start MLflow run
            with mlflow.start_run() as run:
                # Configure XGBoost parameters
                params = {
                    'max_depth': self.config.get("max_depth", 6),
                    'learning_rate': self.config.get("learning_rate", 0.1),
                    'n_estimators': self.config.get("n_estimators", 100),
                    'objective': self.config.get("objective", 'binary:logistic'),
                    'subsample': self.config.get("subsample", 0.8),
                    'colsample_bytree': self.config.get("colsample_bytree", 0.8),
                    'eval_metric': self.config.get("eval_metric", 'auc'),
                    'seed': self.config.get("random_seed", 42)
                }
                
                # Log parameters
                for param, value in params.items():
                    mlflow.log_param(param, value)
                
                # Create DMatrix for training
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)
                
                # Train model
                self.model = xgb.train(
                    params,
                    dtrain,
                    evals=[(dtrain, 'train'), (dtest, 'test')],
                    early_stopping_rounds=self.config.get("early_stopping_rounds", 10),
                    verbose_eval=False
                )
                
                # Make predictions for evaluation
                y_pred = self.model.predict(dtest)
                
                # Calculate metrics
                auc = roc_auc_score(y_test, y_pred)
                precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
                f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
                best_threshold = thresholds[np.argmax(f1_scores)]
                
                # Log metrics
                mlflow.log_metric("auc", auc)
                mlflow.log_metric("best_f1", np.max(f1_scores))
                mlflow.log_metric("best_threshold", best_threshold)
                
                # Log model
                mlflow.xgboost.log_model(self.model, "model")
                
                # Save threshold for inference
                self.config["prediction_threshold"] = float(best_threshold)
                with open('config/model_config.json', 'w') as f:
                    json.dump(self.config, f)
                
                logger.info(f"Model trained successfully. AUC: {auc:.4f}, Best F1: {np.max(f1_scores):.4f}")
                return self.model, run.info.run_id
                
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def predict(self, new_data):
        """Generate predictions for new data."""
        logger.info(f"Generating predictions for {len(new_data)} records")
        
        try:
            # Preprocess new data the same way as training data
            numerical_cols = self.config.get("numerical_columns", [])
            new_data[numerical_cols] = self.scaler.transform(new_data[numerical_cols])
            
            # Convert to DMatrix
            dmatrix = xgb.DMatrix(new_data)
            
            # Generate predictions
            predictions_prob = self.model.predict(dmatrix)
            threshold = self.config.get("prediction_threshold", 0.5)
            predictions_binary = (predictions_prob >= threshold).astype(int)
            
            # Create results dataframe
            results = pd.DataFrame({
                'probability': predictions_prob,
                'prediction': predictions_binary
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def save_predictions(self, user_ids, predictions, output_bucket=None, output_prefix=None):
        """Save predictions to S3."""
        logger.info("Saving predictions to S3")
        
        try:
            # Combine user_ids with predictions
            if isinstance(user_ids, pd.Series):
                user_ids = user_ids.values
                
            results_df = pd.DataFrame({
                'user_id': user_ids,
                'probability': predictions['probability'],
                'prediction': predictions['prediction'],
                'prediction_date': datetime.now().strftime('%Y-%m-%d')
            })
            
            # Add timestamp partition
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Get S3 location from config or parameters
            bucket = output_bucket or self.config.get("output_bucket")
            prefix = output_prefix or self.config.get("output_prefix")
            s3_path = f"s3://{bucket}/{prefix}/prediction_date={datetime.now().strftime('%Y-%m-%d')}/predictions_{timestamp}.parquet"
            
            # Convert to Spark and save
            spark_df = self.spark.createDataFrame(results_df)
            spark_df.write.parquet(s3_path, mode="overwrite")
            
            logger.info(f"Predictions saved to {s3_path}")
            return s3_path
            
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
            raise
    
    def run_pipeline(self):
        """Run the complete pipeline from data ingestion to saving predictions."""
        logger.info("Starting complete pipeline")
        
        try:
            # Ingest data
            df = self.ingest_data()
            
            # Preprocess data
            X, y = self.preprocess_data(df)
            
            # Train model if needed
            if self.model is None:
                self.model, run_id = self.train_model(X, y)
                logger.info(f"Trained new model with run_id: {run_id}")
            
            # Generate predictions (in production, this would be on new data)
            predictions = self.predict(X)
            
            # Save predictions
            user_ids = df.select("user_id").toPandas()["user_id"]
            output_path = self.save_predictions(user_ids, predictions)
            
            logger.info(f"Pipeline completed successfully. Predictions saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise

# Script entry point
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        # Initialize model
        model = StockPropensityModel()
        
        # Run pipeline
        output_path = model.run_pipeline()
        logger.info(f"Stock Propensity Model pipeline completed. Results: {output_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
