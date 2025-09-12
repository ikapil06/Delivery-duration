"""
Model training module for delivery duration prediction
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from config import *


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.best_model = None
        self.best_score = float('inf')
    
    def initialize_models(self):
        """Initialize all models to be trained"""
        self.models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=RANDOM_STATE),
            'DecisionTree': DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
            'RandomForest': RandomForestRegressor(**RANDOM_FOREST_PARAMS),
            'XGBoost': XGBRegressor(**XGB_PARAMS),
            'LGBM': LGBMRegressor(**LGBM_PARAMS),
            'MLP': MLPRegressor(random_state=RANDOM_STATE, max_iter=1000)
        }
        print(f"Initialized {len(self.models)} models")
    
    def train_single_model(self, X_train, y_train, X_test, y_test, model_name, model):
        """Train a single model and return results"""
        print(f"Training {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        results = {
            'model_name': model_name,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'model': model
        }
        
        print(f"{model_name} - Test RMSE: {test_rmse:.2f}, Test R²: {test_r2:.4f}")
        return results
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models and compare performance"""
        print("Training all models...")
        
        self.initialize_models()
        self.results = {}
        
        for model_name, model in self.models.items():
            try:
                result = self.train_single_model(X_train, y_train, X_test, y_test, model_name, model)
                self.results[model_name] = result
                self.trained_models[model_name] = result['model']
                
                # Track best model
                if result['test_rmse'] < self.best_score:
                    self.best_score = result['test_rmse']
                    self.best_model = model_name
                    
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        print(f"Training completed. Best model: {self.best_model} (RMSE: {self.best_score:.2f})")
        return self.results
    
    def train_preparation_time_model(self, X_train, y_train, X_test, y_test):
        """Train model specifically for preparation time prediction"""
        print("Training preparation time model...")
        
        # Use LGBM for preparation time prediction
        prep_model = LGBMRegressor(**LGBM_PARAMS)
        prep_model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = prep_model.predict(X_train)
        y_test_pred = prep_model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        print(f"Preparation time model - Test RMSE: {test_rmse:.2f}")
        
        return prep_model, y_test_pred, test_rmse
    
    def train_final_combination_model(self, X_train, y_train, X_test, y_test):
        """Train final model for combining preparation time with other components"""
        print("Training final combination model...")
        
        # Test different models for final combination
        final_models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=RANDOM_STATE),
            'DecisionTree': DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
            'RandomForest': RandomForestRegressor(**RANDOM_FOREST_PARAMS),
            'XGBoost': XGBRegressor(**XGB_PARAMS),
            'LGBM': LGBMRegressor(**LGBM_PARAMS),
            'MLP': MLPRegressor(random_state=RANDOM_STATE, max_iter=1000)
        }
        
        best_final_model = None
        best_final_score = float('inf')
        final_results = {}
        
        for model_name, model in final_models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                final_results[model_name] = {
                    'rmse': rmse,
                    'r2': r2,
                    'model': model
                }
                
                print(f"Final {model_name} - RMSE: {rmse:.2f}, R²: {r2:.4f}")
                
                if rmse < best_final_score:
                    best_final_score = rmse
                    best_final_model = model_name
                    
            except Exception as e:
                print(f"Error training final {model_name}: {str(e)}")
                continue
        
        print(f"Best final model: {best_final_model} (RMSE: {best_final_score:.2f})")
        return final_results, best_final_model, best_final_score
    
    def save_model(self, model, model_name, filepath=None):
        """Save trained model to disk"""
        if filepath is None:
            filepath = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, model_name, filepath=None):
        """Load trained model from disk"""
        if filepath is None:
            filepath = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    
    def save_best_model(self, model_name, model, metrics, report_dir):
        """Save the best performing model with metadata"""
        # Save in both .joblib and .pkl formats
        joblib_path = os.path.join(report_dir, f"best_model_{model_name}.joblib")
        pkl_path = os.path.join(MODEL_DIR, f"best_model_{model_name}.pkl")
        metadata_path = os.path.join(report_dir, f"best_model_{model_name}_metadata.json")
        
        # Save model in both formats
        joblib.dump(model, joblib_path)
        joblib.dump(model, pkl_path)
        
        # Save metadata
        import json
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj
        
        metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'metrics': convert_numpy_types(metrics),
            'timestamp': pd.Timestamp.now().isoformat(),
            'joblib_path': joblib_path,
            'pkl_path': pkl_path
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Best model saved to {joblib_path}")
        print(f"Best model saved to {pkl_path}")
        print(f"Model metadata saved to {metadata_path}")
        
        return pkl_path, metadata_path
    
    def get_results_summary(self):
        """Get summary of all model results"""
        if not self.results:
            return "No results available"
        
        summary_df = pd.DataFrame([
            {
                'Model': name,
                'Test_RMSE': result['test_rmse'],
                'Test_R2': result['test_r2'],
                'Test_MAE': result['test_mae']
            }
            for name, result in self.results.items()
        ]).sort_values('Test_RMSE')
        
        return summary_df
