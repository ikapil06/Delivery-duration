"""
Data preprocessing module for delivery duration prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime
from config import *


class DataPreprocessor:
    """Handles data loading, cleaning, and initial preprocessing"""
    
    def __init__(self):
        self.data = None
        self.store_categories = {}
    
    def load_data(self, file_path=HISTORICAL_DATA_PATH):
        """Load historical delivery data"""
        print("Loading data...")
        self.data = pd.read_csv(file_path)
        print(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data
    
    def create_target_variable(self):
        """Create the target variable for regression"""
        print("Creating target variable...")
        
        # Convert datetime columns
        self.data['created_at'] = pd.to_datetime(self.data['created_at'])
        self.data['actual_delivery_time'] = pd.to_datetime(self.data['actual_delivery_time'])
        
        # Create target variable (delivery duration in seconds)
        self.data[TARGET_COLUMN] = (
            self.data['actual_delivery_time'] - self.data['created_at']
        ).dt.total_seconds()
        
        print(f"Target variable created. Range: {self.data[TARGET_COLUMN].min():.0f} - {self.data[TARGET_COLUMN].max():.0f} seconds")
        return self.data
    
    def fill_store_categories(self):
        """Fill missing store categories using mode for each store"""
        print("Filling missing store categories...")
        
        # Create dictionary with most repeated categories for each store
        store_ids = self.data['store_id'].unique().tolist()
        self.store_categories = {
            store_id: self.data[self.data.store_id == store_id].store_primary_category.mode() 
            for store_id in store_ids
        }
        
        def fill_category(store_id):
            try:
                return self.store_categories[store_id].values[0]
            except:
                return np.nan
        
        # Fill null values
        self.data['nan_free_store_primary_category'] = self.data.store_id.apply(fill_category)
        
        original_nulls = self.data['store_primary_category'].isnull().sum()
        filled_nulls = self.data['nan_free_store_primary_category'].isnull().sum()
        
        print(f"Store categories filled. Nulls reduced from {original_nulls} to {filled_nulls}")
        return self.data
    
    def create_basic_features(self):
        """Create basic engineered features"""
        print("Creating basic features...")
        
        # Create estimated non-prep duration
        self.data['estimated_non_prep_duration'] = (
            self.data['estimated_store_to_consumer_driving_duration'] + 
            self.data['estimated_order_place_duration']
        )
        
        # Create busy dashers ratio
        self.data['busy_dashers_ratio'] = (
            self.data['total_busy_dashers'] / self.data['total_onshift_dashers']
        )
        
        print("Basic features created")
        return self.data
    
    def clean_data(self):
        """Clean data by handling infinity values and nulls"""
        print("Cleaning data...")
        
        # Replace infinity values with NaN
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop rows with any NaN values
        initial_shape = self.data.shape
        self.data.dropna(inplace=True)
        final_shape = self.data.shape
        
        print(f"Data cleaned. Shape changed from {initial_shape} to {final_shape}")
        print(f"Remaining NaN values: {self.data.isna().sum().sum()}")
        
        return self.data
    
    def preprocess(self, file_path=HISTORICAL_DATA_PATH):
        """Run complete preprocessing pipeline"""
        print("Starting data preprocessing pipeline...")
        
        self.load_data(file_path)
        self.create_target_variable()
        self.fill_store_categories()
        self.create_basic_features()
        self.clean_data()
        
        print("Data preprocessing completed!")
        return self.data
