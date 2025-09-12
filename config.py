"""
Configuration file for Delivery Duration Prediction Project
"""

import os

# Data paths
DATA_DIR = "."
HISTORICAL_DATA_PATH = os.path.join(DATA_DIR, "historical_data.csv")
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VIF_THRESHOLD = 20
TOP_FEATURES = 40

# Feature engineering parameters
CORRELATION_THRESHOLD = 0.9

# Model hyperparameters
LGBM_PARAMS = {
    'random_state': RANDOM_STATE,
    'verbose': -1
}

XGB_PARAMS = {
    'random_state': RANDOM_STATE,
    'verbosity': 0
}

RANDOM_FOREST_PARAMS = {
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Target column
TARGET_COLUMN = 'actual_total_delivery_duration'
PREP_TIME_COLUMN = 'prep_time'

# Columns to drop
COLS_TO_DROP = [
    'created_at', 'market_id', 'store_id', 'store_primary_category', 
    'actual_delivery_time', 'nan_free_store_primary_category', 'order_protocol'
]

# Highly correlated columns to remove
HIGH_CORR_COLS = [
    'total_onshift_dashers', 'total_busy_dashers', 'category_indonesian', 
    'estimated_non_prep_duration'
]

# VIF columns to remove
VIF_COLS_TO_REMOVE = ['percent_distinct_item_of_total']

# Features to preserve (important features that should not be removed by VIF)
PRESERVE_FEATURES = [
    'estimated_order_place_duration',
    'estimated_store_to_consumer_driving_duration',
    'total_outstanding_orders',
    'busy_dashers_ratio',
    'total_items',
    'subtotal',
    'num_distinct_items',
    'min_item_price',
    'max_item_price'
]
