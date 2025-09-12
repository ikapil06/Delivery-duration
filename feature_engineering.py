"""
Feature engineering module for delivery duration prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config import *


class FeatureEngineer:
    """Handles feature engineering and selection"""
    
    def __init__(self):
        self.data = None
        self.scaler = None
        self.selected_features = None
        self.feature_importances = None
    
    def create_dummy_variables(self, data):
        """Create dummy variables for categorical features"""
        print("Creating dummy variables...")
        
        # Create dummies for order protocol
        order_protocol_dummies = pd.get_dummies(data.order_protocol, prefix='order_protocol')
        
        # Create dummies for market_id
        market_id_dummies = pd.get_dummies(data.market_id, prefix='market_id')
        
        # Create dummies for store primary category
        store_category_dummies = pd.get_dummies(data.nan_free_store_primary_category, prefix='category')
        
        print(f"Created dummies: order_protocol({order_protocol_dummies.shape[1]}), "
              f"market_id({market_id_dummies.shape[1]}), "
              f"store_category({store_category_dummies.shape[1]})")
        
        return order_protocol_dummies, market_id_dummies, store_category_dummies
    
    def create_engineered_features(self, data):
        """Create advanced engineered features"""
        print("Creating engineered features...")
        
        # Price-related features
        data['percent_distinct_item_of_total'] = data['num_distinct_items'] / data['total_items']
        data['avg_price_per_item'] = data['subtotal'] / data['total_items']
        data['price_range_of_items'] = data['max_item_price'] - data['min_item_price']
        
        print("Engineered features created")
        return data
    
    def remove_highly_correlated_features(self, data):
        """Remove highly correlated features"""
        print("Removing highly correlated features...")
        
        # Check which columns actually exist before dropping
        existing_cols = [col for col in HIGH_CORR_COLS if col in data.columns]
        if existing_cols:
            data = data.drop(columns=existing_cols)
            print(f"Removed highly correlated features: {existing_cols}")
        else:
            print("No highly correlated features found to remove")
        
        return data
    
    def remove_vif_features(self, data):
        """Remove features with high VIF (Variance Inflation Factor)"""
        print("Removing high VIF features...")
        
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        def compute_vif(features):
            vif_data = pd.DataFrame()
            vif_data['feature'] = features
            vif_data['VIF'] = [variance_inflation_factor(data[features].values, i) 
                              for i in range(len(features))]
            return vif_data.sort_values(by=['VIF']).reset_index(drop=True)
        
        # Get features excluding target
        target_cols = [col for col in [TARGET_COLUMN, PREP_TIME_COLUMN] if col in data.columns]
        features = data.drop(columns=target_cols).columns.to_list()
        
        if len(features) > 0:
            vif_data = compute_vif(features)
            
            # Remove features with VIF > threshold, but preserve important features
            high_vif_features = vif_data[vif_data.VIF > VIF_THRESHOLD]['feature'].tolist()
            # Filter out features that should be preserved
            features_to_remove = [f for f in high_vif_features if f not in PRESERVE_FEATURES]
            
            if features_to_remove:
                data = data.drop(columns=features_to_remove)
                print(f"Removed high VIF features: {features_to_remove}")
            else:
                print("No removable high VIF features found (all high VIF features are preserved)")
        else:
            print("No features available for VIF analysis")
        
        return data
    
    def select_features_by_importance(self, data, model=None):
        """Select top features based on importance"""
        print("Selecting features by importance...")
        
        if model is None:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        
        # Prepare features and target
        X = data.drop(columns=[TARGET_COLUMN])
        y = data[TARGET_COLUMN]
        
        # Fit model to get feature importances
        model.fit(X, y)
        
        # Get feature importances
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features
        self.selected_features = feature_importance_df.head(TOP_FEATURES)['feature'].tolist()
        self.feature_importances = feature_importance_df
        
        print(f"Selected top {len(self.selected_features)} features")
        return self.selected_features
    
    def create_preparation_time_target(self, data):
        """Create preparation time as target variable"""
        print("Creating preparation time target...")
        
        data[PREP_TIME_COLUMN] = (
            data[TARGET_COLUMN] - 
            data['estimated_store_to_consumer_driving_duration'] - 
            data['estimated_order_place_duration']
        )
        
        print(f"Preparation time created. Range: {data[PREP_TIME_COLUMN].min():.0f} - {data[PREP_TIME_COLUMN].max():.0f} seconds")
        return data
    
    def scale_features(self, X, y=None, scaler_type='standard'):
        """Scale features using specified scaler"""
        print(f"Scaling features using {scaler_type} scaler...")
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        X_scaled = self.scaler.fit_transform(X)
        
        if y is not None:
            y_scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
            y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))
            return X_scaled, y_scaled, self.scaler, y_scaler
        else:
            return X_scaled, self.scaler
    
    def engineer_features(self, data, remove_correlated=True, remove_vif=True, 
                         select_features=True, create_prep_time=True):
        """Run complete feature engineering pipeline"""
        print("Starting feature engineering pipeline...")
        
        # Create dummy variables
        order_dummies, market_dummies, store_dummies = self.create_dummy_variables(data)
        
        # Drop unnecessary columns and combine with dummies
        train_df = data.drop(columns=COLS_TO_DROP)
        train_df = pd.concat([train_df, order_dummies, market_dummies, store_dummies], axis=1)
        
        # Convert to float32 for memory efficiency
        train_df = train_df.astype('float32')
        
        # Remove highly correlated features
        if remove_correlated:
            train_df = self.remove_highly_correlated_features(train_df)
        
        # Create engineered features
        train_df = self.create_engineered_features(train_df)
        
        # Remove VIF features
        if remove_vif:
            train_df = self.remove_vif_features(train_df)
        
        # Create preparation time target
        if create_prep_time:
            train_df = self.create_preparation_time_target(train_df)
        
        # Select features by importance
        if select_features:
            self.select_features_by_importance(train_df)
            if self.selected_features:
                # Keep only selected features plus target
                features_to_keep = self.selected_features + [TARGET_COLUMN]
                if PREP_TIME_COLUMN in train_df.columns:
                    features_to_keep.append(PREP_TIME_COLUMN)
                # Only keep features that actually exist
                features_to_keep = [col for col in features_to_keep if col in train_df.columns]
                train_df = train_df[features_to_keep]
        
        print(f"Feature engineering completed. Final shape: {train_df.shape}")
        return train_df
