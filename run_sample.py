"""
Run delivery duration prediction on sample data with image and model saving
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import custom modules
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from evaluation import ModelEvaluator
from config import *


class SampleDeliveryPredictor:
    """Sample version of the delivery duration predictor for testing"""
    
    def __init__(self, sample_file="sample_data.csv"):
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.sample_file = sample_file
        
        self.raw_data = None
        self.processed_data = None
        self.features_data = None
        self.results = {}
        
    def run_sample_pipeline(self):
        """Run the complete pipeline on sample data"""
        print("="*60)
        print("SAMPLE DELIVERY DURATION PREDICTION PIPELINE")
        print("="*60)
        
        # Step 1: Data Preprocessing
        print("\n1. DATA PREPROCESSING")
        print("-" * 30)
        self.raw_data = self.preprocessor.preprocess(self.sample_file)
        self.processed_data = self.raw_data.copy()
        
        # Step 2: Feature Engineering
        print("\n2. FEATURE ENGINEERING")
        print("-" * 30)
        self.features_data = self.feature_engineer.engineer_features(
            self.processed_data, 
            remove_correlated=True, 
            remove_vif=False,  # Skip VIF for small sample
            select_features=False,  # Skip feature selection for small sample
            create_prep_time=True
        )
        
        # Step 3: Direct Model Training
        print("\n3. DIRECT MODEL TRAINING")
        print("-" * 30)
        self.run_direct_approach()
        
        # Step 4: Two-Stage Approach
        print("\n4. TWO-STAGE APPROACH")
        print("-" * 30)
        self.run_two_stage_approach()
        
        # Step 5: Generate Reports and Save Images/Models
        print("\n5. GENERATING REPORTS AND SAVING OUTPUTS")
        print("-" * 30)
        self.generate_final_reports()
        
        print("\n" + "="*60)
        print("SAMPLE PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
    
    def run_direct_approach(self):
        """Run direct prediction approach"""
        print("Running direct prediction approach...")
        
        # Prepare data for direct approach
        X = self.features_data.drop(columns=[TARGET_COLUMN])
        y = self.features_data[TARGET_COLUMN]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE  # Use 30% for test with small sample
        )
        
        # Scale features
        X_train_scaled, y_train_scaled, X_scaler, y_scaler = self.feature_engineer.scale_features(
            X_train, y_train, scaler_type='standard'
        )
        X_test_scaled = X_scaler.transform(X_test)
        
        # Train models
        direct_results = self.model_trainer.train_models(
            X_train_scaled, y_train_scaled[:, 0], X_test_scaled, y_test
        )
        
        # Evaluate best model
        best_model_name = self.model_trainer.best_model
        best_model = self.model_trainer.trained_models[best_model_name]
        
        # Inverse transform predictions for evaluation
        y_pred_scaled = best_model.predict(X_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))[:, 0]
        
        # Calculate final metrics
        metrics = self.evaluator.calculate_metrics(y_test, y_pred)
        self.results['direct_approach'] = {
            'best_model': best_model_name,
            'metrics': metrics,
            'predictions': y_pred,
            'actual': y_test,
            'model': best_model
        }
        
        print(f"Direct approach - Best model: {best_model_name}")
        print(f"RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R2']:.4f}")
        
        return direct_results
    
    def run_two_stage_approach(self):
        """Run two-stage prediction approach"""
        print("Running two-stage prediction approach...")
        
        # Prepare data for preparation time prediction
        X = self.features_data.drop(columns=[TARGET_COLUMN, PREP_TIME_COLUMN])
        y = self.features_data[PREP_TIME_COLUMN]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE
        )
        
        # Scale features
        X_train_scaled, y_train_scaled, X_scaler, y_scaler = self.feature_engineer.scale_features(
            X_train, y_train, scaler_type='standard'
        )
        X_test_scaled = X_scaler.transform(X_test)
        
        # Train preparation time model
        prep_model, prep_predictions_scaled, prep_rmse = self.model_trainer.train_preparation_time_model(
            X_train_scaled, y_train_scaled[:, 0], X_test_scaled, y_test
        )
        
        # Inverse transform preparation time predictions
        prep_predictions = y_scaler.inverse_transform(prep_predictions_scaled.reshape(-1, 1))[:, 0]
        
        # Create final prediction dataframe
        test_indices = X_test.index
        pred_df = pd.DataFrame({
            'actual_total_delivery_duration': y_test.values,
            'prep_duration_prediction': prep_predictions,
            'estimated_store_to_consumer_driving_duration': X_test['estimated_store_to_consumer_driving_duration'].values,
            'estimated_order_place_duration': X_test['estimated_order_place_duration'].values
        })
        
        # Sum up components
        pred_df['sum_total_delivery_duration'] = (
            pred_df['prep_duration_prediction'] + 
            pred_df['estimated_store_to_consumer_driving_duration'] + 
            pred_df['estimated_order_place_duration']
        )
        
        # Train final combination model
        X_final = pred_df[['prep_duration_prediction', 'estimated_store_to_consumer_driving_duration', 'estimated_order_place_duration']]
        y_final = pred_df['actual_total_delivery_duration']
        
        X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
            X_final, y_final, test_size=0.3, random_state=RANDOM_STATE
        )
        
        final_results, best_final_model, best_final_score = self.model_trainer.train_final_combination_model(
            X_train_final, y_train_final, X_test_final, y_test_final
        )
        
        # Get final predictions
        best_final_model_obj = final_results[best_final_model]['model']
        final_predictions = best_final_model_obj.predict(X_test_final)
        
        # Calculate final metrics
        metrics = self.evaluator.calculate_metrics(y_test_final, final_predictions)
        self.results['two_stage_approach'] = {
            'prep_model_rmse': prep_rmse,
            'final_model': best_final_model,
            'final_metrics': metrics,
            'final_predictions': final_predictions,
            'actual': y_test_final,
            'model': best_final_model_obj,
            'simple_sum_rmse': np.sqrt(np.mean((y_test_final - pred_df['sum_total_delivery_duration'].iloc[:len(y_test_final)])**2))
        }
        
        print(f"Two-stage approach - Final model: {best_final_model}")
        print(f"RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R2']:.4f}")
        print(f"Simple sum RMSE: {self.results['two_stage_approach']['simple_sum_rmse']:.2f}")
        
        return final_results
    
    def generate_final_reports(self):
        """Generate final evaluation reports with images and model saving"""
        print("Generating final reports with images and model saving...")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(OUTPUT_DIR, f"sample_report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate comparison report
        self.generate_comparison_report(report_dir)
        
        # Save images and models
        self.save_images_and_models(report_dir)
        
        # Save results
        self.save_results(report_dir)
        
        print(f"All outputs saved to {report_dir}")
    
    def save_images_and_models(self, report_dir):
        """Save prediction images and best models"""
        print("Saving images and models...")
        
        # Save direct approach results
        if 'direct_approach' in self.results:
            direct = self.results['direct_approach']
            
            # Save prediction plot
            plot_path = os.path.join(report_dir, f"direct_approach_{direct['best_model']}_predictions.png")
            self.evaluator.plot_predictions(
                direct['actual'], direct['predictions'], 
                f"Direct_{direct['best_model']}", plot_path
            )
            
            # Save model
            model_path, metadata_path = self.model_trainer.save_best_model(
                f"direct_{direct['best_model']}", direct['model'], 
                direct['metrics'], report_dir
            )
        
        # Save two-stage approach results
        if 'two_stage_approach' in self.results:
            two_stage = self.results['two_stage_approach']
            
            # Save prediction plot
            plot_path = os.path.join(report_dir, f"two_stage_approach_{two_stage['final_model']}_predictions.png")
            self.evaluator.plot_predictions(
                two_stage['actual'], two_stage['final_predictions'], 
                f"TwoStage_{two_stage['final_model']}", plot_path
            )
            
            # Save model
            model_path, metadata_path = self.model_trainer.save_best_model(
                f"two_stage_{two_stage['final_model']}", two_stage['model'], 
                two_stage['final_metrics'], report_dir
            )
        
        # Save feature importance if available
        if self.feature_engineer.feature_importances is not None:
            importance_path = os.path.join(report_dir, "feature_importance.png")
            self.evaluator.plot_feature_importance(
                self.feature_engineer.feature_importances, 
                top_n=10, save_path=importance_path
            )
    
    def generate_comparison_report(self, report_dir):
        """Generate comparison report between approaches"""
        report_path = os.path.join(report_dir, "comparison_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("SAMPLE DELIVERY DURATION PREDICTION - COMPARISON REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Sample Data: {self.sample_file}\n")
            f.write(f"Total Samples: {len(self.raw_data)}\n")
            f.write(f"Features Used: {self.features_data.shape[1]}\n\n")
            
            f.write("1. DIRECT APPROACH\n")
            f.write("-" * 20 + "\n")
            if 'direct_approach' in self.results:
                direct = self.results['direct_approach']
                f.write(f"Best Model: {direct['best_model']}\n")
                f.write(f"RMSE: {direct['metrics']['RMSE']:.2f}\n")
                f.write(f"R²: {direct['metrics']['R2']:.4f}\n")
                f.write(f"MAE: {direct['metrics']['MAE']:.2f}\n\n")
            
            f.write("2. TWO-STAGE APPROACH\n")
            f.write("-" * 20 + "\n")
            if 'two_stage_approach' in self.results:
                two_stage = self.results['two_stage_approach']
                f.write(f"Preparation Model RMSE: {two_stage['prep_model_rmse']:.2f}\n")
                f.write(f"Final Model: {two_stage['final_model']}\n")
                f.write(f"Final RMSE: {two_stage['final_metrics']['RMSE']:.2f}\n")
                f.write(f"Final R²: {two_stage['final_metrics']['R2']:.4f}\n")
                f.write(f"Final MAE: {two_stage['final_metrics']['MAE']:.2f}\n")
                f.write(f"Simple Sum RMSE: {two_stage['simple_sum_rmse']:.2f}\n\n")
            
            f.write("3. RECOMMENDATION\n")
            f.write("-" * 20 + "\n")
            if 'direct_approach' in self.results and 'two_stage_approach' in self.results:
                direct_rmse = self.results['direct_approach']['metrics']['RMSE']
                two_stage_rmse = self.results['two_stage_approach']['final_metrics']['RMSE']
                
                if two_stage_rmse < direct_rmse:
                    improvement = ((direct_rmse - two_stage_rmse) / direct_rmse) * 100
                    f.write(f"Two-stage approach is better by {improvement:.1f}%\n")
                    f.write(f"Recommended approach: Two-stage with {self.results['two_stage_approach']['final_model']}\n")
                else:
                    f.write(f"Direct approach is better\n")
                    f.write(f"Recommended approach: Direct with {self.results['direct_approach']['best_model']}\n")
    
    def save_results(self, report_dir):
        """Save results to files"""
        # Save feature importances
        if self.feature_engineer.feature_importances is not None:
            feature_path = os.path.join(report_dir, "feature_importances.csv")
            self.feature_engineer.feature_importances.to_csv(feature_path, index=False)
        
        # Save model results
        results_path = os.path.join(report_dir, "model_results.csv")
        if self.model_trainer.results:
            results_df = self.model_trainer.get_results_summary()
            results_df.to_csv(results_path, index=False)
        
        # Save predictions
        if 'two_stage_approach' in self.results:
            pred_path = os.path.join(report_dir, "predictions.csv")
            pred_df = pd.DataFrame({
                'actual': self.results['two_stage_approach']['actual'],
                'predicted': self.results['two_stage_approach']['final_predictions']
            })
            pred_df.to_csv(pred_path, index=False)


def main():
    """Main function to run the sample pipeline"""
    try:
        # Create sample data if it doesn't exist
        if not os.path.exists("sample_data.csv"):
            print("Creating sample data...")
            from create_sample_data import create_sample_data
            create_sample_data()
        
        # Initialize predictor
        predictor = SampleDeliveryPredictor()
        
        # Run complete pipeline
        predictor.run_sample_pipeline()
        
        print("\nSample pipeline execution completed successfully!")
        
    except Exception as e:
        print(f"Error in sample pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
