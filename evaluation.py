"""
Model evaluation and prediction module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from config import *


class ModelEvaluator:
    """Handles model evaluation and visualization"""
    
    def __init__(self):
        self.results = {}
        self.predictions = {}
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mse = mean_squared_error(y_true, y_pred)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'MSE': mse
        }
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
        y_pred = model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        self.results[model_name] = metrics
        self.predictions[model_name] = y_pred
        
        print(f"\n{model_name} Evaluation:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics, y_pred
    
    def compare_models(self, results_dict):
        """Compare multiple models"""
        comparison_df = pd.DataFrame(results_dict).T
        comparison_df = comparison_df.sort_values('RMSE')
        
        print("\nModel Comparison (sorted by RMSE):")
        print(comparison_df.round(4))
        
        return comparison_df
    
    def plot_predictions(self, y_true, y_pred, model_name, save_path=None):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(12, 10))
        
        # Scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5, s=20)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Delivery Duration (seconds)')
        plt.ylabel('Predicted Delivery Duration (seconds)')
        plt.title(f'{model_name} - Actual vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # Residuals plot
        plt.subplot(2, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5, s=20)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Delivery Duration (seconds)')
        plt.ylabel('Residuals (seconds)')
        plt.title(f'{model_name} - Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        # Distribution of residuals
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Residuals (seconds)')
        plt.ylabel('Frequency')
        plt.title(f'{model_name} - Residuals Distribution')
        plt.grid(True, alpha=0.3)
        
        # Q-Q plot
        plt.subplot(2, 2, 4)
        try:
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title(f'{model_name} - Q-Q Plot')
            plt.grid(True, alpha=0.3)
        except:
            plt.text(0.5, 0.5, 'Q-Q Plot not available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{model_name} - Q-Q Plot (Not Available)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importances, top_n=20, save_path=None):
        """Plot feature importance"""
        if feature_importances is None:
            print("No feature importance data available")
            return
        
        # Get top N features
        top_features = feature_importances.head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, data, save_path=None):
        """Plot correlation matrix"""
        plt.figure(figsize=(15, 12))
        
        # Calculate correlation matrix
        corr = data.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Create heatmap
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
        
        plt.title('Feature Correlation Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, model_name, metrics, y_true, y_pred, save_path=None):
        """Generate comprehensive evaluation report"""
        report = f"""
# Model Evaluation Report: {model_name}

## Performance Metrics
- RMSE: {metrics['RMSE']:.4f}
- MAE: {metrics['MAE']:.4f}
- R²: {metrics['R2']:.4f}
- MAPE: {metrics['MAPE']:.2f}%

## Data Summary
- Number of test samples: {len(y_true)}
- Actual values range: {y_true.min():.2f} - {y_true.max():.2f}
- Predicted values range: {y_pred.min():.2f} - {y_pred.max():.2f}

## Error Analysis
- Mean absolute error: {metrics['MAE']:.2f} seconds
- Root mean square error: {metrics['RMSE']:.2f} seconds
- Mean absolute percentage error: {metrics['MAPE']:.2f}%

## Model Performance
- R² Score: {metrics['R2']:.4f} ({metrics['R2']*100:.2f}% of variance explained)
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report
    
    def evaluate_two_stage_approach(self, prep_predictions, driving_times, order_times, actual_times):
        """Evaluate the two-stage approach"""
        print("Evaluating two-stage approach...")
        
        # Calculate final predictions
        final_predictions = prep_predictions + driving_times + order_times
        
        # Calculate metrics
        metrics = self.calculate_metrics(actual_times, final_predictions)
        
        print("Two-stage approach results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics, final_predictions
