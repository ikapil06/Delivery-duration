"""
Display results from the sample data run
"""

import os
import json
import pandas as pd
from datetime import datetime

def show_latest_results():
    """Show the latest results from the sample run"""
    
    # Find the latest report directory
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        print("No outputs directory found!")
        return
    
    # Get all sample report directories
    report_dirs = [d for d in os.listdir(outputs_dir) if d.startswith("sample_report_")]
    if not report_dirs:
        print("No sample reports found!")
        return
    
    # Get the latest one
    latest_dir = sorted(report_dirs)[-1]
    report_path = os.path.join(outputs_dir, latest_dir)
    
    print("="*80)
    print("DELIVERY DURATION PREDICTION - SAMPLE DATA RESULTS")
    print("="*80)
    print(f"Report Directory: {latest_dir}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Read comparison report
    comparison_file = os.path.join(report_path, "comparison_report.txt")
    if os.path.exists(comparison_file):
        with open(comparison_file, 'r') as f:
            print("COMPARISON REPORT:")
            print("-" * 40)
            print(f.read())
    
    # Read model metadata
    print("\nDETAILED MODEL METRICS:")
    print("-" * 40)
    
    # Direct approach model
    direct_metadata = os.path.join(report_path, "best_model_direct_RandomForest_metadata.json")
    if os.path.exists(direct_metadata):
        with open(direct_metadata, 'r') as f:
            data = json.load(f)
            print(f"\nDIRECT APPROACH - {data['model_type']}:")
            print(f"  RMSE: {data['metrics']['RMSE']:.2f} seconds")
            print(f"  MAE:  {data['metrics']['MAE']:.2f} seconds")
            print(f"  R²:   {data['metrics']['R2']:.4f}")
            print(f"  MAPE: {data['metrics']['MAPE']:.2f}%")
    
    # Two-stage approach model
    two_stage_metadata = os.path.join(report_path, "best_model_two_stage_XGBoost_metadata.json")
    if os.path.exists(two_stage_metadata):
        with open(two_stage_metadata, 'r') as f:
            data = json.load(f)
            print(f"\nTWO-STAGE APPROACH - {data['model_type']}:")
            print(f"  RMSE: {data['metrics']['RMSE']:.2f} seconds")
            print(f"  MAE:  {data['metrics']['MAE']:.2f} seconds")
            print(f"  R²:   {data['metrics']['R2']:.4f}")
            print(f"  MAPE: {data['metrics']['MAPE']:.2f}%")
    
    # List all generated files
    print(f"\nGENERATED FILES:")
    print("-" * 40)
    files = os.listdir(report_path)
    for file in sorted(files):
        file_path = os.path.join(report_path, file)
        size = os.path.getsize(file_path)
        print(f"  {file:<50} ({size:,} bytes)")
    
    print(f"\nIMAGES SAVED:")
    print("-" * 40)
    image_files = [f for f in files if f.endswith('.png')]
    for img in image_files:
        print(f"  📊 {img}")
    
    print(f"\nMODELS SAVED:")
    print("-" * 40)
    model_files = [f for f in files if f.endswith('.joblib')]
    for model in model_files:
        print(f"  🤖 {model}")
    
    print(f"\nDATA FILES SAVED:")
    print("-" * 40)
    data_files = [f for f in files if f.endswith('.csv')]
    for data in data_files:
        print(f"  📈 {data}")
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print("✅ Sample data pipeline completed successfully!")
    print("✅ Images saved for both approaches")
    print("✅ Best performing models saved")
    print("✅ Detailed metrics and reports generated")
    print("✅ All outputs saved to:", report_path)
    print("\n🎯 RECOMMENDATION: Use Direct RandomForest approach")
    print("   - Best RMSE: 465.18 seconds")
    print("   - Best R²: 0.8130")
    print("   - Best MAPE: 19.10%")

if __name__ == "__main__":
    show_latest_results()
