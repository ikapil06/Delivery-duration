"""
Example script showing how to use the delivery duration prediction system
"""

import os
import sys
from main import DeliveryDurationPredictor

def run_example():
    """Run a complete example of the prediction system"""
    
    print("🚀 Starting Delivery Duration Prediction Example")
    print("=" * 60)
    
    # Check if data file exists
    if not os.path.exists("historical_data.csv"):
        print("❌ Error: historical_data.csv not found!")
        print("Please ensure the data file is in the current directory.")
        return
    
    try:
        # Initialize the predictor
        predictor = DeliveryDurationPredictor()
        
        # Run the complete pipeline
        predictor.run_complete_pipeline()
        
        print("\n✅ Example completed successfully!")
        print("\n📊 Check the 'outputs' directory for results and reports.")
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        print("\nPlease check the error message and try again.")

if __name__ == "__main__":
    run_example()
