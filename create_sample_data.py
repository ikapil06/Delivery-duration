"""
Create sample data for testing with 50 rows
"""

import pandas as pd
import numpy as np
from config import *

def create_sample_data(input_file="historical_data.csv", output_file="sample_data.csv", n_samples=50):
    """Create a sample dataset with specified number of rows"""
    
    print(f"Creating sample data with {n_samples} rows...")
    
    # Load the full dataset
    df = pd.read_csv(input_file)
    
    # Take a random sample
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=RANDOM_STATE)
    
    # Reset index
    sample_df = sample_df.reset_index(drop=True)
    
    # Save sample data
    sample_df.to_csv(output_file, index=False)
    
    print(f"Sample data created: {sample_df.shape[0]} rows, {sample_df.shape[1]} columns")
    print(f"Saved to: {output_file}")
    
    return sample_df

if __name__ == "__main__":
    create_sample_data()
