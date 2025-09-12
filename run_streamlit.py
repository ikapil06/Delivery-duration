#!/usr/bin/env python3
"""
Script to run the Streamlit dashboard for Delivery Duration Prediction
"""

import subprocess
import sys
import os

def run_streamlit():
    """Run the Streamlit application"""
    try:
        # Check if streamlit is installed
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_streamlit.txt"])
        print("✅ Streamlit installed successfully")
    
    # Check if required files exist
    required_files = ["streamlit_app.py", "outputs", "models"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files/directories: {missing_files}")
        print("💡 Please run 'python main.py' first to generate the required outputs")
        return
    
    print("🚀 Starting Streamlit dashboard...")
    print("📱 The dashboard will open in your default web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n" + "="*50)
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ])

if __name__ == "__main__":
    run_streamlit()
