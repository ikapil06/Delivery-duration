# 🚚 Delivery Duration Prediction - Streamlit Dashboard

A comprehensive web dashboard for visualizing and analyzing delivery duration prediction results.

## 🎯 Features

### 📊 Interactive Dashboard Pages

1. **🏠 Overview**
   - Project description and technical stack
   - Latest results summary with key metrics
   - Complete comparison report

2. **📈 Model Performance**
   - Interactive model comparison charts (RMSE, R², MAE, MAPE)
   - Detailed model results table
   - Performance metrics summary

3. **🔍 Feature Analysis**
   - Feature importance visualization
   - Top 5 most important features
   - Detailed feature importance table

4. **🎯 Predictions**
   - Actual vs Predicted scatter plot
   - Sample predictions table
   - Download full predictions CSV

5. **🤖 Model Details**
   - Best model metadata and information
   - Model performance metrics
   - Download best model (.pkl file)

### 🎨 Visualizations

- **Interactive Charts**: Built with Plotly for zoom, pan, and hover interactions
- **Responsive Design**: Adapts to different screen sizes
- **Professional Styling**: Custom CSS for a polished look
- **Real-time Updates**: Automatically loads latest results

## 🚀 Quick Start

### Method 1: Using the Run Script (Recommended)
```bash
python run_streamlit.py
```

### Method 2: Direct Streamlit Command
```bash
# Install requirements
pip install -r requirements_streamlit.txt

# Run the app
streamlit run streamlit_app.py
```

### Method 3: Manual Installation
```bash
# Install Streamlit
pip install streamlit

# Install other dependencies
pip install pandas numpy matplotlib seaborn plotly scikit-learn joblib

# Run the app
streamlit run streamlit_app.py
```

## 📋 Prerequisites

Before running the Streamlit dashboard, ensure you have:

1. **Generated Results**: Run `python main.py` to create the outputs
2. **Required Files**:
   - `outputs/` directory with latest report
   - `models/` directory with best model
   - `streamlit_app.py` (the main dashboard file)

## 🎛️ Dashboard Navigation

### Sidebar Menu
- **🏠 Overview**: Project summary and latest results
- **📈 Model Performance**: Model comparison and metrics
- **🔍 Feature Analysis**: Feature importance analysis
- **🎯 Predictions**: Prediction analysis and downloads
- **🤖 Model Details**: Best model information and download

### Interactive Features
- **Hover Tooltips**: Detailed information on hover
- **Zoom & Pan**: Interactive chart navigation
- **Download Buttons**: Export data and models
- **Responsive Layout**: Works on desktop and mobile

## 📊 Data Sources

The dashboard automatically loads data from:

- **Model Results**: `outputs/report_*/model_results.csv`
- **Feature Importances**: `outputs/report_*/feature_importances.csv`
- **Predictions**: `outputs/report_*/predictions.csv`
- **Comparison Report**: `outputs/report_*/comparison_report.txt`
- **Best Model**: `models/best_*.pkl` and `models/best_*_metadata.json`

## 🎨 Customization

### Styling
The dashboard uses custom CSS for styling. You can modify the styles in the `streamlit_app.py` file:

```python
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)
```

### Adding New Pages
To add new pages, modify the sidebar selectbox and add corresponding page logic:

```python
page = st.sidebar.selectbox(
    "Select Page",
    ["🏠 Overview", "📈 Model Performance", "🔍 Feature Analysis", "🎯 Predictions", "🤖 Model Details", "🆕 New Page"]
)

# Add new page logic
elif page == "🆕 New Page":
    st.header("New Page Title")
    # Your page content here
```

## 🔧 Troubleshooting

### Common Issues

1. **"No results found" Error**
   - Solution: Run `python main.py` first to generate outputs

2. **Missing Dependencies**
   - Solution: Install requirements with `pip install -r requirements_streamlit.txt`

3. **Port Already in Use**
   - Solution: Change port in `run_streamlit.py` or kill existing Streamlit processes

4. **Charts Not Displaying**
   - Solution: Check if Plotly is installed: `pip install plotly`

### Performance Tips

- **Large Datasets**: The dashboard loads data on each page visit. For very large datasets, consider implementing caching
- **Memory Usage**: Close unused browser tabs to free up memory
- **Network**: The dashboard runs locally, so no network issues should occur

## 📱 Browser Compatibility

- **Chrome**: ✅ Fully supported
- **Firefox**: ✅ Fully supported  
- **Safari**: ✅ Fully supported
- **Edge**: ✅ Fully supported
- **Mobile**: ✅ Responsive design works on mobile devices

## 🚀 Deployment Options

### Local Development
```bash
streamlit run streamlit_app.py
```

### Production Deployment
For production deployment, consider:
- **Streamlit Cloud**: Free hosting for Streamlit apps
- **Heroku**: Deploy with Procfile
- **Docker**: Containerize the application
- **AWS/GCP/Azure**: Cloud deployment options

## 📞 Support

If you encounter any issues:

1. Check the console output for error messages
2. Ensure all required files are present
3. Verify Python and package versions
4. Check the Streamlit documentation: https://docs.streamlit.io/

## 🎉 Enjoy Your Dashboard!

The Streamlit dashboard provides a comprehensive view of your delivery duration prediction results with interactive visualizations and easy data access. Happy analyzing! 🚚📊
