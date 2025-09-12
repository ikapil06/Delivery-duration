import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import joblib
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Delivery Duration Prediction Dashboard",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_latest_results():
    """Load the latest results from outputs directory"""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return None, None, None
    
    # Find the latest report directory
    report_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("report_")]
    if not report_dirs:
        return None, None, None
    
    latest_dir = max(report_dirs, key=lambda x: x.name)
    
    # Load files
    comparison_report = None
    model_results = None
    feature_importances = None
    predictions = None
    
    try:
        if (latest_dir / "comparison_report.txt").exists():
            with open(latest_dir / "comparison_report.txt", 'r') as f:
                comparison_report = f.read()
        
        if (latest_dir / "model_results.csv").exists():
            model_results = pd.read_csv(latest_dir / "model_results.csv")
        
        if (latest_dir / "feature_importances.csv").exists():
            feature_importances = pd.read_csv(latest_dir / "feature_importances.csv")
        
        if (latest_dir / "predictions.csv").exists():
            predictions = pd.read_csv(latest_dir / "predictions.csv")
            
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None, None, None
    
    return latest_dir, {
        'comparison_report': comparison_report,
        'model_results': model_results,
        'feature_importances': feature_importances,
        'predictions': predictions
    }, latest_dir.name

def load_best_model():
    """Load the best model and its metadata"""
    models_dir = Path("models")
    if not models_dir.exists():
        return None, None
    
    # Find the best model files
    model_files = list(models_dir.glob("best_*.pkl"))
    metadata_files = list(models_dir.glob("best_*_metadata.json"))
    
    if not model_files or not metadata_files:
        return None, None
    
    try:
        # Load the first model (assuming there's only one best model)
        model_path = model_files[0]
        metadata_path = metadata_files[0]
        
        model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def create_model_comparison_chart(model_results):
    """Create interactive model comparison chart"""
    if model_results is None or model_results.empty:
        return None
    
    # Check column names and use appropriate ones
    if 'Test_RMSE' in model_results.columns:
        rmse_col = 'Test_RMSE'
        r2_col = 'Test_R2'
        mae_col = 'Test_MAE'
        mape_col = 'Test_MAPE' if 'Test_MAPE' in model_results.columns else None
    else:
        rmse_col = 'RMSE'
        r2_col = 'R²'
        mae_col = 'MAE'
        mape_col = 'MAPE' if 'MAPE' in model_results.columns else None
    
    # Create subplots (only for available metrics)
    available_metrics = []
    metric_cols = []
    metric_names = []
    
    if rmse_col in model_results.columns:
        available_metrics.append(rmse_col)
        metric_cols.append(rmse_col)
        metric_names.append('RMSE')
    
    if r2_col in model_results.columns:
        available_metrics.append(r2_col)
        metric_cols.append(r2_col)
        metric_names.append('R²')
    
    if mae_col in model_results.columns:
        available_metrics.append(mae_col)
        metric_cols.append(mae_col)
        metric_names.append('MAE')
    
    if mape_col and mape_col in model_results.columns:
        available_metrics.append(mape_col)
        metric_cols.append(mape_col)
        metric_names.append('MAPE')
    
    if not available_metrics:
        return None
    
    # Create subplots based on available metrics
    n_metrics = len(available_metrics)
    if n_metrics == 1:
        rows, cols = 1, 1
    elif n_metrics == 2:
        rows, cols = 1, 2
    elif n_metrics == 3:
        rows, cols = 2, 2
    else:
        rows, cols = 2, 2
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'{name} Comparison' for name in metric_names],
        specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (metric_col, metric_name) in enumerate(zip(metric_cols, metric_names)):
        if i >= 4:  # Limit to 4 subplots
            break
            
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        fig.add_trace(
            go.Bar(
                x=model_results['Model'],
                y=model_results[metric_col],
                name=metric_name,
                marker_color=colors[i % len(colors)],
                text=model_results[metric_col].round(4),
                textposition='auto'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Model Performance Comparison",
        title_x=0.5
    )
    
    return fig

def create_feature_importance_chart(feature_importances):
    """Create feature importance chart"""
    if feature_importances is None or feature_importances.empty:
        return None
    
    # Check column names and use appropriate ones
    if 'importance' in feature_importances.columns:
        importance_col = 'importance'
        feature_col = 'feature'
    else:
        importance_col = 'Importance'
        feature_col = 'Feature'
    
    fig = px.bar(
        feature_importances,
        x=importance_col,
        y=feature_col,
        orientation='h',
        title="Feature Importance Analysis",
        color=importance_col,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_prediction_scatter(predictions):
    """Create prediction scatter plot"""
    if predictions is None or predictions.empty:
        return None
    
    # Check if we have actual and predicted columns
    if 'actual' in predictions.columns and 'predicted' in predictions.columns:
        fig = px.scatter(
            predictions,
            x='actual',
            y='predicted',
            title="Actual vs Predicted Values",
            labels={'actual': 'Actual Delivery Duration', 'predicted': 'Predicted Delivery Duration'},
            opacity=0.6
        )
        
        # Add perfect prediction line
        min_val = min(predictions['actual'].min(), predictions['predicted'].min())
        max_val = max(predictions['actual'].max(), predictions['predicted'].max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            )
        )
        
        fig.update_layout(height=500)
        return fig
    
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🚚 Delivery Duration Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Overview", "📈 Model Performance", "🔍 Feature Analysis", "🎯 Predictions", "🤖 Model Details"]
    )
    
    # Load data
    with st.spinner("Loading latest results..."):
        latest_dir, results, report_name = load_latest_results()
        best_model, model_metadata = load_best_model()
    
    if latest_dir is None:
        st.error("❌ No results found! Please run the main pipeline first.")
        st.info("💡 Run `python main.py` to generate results.")
        return
    
    # Overview Page
    if page == "🏠 Overview":
        st.header("📋 Project Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Project Description
            This dashboard presents the results of a comprehensive **Delivery Duration Prediction** system 
            built using machine learning techniques. The system predicts delivery times for food delivery 
            services based on various factors like order details, store information, and dasher availability.
            """)
            
            st.markdown("""
            ### 🛠️ Technical Stack
            - **Data Processing**: Pandas, NumPy
            - **Machine Learning**: Scikit-learn, XGBoost, LightGBM
            - **Visualization**: Matplotlib, Seaborn, Plotly
            - **Model Persistence**: Joblib
            - **Web Interface**: Streamlit
            """)
        
        with col2:
            st.markdown("### 📊 Latest Results Summary")
            st.info(f"**Report Generated**: {report_name}")
            
            if results['model_results'] is not None:
                # Check column names and use appropriate ones
                if 'Test_RMSE' in results['model_results'].columns:
                    rmse_col = 'Test_RMSE'
                    r2_col = 'Test_R2'
                    mae_col = 'Test_MAE'
                else:
                    rmse_col = 'RMSE'
                    r2_col = 'R²'
                    mae_col = 'MAE'
                
                best_model_name = results['model_results'].loc[results['model_results'][rmse_col].idxmin(), 'Model']
                best_rmse = results['model_results'][rmse_col].min()
                best_r2 = results['model_results'][r2_col].max()
                
                st.metric("🏆 Best Model", best_model_name)
                st.metric("📉 Best RMSE", f"{best_rmse:.4f}")
                st.metric("📈 Best R²", f"{best_r2:.4f}")
        
        # Show comparison report
        if results['comparison_report']:
            st.header("📄 Detailed Comparison Report")
            st.text(results['comparison_report'])
    
    # Model Performance Page
    elif page == "📈 Model Performance":
        st.header("📈 Model Performance Analysis")
        
        if results['model_results'] is not None:
            # Model comparison chart
            fig = create_model_comparison_chart(results['model_results'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Model results table
            st.subheader("📊 Detailed Model Results")
            st.dataframe(results['model_results'], width='stretch')
            
            # Performance metrics summary
            # Check column names and use appropriate ones
            if 'Test_RMSE' in results['model_results'].columns:
                rmse_col = 'Test_RMSE'
                r2_col = 'Test_R2'
                mae_col = 'Test_MAE'
                mape_col = 'Test_MAPE' if 'Test_MAPE' in results['model_results'].columns else None
            else:
                rmse_col = 'RMSE'
                r2_col = 'R²'
                mae_col = 'MAE'
                mape_col = 'MAPE' if 'MAPE' in results['model_results'].columns else None
            
            # Create columns based on available metrics
            available_metrics = []
            if rmse_col in results['model_results'].columns:
                available_metrics.append(('🏆 Best RMSE', results['model_results'][rmse_col].min(), 'min'))
            if r2_col in results['model_results'].columns:
                available_metrics.append(('📈 Best R²', results['model_results'][r2_col].max(), 'max'))
            if mae_col in results['model_results'].columns:
                available_metrics.append(('📉 Best MAE', results['model_results'][mae_col].min(), 'min'))
            if mape_col and mape_col in results['model_results'].columns:
                available_metrics.append(('📊 Best MAPE', results['model_results'][mape_col].min(), 'min'))
            
            # Display metrics in columns
            cols = st.columns(len(available_metrics))
            for i, (title, value, _) in enumerate(available_metrics):
                with cols[i]:
                    st.metric(title, f"{value:.4f}")
        else:
            st.warning("No model results found!")
    
    # Feature Analysis Page
    elif page == "🔍 Feature Analysis":
        st.header("🔍 Feature Importance Analysis")
        
        if results['feature_importances'] is not None:
            # Feature importance chart
            fig = create_feature_importance_chart(results['feature_importances'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.subheader("📊 Feature Importance Details")
            st.dataframe(results['feature_importances'], width='stretch')
            
            # Top features summary
            top_features = results['feature_importances'].head(5)
            st.subheader("🏆 Top 5 Most Important Features")
            
            # Check column names and use appropriate ones
            if 'importance' in results['feature_importances'].columns:
                importance_col = 'importance'
                feature_col = 'feature'
            else:
                importance_col = 'Importance'
                feature_col = 'Feature'
            
            for idx, row in top_features.iterrows():
                st.write(f"**{idx+1}.** {row[feature_col]}: {row[importance_col]:.4f}")
        else:
            st.warning("No feature importance data found!")
    
    # Predictions Page
    elif page == "🎯 Predictions":
        st.header("🎯 Prediction Analysis")
        
        if results['predictions'] is not None:
            # Prediction scatter plot
            fig = create_prediction_scatter(results['predictions'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Predictions table
            st.subheader("📊 Sample Predictions")
            st.dataframe(results['predictions'].head(20), width='stretch')
            
            # Download predictions
            csv = results['predictions'].to_csv(index=False)
            st.download_button(
                label="📥 Download Full Predictions CSV",
                data=csv,
                file_name=f"predictions_{report_name}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No prediction data found!")
    
    # Model Details Page
    elif page == "🤖 Model Details":
        st.header("🤖 Best Model Information")
        
        if best_model is not None and model_metadata is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📋 Model Metadata")
                st.json(model_metadata)
            
            with col2:
                st.subheader("🔧 Model Information")
                st.write(f"**Model Type**: {model_metadata.get('model_type', 'Unknown')}")
                st.write(f"**Model Name**: {model_metadata.get('model_name', 'Unknown')}")
                st.write(f"**Saved At**: {model_metadata.get('timestamp', 'Unknown')}")
                
                if 'metrics' in model_metadata:
                    st.subheader("📊 Model Metrics")
                    metrics = model_metadata['metrics']
                    for metric, value in metrics.items():
                        st.write(f"**{metric}**: {value:.4f}")
            
            # Model download
            st.subheader("📥 Download Model")
            if os.path.exists(model_metadata.get('model_path', '')):
                with open(model_metadata['model_path'], 'rb') as f:
                    model_data = f.read()
                
                st.download_button(
                    label="📥 Download Best Model (.pkl)",
                    data=model_data,
                    file_name=f"best_model_{model_metadata.get('model_name', 'unknown')}.pkl",
                    mime="application/octet-stream"
                )
        else:
            st.warning("No model information found!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚚 Delivery Duration Prediction Dashboard | Generated with Streamlit</p>
        <p>Report: {report_name}</p>
    </div>
    """.format(report_name=report_name), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
