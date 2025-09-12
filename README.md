# Delivery Duration Prediction Project

A comprehensive machine learning project for predicting food delivery duration using historical data.

## Project Structure

```
delivery-duration-prediction/
├── main.py                 # Main orchestrator script
├── config.py              # Configuration parameters
├── data_preprocessing.py  # Data loading and cleaning
├── feature_engineering.py # Feature creation and selection
├── model_training.py      # Model training and evaluation
├── evaluation.py          # Model evaluation and visualization
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── historical_data.csv   # Input data
├── models/               # Saved models directory
└── outputs/              # Results and reports directory
```

## Features

- **Data Preprocessing**: Comprehensive data cleaning and preparation
- **Feature Engineering**: Advanced feature creation and selection
- **Multiple Models**: Support for various ML algorithms (LGBM, XGBoost, Random Forest, etc.)
- **Two-Stage Approach**: Innovative preparation time prediction + combination
- **Comprehensive Evaluation**: Detailed metrics and visualization
- **Modular Design**: Clean, maintainable code structure

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd delivery-duration-prediction
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv vnev
   source vnev/bin/activate  # On Windows: vnev\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

Run the complete pipeline:

```bash
python main.py
```

### Individual Components

You can also run individual components:

```python
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer

# Data preprocessing
preprocessor = DataPreprocessor()
data = preprocessor.preprocess()

# Feature engineering
feature_engineer = FeatureEngineer()
features = feature_engineer.engineer_features(data)

# Model training
trainer = ModelTrainer()
# ... training code
```

## Configuration

Modify `config.py` to adjust:
- Data paths
- Model parameters
- Feature selection criteria
- Evaluation metrics

## Models Supported

- **Linear Regression**
- **Ridge Regression**
- **Decision Tree**
- **Random Forest**
- **XGBoost**
- **LightGBM**
- **Multi-layer Perceptron (MLP)**

## Approaches

### 1. Direct Approach
Directly predicts total delivery duration using all features.

### 2. Two-Stage Approach (Recommended)
1. **Stage 1**: Predict preparation time using ML model
2. **Stage 2**: Combine with driving time and order place duration

## Results

The two-stage approach typically achieves:
- **RMSE**: ~983 seconds
- **R²**: ~0.85
- **MAE**: ~650 seconds

## Output

The pipeline generates:
- Model performance comparisons
- Feature importance analysis
- Prediction visualizations
- Comprehensive evaluation reports
- Saved models for deployment

## Key Features

### Data Preprocessing
- Missing value handling
- Datetime conversion
- Outlier detection
- Data type optimization

### Feature Engineering
- Categorical encoding
- Feature interaction creation
- Correlation analysis
- VIF-based feature selection
- Domain-specific features

### Model Evaluation
- Cross-validation
- Multiple metrics (RMSE, MAE, R², MAPE)
- Residual analysis
- Feature importance ranking

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- lightgbm
- statsmodels

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please open an issue in the repository.