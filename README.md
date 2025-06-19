# Customer Churn Prediction

## Project Overview
This project implements a machine learning application to predict customer churn based on various features including demographic data, purchase history, and customer interaction metrics. The application provides comprehensive data analysis, model training, evaluation, and visualization capabilities.

## Project Structure
```
Customer Churn Prediction/
├── data/                           # Data files
│   ├── raw/                       # Raw dataset
│   └── processed/                 # Processed data
├── models/                        # Trained models
├── notebooks/                     # Jupyter notebooks
├── src/                          # Source code
│   ├── data_preprocessing.py     # Data preprocessing functions
│   ├── model_training.py         # Model training and evaluation
│   ├── visualization.py          # Data visualization functions
│   ├── feature_analysis.py       # Feature importance and SHAP analysis
│   └── web_app.py                # Flask web application
├── static/                       # Static files for web app
├── templates/                    # HTML templates
├── results/                      # Generated visualizations and results
├── requirements.txt              # Python dependencies
├── main.py                       # Main execution script
└── README.md                     # Project documentation
```

## Dataset
The project uses the "Customer Churn" dataset which includes the following features:
- **Customer ID**: Unique identifier for each customer
- **Tenure**: Number of months the customer has been with the company
- **Age**: Customer's age
- **Gender**: Customer's gender (Male/Female)
- **Contract Type**: Type of contract (Month-to-month, One year, Two year)
- **Monthly Charges**: Monthly billing amount
- **Total Charges**: Total amount charged to the customer
- **Churn Status**: Target variable (Yes/No)

## Model Selection
The following machine learning models are implemented and compared:
1. **Logistic Regression**: Baseline model for binary classification
2. **Random Forest Classifier**: Ensemble method with good interpretability
3. **Support Vector Machine (SVM)**: Effective for high-dimensional data
4. **XGBoost**: Gradient boosting algorithm with high performance

## Key Features
- **Data Preprocessing**: Handles missing data, feature scaling, and categorical encoding
- **Model Training**: Implements multiple algorithms with hyperparameter optimization
- **Model Evaluation**: Comprehensive evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
- **Feature Analysis**: SHAP analysis for model interpretability
- **Visualization**: Interactive plots and charts for data exploration
- **Web Interface**: Flask-based web application for predictions

## Installation and Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Customer-Churn-Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main application:
```bash
python main.py
```

4. For the web interface:
```bash
python src/web_app.py
```

## Usage

### Data Analysis
The application automatically downloads and analyzes the dataset, providing insights into:
- Missing data patterns
- Feature distributions
- Correlation analysis
- Churn rate analysis

### Model Training
Models are trained with the following process:
1. Data preprocessing and feature engineering
2. Train-test split (80/20)
3. Hyperparameter optimization using GridSearchCV
4. Model evaluation and comparison

### Model Evaluation
Each model is evaluated using:
- Accuracy score
- Precision, Recall, and F1-score
- Confusion matrix
- ROC curve and AUC score

### Feature Importance
SHAP analysis provides insights into:
- Feature importance rankings
- Individual prediction explanations
- Feature interaction effects

## Key Findings

### Data Insights
- **Churn Rate**: Approximately 26.5% of customers churn
- **Key Correlates**: Contract type, monthly charges, and tenure show strong correlation with churn
- **Missing Data**: Minimal missing values, handled through appropriate imputation

### Model Performance
- **Best Model**: XGBoost achieves the highest performance with ~85% accuracy
- **Feature Importance**: Contract type, monthly charges, and tenure are the most important predictors
- **Model Interpretability**: SHAP analysis reveals non-linear relationships between features and churn

### Business Recommendations
1. **Contract Optimization**: Focus on converting month-to-month customers to longer contracts
2. **Pricing Strategy**: Monitor monthly charges and consider loyalty discounts
3. **Retention Programs**: Target customers with shorter tenure for retention initiatives
4. **Personalized Offers**: Use model predictions to identify high-risk customers

## Web Application
The Flask web application allows users to:
- Input customer data through a user-friendly interface
- Receive instant churn predictions
- View feature importance for individual predictions
- Access model performance metrics

## Future Enhancements
- Real-time data integration
- Advanced feature engineering
- Model retraining pipeline
- A/B testing framework
- Customer segmentation analysis

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

## License
This project is licensed under the MIT License. 