"""
Flask Web Application for Customer Churn Prediction

This module provides a web interface for:
- Inputting customer data
- Receiving churn predictions
- Viewing feature importance
- Accessing model performance metrics
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class ChurnPredictor:
    """
    A class to handle churn predictions using trained models.
    """
    
    def __init__(self, models_dir='models', preprocessor_path=None):
        """
        Initialize the ChurnPredictor.
        
        Args:
            models_dir: Directory containing trained models
            preprocessor_path: Path to the preprocessor
        """
        self.models_dir = models_dir
        self.preprocessor_path = preprocessor_path
        self.models = {}
        self.preprocessor = None
        self.feature_names = None
        self.best_model = None
        self.load_models()
        
    def load_models(self):
        """
        Load trained models and preprocessor.
        """
        try:
            # Load preprocessor
            if self.preprocessor_path and os.path.exists(self.preprocessor_path):
                self.preprocessor = joblib.load(self.preprocessor_path)
            
            # Load models
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.joblib')]
            
            for model_file in model_files:
                model_name = model_file.replace('.joblib', '')
                model_path = os.path.join(self.models_dir, model_file)
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} from {model_path}")
            
            # Load model scores to determine best model
            scores_path = os.path.join(self.models_dir, 'model_scores.json')
            if os.path.exists(scores_path):
                with open(scores_path, 'r') as f:
                    model_scores = json.load(f)
                
                # Find best model based on CV F1 score
                best_model_name = max(model_scores.keys(), 
                                    key=lambda x: model_scores[x]['cv_f1'])
                self.best_model = self.models[best_model_name]
                print(f"Best model: {best_model_name}")
            
            # Load feature names
            feature_names_path = 'data/processed/feature_names.csv'
            if os.path.exists(feature_names_path):
                feature_names_df = pd.read_csv(feature_names_path)
                self.feature_names = feature_names_df['feature_name'].tolist()
                print(f"Loaded {len(self.feature_names)} feature names")
            
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def prepare_input_data(self, customer_data):
        """
        Prepare input data for prediction.
        
        Args:
            customer_data: Dictionary with customer features
            
        Returns:
            DataFrame: Prepared data
        """
        # Create DataFrame with all required features
        required_features = [
            'tenure', 'age', 'gender', 'contract_type', 'monthly_charges', 
            'total_charges', 'internet_service', 'online_security', 
            'online_backup', 'device_protection', 'tech_support', 
            'streaming_tv', 'streaming_movies', 'paperless_billing', 
            'payment_method'
        ]
        
        # Initialize with default values
        data = {}
        for feature in required_features:
            if feature in customer_data:
                data[feature] = [customer_data[feature]]
            else:
                # Set default values
                if feature in ['tenure', 'age', 'monthly_charges', 'total_charges']:
                    data[feature] = [0]
                else:
                    data[feature] = ['No']
        
        # Create engineered features
        df = pd.DataFrame(data)
        
        # Add engineered features
        df['avg_monthly_charges'] = df['total_charges'] / df['tenure'].replace(0, 1)
        
        df['tenure_category'] = pd.cut(
            df['tenure'], 
            bins=[0, 12, 24, 48, 72], 
            labels=['New', 'Short-term', 'Medium-term', 'Long-term']
        )
        
        df['age_category'] = pd.cut(
            df['age'], 
            bins=[0, 30, 50, 70, 100], 
            labels=['Young', 'Adult', 'Senior', 'Elderly']
        )
        
        df['monthly_charges_category'] = pd.cut(
            df['monthly_charges'], 
            bins=[0, 40, 70, 100, 150], 
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        service_columns = ['online_security', 'online_backup', 'device_protection', 
                          'tech_support', 'streaming_tv', 'streaming_movies']
        df['num_services'] = df[service_columns].apply(
            lambda x: (x == 'Yes').sum(), axis=1
        )
        
        df['has_internet'] = (df['internet_service'] != 'No').astype(int)
        df['is_month_to_month'] = (df['contract_type'] == 'Month-to-month').astype(int)
        
        return df
    
    def predict_churn(self, customer_data):
        """
        Predict churn probability for a customer.
        
        Args:
            customer_data: Dictionary with customer features
            
        Returns:
            dict: Prediction results
        """
        try:
            # Prepare input data
            input_df = self.prepare_input_data(customer_data)
            
            # Transform data using preprocessor
            if self.preprocessor:
                X_transformed = self.preprocessor.transform(input_df)
            else:
                # If no preprocessor, use raw features (basic approach)
                X_transformed = input_df.select_dtypes(include=[np.number]).values
            
            # Make prediction
            if self.best_model:
                prediction = self.best_model.predict(X_transformed)[0]
                prediction_proba = self.best_model.predict_proba(X_transformed)[0]
                
                churn_probability = prediction_proba[1]
                churn_status = "Yes" if prediction == 1 else "No"
                
                # Get feature importance if available
                feature_importance = None
                if hasattr(self.best_model, 'feature_importances_') and self.feature_names:
                    importance_scores = self.best_model.feature_importances_
                    feature_importance = dict(zip(self.feature_names, importance_scores))
                    feature_importance = dict(sorted(feature_importance.items(), 
                                                   key=lambda x: x[1], reverse=True))
                
                result = {
                    'churn_status': churn_status,
                    'churn_probability': float(churn_probability),
                    'no_churn_probability': float(prediction_proba[0]),
                    'confidence': float(max(prediction_proba)),
                    'feature_importance': feature_importance,
                    'model_used': 'best_model',
                    'timestamp': datetime.now().isoformat()
                }
                
                return result
            else:
                return {'error': 'No trained model available'}
                
        except Exception as e:
            return {'error': f'Prediction error: {str(e)}'}
    
    def get_model_performance(self):
        """
        Get model performance metrics.
        
        Returns:
            dict: Model performance information
        """
        try:
            scores_path = os.path.join(self.models_dir, 'model_scores.json')
            if os.path.exists(scores_path):
                with open(scores_path, 'r') as f:
                    model_scores = json.load(f)
                return model_scores
            else:
                return {'error': 'Model scores not available'}
        except Exception as e:
            return {'error': f'Error loading model performance: {str(e)}'}

# Initialize predictor
predictor = ChurnPredictor()

@app.route('/')
def index():
    """
    Main page with prediction form.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    """
    try:
        # Get form data
        customer_data = {
            'tenure': int(request.form.get('tenure', 0)),
            'age': int(request.form.get('age', 0)),
            'gender': request.form.get('gender', 'Male'),
            'contract_type': request.form.get('contract_type', 'Month-to-month'),
            'monthly_charges': float(request.form.get('monthly_charges', 0)),
            'total_charges': float(request.form.get('total_charges', 0)),
            'internet_service': request.form.get('internet_service', 'No'),
            'online_security': request.form.get('online_security', 'No'),
            'online_backup': request.form.get('online_backup', 'No'),
            'device_protection': request.form.get('device_protection', 'No'),
            'tech_support': request.form.get('tech_support', 'No'),
            'streaming_tv': request.form.get('streaming_tv', 'No'),
            'streaming_movies': request.form.get('streaming_movies', 'No'),
            'paperless_billing': request.form.get('paperless_billing', 'No'),
            'payment_method': request.form.get('payment_method', 'Electronic check')
        }
        
        # Make prediction
        result = predictor.predict_churn(customer_data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'})

@app.route('/performance')
def performance():
    """
    Display model performance metrics.
    """
    performance_data = predictor.get_model_performance()
    return render_template('performance.html', performance=performance_data)

@app.route('/api/performance')
def api_performance():
    """
    API endpoint for model performance.
    """
    return jsonify(predictor.get_model_performance())

@app.route('/about')
def about():
    """
    About page with project information.
    """
    return render_template('about.html')

if __name__ == '__main__':
    print("Starting Customer Churn Prediction Web Application...")
    print("Make sure you have trained models in the 'models' directory")
    print("Access the application at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 