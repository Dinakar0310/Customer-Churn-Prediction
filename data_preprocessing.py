"""
Data Preprocessing Module for Customer Churn Prediction

This module handles all data preprocessing tasks including:
- Data loading and exploration
- Missing data handling
- Feature engineering
- Categorical encoding
- Feature scaling
- Train-test splitting
"""

import pandas as pd
import numpy as np
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    A comprehensive data preprocessing class for customer churn prediction.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the DataPreprocessor.
        
        Args:
            data_path (str): Path to the data file. If None, will download from Kaggle.
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.feature_names = None
        
    def download_dataset(self):
        """
        Download the customer churn dataset from Kaggle or create synthetic data.
        """
        # Create synthetic customer churn data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic data
        data = {
            'customer_id': range(1, n_samples + 1),
            'tenure': np.random.randint(1, 73, n_samples),
            'age': np.random.randint(18, 85, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
            'monthly_charges': np.random.uniform(20, 120, n_samples),
            'total_charges': np.random.uniform(100, 8000, n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
            'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
            'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.265, 0.735])  # 26.5% churn rate
        }
        
        # Create realistic churn patterns based on features
        for i in range(n_samples):
            # Higher churn for month-to-month contracts
            if data['contract_type'][i] == 'Month-to-month':
                data['churn'][i] = np.random.choice(['Yes', 'No'], p=[0.4, 0.6])
            
            # Higher churn for higher monthly charges
            if data['monthly_charges'][i] > 80:
                data['churn'][i] = np.random.choice(['Yes', 'No'], p=[0.35, 0.65])
            
            # Higher churn for shorter tenure
            if data['tenure'][i] < 12:
                data['churn'][i] = np.random.choice(['Yes', 'No'], p=[0.4, 0.6])
        
        self.raw_data = pd.DataFrame(data)
        
        # Save to data/raw directory
        os.makedirs('data/raw', exist_ok=True)
        self.raw_data.to_csv('data/raw/customer_churn_data.csv', index=False)
        print("Dataset created and saved to data/raw/customer_churn_data.csv")
        
        return self.raw_data
    
    def load_data(self):
        """
        Load the dataset from file or download if not available.
        """
        if self.data_path and os.path.exists(self.data_path):
            self.raw_data = pd.read_csv(self.data_path)
        else:
            self.raw_data = self.download_dataset()
        
        print(f"Dataset loaded with shape: {self.raw_data.shape}")
        return self.raw_data
    
    def explore_data(self):
        """
        Perform initial data exploration and analysis.
        """
        print("=" * 50)
        print("DATA EXPLORATION")
        print("=" * 50)
        
        print(f"\nDataset Shape: {self.raw_data.shape}")
        print(f"Number of features: {self.raw_data.shape[1]}")
        print(f"Number of samples: {self.raw_data.shape[0]}")
        
        print("\n" + "=" * 30)
        print("MISSING DATA ANALYSIS")
        print("=" * 30)
        missing_data = self.raw_data.isnull().sum()
        missing_percentage = (missing_data / len(self.raw_data)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing_data,
            'Percentage': missing_percentage
        })
        print(missing_df[missing_df['Missing Values'] > 0])
        
        print("\n" + "=" * 30)
        print("TARGET VARIABLE ANALYSIS")
        print("=" * 30)
        churn_counts = self.raw_data['churn'].value_counts()
        churn_percentage = (churn_counts / len(self.raw_data)) * 100
        print("Churn Distribution:")
        for churn_status, count in churn_counts.items():
            percentage = churn_percentage[churn_status]
            print(f"  {churn_status}: {count} ({percentage:.1f}%)")
        
        print("\n" + "=" * 30)
        print("FEATURE TYPES")
        print("=" * 30)
        print(self.raw_data.dtypes)
        
        print("\n" + "=" * 30)
        print("NUMERICAL FEATURES SUMMARY")
        print("=" * 30)
        numerical_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        print(self.raw_data[numerical_cols].describe())
        
        print("\n" + "=" * 30)
        print("CATEGORICAL FEATURES SUMMARY")
        print("=" * 30)
        categorical_cols = self.raw_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'churn':
                print(f"\n{col}:")
                print(self.raw_data[col].value_counts())
        
        return {
            'missing_data': missing_df,
            'churn_distribution': churn_counts,
            'numerical_features': numerical_cols.tolist(),
            'categorical_features': categorical_cols.tolist()
        }
    
    def handle_missing_data(self):
        """
        Handle missing data using appropriate imputation strategies.
        """
        print("\nHandling missing data...")
        
        # Check for missing values
        missing_data = self.raw_data.isnull().sum()
        if missing_data.sum() == 0:
            print("No missing data found.")
            return self.raw_data
        
        # Create a copy for processing
        data_cleaned = self.raw_data.copy()
        
        # Handle missing values in numerical columns
        numerical_cols = data_cleaned.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if data_cleaned[col].isnull().sum() > 0:
                # Use median for numerical columns
                median_val = data_cleaned[col].median()
                data_cleaned[col].fillna(median_val, inplace=True)
                print(f"Filled missing values in {col} with median: {median_val:.2f}")
        
        # Handle missing values in categorical columns
        categorical_cols = data_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data_cleaned[col].isnull().sum() > 0:
                # Use mode for categorical columns
                mode_val = data_cleaned[col].mode()[0]
                data_cleaned[col].fillna(mode_val, inplace=True)
                print(f"Filled missing values in {col} with mode: {mode_val}")
        
        self.raw_data = data_cleaned
        return self.raw_data
    
    def feature_engineering(self):
        """
        Create new features and perform feature engineering.
        """
        print("\nPerforming feature engineering...")
        
        data_engineered = self.raw_data.copy()
        
        # Create new features
        # 1. Average monthly charges per tenure
        data_engineered['avg_monthly_charges'] = data_engineered['total_charges'] / data_engineered['tenure']
        
        # 2. Tenure categories
        data_engineered['tenure_category'] = pd.cut(
            data_engineered['tenure'], 
            bins=[0, 12, 24, 48, 72], 
            labels=['New', 'Short-term', 'Medium-term', 'Long-term']
        )
        
        # 3. Age categories
        data_engineered['age_category'] = pd.cut(
            data_engineered['age'], 
            bins=[0, 30, 50, 70, 100], 
            labels=['Young', 'Adult', 'Senior', 'Elderly']
        )
        
        # 4. Monthly charges categories
        data_engineered['monthly_charges_category'] = pd.cut(
            data_engineered['monthly_charges'], 
            bins=[0, 40, 70, 100, 150], 
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # 5. Number of services (count of Yes responses for service features)
        service_columns = ['online_security', 'online_backup', 'device_protection', 
                          'tech_support', 'streaming_tv', 'streaming_movies']
        data_engineered['num_services'] = data_engineered[service_columns].apply(
            lambda x: (x == 'Yes').sum(), axis=1
        )
        
        # 6. Has internet service (binary)
        data_engineered['has_internet'] = (data_engineered['internet_service'] != 'No').astype(int)
        
        # 7. Contract type binary (month-to-month vs others)
        data_engineered['is_month_to_month'] = (data_engineered['contract_type'] == 'Month-to-month').astype(int)
        
        print("Feature engineering completed. New features created:")
        new_features = ['avg_monthly_charges', 'tenure_category', 'age_category', 
                       'monthly_charges_category', 'num_services', 'has_internet', 'is_month_to_month']
        for feature in new_features:
            print(f"  - {feature}")
        
        self.raw_data = data_engineered
        return self.raw_data
    
    def prepare_features(self):
        """
        Prepare features for model training by separating features and target.
        """
        print("\nPreparing features for model training...")
        
        # Remove customer_id as it's not useful for prediction
        features_data = self.raw_data.drop(['customer_id', 'churn'], axis=1)
        target_data = self.raw_data['churn']
        
        # Separate numerical and categorical features
        numerical_features = features_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = features_data.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numerical features: {len(numerical_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        # Fit and transform the data
        X_processed = self.preprocessor.fit_transform(features_data)
        
        # Get feature names after preprocessing
        numerical_feature_names = numerical_features
        categorical_feature_names = []
        
        if categorical_features:
            categorical_encoder = self.preprocessor.named_transformers_['cat']['onehot']
            categorical_feature_names = categorical_encoder.get_feature_names_out(categorical_features).tolist()
        
        self.feature_names = numerical_feature_names + categorical_feature_names
        
        # Convert target to binary
        label_encoder = LabelEncoder()
        y_processed = label_encoder.fit_transform(target_data)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")
        print(f"Number of features after preprocessing: {len(self.feature_names)}")
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        np.save('data/processed/X_train.npy', self.X_train)
        np.save('data/processed/X_test.npy', self.X_test)
        np.save('data/processed/y_train.npy', self.y_train)
        np.save('data/processed/y_test.npy', self.y_test)
        
        # Save feature names
        feature_names_df = pd.DataFrame({'feature_name': self.feature_names})
        feature_names_df.to_csv('data/processed/feature_names.csv', index=False)
        
        print("Processed data saved to data/processed/")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_feature_importance_data(self):
        """
        Get the original data for feature importance analysis.
        """
        return self.raw_data
    
    def get_preprocessor(self):
        """
        Get the fitted preprocessor for transforming new data.
        """
        return self.preprocessor
    
    def get_feature_names(self):
        """
        Get the feature names after preprocessing.
        """
        return self.feature_names
    
    def transform_new_data(self, new_data):
        """
        Transform new data using the fitted preprocessor.
        
        Args:
            new_data (pd.DataFrame): New data to transform
            
        Returns:
            np.array: Transformed features
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call prepare_features() first.")
        
        return self.preprocessor.transform(new_data)

def main():
    """
    Main function to demonstrate data preprocessing pipeline.
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    preprocessor.load_data()
    
    # Explore data
    preprocessor.explore_data()
    
    # Handle missing data
    preprocessor.handle_missing_data()
    
    # Feature engineering
    preprocessor.feature_engineering()
    
    # Prepare features for modeling
    X_train, X_test, y_train, y_test = preprocessor.prepare_features()
    
    print("\n" + "=" * 50)
    print("DATA PREPROCESSING COMPLETED")
    print("=" * 50)
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Features: {len(preprocessor.get_feature_names())}")
    
    return preprocessor, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()