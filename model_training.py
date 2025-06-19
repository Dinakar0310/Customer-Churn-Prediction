"""
Model Training Module for Customer Churn Prediction

This module implements multiple machine learning models including:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- XGBoost

Features include:
- Hyperparameter optimization using GridSearchCV
- Comprehensive model evaluation
- Model comparison and selection
- Model persistence
"""

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    A comprehensive model training class for customer churn prediction.
    """
    
    def __init__(self, X_train, X_test, y_train, y_test, feature_names=None):
        """
        Initialize the ModelTrainer.
        
        Args:
            X_train, X_test, y_train, y_test: Training and testing data
            feature_names: List of feature names
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.models = {}
        self.best_models = {}
        self.model_scores = {}
        self.cv_scores = {}
        
    def train_logistic_regression(self):
        """
        Train Logistic Regression model with hyperparameter optimization.
        """
        print("Training Logistic Regression...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        }
        
        # Initialize model
        lr = LogisticRegression(random_state=42)
        
        # Grid search
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0
        )
        grid_search.fit(self.X_train, self.y_train)
        
        # Get best model
        best_lr = grid_search.best_estimator_
        self.best_models['logistic_regression'] = best_lr
        
        # Evaluate
        train_score = best_lr.score(self.X_train, self.y_train)
        test_score = best_lr.score(self.X_test, self.y_test)
        cv_score = grid_search.best_score_
        
        self.model_scores['logistic_regression'] = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_f1': cv_score,
            'best_params': grid_search.best_params_
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"CV F1 Score: {cv_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        
        return best_lr
    
    def train_random_forest(self):
        """
        Train Random Forest model with hyperparameter optimization.
        """
        print("Training Random Forest...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Initialize model
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0
        )
        grid_search.fit(self.X_train, self.y_train)
        
        # Get best model
        best_rf = grid_search.best_estimator_
        self.best_models['random_forest'] = best_rf
        
        # Evaluate
        train_score = best_rf.score(self.X_train, self.y_train)
        test_score = best_rf.score(self.X_test, self.y_test)
        cv_score = grid_search.best_score_
        
        self.model_scores['random_forest'] = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_f1': cv_score,
            'best_params': grid_search.best_params_
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"CV F1 Score: {cv_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        
        return best_rf
    
    def train_svm(self):
        """
        Train Support Vector Machine model with hyperparameter optimization.
        """
        print("Training Support Vector Machine...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
        
        # Initialize model
        svm = SVC(random_state=42, probability=True)
        
        # Grid search
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0
        )
        grid_search.fit(self.X_train, self.y_train)
        
        # Get best model
        best_svm = grid_search.best_estimator_
        self.best_models['svm'] = best_svm
        
        # Evaluate
        train_score = best_svm.score(self.X_train, self.y_train)
        test_score = best_svm.score(self.X_test, self.y_test)
        cv_score = grid_search.best_score_
        
        self.model_scores['svm'] = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_f1': cv_score,
            'best_params': grid_search.best_params_
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"CV F1 Score: {cv_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        
        return best_svm
    
    def train_xgboost(self):
        """
        Train XGBoost model with hyperparameter optimization.
        """
        print("Training XGBoost...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Initialize model
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        # Grid search
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0
        )
        grid_search.fit(self.X_train, self.y_train)
        
        # Get best model
        best_xgb = grid_search.best_estimator_
        self.best_models['xgboost'] = best_xgb
        
        # Evaluate
        train_score = best_xgb.score(self.X_train, self.y_train)
        test_score = best_xgb.score(self.X_test, self.y_test)
        cv_score = grid_search.best_score_
        
        self.model_scores['xgboost'] = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_f1': cv_score,
            'best_params': grid_search.best_params_
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"CV F1 Score: {cv_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        
        return best_xgb
    
    def train_all_models(self):
        """
        Train all models and return results.
        """
        print("=" * 50)
        print("TRAINING ALL MODELS")
        print("=" * 50)
        
        # Train each model
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_svm()
        self.train_xgboost()
        
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED")
        print("=" * 50)
        
        return self.best_models
    
    def evaluate_model(self, model, model_name):
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model: Trained model
            model_name: Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Classification report
        report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        
        evaluation_results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_curve': {'fpr': fpr, 'tpr': tpr},
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return evaluation_results
    
    def evaluate_all_models(self):
        """
        Evaluate all trained models and return comprehensive results.
        """
        print("\n" + "=" * 50)
        print("EVALUATING ALL MODELS")
        print("=" * 50)
        
        evaluation_results = {}
        
        for model_name, model in self.best_models.items():
            print(f"\nEvaluating {model_name}...")
            results = self.evaluate_model(model, model_name)
            evaluation_results[model_name] = results
            
            # Print key metrics
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"F1 Score: {results['f1_score']:.4f}")
            print(f"AUC: {results['auc']:.4f}")
        
        return evaluation_results
    
    def compare_models(self, evaluation_results):
        """
        Compare all models and identify the best performer.
        
        Args:
            evaluation_results: Results from evaluate_all_models()
            
        Returns:
            dict: Comparison summary
        """
        print("\n" + "=" * 50)
        print("MODEL COMPARISON")
        print("=" * 50)
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, results in evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1_score'],
                'AUC': results['auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
        
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Identify best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = self.best_models[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best F1 Score: {comparison_df.iloc[0]['F1 Score']:.4f}")
        print(f"Best AUC: {comparison_df.iloc[0]['AUC']:.4f}")
        
        comparison_summary = {
            'comparison_df': comparison_df,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'best_metrics': comparison_df.iloc[0].to_dict()
        }
        
        return comparison_summary
    
    def save_models(self, save_dir='models'):
        """
        Save all trained models to disk.
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nSaving models to {save_dir}/...")
        
        for model_name, model in self.best_models.items():
            model_path = os.path.join(save_dir, f'{model_name}.joblib')
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")
        
        # Save model scores
        scores_path = os.path.join(save_dir, 'model_scores.json')
        import json
        with open(scores_path, 'w') as f:
            json.dump(self.model_scores, f, indent=4, default=str)
        
        print(f"Saved model scores to {scores_path}")
    
    def load_models(self, load_dir='models'):
        """
        Load trained models from disk.
        
        Args:
            load_dir: Directory containing saved models
        """
        print(f"Loading models from {load_dir}/...")
        
        model_files = [f for f in os.listdir(load_dir) if f.endswith('.joblib')]
        
        for model_file in model_files:
            model_name = model_file.replace('.joblib', '')
            model_path = os.path.join(load_dir, model_file)
            model = joblib.load(model_path)
            self.best_models[model_name] = model
            print(f"Loaded {model_name} from {model_path}")
    
    def get_feature_importance(self, model_name='random_forest'):
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model to get feature importance from
            
        Returns:
            dict: Feature importance scores
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model = self.best_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
            
            if self.feature_names:
                feature_importance = dict(zip(self.feature_names, importance_scores))
            else:
                feature_importance = {f'feature_{i}': score for i, score in enumerate(importance_scores)}
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            return feature_importance
        else:
            print(f"Model {model_name} does not support feature importance")
            return None
    
    def predict_proba(self, X, model_name=None):
        """
        Get probability predictions for new data.
        
        Args:
            X: Input features
            model_name: Specific model to use (if None, uses best model)
            
        Returns:
            np.array: Probability predictions
        """
        if model_name is None:
            # Use best model based on F1 score
            best_model_name = max(self.model_scores.keys(), 
                                key=lambda x: self.model_scores[x]['cv_f1'])
            model = self.best_models[best_model_name]
        else:
            model = self.best_models[model_name]
        
        return model.predict_proba(X)

def main():
    """
    Main function to demonstrate model training pipeline.
    """
    # Load preprocessed data
    try:
        X_train = np.load('data/processed/X_train.npy')
        X_test = np.load('data/processed/X_test.npy')
        y_train = np.load('data/processed/y_train.npy')
        y_test = np.load('data/processed/y_test.npy')
        
        # Load feature names
        feature_names_df = pd.read_csv('data/processed/feature_names.csv')
        feature_names = feature_names_df['feature_name'].tolist()
        
        print("Loaded preprocessed data successfully!")
        
    except FileNotFoundError:
        print("Preprocessed data not found. Please run data preprocessing first.")
        return
    
    # Initialize model trainer
    trainer = ModelTrainer(X_train, X_test, y_train, y_test, feature_names)
    
    # Train all models
    models = trainer.train_all_models()
    
    # Evaluate all models
    evaluation_results = trainer.evaluate_all_models()
    
    # Compare models
    comparison_summary = trainer.compare_models(evaluation_results)
    
    # Save models
    trainer.save_models()
    
    # Get feature importance
    feature_importance = trainer.get_feature_importance()
    if feature_importance:
        print("\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
            print(f"{i+1}. {feature}: {importance:.4f}")
    
    print("\n" + "=" * 50)
    print("MODEL TRAINING COMPLETED")
    print("=" * 50)
    
    return trainer, evaluation_results, comparison_summary

if __name__ == "__main__":
    main() 