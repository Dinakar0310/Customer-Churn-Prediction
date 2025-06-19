"""
Feature Analysis Module for Customer Churn Prediction

This module provides comprehensive feature analysis including:
- SHAP (SHapley Additive exPlanations) analysis
- Feature importance analysis
- Individual prediction explanations
- Feature interaction analysis
- Business insights and recommendations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings
warnings.filterwarnings('ignore')

class FeatureAnalyzer:
    """
    A comprehensive feature analysis class for customer churn prediction.
    """
    
    def __init__(self, model, X_train, X_test, feature_names, save_dir='results'):
        """
        Initialize the FeatureAnalyzer.
        
        Args:
            model: Trained model for SHAP analysis
            X_train, X_test: Training and testing data
            feature_names: List of feature names
            save_dir: Directory to save analysis results
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize SHAP explainer
        self.explainer = None
        self.shap_values = None
        
    def create_shap_explainer(self, background_samples=100):
        """
        Create SHAP explainer for the model.
        
        Args:
            background_samples: Number of background samples for SHAP
        """
        print("Creating SHAP explainer...")
        
        # Select background samples
        background = shap.sample(self.X_train, background_samples)
        
        # Create explainer based on model type
        if hasattr(self.model, 'predict_proba'):
            self.explainer = shap.TreeExplainer(self.model) if hasattr(self.model, 'feature_importances_') else \
                           shap.KernelExplainer(self.model.predict_proba, background)
        else:
            self.explainer = shap.TreeExplainer(self.model) if hasattr(self.model, 'feature_importances_') else \
                           shap.KernelExplainer(self.model.predict, background)
        
        print("SHAP explainer created successfully!")
        
    def calculate_shap_values(self, X=None):
        """
        Calculate SHAP values for the given data.
        
        Args:
            X: Data to calculate SHAP values for (default: X_test)
        """
        if X is None:
            X = self.X_test
        
        if self.explainer is None:
            self.create_shap_explainer()
        
        print("Calculating SHAP values...")
        self.shap_values = self.explainer.shap_values(X)
        
        # Handle different SHAP output formats
        if isinstance(self.shap_values, list):
            # For models with predict_proba, get values for positive class
            self.shap_values = self.shap_values[1]
        
        print(f"SHAP values calculated for {X.shape[0]} samples")
        
    def plot_shap_summary(self, X=None, save=True):
        """
        Create SHAP summary plot.
        
        Args:
            X: Data to plot (default: X_test)
            save: Whether to save the plot
        """
        if X is None:
            X = self.X_test
        
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, 
                         show=False, plot_size=(12, 8))
        plt.title('SHAP Summary Plot', fontsize=16, pad=20)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'shap_summary.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_shap_bar(self, X=None, save=True):
        """
        Create SHAP bar plot showing mean absolute SHAP values.
        
        Args:
            X: Data to plot (default: X_test)
            save: Whether to save the plot
        """
        if X is None:
            X = self.X_test
        
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, 
                         plot_type="bar", show=False)
        plt.title('SHAP Feature Importance (Mean Absolute Impact)', fontsize=16, pad=20)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'shap_bar.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_shap_waterfall(self, sample_idx=0, X=None, save=True):
        """
        Create SHAP waterfall plot for a specific sample.
        
        Args:
            sample_idx: Index of the sample to explain
            X: Data to plot (default: X_test)
            save: Whether to save the plot
        """
        if X is None:
            X = self.X_test
        
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(shap.Explanation(values=self.shap_values[sample_idx],
                                            base_values=self.explainer.expected_value,
                                            data=X[sample_idx],
                                            feature_names=self.feature_names),
                           show=False)
        plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}', fontsize=16, pad=20)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, f'shap_waterfall_sample_{sample_idx}.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_shap_dependence(self, feature_name, X=None, save=True):
        """
        Create SHAP dependence plot for a specific feature.
        
        Args:
            feature_name: Name of the feature to analyze
            X: Data to plot (default: X_test)
            save: Whether to save the plot
        """
        if X is None:
            X = self.X_test
        
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Find feature index
        try:
            feature_idx = self.feature_names.index(feature_name)
        except ValueError:
            print(f"Feature '{feature_name}' not found in feature names")
            return
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature_idx, self.shap_values, X, 
                           feature_names=self.feature_names, show=False)
        plt.title(f'SHAP Dependence Plot - {feature_name}', fontsize=16, pad=20)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, f'shap_dependence_{feature_name}.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_shap_interaction(self, feature1, feature2, X=None, save=True):
        """
        Create SHAP interaction plot between two features.
        
        Args:
            feature1, feature2: Names of the features to analyze
            X: Data to plot (default: X_test)
            save: Whether to save the plot
        """
        if X is None:
            X = self.X_test
        
        # Find feature indices
        try:
            feature1_idx = self.feature_names.index(feature1)
            feature2_idx = self.feature_names.index(feature2)
        except ValueError as e:
            print(f"Feature not found: {e}")
            return
        
        # Calculate interaction values
        interaction_values = shap.TreeExplainer(self.model).shap_interaction_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature1_idx, self.shap_values, X, 
                           interaction_index=feature2_idx,
                           feature_names=self.feature_names, show=False)
        plt.title(f'SHAP Interaction Plot - {feature1} vs {feature2}', fontsize=16, pad=20)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, f'shap_interaction_{feature1}_{feature2}.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def get_feature_importance_ranking(self, X=None):
        """
        Get feature importance ranking based on mean absolute SHAP values.
        
        Args:
            X: Data to analyze (default: X_test)
            
        Returns:
            DataFrame: Feature importance ranking
        """
        if X is None:
            X = self.X_test
        
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        
        # Create ranking dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
        
    def analyze_individual_prediction(self, sample_idx=0, X=None):
        """
        Analyze individual prediction and provide explanation.
        
        Args:
            sample_idx: Index of the sample to analyze
            X: Data to analyze (default: X_test)
            
        Returns:
            dict: Analysis results
        """
        if X is None:
            X = self.X_test
        
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Get prediction
        prediction = self.model.predict(X[sample_idx:sample_idx+1])[0]
        prediction_proba = self.model.predict_proba(X[sample_idx:sample_idx+1])[0]
        
        # Get SHAP values for this sample
        sample_shap_values = self.shap_values[sample_idx]
        
        # Create feature contribution dataframe
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'value': X[sample_idx],
            'shap_value': sample_shap_values,
            'abs_shap': np.abs(sample_shap_values)
        })
        
        # Sort by absolute SHAP value
        contributions = contributions.sort_values('abs_shap', ascending=False)
        
        # Get top contributing features
        top_positive = contributions[contributions['shap_value'] > 0].head(5)
        top_negative = contributions[contributions['shap_value'] < 0].head(5)
        
        analysis = {
            'sample_idx': sample_idx,
            'prediction': prediction,
            'prediction_proba': prediction_proba,
            'churn_probability': prediction_proba[1],
            'base_value': self.explainer.expected_value,
            'top_positive_features': top_positive,
            'top_negative_features': top_negative,
            'all_contributions': contributions
        }
        
        return analysis
        
    def generate_business_insights(self, data, top_n=10):
        """
        Generate business insights based on SHAP analysis.
        
        Args:
            data: Original dataset
            top_n: Number of top features to analyze
            
        Returns:
            dict: Business insights and recommendations
        """
        print("Generating business insights...")
        
        # Get feature importance ranking
        importance_df = self.get_feature_importance_ranking()
        top_features = importance_df.head(top_n)
        
        insights = {
            'top_features': top_features,
            'recommendations': [],
            'risk_factors': [],
            'opportunities': []
        }
        
        # Analyze each top feature
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['mean_abs_shap']
            
            # Get feature statistics
            if feature in data.columns:
                feature_stats = data[feature].describe()
                
                # Generate insights based on feature type
                if data[feature].dtype in ['int64', 'float64']:
                    # Numerical feature
                    if 'tenure' in feature.lower():
                        insights['recommendations'].append({
                            'feature': feature,
                            'insight': f"Customer tenure is a key predictor (importance: {importance:.4f}). Focus on retaining long-term customers.",
                            'action': "Implement loyalty programs for customers with longer tenure."
                        })
                    elif 'charges' in feature.lower():
                        insights['recommendations'].append({
                            'feature': feature,
                            'insight': f"Monthly charges significantly impact churn (importance: {importance:.4f}).",
                            'action': "Review pricing strategy and consider tiered pricing options."
                        })
                    elif 'age' in feature.lower():
                        insights['recommendations'].append({
                            'feature': feature,
                            'insight': f"Customer age affects churn prediction (importance: {importance:.4f}).",
                            'action': "Develop age-specific marketing campaigns and service offerings."
                        })
                else:
                    # Categorical feature
                    if 'contract' in feature.lower():
                        insights['recommendations'].append({
                            'feature': feature,
                            'insight': f"Contract type is crucial for churn prediction (importance: {importance:.4f}).",
                            'action': "Focus on converting month-to-month customers to longer contracts."
                        })
                    elif 'service' in feature.lower():
                        insights['recommendations'].append({
                            'feature': feature,
                            'insight': f"Service type influences churn (importance: {importance:.4f}).",
                            'action': "Improve service quality and offer bundle deals."
                        })
        
        # Generate risk factors
        insights['risk_factors'] = [
            "High monthly charges relative to service value",
            "Month-to-month contracts with no long-term commitment",
            "Short customer tenure (less than 12 months)",
            "Limited service usage or engagement",
            "Poor customer service experience"
        ]
        
        # Generate opportunities
        insights['opportunities'] = [
            "Implement predictive churn models for early intervention",
            "Develop personalized retention strategies",
            "Create loyalty programs and rewards",
            "Improve customer service and support",
            "Offer competitive pricing and value propositions"
        ]
        
        return insights
        
    def create_comprehensive_report(self, data, save=True):
        """
        Create a comprehensive feature analysis report.
        
        Args:
            data: Original dataset
            save: Whether to save the report
        """
        print("Creating comprehensive feature analysis report...")
        
        # Create SHAP explainer and calculate values
        self.create_shap_explainer()
        self.calculate_shap_values()
        
        # Generate all plots
        self.plot_shap_summary()
        self.plot_shap_bar()
        self.plot_shap_waterfall()
        
        # Plot dependence plots for top features
        importance_df = self.get_feature_importance_ranking()
        top_features = importance_df.head(5)['feature'].tolist()
        
        for feature in top_features:
            self.plot_shap_dependence(feature)
        
        # Generate business insights
        insights = self.generate_business_insights(data)
        
        # Save insights to file
        if save:
            insights_file = os.path.join(self.save_dir, 'business_insights.txt')
            with open(insights_file, 'w') as f:
                f.write("CUSTOMER CHURN PREDICTION - BUSINESS INSIGHTS\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("TOP FEATURES BY IMPORTANCE:\n")
                f.write("-" * 30 + "\n")
                for _, row in insights['top_features'].iterrows():
                    f.write(f"{row['rank']}. {row['feature']}: {row['mean_abs_shap']:.4f}\n")
                
                f.write("\n\nRECOMMENDATIONS:\n")
                f.write("-" * 20 + "\n")
                for rec in insights['recommendations']:
                    f.write(f"Feature: {rec['feature']}\n")
                    f.write(f"Insight: {rec['insight']}\n")
                    f.write(f"Action: {rec['action']}\n\n")
                
                f.write("RISK FACTORS:\n")
                f.write("-" * 15 + "\n")
                for risk in insights['risk_factors']:
                    f.write(f"• {risk}\n")
                
                f.write("\nOPPORTUNITIES:\n")
                f.write("-" * 15 + "\n")
                for opp in insights['opportunities']:
                    f.write(f"• {opp}\n")
        
        print(f"Comprehensive report created and saved to {self.save_dir}/")
        
        return insights

def main():
    """
    Main function to demonstrate feature analysis capabilities.
    """
    print("Feature Analysis module ready for use!")
    print("Use FeatureAnalyzer class to perform SHAP analysis and generate insights.")

if __name__ == "__main__":
    main() 