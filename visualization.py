"""
Visualization Module for Customer Churn Prediction

This module provides comprehensive visualization capabilities including:
- Data exploration plots
- Model performance comparison
- Feature importance visualization
- ROC curves and confusion matrices
- Interactive plots using Plotly
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataVisualizer:
    """
    A comprehensive visualization class for customer churn prediction.
    """
    
    def __init__(self, save_dir='results'):
        """
        Initialize the DataVisualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_correlation_heatmap(self, data, save=True):
        """
        Create correlation heatmap for numerical features.
        
        Args:
            data: DataFrame with numerical features
            save: Whether to save the plot
        """
        # Select numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numerical_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Heatmap of Numerical Features', fontsize=16, pad=20)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'correlation_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_feature_distributions(self, data, save=True):
        """
        Create distribution plots for numerical features.
        
        Args:
            data: DataFrame with features
            save: Whether to save the plot
        """
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        n_cols = 3
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            sns.histplot(data[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'feature_distributions.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_churn_distribution(self, data, save=True):
        """
        Create churn distribution plots.
        
        Args:
            data: DataFrame with churn column
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Churn count plot
        churn_counts = data['churn'].value_counts()
        colors = ['#2E8B57', '#CD5C5C']
        axes[0].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
        axes[0].set_title('Churn Distribution', fontsize=14)
        
        # Churn bar plot
        sns.countplot(data=data, x='churn', ax=axes[1], palette=colors)
        axes[1].set_title('Churn Count', fontsize=14)
        axes[1].set_xlabel('Churn Status')
        axes[1].set_ylabel('Count')
        
        # Add count labels on bars
        for i, v in enumerate(churn_counts.values):
            axes[1].text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'churn_distribution.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_categorical_features(self, data, save=True):
        """
        Create plots for categorical features vs churn.
        
        Args:
            data: DataFrame with categorical features
            save: Whether to save the plot
        """
        categorical_cols = data.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'churn']
        
        n_cols = 2
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(categorical_cols):
            row = i // n_cols
            col_idx = i % n_cols
            
            # Create cross-tabulation
            crosstab = pd.crosstab(data[col], data['churn'], normalize='index') * 100
            
            crosstab.plot(kind='bar', ax=axes[row, col_idx], color=['#2E8B57', '#CD5C5C'])
            axes[row, col_idx].set_title(f'Churn Rate by {col}')
            axes[row, col_idx].set_xlabel(col)
            axes[row, col_idx].set_ylabel('Percentage')
            axes[row, col_idx].legend(title='Churn')
            axes[row, col_idx].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(len(categorical_cols), n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'categorical_features_churn.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_model_comparison(self, evaluation_results, save=True):
        """
        Create model comparison plots.
        
        Args:
            evaluation_results: Dictionary with model evaluation results
            save: Whether to save the plot
        """
        # Extract metrics
        models = list(evaluation_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [evaluation_results[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Hide the last empty subplot
        axes[-1].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'model_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_roc_curves(self, evaluation_results, save=True):
        """
        Create ROC curves for all models.
        
        Args:
            evaluation_results: Dictionary with model evaluation results
            save: Whether to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            fpr = results['roc_curve']['fpr']
            tpr = results['roc_curve']['tpr']
            auc_score = results['auc']
            
            plt.plot(fpr, tpr, color=colors[i], lw=2, 
                    label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Models')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'roc_curves.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrices(self, evaluation_results, save=True):
        """
        Create confusion matrices for all models.
        
        Args:
            evaluation_results: Dictionary with model evaluation results
            save: Whether to save the plot
        """
        n_models = len(evaluation_results)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            row = i // n_cols
            col_idx = i % n_cols
            
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'Confusion Matrix - {model_name}')
            axes[row, col_idx].set_xlabel('Predicted')
            axes[row, col_idx].set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'confusion_matrices.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_feature_importance(self, feature_importance, top_n=15, save=True):
        """
        Create feature importance plot.
        
        Args:
            feature_importance: Dictionary with feature importance scores
            top_n: Number of top features to show
            save: Whether to save the plot
        """
        # Get top N features
        top_features = dict(list(feature_importance.items())[:top_n])
        
        plt.figure(figsize=(12, 8))
        features = list(top_features.keys())
        importance_scores = list(top_features.values())
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(features)), importance_scores, color='skyblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, importance_scores)):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{score:.4f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'feature_importance.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_interactive_dashboard(self, data, evaluation_results, feature_importance):
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            data: Original dataset
            evaluation_results: Model evaluation results
            feature_importance: Feature importance scores
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Churn Distribution', 'Model Performance Comparison',
                          'ROC Curves', 'Feature Importance',
                          'Tenure vs Monthly Charges', 'Age Distribution by Churn'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "box"}]]
        )
        
        # 1. Churn Distribution (Pie Chart)
        churn_counts = data['churn'].value_counts()
        fig.add_trace(
            go.Pie(labels=churn_counts.index, values=churn_counts.values, name="Churn"),
            row=1, col=1
        )
        
        # 2. Model Performance Comparison (Bar Chart)
        models = list(evaluation_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        for i, metric in enumerate(metrics):
            values = [evaluation_results[model][metric] for model in models]
            fig.add_trace(
                go.Bar(name=metric.replace('_', ' ').title(), x=models, y=values),
                row=1, col=2
            )
        
        # 3. ROC Curves
        colors = ['red', 'blue', 'green', 'orange']
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            fpr = results['roc_curve']['fpr']
            tpr = results['roc_curve']['tpr']
            auc_score = results['auc']
            
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC={auc_score:.3f})',
                          line=dict(color=colors[i])),
                row=2, col=1
            )
        
        # Add diagonal line for ROC
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                      line=dict(color='black', dash='dash')),
            row=2, col=1
        )
        
        # 4. Feature Importance
        top_features = dict(list(feature_importance.items())[:10])
        fig.add_trace(
            go.Bar(x=list(top_features.values()), y=list(top_features.keys()),
                  orientation='h', name='Feature Importance'),
            row=2, col=2
        )
        
        # 5. Tenure vs Monthly Charges
        fig.add_trace(
            go.Scatter(x=data['tenure'], y=data['monthly_charges'], mode='markers',
                      marker=dict(color=data['churn'].map({'Yes': 'red', 'No': 'blue'}),
                                opacity=0.6),
                      name='Tenure vs Charges'),
            row=3, col=1
        )
        
        # 6. Age Distribution by Churn
        for churn_status in ['Yes', 'No']:
            subset = data[data['churn'] == churn_status]
            fig.add_trace(
                go.Box(y=subset['age'], name=f'Age - {churn_status}'),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Customer Churn Prediction Dashboard",
            showlegend=True,
            height=1200
        )
        
        # Save interactive plot
        fig.write_html(os.path.join(self.save_dir, 'interactive_dashboard.html'))
        
        return fig
        
    def create_summary_report(self, data, evaluation_results, feature_importance):
        """
        Create a comprehensive summary report with all visualizations.
        
        Args:
            data: Original dataset
            evaluation_results: Model evaluation results
            feature_importance: Feature importance scores
        """
        print("Creating comprehensive visualization report...")
        
        # Create all plots
        self.plot_correlation_heatmap(data)
        self.plot_feature_distributions(data)
        self.plot_churn_distribution(data)
        self.plot_categorical_features(data)
        self.plot_model_comparison(evaluation_results)
        self.plot_roc_curves(evaluation_results)
        self.plot_confusion_matrices(evaluation_results)
        self.plot_feature_importance(feature_importance)
        
        # Create interactive dashboard
        self.create_interactive_dashboard(data, evaluation_results, feature_importance)
        
        print(f"All visualizations saved to {self.save_dir}/")
        print("Interactive dashboard saved as interactive_dashboard.html")

def main():
    """
    Main function to demonstrate visualization capabilities.
    """
    # This would typically be called after data preprocessing and model training
    print("Visualization module ready for use!")
    print("Use DataVisualizer class to create comprehensive visualizations.")

if __name__ == "__main__":
    main() 