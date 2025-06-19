"""
Main Execution Script for Customer Churn Prediction

This script orchestrates the entire pipeline including:
- Data preprocessing
- Model training and evaluation
- Visualization generation
- Feature analysis with SHAP
- Web application deployment
"""

import os
import sys
import time
from datetime import datetime

# Add src directory to path
sys.path.append('src')

def main():
    """
    Main execution function for the customer churn prediction pipeline.
    """
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION SYSTEM")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Step 1: Data Preprocessing
        print("STEP 1: DATA PREPROCESSING")
        print("-" * 30)
        from src.data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        preprocessor.load_data()
        preprocessor.explore_data()
        preprocessor.handle_missing_data()
        preprocessor.feature_engineering()
        X_train, X_test, y_train, y_test = preprocessor.prepare_features()
        
        print("✓ Data preprocessing completed successfully!")
        print()
        
        # Step 2: Model Training
        print("STEP 2: MODEL TRAINING")
        print("-" * 30)
        from src.model_training import ModelTrainer
        
        trainer = ModelTrainer(X_train, X_test, y_train, y_test, preprocessor.get_feature_names())
        models = trainer.train_all_models()
        evaluation_results = trainer.evaluate_all_models()
        comparison_summary = trainer.compare_models(evaluation_results)
        trainer.save_models()
        
        print("✓ Model training completed successfully!")
        print()
        
        # Step 3: Visualization
        print("STEP 3: VISUALIZATION")
        print("-" * 30)
        from src.visualization import DataVisualizer
        
        visualizer = DataVisualizer()
        visualizer.create_summary_report(
            preprocessor.get_feature_importance_data(),
            evaluation_results,
            trainer.get_feature_importance()
        )
        
        print("✓ Visualization completed successfully!")
        print()
        
        # Step 4: Feature Analysis with SHAP
        print("STEP 4: FEATURE ANALYSIS (SHAP)")
        print("-" * 30)
        from src.feature_analysis import FeatureAnalyzer
        
        # Use the best model for SHAP analysis
        best_model = comparison_summary['best_model']
        analyzer = FeatureAnalyzer(
            best_model, 
            X_train, 
            X_test, 
            preprocessor.get_feature_names()
        )
        insights = analyzer.create_comprehensive_report(preprocessor.get_feature_importance_data())
        
        print("✓ Feature analysis completed successfully!")
        print()
        
        # Step 5: Generate Summary Report
        print("STEP 5: GENERATING SUMMARY REPORT")
        print("-" * 30)
        generate_summary_report(preprocessor, trainer, comparison_summary, insights)
        
        print("✓ Summary report generated successfully!")
        print()
        
        # Step 6: Display Results
        print("STEP 6: FINAL RESULTS")
        print("-" * 30)
        display_final_results(comparison_summary, insights)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 7: Web Application Instructions
        print("\n" + "=" * 60)
        print("WEB APPLICATION")
        print("=" * 60)
        print("To start the web application, run:")
        print("python src/web_app.py")
        print("\nThen open your browser and go to: http://localhost:5000")
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {str(e)}")
        print("Please check the error and try again.")
        return False
    
    return True

def generate_summary_report(preprocessor, trainer, comparison_summary, insights):
    """
    Generate a comprehensive summary report.
    """
    report_file = "results/summary_report.txt"
    os.makedirs("results", exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write("CUSTOMER CHURN PREDICTION - SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Best Model: {comparison_summary['best_model_name']}\n")
        f.write(f"Best F1 Score: {comparison_summary['best_metrics']['F1 Score']:.4f}\n")
        f.write(f"Best AUC: {comparison_summary['best_metrics']['AUC']:.4f}\n")
        f.write(f"Dataset Size: {preprocessor.raw_data.shape[0]} samples\n")
        f.write(f"Features: {len(preprocessor.get_feature_names())}\n\n")
        
        f.write("MODEL PERFORMANCE COMPARISON\n")
        f.write("-" * 30 + "\n")
        f.write(comparison_summary['comparison_df'].to_string())
        f.write("\n\n")
        
        f.write("TOP FEATURES BY IMPORTANCE\n")
        f.write("-" * 25 + "\n")
        feature_importance = trainer.get_feature_importance()
        if feature_importance:
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
                f.write(f"{i+1}. {feature}: {importance:.4f}\n")
        f.write("\n")
        
        f.write("BUSINESS INSIGHTS\n")
        f.write("-" * 18 + "\n")
        f.write("Key Recommendations:\n")
        for rec in insights['recommendations']:
            f.write(f"• {rec['action']}\n")
        f.write("\n")
        
        f.write("Risk Factors:\n")
        for risk in insights['risk_factors']:
            f.write(f"• {risk}\n")
        f.write("\n")
        
        f.write("Opportunities:\n")
        for opp in insights['opportunities']:
            f.write(f"• {opp}\n")
    
    print(f"Summary report saved to: {report_file}")

def display_final_results(comparison_summary, insights):
    """
    Display final results in a formatted way.
    """
    print("🏆 BEST MODEL RESULTS")
    print(f"   Model: {comparison_summary['best_model_name']}")
    print(f"   F1 Score: {comparison_summary['best_metrics']['F1 Score']:.4f}")
    print(f"   Accuracy: {comparison_summary['best_metrics']['Accuracy']:.4f}")
    print(f"   AUC: {comparison_summary['best_metrics']['AUC']:.4f}")
    
    print("\n📊 MODEL COMPARISON")
    print(comparison_summary['comparison_df'].to_string(index=False, float_format='%.4f'))
    
    print("\n💡 KEY BUSINESS INSIGHTS")
    print("Top Recommendations:")
    for i, rec in enumerate(insights['recommendations'][:3]):
        print(f"   {i+1}. {rec['action']}")
    
    print("\n⚠️  RISK FACTORS")
    for i, risk in enumerate(insights['risk_factors'][:3]):
        print(f"   {i+1}. {risk}")
    
    print("\n🎯 OPPORTUNITIES")
    for i, opp in enumerate(insights['opportunities'][:3]):
        print(f"   {i+1}. {opp}")

def run_individual_components():
    """
    Function to run individual components for testing.
    """
    print("Available components:")
    print("1. Data Preprocessing")
    print("2. Model Training")
    print("3. Visualization")
    print("4. Feature Analysis")
    print("5. Web Application")
    
    choice = input("\nEnter component number (1-5): ")
    
    if choice == "1":
        print("\nRunning Data Preprocessing...")
        from src.data_preprocessing import main as preprocess_main
        preprocess_main()
    
    elif choice == "2":
        print("\nRunning Model Training...")
        from src.model_training import main as train_main
        train_main()
    
    elif choice == "3":
        print("\nRunning Visualization...")
        from src.visualization import main as viz_main
        viz_main()
    
    elif choice == "4":
        print("\nRunning Feature Analysis...")
        from src.feature_analysis import main as analysis_main
        analysis_main()
    
    elif choice == "5":
        print("\nStarting Web Application...")
        print("The web application will be available at: http://localhost:5000")
        os.system("python src/web_app.py")
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Customer Churn Prediction System')
    parser.add_argument('--component', action='store_true', 
                       help='Run individual components instead of full pipeline')
    
    args = parser.parse_args()
    
    if args.component:
        run_individual_components()
    else:
        main() 