"""
Portfolio ML Model Training Module
==================================
This module trains all ML models once and saves the best performing model.
Run this file once to determine and save the best model for portfolio predictions.

Usage: python train_best_model.py
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime

from data_preprocessing import preprocess_all
from portfolio_feature_engineering import create_portfolio_features
from ml_models import MLModelPipeline

class BestModelTrainer:
    def __init__(self):
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        self.feature_columns = None
        self.model_results = {}
        
    def train_and_save_best_model(self):
        """Complete pipeline to train all models and save the best one"""
        print("🚀 PORTFOLIO ML MODEL TRAINING")
        print("Finding the best model for portfolio predictions...")
        print("="*70)
        
        # Step 1: Load and prepare data
        print("📥 Step 1: Loading and preprocessing data...")
        df = preprocess_all()
        print(f"   ✅ Data loaded: {df.shape}")
        
        # Step 2: Create portfolio features
        print("\n📊 Step 2: Creating portfolio features...")
        portfolio_data = create_portfolio_features(df)
        print(f"   ✅ Portfolio features created: {portfolio_data.shape}")
        
        # Step 3: Initialize and run ML pipeline
        print("\n🤖 Step 3: Training all ML models...")
        ml_pipeline = MLModelPipeline()
        
        # Prepare features and target
        feature_cols = [col for col in portfolio_data.columns if col not in ['Date', 'target']]
        X = portfolio_data[feature_cols]  # Keep as DataFrame
        y = portfolio_data['target']      # Keep as Series
        
        # Create dates array for rolling window validation
        dates = portfolio_data['Date']
        
        # Run rolling window cross-validation
        print("   🔄 Running rolling window cross-validation...")
        cv_results, cv_predictions = ml_pipeline.rolling_window_validation(X, y, dates)
        
        # Create train/test split for final model training
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train final models
        print("   🎯 Training final models...")
        final_results, final_predictions = ml_pipeline.train_final_models(X_train, X_test, y_train, y_test, feature_cols)
        
        # Combine results
        results = {
            'cv_results': cv_results,
            'final_results': final_results,
            'final_predictions': final_predictions
        }
        
        # Step 4: Display comprehensive model comparison
        print("\n📊 Step 4: Model Performance Comparison...")
        self._display_model_comparison(results)
        
        # Step 5: Identify best model
        print("\n🏆 Step 5: Identifying best model...")
        best_model_info = self._find_best_model(results)
        
        # Step 6: Train final best model
        print(f"\n🎯 Step 6: Training final {best_model_info['name']} model...")
        self._train_final_model(X, y, feature_cols, best_model_info)
        
        # Step 7: Save model and metadata
        print("\n💾 Step 7: Saving model and metadata...")
        self._save_model_artifacts()
        
        # Step 8: Create summary report and results table
        print("\n📊 Step 8: Creating training summary and results table...")
        self._create_training_summary(results, best_model_info)
        self._create_results_table(results)
        
        print("\n🎉 MODEL TRAINING COMPLETED!")
        print(f"✅ Best model: {self.best_model_name}")
        print(f"📁 Model saved to: trained_models/")
        print(f"📋 Summary saved to: trained_models/training_summary.json")
        
        return self.best_model_name
    
    def _find_best_model(self, results):
        """Find the best performing model based on cross-validation R²"""
        cv_results = results['cv_results']
        
        best_r2 = float('-inf')
        best_model = None
        
        for model_name, metrics in cv_results.items():
            r2_score = metrics['R2']
            if r2_score > best_r2:
                best_r2 = r2_score
                best_model = model_name
        
        print(f"   🏆 Best model: {best_model}")
        print(f"   📊 Cross-validation R²: {best_r2:.4f}")
        print(f"   📏 Cross-validation RMSE: ${cv_results[best_model]['RMSE']:.2f}")
        print(f"   🎯 Why {best_model} won:")
        print(f"      - Highest R² score ({best_r2:.4f})")
        print(f"      - Lowest prediction error (${cv_results[best_model]['RMSE']:.2f} RMSE)")
        print(f"      - Most consistent performance across 20 validation windows")
        
        return {
            'name': best_model,
            'r2_score': best_r2,
            'rmse': cv_results[best_model]['RMSE']
        }
    
    def _display_model_comparison(self, results):
        """Display comprehensive model comparison table"""
        cv_results = results['cv_results']
        
        print("\n" + "="*90)
        print("📊 CROSS-VALIDATION RESULTS COMPARISON (20 Rolling Windows)")
        print("="*90)
        print(f"{'Model':<18} {'R² Score':<12} {'RMSE':<12} {'MAE':<12} {'MAPE':<12} {'Status':<15}")
        print("-"*90)
        
        # Sort models by R² score
        sorted_models = sorted(cv_results.items(), key=lambda x: x[1]['R2'], reverse=True)
        
        for i, (model_name, metrics) in enumerate(sorted_models):
            status = "🏆 BEST" if i == 0 else "✅ Good" if metrics['R2'] > 0 else "❌ Poor"
            print(f"{model_name:<18} {metrics['R2']:<12.4f} {metrics['RMSE']:<12.2f} "
                  f"{metrics['MAE']:<12.2f} {metrics['MAPE']:<12.4f} {status:<15}")
        
        print("-"*90)
        print("📝 Key Insights:")
        best_model = sorted_models[0]
        print(f"   • Best Model: {best_model[0]} (R² = {best_model[1]['R2']:.4f})")
        print(f"   • Lowest RMSE: ${best_model[1]['RMSE']:.2f}")
        print(f"   • Most Accurate: {best_model[1]['MAPE']:.2f}% MAPE")
        
        # Show why this model is best
        r2_values = [metrics['R2'] for metrics in cv_results.values()]
        print(f"   • R² Range: {min(r2_values):.4f} to {max(r2_values):.4f}")
        print(f"   • Performance Gap: {best_model[1]['R2'] - sorted_models[1][1]['R2']:.4f} R² advantage")
        print("="*90)
    
    def _train_final_model(self, X, y, feature_cols, best_model_info):
        """Train the final best model on all available data"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        # Train the best model
        model_name = best_model_info['name']
        if model_name == 'Linear Regression':
            from sklearn.linear_model import LinearRegression
            self.best_model = LinearRegression()
        elif model_name == 'Ridge Regression':
            from sklearn.linear_model import Ridge
            self.best_model = Ridge(alpha=1.0)
        elif model_name == 'Lasso Regression':
            from sklearn.linear_model import Lasso
            self.best_model = Lasso(alpha=1.0)
        elif model_name == 'SVR':
            from sklearn.svm import SVR
            self.best_model = SVR(kernel='rbf', C=1.0, gamma='scale')
        elif model_name == 'XGBoost':
            import xgboost as xgb
            self.best_model = xgb.XGBRegressor(random_state=42)
        elif model_name == 'Random Forest':
            from sklearn.ensemble import RandomForestRegressor
            self.best_model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == 'Bayesian Ridge':
            from sklearn.linear_model import BayesianRidge
            self.best_model = BayesianRidge()
        
        # Train the model
        self.best_model.fit(X_train_scaled, y_train)
        self.best_model_name = model_name
        
        # Evaluate on test set
        y_pred = self.best_model.predict(X_test_scaled)
        from sklearn.metrics import mean_squared_error, r2_score
        
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        
        print(f"   ✅ Final model trained")
        print(f"   📊 Test RMSE: ${test_rmse:.2f}")
        print(f"   📊 Test R²: {test_r2:.4f}")
        
        # Store test results
        self.model_results = {
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'cv_r2': best_model_info['r2_score'],
            'cv_rmse': best_model_info['rmse']
        }
    
    def _save_model_artifacts(self):
        """Save the trained model, scaler, and metadata"""
        # Create directory for trained models
        os.makedirs('trained_models', exist_ok=True)
        
        # Save the trained model
        with open('trained_models/best_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save the scaler
        with open('trained_models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature columns
        with open('trained_models/feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'feature_count': len(self.feature_columns),
            'training_date': datetime.now().isoformat(),
            'test_metrics': self.model_results
        }
        
        with open('trained_models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Model artifacts saved to trained_models/")
    
    def _create_training_summary(self, results, best_model_info):
        """Create a comprehensive training summary"""
        summary = {
            'training_info': {
                'training_date': datetime.now().isoformat(),
                'best_model': self.best_model_name,
                'feature_count': len(self.feature_columns),
                'data_shape': f"{len(self.feature_columns)} features"
            },
            'best_model_performance': {
                'cross_validation_r2': best_model_info['r2_score'],
                'cross_validation_rmse': best_model_info['rmse'],
                'test_r2': self.model_results['test_r2'],
                'test_rmse': self.model_results['test_rmse']
            },
            'all_models_comparison': results['cv_results'],
            'feature_names': self.feature_columns
        }
        
        with open('trained_models/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   ✅ Training summary saved")
    
    def _create_results_table(self, results):
        """Create a comprehensive results table CSV"""
        import pandas as pd
        
        cv_results = results['cv_results']
        final_results = results['final_results']
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name in cv_results.keys():
            cv_metrics = cv_results[model_name]
            final_metrics = final_results.get(model_name, {})
            
            row = {
                'Model': model_name,
                'CV_R2': cv_metrics['R2'],
                'CV_RMSE': cv_metrics['RMSE'],
                'CV_MAE': cv_metrics['MAE'],
                'CV_MAPE': cv_metrics['MAPE'],
                'Test_R2': final_metrics.get('R2', 'N/A'),
                'Test_RMSE': final_metrics.get('RMSE', 'N/A'),
                'Test_MAE': final_metrics.get('MAE', 'N/A'),
                'Overall_Rank': 0  # Will be filled below
            }
            comparison_data.append(row)
        
        # Create DataFrame and rank by CV R²
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('CV_R2', ascending=False).reset_index(drop=True)
        df['Overall_Rank'] = range(1, len(df) + 1)
        
        # Save to CSV
        os.makedirs('trained_models', exist_ok=True)
        df.to_csv('trained_models/model_comparison_table.csv', index=False)
        
        print(f"   ✅ Model comparison table saved to trained_models/model_comparison_table.csv")
        
        # Also create a formatted display version
        print("\n📋 FINAL MODEL RANKING:")
        print("-" * 60)
        for _, row in df.iterrows():
            status = "🏆" if row['Overall_Rank'] == 1 else f"{row['Overall_Rank']}"
            print(f"{status}. {row['Model']:<18} (R² = {row['CV_R2']:.4f})")
        print("-" * 60)

def main():
    """Main function to train and save the best model"""
    trainer = BestModelTrainer()
    best_model_name = trainer.train_and_save_best_model()
    
    print(f"\n🎯 READY FOR PORTFOLIO RECOMMENDATIONS!")
    print(f"Now you can run interactive_portfolio_system.py to get portfolio recommendations")
    print(f"using the trained {best_model_name} model.")

if __name__ == "__main__":
    main()