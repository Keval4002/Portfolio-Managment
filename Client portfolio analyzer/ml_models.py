import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class MLModelPipeline:
    def __init__(self, models=None):
        """
        Initialize ML pipeline with regression models
        """
        if models is None:
            self.models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Lasso Regression': Lasso(alpha=0.1),
                'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
                'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                'Bayesian Ridge': BayesianRidge()
            }
        else:
            self.models = models
        
        self.results = {}
        self.predictions = {}
        self.feature_importance = {}
        
    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        return metrics
    
    def rolling_window_validation(self, X, y, dates, initial_window=200, step_size=50):
        """
        Perform rolling window validation for time series data
        """
        results = {}
        all_predictions = {}
        
        print(f"🔄 Starting rolling window validation...")
        print(f"   Initial window: {initial_window} days")
        print(f"   Step size: {step_size} days")
        print(f"   Total data points: {len(X)}")
        
        # Calculate number of windows
        n_windows = max(1, (len(X) - initial_window) // step_size)
        print(f"   Number of validation windows: {n_windows}")
        
        for model_name, model in self.models.items():
            print(f"\n🤖 Training {model_name}...")
            
            window_metrics = []
            model_predictions = []
            
            for i in range(n_windows):
                start_idx = i * step_size
                train_end_idx = start_idx + initial_window
                test_start_idx = train_end_idx
                test_end_idx = min(test_start_idx + step_size, len(X))
                
                if test_end_idx <= test_start_idx:
                    break
                
                # Split data
                X_train = X.iloc[start_idx:train_end_idx]
                X_test = X.iloc[test_start_idx:test_end_idx]
                y_train = y.iloc[start_idx:train_end_idx]
                y_test = y.iloc[test_start_idx:test_end_idx]
                
                try:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    window_metric = self.calculate_metrics(y_test, y_pred)
                    window_metrics.append(window_metric)
                    
                    # Store predictions with dates
                    test_dates = dates.iloc[test_start_idx:test_end_idx]
                    for j, (date, actual, pred) in enumerate(zip(test_dates, y_test, y_pred)):
                        model_predictions.append({
                            'Date': date,
                            'Actual': actual,
                            'Predicted': pred,
                            'Window': i
                        })
                
                except Exception as e:
                    print(f"   ⚠️ Error in window {i}: {str(e)}")
                    continue
            
            if window_metrics:
                # Average metrics across all windows
                avg_metrics = {}
                for metric in window_metrics[0].keys():
                    avg_metrics[metric] = np.mean([w[metric] for w in window_metrics])
                    avg_metrics[f'{metric}_std'] = np.std([w[metric] for w in window_metrics])
                
                results[model_name] = avg_metrics
                all_predictions[model_name] = pd.DataFrame(model_predictions)
                
                print(f"   ✅ {model_name} completed. RMSE: {avg_metrics['RMSE']:.4f}, R²: {avg_metrics['R2']:.4f}")
            else:
                print(f"   ❌ {model_name} failed - no valid windows")
        
        self.results = results
        self.predictions = all_predictions
        
        return results, all_predictions
    
    def train_final_models(self, X_train, X_test, y_train, y_test, feature_names):
        """
        Train final models on the last train/test split
        """
        print(f"\n🎯 Training final models on train/test split...")
        print(f"   Training set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
        
        final_results = {}
        final_predictions = {}
        
        for model_name, model in self.models.items():
            try:
                print(f"\n📊 Training final {model_name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_test, y_pred)
                final_results[model_name] = metrics
                final_predictions[model_name] = {
                    'actual': y_test,
                    'predicted': y_pred
                }
                
                # Feature importance (for tree-based models)
                if hasattr(model, 'feature_importances_'):
                    feature_imp = pd.DataFrame({
                        'feature': feature_names,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    self.feature_importance[model_name] = feature_imp
                
                print(f"   ✅ Final {model_name} - RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error training {model_name}: {str(e)}")
        
        return final_results, final_predictions
    
    def create_results_table(self, cv_results=None, final_results=None):
        """
        Create comparison table of all models
        """
        if cv_results is None:
            cv_results = self.results
        
        # Create comparison DataFrame
        comparison_data = []
        
        for model_name in cv_results.keys():
            row = {'Model': model_name}
            
            # Cross-validation metrics
            cv_metrics = cv_results[model_name]
            row['CV_RMSE'] = f"{cv_metrics['RMSE']:.4f} ± {cv_metrics.get('RMSE_std', 0):.4f}"
            row['CV_MAE'] = f"{cv_metrics['MAE']:.4f} ± {cv_metrics.get('MAE_std', 0):.4f}"
            row['CV_R2'] = f"{cv_metrics['R2']:.4f} ± {cv_metrics.get('R2_std', 0):.4f}"
            row['CV_MAPE'] = f"{cv_metrics['MAPE']:.4f} ± {cv_metrics.get('MAPE_std', 0):.4f}%"
            
            # Final test metrics
            if final_results and model_name in final_results:
                final_metrics = final_results[model_name]
                row['Final_RMSE'] = f"{final_metrics['RMSE']:.4f}"
                row['Final_MAE'] = f"{final_metrics['MAE']:.4f}"
                row['Final_R2'] = f"{final_metrics['R2']:.4f}"
                row['Final_MAPE'] = f"{final_metrics['MAPE']:.4f}%"
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by CV R² score (descending)
        comparison_df['R2_numeric'] = comparison_df['CV_R2'].str.split(' ±').str[0].astype(float)
        comparison_df = comparison_df.sort_values('R2_numeric', ascending=False).drop('R2_numeric', axis=1)
        
        return comparison_df
    
    def plot_results(self, stock_ticker, save_dir='result'):
        """
        Create visualization plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot 1: Model comparison
        if self.results:
            plt.figure(figsize=(12, 8))
            
            models = list(self.results.keys())
            metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, metric in enumerate(metrics):
                values = [self.results[model][metric] for model in models]
                stds = [self.results[model].get(f'{metric}_std', 0) for model in models]
                
                bars = axes[i].bar(models, values, yerr=stds, capsize=5, alpha=0.7)
                axes[i].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + max(stds)/2,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.suptitle(f'Model Performance Comparison - {stock_ticker}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/model_comparison_{stock_ticker}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 2: Best model predictions
        if self.predictions:
            best_model = max(self.results.keys(), key=lambda x: self.results[x]['R2'])
            
            if best_model in self.predictions:
                pred_df = self.predictions[best_model]
                
                plt.figure(figsize=(15, 8))
                plt.plot(pred_df['Date'], pred_df['Actual'], label='Actual', alpha=0.7, linewidth=2)
                plt.plot(pred_df['Date'], pred_df['Predicted'], label='Predicted', alpha=0.7, linewidth=2)
                
                plt.title(f'Best Model Predictions: {best_model} - {stock_ticker}', fontsize=14, fontweight='bold')
                plt.xlabel('Date')
                plt.ylabel('Price ($)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'{save_dir}/best_model_predictions_{stock_ticker}.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"📊 Plots saved to {save_dir}/")
    
    def get_best_model(self):
        """
        Get the best performing model based on R² score
        """
        if not self.results:
            return None
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['R2'])
        best_model = self.models[best_model_name]
        best_metrics = self.results[best_model_name]
        
        return best_model_name, best_model, best_metrics