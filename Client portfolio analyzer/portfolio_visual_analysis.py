"""
Portfolio Recommendation Visual Analysis
========================================
Creates comprehensive visualizations of portfolio recommendations and model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PortfolioVisualAnalyzer:
    def __init__(self, user_id="default_user", session_timestamp=None):
        self.user_id = user_id
        self.session_timestamp = session_timestamp if session_timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.create_results_structure()
        self.recommendations_data = []
        self.model_results = {}
        self.color_palette = {
            'Technology': '#2E86C1',
            'Healthcare': '#28B463', 
            'Finance': '#F39C12',
            'Aviation': '#E74C3C',
            'Conservative': '#27AE60',
            'Moderate': '#F39C12',
            'Aggressive': '#E74C3C'
        }
    
    def create_results_structure(self):
        """Create systematic folder structure for results"""
        # Create main results directory
        results_dir = os.path.join("..", "Portfolio_Analysis_Results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Create user-specific directory
        user_dir = os.path.join(results_dir, f"User_{self.user_id}")
        os.makedirs(user_dir, exist_ok=True)
        
        # Create session directory with timestamp
        session_dir = os.path.join(user_dir, f"Session_{self.session_timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            "visualizations",
            "recommendations", 
            "model_analysis",
            "performance_metrics",
            "raw_data"
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(session_dir, subdir), exist_ok=True)
        
        print(f"📁 Results directory created: {session_dir}")
        return session_dir
        
    def load_data(self):
        """Load all portfolio recommendations and model results"""
        print("📥 Loading portfolio recommendations and model data...")
        
        # Load recommendations
        recommendations_dir = 'recommendations'
        if os.path.exists(recommendations_dir):
            for file in os.listdir(recommendations_dir):
                if file.endswith('.json'):
                    with open(os.path.join(recommendations_dir, file), 'r') as f:
                        data = json.load(f)
                        self.recommendations_data.append(data)
        
        # Load model results
        if os.path.exists('trained_models/training_summary.json'):
            with open('trained_models/training_summary.json', 'r') as f:
                self.model_results = json.load(f)
        
        print(f"   ✅ Loaded {len(self.recommendations_data)} recommendations")
        print(f"   ✅ Loaded model results for {len(self.model_results.get('all_models_comparison', {}))} models")
    
    def create_model_performance_plots(self):
        """Create comprehensive model performance visualizations"""
        if not self.model_results:
            print("❌ No model results found")
            return
        
        model_comparison = self.model_results.get('all_models_comparison', {})
        
        # Prepare data
        models = list(model_comparison.keys())
        r2_scores = [model_comparison[model]['R2'] for model in models]
        rmse_scores = [model_comparison[model]['RMSE'] for model in models]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🤖 ML Model Performance Analysis - Portfolio Prediction', fontsize=16, fontweight='bold')
        
        # 1. R² Score Comparison
        colors = ['#27AE60' if r2 > 0.5 else '#F39C12' if r2 > 0 else '#E74C3C' for r2 in r2_scores]
        bars1 = ax1.bar(range(len(models)), r2_scores, color=colors, alpha=0.8)
        ax1.set_title('📊 R² Score Comparison (Higher = Better)', fontweight='bold')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('R² Score')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels([m.replace(' ', '\\n') for m in models], rotation=45, ha='right')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars1, r2_scores)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. RMSE Comparison  
        bars2 = ax2.bar(range(len(models)), rmse_scores, color='#3498DB', alpha=0.8)
        ax2.set_title('📈 RMSE Comparison (Lower = Better)', fontweight='bold')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('RMSE ($)')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.replace(' ', '\\n') for m in models], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars2, rmse_scores)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(rmse_scores)*0.01,
                    f'${score:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Model Performance Matrix
        metrics = ['R2', 'RMSE', 'MAE', 'MAPE']
        model_matrix = []
        for model in models:
            row = []
            for metric in metrics:
                value = model_comparison[model][metric]
                if metric == 'R2':
                    # Normalize R2 to 0-1 scale for visualization
                    normalized = max(0, (value + 15) / 16)  # Adjust for negative values
                else:
                    # For error metrics, invert so lower is better shows as higher
                    max_val = max([model_comparison[m][metric] for m in models])
                    normalized = 1 - (value / max_val) if max_val > 0 else 0
                row.append(normalized)
            model_matrix.append(row)
        
        im3 = ax3.imshow(model_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax3.set_title('🎯 Model Performance Heatmap', fontweight='bold')
        ax3.set_xticks(range(len(metrics)))
        ax3.set_xticklabels(metrics)
        ax3.set_yticks(range(len(models)))
        ax3.set_yticklabels([m.replace(' ', '\\n') for m in models])
        
        # Add colorbar
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label('Performance Score\\n(Green = Better)', rotation=270, labelpad=20)
        
        # 4. Best Model Highlight
        best_model_data = self.model_results.get('best_model_performance', {})
        cv_r2 = best_model_data.get('cross_validation_r2', 0)
        test_r2 = best_model_data.get('test_r2', 0)
        
        categories = ['Cross-Validation', 'Test Set']
        scores = [cv_r2, test_r2]
        
        bars4 = ax4.bar(categories, scores, color=['#E67E22', '#27AE60'], alpha=0.8)
        ax4.set_title('🏆 Bayesian Ridge - Best Model Performance', fontweight='bold')
        ax4.set_ylabel('R² Score')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars4, scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.4f}\\n({score*100:.2f}%)', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.session_dir, 'model_analysis', 'model_performance_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"   ✅ Model performance plots saved: {save_path}")
    
    def create_portfolio_allocation_plots(self):
        """Create portfolio allocation visualizations"""
        if not self.recommendations_data:
            print("❌ No recommendation data found")
            return
        
        # Analyze latest recommendation
        latest_rec = self.recommendations_data[-1]
        allocation = latest_rec['portfolio_allocation']
        user_prefs = latest_rec['user_preferences']
        
        # Prepare data
        stocks = [item['stock'] for item in allocation]
        weights = [item['weight'] * 100 for item in allocation]
        sectors = [item['sector'] for item in allocation]
        investments = [item['investment'] for item in allocation]
        returns = [item['expected_return'] * 100 for item in allocation]
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'💼 Portfolio Analysis - {user_prefs["risk_profile"]} Investor (${user_prefs["investment_amount"]:,.0f})', 
                    fontsize=16, fontweight='bold')
        
        # 1. Portfolio Allocation Pie Chart
        colors = [self.color_palette.get(sector, '#95A5A6') for sector in sectors]
        wedges, texts, autotexts = ax1.pie(weights, labels=stocks, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('🥧 Stock Allocation Distribution', fontweight='bold')
        
        # Make percentage labels bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        # 2. Sector Allocation
        sector_weights = {}
        for item in allocation:
            sector = item['sector']
            if sector in sector_weights:
                sector_weights[sector] += item['weight'] * 100
            else:
                sector_weights[sector] = item['weight'] * 100
        
        sector_names = list(sector_weights.keys())
        sector_values = list(sector_weights.values())
        sector_colors = [self.color_palette.get(sector, '#95A5A6') for sector in sector_names]
        
        bars2 = ax2.bar(sector_names, sector_values, color=sector_colors, alpha=0.8)
        ax2.set_title('🏢 Sector Allocation', fontweight='bold')
        ax2.set_ylabel('Allocation (%)')
        ax2.set_xlabel('Sectors')
        
        # Add value labels
        for bar, value in zip(bars2, sector_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Investment Amount by Stock
        bars3 = ax3.bar(stocks, [inv/1000 for inv in investments], 
                       color=[self.color_palette.get(sector, '#95A5A6') for sector in sectors], alpha=0.8)
        ax3.set_title('💰 Investment Distribution', fontweight='bold')
        ax3.set_ylabel('Investment Amount ($1000s)')
        ax3.set_xlabel('Stocks')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, inv in zip(bars3, investments):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(investments)/1000*0.01,
                    f'${inv/1000:.0f}k', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 4. Expected Returns vs Risk
        volatilities = [item['volatility'] * 100 for item in allocation]
        
        scatter = ax4.scatter(volatilities, returns, s=[w*20 for w in weights], 
                             c=[self.color_palette.get(sector, '#95A5A6') for sector in sectors], 
                             alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add stock labels
        for i, stock in enumerate(stocks):
            ax4.annotate(stock, (volatilities[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold', fontsize=9)
        
        ax4.set_title('📈 Risk vs Return Profile', fontweight='bold')
        ax4.set_xlabel('Volatility (%)')
        ax4.set_ylabel('Expected Return (%)')
        ax4.grid(True, alpha=0.3)
        
        # Add legend for sectors
        sector_legend = []
        for sector in set(sectors):
            sector_legend.append(plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=self.color_palette.get(sector, '#95A5A6'), 
                                          markersize=10, label=sector))
        ax4.legend(handles=sector_legend, title='Sectors', loc='upper left')
        
        plt.tight_layout()
        save_path = os.path.join(self.session_dir, 'visualizations', 'portfolio_allocation_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"   ✅ Portfolio allocation plots saved: {save_path}")
    
    def create_recommendation_comparison(self):
        """Compare multiple portfolio recommendations if available"""
        if len(self.recommendations_data) < 2:
            print("⚠️ Need at least 2 recommendations for comparison")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🔍 Portfolio Recommendations Comparison', fontsize=16, fontweight='bold')
        
        # Prepare comparison data
        rec_names = []
        total_returns = []
        volatilities = []
        sharpe_ratios = []
        risk_profiles = []
        
        for i, rec in enumerate(self.recommendations_data):
            rec_names.append(f"Rec {i+1}")
            
            # Calculate portfolio metrics
            allocations = rec['portfolio_allocation']
            weights = [item['weight'] for item in allocations]
            returns = [item['expected_return'] for item in allocations]
            vols = [item['volatility'] for item in allocations]
            
            portfolio_return = sum(w * r for w, r in zip(weights, returns))
            portfolio_vol = np.sqrt(sum((w * v)**2 for w, v in zip(weights, vols)))
            sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            
            total_returns.append(portfolio_return * 100)
            volatilities.append(portfolio_vol * 100)
            sharpe_ratios.append(sharpe)
            risk_profiles.append(rec['user_preferences']['risk_profile'])
        
        # 1. Expected Returns Comparison
        colors = [self.color_palette.get(profile, '#95A5A6') for profile in risk_profiles]
        bars1 = ax1.bar(rec_names, total_returns, color=colors, alpha=0.8)
        ax1.set_title('📊 Expected Portfolio Returns', fontweight='bold')
        ax1.set_ylabel('Expected Return (%)')
        
        for bar, ret, profile in zip(bars1, total_returns, risk_profiles):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(total_returns)*0.01,
                    f'{ret:.1f}%\\n{profile}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 2. Risk Comparison
        bars2 = ax2.bar(rec_names, volatilities, color=colors, alpha=0.8)
        ax2.set_title('📈 Portfolio Volatility (Risk)', fontweight='bold')
        ax2.set_ylabel('Volatility (%)')
        
        for bar, vol in zip(bars2, volatilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(volatilities)*0.01,
                    f'{vol:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Sharpe Ratio
        bars3 = ax3.bar(rec_names, sharpe_ratios, color='#3498DB', alpha=0.8)
        ax3.set_title('🎯 Risk-Adjusted Returns (Sharpe Ratio)', fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        
        for bar, sharpe in zip(bars3, sharpe_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(sharpe_ratios)*0.01,
                    f'{sharpe:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Risk vs Return Scatter
        ax4.scatter(volatilities, total_returns, s=200, c=colors, alpha=0.8, edgecolors='black', linewidth=2)
        
        for i, (vol, ret, name, profile) in enumerate(zip(volatilities, total_returns, rec_names, risk_profiles)):
            ax4.annotate(f'{name}\\n({profile})', (vol, ret), 
                        xytext=(10, 10), textcoords='offset points', fontweight='bold', fontsize=9)
        
        ax4.set_title('🎪 Risk vs Return Positioning', fontweight='bold')
        ax4.set_xlabel('Portfolio Volatility (%)')
        ax4.set_ylabel('Expected Return (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.session_dir, 'visualizations', 'recommendation_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"   ✅ Recommendation comparison saved: {save_path}")
    
    def create_performance_summary_report(self):
        """Create a comprehensive summary report"""
        if not self.recommendations_data or not self.model_results:
            print("❌ Insufficient data for summary report")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📋 Portfolio ML System - Performance Summary Report', fontsize=16, fontweight='bold')
        
        # 1. Model Accuracy Timeline
        best_perf = self.model_results.get('best_model_performance', {})
        cv_r2 = best_perf.get('cross_validation_r2', 0) * 100
        test_r2 = best_perf.get('test_r2', 0) * 100
        
        phases = ['Cross\\nValidation', 'Test Set']
        accuracies = [cv_r2, test_r2]
        colors = ['#E67E22', '#27AE60']
        
        bars1 = ax1.bar(phases, accuracies, color=colors, alpha=0.8)
        ax1.set_title('🎯 Bayesian Ridge Model - R² Performance', fontweight='bold')
        ax1.set_ylabel('R² Score (%)')
        ax1.set_ylim(0, 100)
        
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # 2. Portfolio Diversification Analysis
        latest_rec = self.recommendations_data[-1]
        allocation = latest_rec['portfolio_allocation']
        
        sector_diversity = {}
        for item in allocation:
            sector = item['sector']
            if sector in sector_diversity:
                sector_diversity[sector] += 1
            else:
                sector_diversity[sector] = 1
        
        sectors = list(sector_diversity.keys())
        counts = list(sector_diversity.values())
        colors = [self.color_palette.get(sector, '#95A5A6') for sector in sectors]
        
        wedges, texts, autotexts = ax2.pie(counts, labels=sectors, autopct='%1.0f stocks', 
                                          colors=colors, startangle=90)
        ax2.set_title('🏢 Portfolio Sector Diversification', fontweight='bold')
        
        # 3. Investment Risk Assessment
        risk_profile = latest_rec['user_preferences']['risk_profile']
        portfolio_summary = latest_rec.get('portfolio_summary', {})
        
        expected_return = portfolio_summary.get('expected_return', 0) * 100
        volatility = portfolio_summary.get('volatility', 0) * 100
        sharpe_ratio = portfolio_summary.get('sharpe_ratio', 0)
        
        risk_metrics = ['Expected\\nReturn (%)', 'Volatility\\n(%)', 'Sharpe\\nRatio']
        risk_values = [expected_return, volatility, sharpe_ratio * 10]  # Scale Sharpe for visualization
        risk_colors = ['#27AE60', '#E74C3C', '#3498DB']
        
        bars3 = ax3.bar(risk_metrics, risk_values, color=risk_colors, alpha=0.8)
        ax3.set_title(f'📊 {risk_profile} Portfolio - Risk Metrics', fontweight='bold')
        ax3.set_ylabel('Metric Value')
        
        # Add actual values as labels
        actual_labels = [f'{expected_return:.1f}%', f'{volatility:.1f}%', f'{sharpe_ratio:.2f}']
        for bar, label in zip(bars3, actual_labels):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(risk_values)*0.01,
                    label, ha='center', va='bottom', fontweight='bold')
        
        # 4. System Performance Summary
        training_info = self.model_results.get('training_info', {})
        feature_count = training_info.get('feature_count', 0)
        best_model = training_info.get('best_model', 'Unknown')
        
        summary_data = {
            'Features\\nEngineered': feature_count,
            'Models\\nTested': len(self.model_results.get('all_models_comparison', {})),
            'Portfolio\\nRecommendations': len(self.recommendations_data),
            'Accuracy\\nAchieved (%)': test_r2
        }
        
        categories = list(summary_data.keys())
        values = list(summary_data.values())
        
        bars4 = ax4.bar(categories, values, color='#8E44AD', alpha=0.8)
        ax4.set_title(f'🚀 ML System Performance Summary\\nBest Model: {best_model}', fontweight='bold')
        ax4.set_ylabel('Count / Percentage')
        
        for bar, value, cat in zip(bars4, values, categories):
            height = bar.get_height()
            if 'Accuracy' in cat:
                label = f'{value:.1f}%'
            else:
                label = f'{int(value)}'
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    label, ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.session_dir, 'performance_metrics', 'performance_summary_report.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"   ✅ Performance summary report saved: {save_path}")
        print("   ✅ Performance summary report saved as 'performance_summary_report.png'")
        
        # Print text summary
        print("\\n" + "="*70)
        print("📋 PORTFOLIO ML SYSTEM - EXECUTIVE SUMMARY")
        print("="*70)
        print(f"🏆 Best Model: {best_model}")
        print(f"📊 Model Accuracy: {test_r2:.2f}% (R² Score)")
        print(f"🔧 Features Engineered: {feature_count}")
        print(f"🤖 Models Compared: {len(self.model_results.get('all_models_comparison', {}))}")
        print(f"💼 Portfolio Recommendations Generated: {len(self.recommendations_data)}")
        print(f"🎯 Risk Profile: {risk_profile}")
        print(f"💰 Investment Amount: ${latest_rec['user_preferences']['investment_amount']:,.0f}")
        print(f"📈 Expected Annual Return: {expected_return:.1f}%")
        print(f"📉 Portfolio Volatility: {volatility:.1f}%")
        print(f"⚖️ Sharpe Ratio: {sharpe_ratio:.2f}")
        print("="*70)
    
    def save_recommendation_data(self, recommendation_data):
        """Save recommendation data to the systematic structure"""
        rec_path = os.path.join(self.session_dir, 'recommendations', 
                               f'portfolio_recommendation_{self.session_timestamp}.json')
        
        with open(rec_path, 'w') as f:
            json.dump(recommendation_data, f, indent=2, default=str)
        
        print(f"💾 Recommendation data saved: {rec_path}")
        return rec_path
    
    def save_session_summary(self):
        """Save session summary with metadata"""
        summary = {
            'user_id': self.user_id,
            'session_timestamp': self.session_timestamp,
            'session_dir': self.session_dir,
            'total_recommendations': len(self.recommendations_data),
            'model_used': self.model_results.get('training_info', {}).get('best_model', 'Unknown'),
            'analysis_date': datetime.now().isoformat(),
            'files_generated': {
                'visualizations': [
                    'portfolio_allocation_analysis.png',
                    'recommendation_comparison.png'
                ],
                'model_analysis': [
                    'model_performance_analysis.png'
                ],
                'performance_metrics': [
                    'performance_summary_report.png'
                ]
            }
        }
        
        summary_path = os.path.join(self.session_dir, 'session_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Session summary saved: {summary_path}")
    
    def run_complete_analysis(self):
        """Run all visual analyses"""
        print(f"🚀 Starting Complete Portfolio Visual Analysis for User: {self.user_id}")
        print("="*70)
        
        self.load_data()
        
        if self.model_results:
            print("\\n📊 Creating model performance visualizations...")
            self.create_model_performance_plots()
        
        if self.recommendations_data:
            print("\\n💼 Creating portfolio allocation visualizations...")
            self.create_portfolio_allocation_plots()
            
            if len(self.recommendations_data) > 1:
                print("\\n🔍 Creating recommendation comparison...")
                self.create_recommendation_comparison()
        
        if self.model_results and self.recommendations_data:
            print("\\n📋 Creating performance summary report...")
            self.create_performance_summary_report()
        
        # Save session summary
        self.save_session_summary()
        
        print("\\n🎉 Visual analysis completed!")
        print(f"📁 All results saved in: {self.session_dir}")

def create_analysis_for_user(user_id, session_timestamp=None):
    """Create analysis for a specific user"""
    analyzer = PortfolioVisualAnalyzer(user_id=user_id, session_timestamp=session_timestamp)
    analyzer.run_complete_analysis()
    return analyzer

if __name__ == "__main__":
    # Example usage with different users
    print("Portfolio Visual Analysis System")
    print("="*50)
    
    # You can specify different user IDs for systematic organization
    user_id = input("Enter User ID (default: 'default_user'): ").strip()
    if not user_id:
        user_id = 'default_user'
    
    # Create analysis
    analyzer = create_analysis_for_user(user_id)
    
    print("\\n" + "="*50)
    print("📊 ANALYSIS COMPLETED!")
    print(f"📁 Results folder: {analyzer.session_dir}")
    print("="*50)
    analyzer = PortfolioVisualAnalyzer()
    analyzer.run_complete_analysis()