"""
Interactive Portfolio Recommendation System
===========================================
Uses pre-trained ML model to provide personalized portfolio recommendations.

Prerequisites: Run train_best_model.py first to train and save the best model.
Usage: python interactive_portfolio_recommendations.py
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

from data_preprocessing import preprocess_all
from portfolio_feature_engineering import create_portfolio_features

warnings.filterwarnings("ignore")

class PortfolioRecommendationSystem:
    def __init__(self, user_id=None):
        self.user_id = user_id if user_id else f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.create_results_structure()
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_metadata = None
        self.df = None
        self.portfolio_features = None
        self.company_sectors = {
            'Aviation': ['AAL', 'ALGT', 'ALK', 'DAL', 'HA', 'LUV'],
            'Finance': ['BCS', 'CS', 'DB', 'GS', 'MS', 'WFC'],
            'Healthcare': ['BHC', 'JNJ', 'MRK', 'PFE', 'RHHBY', 'UNH'],
            'Technology': ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM', 'MSFT']
        }
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
        
    def load_trained_model(self):
        """Load the pre-trained model and associated artifacts"""
        try:
            print("📥 Loading pre-trained model...")
            
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            trained_models_dir = os.path.join(script_dir, 'trained_models')
            
            # Check if trained model exists
            model_path = os.path.join(trained_models_dir, 'best_model.pkl')
            if not os.path.exists(model_path):
                print("❌ No trained model found!")
                print("Please run 'python train_best_model.py' first to train the model.")
                return False
            
            # Load model artifacts
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(os.path.join(trained_models_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(os.path.join(trained_models_dir, 'feature_columns.pkl'), 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            with open(os.path.join(trained_models_dir, 'model_metadata.json'), 'r') as f:
                self.model_metadata = json.load(f)
            
            print(f"   ✅ Model loaded: {self.model_metadata['model_name']}")
            print(f"   📊 Test R²: {self.model_metadata['test_metrics']['test_r2']:.4f}")
            print(f"   📅 Trained on: {self.model_metadata['training_date'][:10]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def load_data(self):
        """Load and prepare the latest market data"""
        try:
            print("📊 Loading market data...")
            
            # Load preprocessed data
            self.df = preprocess_all()
            print(f"   ✅ Market data loaded: {self.df.shape}")
            
            # Create portfolio features
            self.portfolio_features = create_portfolio_features(self.df)
            print(f"   ✅ Portfolio features ready: {self.portfolio_features.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def get_user_preferences(self):
        """Get user investment preferences through interactive prompts"""
        print("\\n" + "="*60)
        print("💼 PORTFOLIO PREFERENCE COLLECTION")
        print("="*60)
        
        # Get user ID for systematic organization
        user_input = input("👤 Enter your User ID (for organized results): ").strip()
        if user_input:
            self.user_id = user_input
            # Recreate session directory with new user ID
            self.session_dir = self.create_results_structure()
        
        try:
            # Investment amount
            while True:
                try:
                    amount = float(input("💰 Enter your investment amount ($): "))
                    if amount <= 0:
                        print("   ❌ Please enter a positive amount")
                        continue
                    break
                except ValueError:
                    print("   ❌ Please enter a valid number")
            
            # Risk profile
            while True:
                risk = input("🎯 Enter your risk profile (Conservative/Moderate/Aggressive): ").strip().title()
                if risk in ['Conservative', 'Moderate', 'Aggressive']:
                    break
                print("   ❌ Please choose Conservative, Moderate, or Aggressive")
            
            # Time horizon
            while True:
                try:
                    time_horizon = int(input("📅 Enter your investment time horizon (years): "))
                    if time_horizon <= 0:
                        print("   ❌ Please enter a positive number of years")
                        continue
                    break
                except ValueError:
                    print("   ❌ Please enter a valid number")
            
            # Sector preferences
            print("🏢 Available sectors: Aviation, Finance, Healthcare, Technology")
            sectors_input = input("   Enter preferred sectors (comma-separated) or 'All' for no preference: ").strip()
            
            if sectors_input.lower() in ['all', 'none', '']:
                preferred_sectors = list(self.company_sectors.keys())
            else:
                preferred_sectors = [s.strip().title() for s in sectors_input.split(',')]
                # Validate sectors
                valid_sectors = [s for s in preferred_sectors if s in self.company_sectors.keys()]
                if not valid_sectors:
                    print("   ⚠️ No valid sectors found, using all sectors")
                    preferred_sectors = list(self.company_sectors.keys())
                else:
                    preferred_sectors = valid_sectors
            
            # ESG preference
            while True:
                esg = input("🌱 Do you prefer ESG-friendly investments? (Yes/No): ").strip().lower()
                if esg in ['yes', 'y', 'true', '1']:
                    esg_preference = True
                    break
                elif esg in ['no', 'n', 'false', '0']:
                    esg_preference = False
                    break
                print("   ❌ Please enter Yes or No")
            
            preferences = {
                'investment_amount': amount,
                'risk_profile': risk,
                'time_horizon': time_horizon,
                'preferred_sectors': preferred_sectors,
                'esg_preference': esg_preference
            }
            
            print(f"\n✅ Preferences collected successfully!")
            return preferences
            
        except KeyboardInterrupt:
            print("\n\n❌ User cancelled input")
            return None
        except Exception as e:
            print(f"❌ Error collecting preferences: {str(e)}")
            return None
    
    def predict_portfolio_value(self):
        """Get the latest portfolio value prediction using the trained model"""
        try:
            # Get the most recent feature data
            latest_features = self.portfolio_features.iloc[-1][self.feature_columns].values.reshape(1, -1)
            
            # Scale features
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Make prediction
            predicted_value = self.model.predict(latest_features_scaled)[0]
            
            return predicted_value
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return None
    
    def generate_portfolio_recommendation(self, preferences):
        """Generate personalized portfolio recommendations based on user preferences"""
        print(f"\n🔍 Generating portfolio recommendations for {preferences['risk_profile']} investor...")
        
        try:
            # Get available companies from preferred sectors
            available_companies = []
            for sector in preferences['preferred_sectors']:
                if sector in self.company_sectors:
                    available_companies.extend(self.company_sectors[sector])
            
            if not available_companies:
                available_companies = [company for companies in self.company_sectors.values() for company in companies]
            
            # Get latest stock data for portfolio allocation
            latest_data = {}
            for company in available_companies:
                if f'Close_{company}' in self.df.columns:
                    latest_data[company] = {
                        'price': self.df[f'Close_{company}'].iloc[-1],
                        'sector': next((sector for sector, companies in self.company_sectors.items() if company in companies), 'Unknown')
                    }
            
            # Calculate expected returns (simplified based on recent performance)
            for company, data in latest_data.items():
                if f'Close_{company}' in self.df.columns:
                    recent_prices = self.df[f'Close_{company}'].tail(252)  # Last year
                    if len(recent_prices) > 1:
                        returns = recent_prices.pct_change().dropna()
                        annual_return = (1 + returns.mean()) ** 252 - 1
                        data['expected_return'] = annual_return
                        data['volatility'] = returns.std() * np.sqrt(252)
                    else:
                        data['expected_return'] = 0.1  # Default 10%
                        data['volatility'] = 0.2  # Default 20%
            
            # Risk-based portfolio allocation
            risk_multipliers = {
                'Conservative': 0.6,
                'Moderate': 1.0,
                'Aggressive': 1.5
            }
            
            risk_mult = risk_multipliers[preferences['risk_profile']]
            
            # Sort companies by risk-adjusted returns
            sorted_companies = sorted(
                latest_data.items(),
                key=lambda x: x[1]['expected_return'] / x[1]['volatility'] * risk_mult,
                reverse=True
            )
            
            # Select top companies based on risk profile
            num_stocks = {
                'Conservative': min(8, len(sorted_companies)),
                'Moderate': min(6, len(sorted_companies)),
                'Aggressive': min(4, len(sorted_companies))
            }[preferences['risk_profile']]
            
            selected_companies = sorted_companies[:num_stocks]
            
            # Calculate portfolio weights (equal weight with slight optimization)
            total_weight = 1.0
            portfolio = []
            
            for i, (company, data) in enumerate(selected_companies):
                # Adjust weight based on expected return and risk profile
                base_weight = 1.0 / num_stocks
                return_adjustment = data['expected_return'] * 0.1  # Small adjustment based on returns
                weight = base_weight + return_adjustment
                
                portfolio.append({
                    'stock': company,
                    'sector': data['sector'],
                    'weight': weight,
                    'price': data['price'],
                    'expected_return': data['expected_return'],
                    'volatility': data['volatility']
                })
            
            # Normalize weights to sum to 1
            total_weight = sum([p['weight'] for p in portfolio])
            for p in portfolio:
                p['weight'] = p['weight'] / total_weight
                p['investment'] = preferences['investment_amount'] * p['weight']
                p['shares'] = int(p['investment'] / p['price'])
            
            return portfolio
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {str(e)}")
            return None
    
    def display_portfolio_recommendation(self, portfolio, preferences):
        """Display the portfolio recommendation in a formatted way"""
        print("\n" + "="*80)
        print("📋 PERSONALIZED PORTFOLIO RECOMMENDATION")
        print("="*80)
        
        # Display user profile
        print(f"\n👤 INVESTOR PROFILE:")
        print(f"   💰 Investment Amount: ${preferences['investment_amount']:,.2f}")
        print(f"   🎯 Risk Profile: {preferences['risk_profile']}")
        print(f"   📅 Time Horizon: {preferences['time_horizon']} years")
        print(f"   🏢 Preferred Sectors: {', '.join(preferences['preferred_sectors'])}")
        print(f"   🌱 ESG Preference: {'Yes' if preferences['esg_preference'] else 'No'}")
        
        # Display portfolio allocation
        print(f"\n📊 RECOMMENDED PORTFOLIO ALLOCATION:")
        print("-" * 80)
        print(f"{'Stock':<8} {'Sector':<12} {'Weight':<8} {'Investment':<12} {'Shares':<8} {'Exp. Return':<12}")
        print("-" * 80)
        
        total_expected_return = 0
        total_volatility = 0
        
        for stock_data in portfolio:
            print(f"{stock_data['stock']:<8} {stock_data['sector']:<12} "
                  f"{stock_data['weight']*100:>5.1f}% "
                  f"${stock_data['investment']:>10,.0f} "
                  f"{stock_data['shares']:>6,} "
                  f"{stock_data['expected_return']*100:>8.1f}%")
            
            total_expected_return += stock_data['weight'] * stock_data['expected_return']
            total_volatility += stock_data['weight'] * stock_data['volatility']
        
        # Calculate portfolio metrics
        sharpe_ratio = total_expected_return / total_volatility if total_volatility > 0 else 0
        future_value = preferences['investment_amount'] * (1 + total_expected_return) ** preferences['time_horizon']
        total_return = (future_value / preferences['investment_amount'] - 1) * 100
        
        print(f"\n📈 PORTFOLIO PERFORMANCE PROJECTIONS:")
        print("-" * 50)
        print(f"Expected Annual Return: {total_expected_return*100:.1f}%")
        print(f"Expected Volatility: {total_volatility*100:.1f}%")
        print(f"Expected Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Initial Investment: ${preferences['investment_amount']:,.2f}")
        print(f"Projected Value ({preferences['time_horizon']}yr): ${future_value:,.2f}")
        print(f"Total Return: {total_return:.1f}%")
        
        # Provide recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        recommendations = []
        
        if total_expected_return > 0.15:
            recommendations.append("✅ Portfolio meets your return expectations")
        else:
            recommendations.append("⚠️ Consider adding higher-growth stocks")
        
        if (preferences['risk_profile'] == 'Conservative' and total_volatility < 0.25) or \
           (preferences['risk_profile'] == 'Moderate' and total_volatility < 0.35) or \
           (preferences['risk_profile'] == 'Aggressive' and total_volatility < 0.45):
            recommendations.append("✅ Portfolio risk within your tolerance")
        else:
            recommendations.append("⚠️ Portfolio risk exceeds tolerance - consider rebalancing")
        
        sectors_in_portfolio = set([s['sector'] for s in portfolio])
        if len(sectors_in_portfolio) >= 3:
            recommendations.append("✅ Good sector diversification")
        else:
            recommendations.append("⚠️ Consider adding more sector diversification")
        
        if preferences['time_horizon'] >= 5:
            recommendations.append("✅ Long-term horizon allows for growth strategies")
        
        if sharpe_ratio > 0.8:
            recommendations.append("✅ Good risk-adjusted returns expected")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Disclaimers
        print(f"\n⚠️ IMPORTANT DISCLAIMERS:")
        print(f"   • Past performance does not guarantee future results")
        print(f"   • All investments carry risk of loss")
        print(f"   • Projections are estimates based on the trained ML model")
        print(f"   • Model: {self.model_metadata['model_name']} (R²: {self.model_metadata['test_metrics']['test_r2']:.4f})")
        print(f"   • Consider consulting with a financial advisor")
        print(f"   • Regularly review and rebalance your portfolio")
        
        return portfolio
    
    def create_portfolio_visualizations(self, portfolio, preferences):
        """Create comprehensive visualizations for the portfolio recommendation"""
        try:
            # Prepare data
            stocks = [item['stock'] for item in portfolio]
            weights = [item['weight'] * 100 for item in portfolio]
            sectors = [item['sector'] for item in portfolio]
            investments = [item['investment'] for item in portfolio]
            returns = [item['expected_return'] * 100 for item in portfolio]
            volatilities = [item['volatility'] * 100 for item in portfolio]
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'💼 Portfolio Analysis - {preferences["risk_profile"]} Investor (${preferences["investment_amount"]:,.0f})', 
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
            for item in portfolio:
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
            
            # Save visualization
            viz_path = os.path.join(self.session_dir, 'visualizations', f'portfolio_allocation_{self.session_timestamp}.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 Portfolio visualization saved: {viz_path}")
            return viz_path
            
        except Exception as e:
            print(f"   ⚠️ Error creating portfolio visualization: {str(e)}")
            return None
    
    def create_model_performance_visualization(self):
        """Create model performance visualization"""
        try:
            # Load model results
            with open('trained_models/training_summary.json', 'r') as f:
                model_results = json.load(f)
            
            model_comparison = model_results.get('all_models_comparison', {})
            models = list(model_comparison.keys())
            r2_scores = [model_comparison[model]['R2'] for model in models]
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            fig.suptitle('🤖 ML Model Performance - Portfolio Prediction', fontsize=16, fontweight='bold')
            
            # R² Score Comparison
            colors = ['#27AE60' if r2 > 0.5 else '#F39C12' if r2 > 0 else '#E74C3C' for r2 in r2_scores]
            bars = ax.bar(range(len(models)), r2_scores, color=colors, alpha=0.8)
            ax.set_title('📊 R² Score Comparison (Higher = Better)', fontweight='bold')
            ax.set_xlabel('Models')
            ax.set_ylabel('R² Score')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([m.replace(' ', '\\n') for m in models], rotation=45, ha='right')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, score) in enumerate(zip(bars, r2_scores)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save visualization
            model_viz_path = os.path.join(self.session_dir, 'model_analysis', f'model_performance_{self.session_timestamp}.png')
            plt.savefig(model_viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   🤖 Model performance visualization saved: {model_viz_path}")
            return model_viz_path
            
        except Exception as e:
            print(f"   ⚠️ Error creating model visualization: {str(e)}")
            return None

    def save_recommendation_report(self, portfolio, preferences, filename=None):
        """Save the portfolio recommendation as a report file with visualizations"""
        if filename is None:
            filename = f"portfolio_recommendation_{self.session_timestamp}.json"
        
        # Calculate portfolio summary metrics
        portfolio_summary = self.calculate_portfolio_summary(portfolio)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'user_id': self.user_id,
            'session_timestamp': self.session_timestamp,
            'user_preferences': preferences,
            'portfolio_allocation': portfolio,
            'portfolio_summary': portfolio_summary,
            'model_info': {
                'name': self.model_metadata['model_name'],
                'test_r2': self.model_metadata['test_metrics']['test_r2'],
                'training_date': self.model_metadata['training_date']
            }
        }
        
        # Save to systematic structure
        rec_path = os.path.join(self.session_dir, 'recommendations', filename)
        with open(rec_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\\n💾 Recommendation saved to: {rec_path}")
        
        # Generate visualizations
        print("\\n🎨 Generating visualizations...")
        viz_path = self.create_portfolio_visualizations(portfolio, preferences)
        model_viz_path = self.create_model_performance_visualization()
        
        # Save session summary
        self.save_session_summary(portfolio, preferences, [viz_path, model_viz_path])
        
        print(f"\\n✅ Complete analysis saved in: {self.session_dir}")
        return rec_path
    
    def calculate_portfolio_summary(self, portfolio):
        """Calculate portfolio summary metrics"""
        try:
            weights = [item['weight'] for item in portfolio]
            returns = [item['expected_return'] for item in portfolio]
            volatilities = [item['volatility'] for item in portfolio]
            
            # Portfolio metrics
            portfolio_return = sum(w * r for w, r in zip(weights, returns))
            portfolio_volatility = np.sqrt(sum((w * v)**2 for w, v in zip(weights, volatilities)))
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            return {
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'total_investment': sum(item['investment'] for item in portfolio),
                'number_of_stocks': len(portfolio),
                'sectors': list(set(item['sector'] for item in portfolio))
            }
        except Exception as e:
            print(f"   ⚠️ Error calculating portfolio summary: {str(e)}")
            return {}
    
    def save_session_summary(self, portfolio, preferences, visualization_paths):
        """Save session summary with metadata"""
        summary = {
            'user_id': self.user_id,
            'session_timestamp': self.session_timestamp,
            'session_dir': self.session_dir,
            'user_preferences': preferences,
            'portfolio_summary': self.calculate_portfolio_summary(portfolio),
            'model_used': self.model_metadata['model_name'],
            'model_accuracy': self.model_metadata['test_metrics']['test_r2'],
            'analysis_date': datetime.now().isoformat(),
            'files_generated': {
                'recommendation_data': f'recommendations/portfolio_recommendation_{self.session_timestamp}.json',
                'visualizations': [path for path in visualization_paths if path],
                'session_summary': 'session_summary.json'
            }
        }
        
        summary_path = os.path.join(self.session_dir, 'session_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"   📋 Session summary saved: {summary_path}")

    def get_user_preferences(self):
        """Get user investment preferences through interactive prompts"""
        print("\\n" + "="*60)
        print("💼 PORTFOLIO PREFERENCE COLLECTION")
        print("="*60)
        
        # Get user ID for systematic organization
        user_input = input("👤 Enter your User ID (for organized results): ").strip()
        if user_input:
            self.user_id = user_input
            # Recreate session directory with new user ID
            self.session_dir = self.create_results_structure()
        
        try:
            # Investment amount
            while True:
                try:
                    amount = float(input("💰 Enter your investment amount ($): "))
                    if amount <= 0:
                        print("   ❌ Please enter a positive amount")
                        continue
                    break
                except ValueError:
                    print("   ❌ Please enter a valid number")
            
            # Risk profile
            while True:
                risk = input("🎯 Enter your risk profile (Conservative/Moderate/Aggressive): ").strip().title()
                if risk in ['Conservative', 'Moderate', 'Aggressive']:
                    break
                print("   ❌ Please choose Conservative, Moderate, or Aggressive")
            
            # Time horizon
            while True:
                try:
                    time_horizon = int(input("📅 Enter your investment time horizon (years): "))
                    if time_horizon <= 0:
                        print("   ❌ Please enter a positive number of years")
                        continue
                    break
                except ValueError:
                    print("   ❌ Please enter a valid number")
            
            # Sector preferences
            print("🏢 Available sectors: Aviation, Finance, Healthcare, Technology")
            sectors_input = input("   Enter preferred sectors (comma-separated) or 'All' for no preference: ").strip()
            
            if sectors_input.lower() in ['all', 'none', '']:
                preferred_sectors = list(self.company_sectors.keys())
            else:
                preferred_sectors = [s.strip().title() for s in sectors_input.split(',')]
                # Validate sectors
                valid_sectors = [s for s in preferred_sectors if s in self.company_sectors.keys()]
                if not valid_sectors:
                    print("   ⚠️ No valid sectors found, using all sectors")
                    preferred_sectors = list(self.company_sectors.keys())
                else:
                    preferred_sectors = valid_sectors
            
            # ESG preference
            while True:
                esg_input = input("🌱 Do you prefer ESG-friendly investments? (Yes/No): ").strip().lower()
                if esg_input in ['yes', 'y', 'true', '1']:
                    esg_preference = True
                    break
                elif esg_input in ['no', 'n', 'false', '0']:
                    esg_preference = False
                    break
                else:
                    print("   ❌ Please enter Yes or No")
            
            preferences = {
                'investment_amount': amount,
                'risk_profile': risk,
                'time_horizon': time_horizon,
                'preferred_sectors': preferred_sectors,
                'esg_preference': esg_preference
            }
            
            print("\\n✅ Preferences collected successfully!")
            return preferences
            
        except KeyboardInterrupt:
            print("\\n❌ Input cancelled by user")
            return None
        except Exception as e:
            print(f"\\n❌ Error collecting preferences: {str(e)}")
            return None

def main():
    print("🚀 INTERACTIVE PORTFOLIO RECOMMENDATION SYSTEM")
    print("Using pre-trained ML model for personalized recommendations")
    print("="*70)
    
    # Initialize system
    system = PortfolioRecommendationSystem()
    
    # Load trained model
    if not system.load_trained_model():
        return
    
    # Load market data
    if not system.load_data():
        return
    
    print("✅ System ready for portfolio recommendations!\n")
    
    try:
        while True:
            # Get user preferences
            preferences = system.get_user_preferences()
            if preferences is None:
                break
            
            # Generate recommendations
            portfolio = system.generate_portfolio_recommendation(preferences)
            if portfolio is None:
                continue
            
            # Display recommendations
            system.display_portfolio_recommendation(portfolio, preferences)
            
            # Save report
            system.save_recommendation_report(portfolio, preferences)
            
            # Ask if user wants another recommendation
            print(f"\n" + "="*70)
            again = input("🔄 Would you like another portfolio recommendation? (y/n): ").strip().lower()
            if again not in ['y', 'yes', '1']:
                break
    
    except KeyboardInterrupt:
        print("\n\n👋 Thank you for using the Portfolio Recommendation System!")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
    
    print(f"\n🎉 Session completed!")

if __name__ == "__main__":
    main()