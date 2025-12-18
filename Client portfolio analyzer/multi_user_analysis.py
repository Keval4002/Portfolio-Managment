"""
Multi-User Portfolio Analysis System
====================================
Generate portfolio recommendations for multiple users and create systematic visual analysis.
"""

import json
import os
from datetime import datetime
from interactive_portfolio_recommendations import PortfolioRecommendationSystem
from portfolio_visual_analysis import PortfolioVisualAnalyzer

def create_sample_users():
    """Create sample user profiles for testing"""
    sample_users = {
        'PatrickJyengar': {
            'name': 'Patrick Jyengar',
            'investment_amount': 1000000,
            'risk_profile': 'Conservative',
            'time_horizon': '5+ years',
            'preferred_sectors': ['Healthcare', 'Technology'],
            'esg_preference': True
        },
        'PeterJyengar': {
            'name': 'Peter Jyengar', 
            'investment_amount': 1000000,
            'risk_profile': 'Aggressive',
            'time_horizon': '2-5 years',
            'preferred_sectors': ['Technology', 'Finance', 'Healthcare', 'Aviation'],
            'esg_preference': False
        },
        'ConservativeInvestor': {
            'name': 'Sarah Miller',
            'investment_amount': 500000,
            'risk_profile': 'Conservative',
            'time_horizon': '10+ years',
            'preferred_sectors': ['Healthcare', 'Technology'],
            'esg_preference': True
        },
        'ModerateInvestor': {
            'name': 'John Smith',
            'investment_amount': 750000,
            'risk_profile': 'Moderate',
            'time_horizon': '5-10 years', 
            'preferred_sectors': ['Technology', 'Finance'],
            'esg_preference': False
        },
        'AggressiveInvestor': {
            'name': 'Mike Johnson',
            'investment_amount': 1500000,
            'risk_profile': 'Aggressive',
            'time_horizon': '2-5 years',
            'preferred_sectors': ['Technology', 'Finance', 'Aviation'],
            'esg_preference': False
        }
    }
    return sample_users

def generate_recommendation_for_user(user_id, user_profile, recommendation_system):
    """Generate portfolio recommendation for a specific user"""
    print(f"\n🎯 Generating recommendation for {user_profile['name']} (ID: {user_id})")
    print("-" * 60)
    
    # Simulate user input
    print(f"💰 Investment Amount: ${user_profile['investment_amount']:,}")
    print(f"⚖️ Risk Profile: {user_profile['risk_profile']}")
    print(f"⏰ Time Horizon: {user_profile['time_horizon']}")
    print(f"🏭 Preferred Sectors: {', '.join(user_profile['preferred_sectors'])}")
    print(f"🌱 ESG Preference: {'Yes' if user_profile['esg_preference'] else 'No'}")
    
    # Generate recommendation using the correct method
    try:
        # Create preferences object
        preferences = {
            'investment_amount': user_profile['investment_amount'],
            'risk_profile': user_profile['risk_profile'],
            'time_horizon': user_profile['time_horizon'],
            'preferred_sectors': user_profile['preferred_sectors'],
            'esg_preference': user_profile['esg_preference']
        }
        
        # Generate portfolio recommendation
        portfolio = recommendation_system.generate_portfolio_recommendation(preferences)
        
        if portfolio:
            # Create complete recommendation object
            recommendation = {
                'user_preferences': preferences,
                'portfolio_allocation': portfolio,
                'timestamp': datetime.now().isoformat()
            }
            print("✅ Recommendation generated successfully!")
            return recommendation
        else:
            print("❌ Failed to generate recommendation")
            return None
            
    except Exception as e:
        print(f"❌ Error generating recommendation: {str(e)}")
        return None

def run_multi_user_analysis():
    """Run complete multi-user analysis"""
    print("🚀 MULTI-USER PORTFOLIO ANALYSIS SYSTEM")
    print("="*70)
    
    # Initialize recommendation system
    print("🔧 Initializing portfolio recommendation system...")
    rec_system = PortfolioRecommendationSystem()
    
    # Get sample users
    sample_users = create_sample_users()
    
    # Generate timestamp for this batch
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"👥 Processing {len(sample_users)} users...")
    
    # Generate recommendations and create analyses for each user
    for user_id, user_profile in sample_users.items():
        try:
            # Generate recommendation
            recommendation = generate_recommendation_for_user(user_id, user_profile, rec_system)
            
            if recommendation:
                # Add user metadata to recommendation
                recommendation['user_profile'] = user_profile
                recommendation['user_id'] = user_id
                recommendation['batch_timestamp'] = batch_timestamp
                
                # Create visual analysis for this user
                print(f"📊 Creating visual analysis for {user_profile['name']}...")
                session_timestamp = f"{batch_timestamp}_{user_id}"
                
                # Initialize analyzer for this user
                analyzer = PortfolioVisualAnalyzer(user_id=user_id, session_timestamp=session_timestamp)
                
                # Save recommendation data to the systematic structure
                analyzer.save_recommendation_data(recommendation)
                
                # Load the data and run analysis
                analyzer.load_data()
                
                # Create visualizations if data is available
                if analyzer.model_results:
                    analyzer.create_model_performance_plots()
                
                if analyzer.recommendations_data:
                    analyzer.create_portfolio_allocation_plots()
                    analyzer.create_performance_summary_report()
                
                # Save session summary
                analyzer.save_session_summary()
                
                print(f"✅ Analysis completed for {user_profile['name']}")
                print(f"📁 Results saved in: {analyzer.session_dir}")
                
            else:
                print(f"⚠️ Skipping analysis for {user_profile['name']} - no recommendation generated")
                
        except Exception as e:
            print(f"❌ Error processing user {user_id}: {str(e)}")
            continue
    
    # Create comparative analysis across all users
    print(f"\n🔍 Creating comparative analysis across all users...")
    create_comparative_analysis(batch_timestamp)
    
    print("\n" + "="*70)
    print("🎉 MULTI-USER ANALYSIS COMPLETED!")
    print(f"📁 Results organized in: ../Portfolio_Analysis_Results/")
    print(f"⏰ Batch timestamp: {batch_timestamp}")
    print("="*70)

def create_comparative_analysis(batch_timestamp):
    """Create comparative analysis across all users in the batch"""
    try:
        # Create comparative analysis directory
        comp_dir = os.path.join("..", "Portfolio_Analysis_Results", f"Comparative_Analysis_{batch_timestamp}")
        os.makedirs(comp_dir, exist_ok=True)
        
        print(f"📊 Comparative analysis directory: {comp_dir}")
        
        # Here you could add code to:
        # 1. Load all recommendations from the batch
        # 2. Create comparison charts
        # 3. Generate batch summary report
        
        print("✅ Comparative analysis framework created")
        
    except Exception as e:
        print(f"❌ Error creating comparative analysis: {str(e)}")

if __name__ == "__main__":
    print("Multi-User Portfolio Analysis")
    print("="*50)
    
    choice = input("Run analysis for:\n1. All sample users\n2. Single user\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_multi_user_analysis()
    elif choice == "2":
        user_id = input("Enter User ID: ").strip()
        if not user_id:
            user_id = 'single_user'
        
        analyzer = PortfolioVisualAnalyzer(user_id=user_id)
        analyzer.run_complete_analysis()
    else:
        print("Invalid choice. Running default analysis...")
        analyzer = PortfolioVisualAnalyzer()
        analyzer.run_complete_analysis()