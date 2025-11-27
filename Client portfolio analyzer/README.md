# Portfolio ML Recommendation System

## 🎯 Quick Start Guide

### Option 1: From Any Directory (Recommended)
```bash
# From anywhere on your system:
"D:/Portfolio Managment/.venv/Scripts/python.exe" "d:/Portfolio Managment/Client portfolio analyzer/interactive_portfolio_recommendations.py"
```

### Option 2: From Project Directory  
```bash
# Navigate to project first:
cd "d:/Portfolio Managment/Client portfolio analyzer"

# Then run:
"D:/Portfolio Managment/.venv/Scripts/python.exe" interactive_portfolio_recommendations.py
```

### Option 3: Quick Test with Sample Data
```bash
cd "d:/Portfolio Managment/Client portfolio analyzer"
"D:/Portfolio Managment/.venv/Scripts/python.exe" quick_portfolio_test.py
```

## 📋 System Components

### 🔧 One-Time Setup (Already Done)
- **`train_best_model.py`** - Trains all 7 models and saves the best one
- **`create_model_analysis.py`** - Creates comprehensive visualizations

### 💼 Interactive Use
- **`interactive_portfolio_recommendations.py`** - Get personalized recommendations
- **`quick_portfolio_test.py`** - Test with pre-defined investor profiles

### 📊 Analysis & Results  
- **`model_analysis/`** - Model comparison charts and tables
- **`trained_models/`** - Best model (Bayesian Ridge) and metadata
- **`recommendations/`** - Individual portfolio recommendation reports

## 🎯 User Input Guide

When running the interactive system, you'll be prompted for:

1. **💰 Investment Amount**: e.g., `500000` (for $500,000)
2. **🎯 Risk Profile**: `Conservative`, `Moderate`, or `Aggressive` 
3. **📅 Time Horizon**: e.g., `5` (for 5 years)
4. **🏢 Preferred Sectors**: 
   - `All` for no preference
   - `Healthcare,Technology` for specific sectors
   - Available: `Aviation`, `Finance`, `Healthcare`, `Technology`
5. **🌱 ESG Preference**: `Yes` or `No`

## 📈 What You Get

✅ **Personalized Portfolio Allocation Table**
✅ **Performance Projections** (expected returns, volatility, Sharpe ratio)
✅ **Smart Recommendations** based on your risk profile
✅ **Saved Report** with timestamp in `recommendations/` folder

## 🏆 Model Information

- **Best Model**: Bayesian Ridge Regression
- **Accuracy**: 97.69% R² on test data
- **Training**: 5 years of data, 7 models compared
- **Validation**: 20 rolling windows cross-validation

## 🔍 Understanding Results

The system uses the proven best ML model to:
1. Analyze 24 companies across 4 sectors
2. Generate 90 portfolio features
3. Predict optimal allocations based on your preferences
4. Provide data-driven investment recommendations

## 📁 File Structure
```
Client portfolio analyzer/
├── interactive_portfolio_recommendations.py  # Main system
├── quick_portfolio_test.py                  # Quick testing
├── trained_models/                          # ML model files
│   ├── best_model.pkl                       # Bayesian Ridge model
│   └── model_metadata.json                 # Performance metrics
├── model_analysis/                          # Comparison charts
│   ├── model_ranking_dashboard.png         # Why Bayesian Ridge won
│   └── selection_criteria_summary.csv      # Selection explanation
└── recommendations/                         # Your portfolio reports
    └── portfolio_recommendation_*.json     # Timestamped reports
```

## 🚀 Ready to Use!

The system is fully trained and ready. Just run the interactive script and follow the prompts to get your personalized portfolio recommendation!