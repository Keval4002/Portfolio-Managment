"""
Portfolio-Level Feature Engineering and ML Pipeline
Uses all companies' data together to predict portfolio performance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_portfolio_features(df):
    """
    Create portfolio-level features using all companies' data
    Target: Predict overall portfolio performance
    """
    print("📊 Creating portfolio-level features from all companies...")
    
    # Get all stock columns
    stock_tickers = ['AAL', 'ALGT', 'ALK', 'DAL', 'HA', 'LUV',  # Aviation
                    'BCS', 'CS', 'DB', 'GS', 'MS', 'WFC',      # Finance  
                    'BHC', 'JNJ', 'MRK', 'PFE', 'RHHBY', 'UNH', # Healthcare
                    'AAPL', 'AMZN', 'FB', 'GOOG', 'IBM', 'MSFT'] # Technology
    
    close_cols = [f'Close_{ticker}' for ticker in stock_tickers]
    volume_cols = [f'Volume_{ticker}' for ticker in stock_tickers]
    
    # Create feature dataframe
    features = pd.DataFrame()
    features['Date'] = df['Date']
    
    # === PORTFOLIO AGGREGATED FEATURES ===
    
    # Portfolio value metrics
    features['portfolio_total_value'] = df[close_cols].sum(axis=1)  # Total portfolio value
    features['portfolio_avg_price'] = df[close_cols].mean(axis=1)   # Average stock price
    features['portfolio_median_price'] = df[close_cols].median(axis=1)  # Median price
    features['portfolio_price_std'] = df[close_cols].std(axis=1)    # Price volatility across stocks
    
    # Target: Next day portfolio total value
    features['target'] = features['portfolio_total_value'].shift(-1)
    
    # Portfolio returns
    features['portfolio_return'] = features['portfolio_total_value'].pct_change()
    features['portfolio_return_lag1'] = features['portfolio_return'].shift(1)
    features['portfolio_return_lag2'] = features['portfolio_return'].shift(2)
    features['portfolio_return_lag3'] = features['portfolio_return'].shift(3)
    
    # Moving averages of portfolio value
    for window in [5, 10, 20]:
        features[f'portfolio_ma_{window}'] = features['portfolio_total_value'].rolling(window).mean()
        features[f'portfolio_to_ma_{window}'] = features['portfolio_total_value'] / features[f'portfolio_ma_{window}']
    
    # Portfolio volatility
    for window in [5, 10, 20]:
        features[f'portfolio_volatility_{window}d'] = features['portfolio_return'].rolling(window).std()
    
    # === VOLUME FEATURES ===
    
    features['portfolio_total_volume'] = df[volume_cols].sum(axis=1)
    features['portfolio_avg_volume'] = df[volume_cols].mean(axis=1)
    features['portfolio_volume_std'] = df[volume_cols].std(axis=1)
    
    # Volume moving averages and ratios
    for window in [5, 10, 20]:
        features[f'volume_ma_{window}'] = features['portfolio_total_volume'].rolling(window).mean()
        features[f'volume_ratio_{window}'] = features['portfolio_total_volume'] / features[f'volume_ma_{window}']
    
    # Volume change
    features['volume_change'] = features['portfolio_total_volume'].pct_change()
    features['volume_change_5d'] = features['volume_change'].rolling(5).mean()
    
    # === SECTOR-WISE FEATURES ===
    
    sectors = {
        'aviation': ['AAL', 'ALGT', 'ALK', 'DAL', 'HA', 'LUV'],
        'finance': ['BCS', 'CS', 'DB', 'GS', 'MS', 'WFC'], 
        'healthcare': ['BHC', 'JNJ', 'MRK', 'PFE', 'RHHBY', 'UNH'],
        'technology': ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM', 'MSFT']
    }
    
    for sector_name, sector_stocks in sectors.items():
        sector_close_cols = [f'Close_{stock}' for stock in sector_stocks]
        sector_volume_cols = [f'Volume_{stock}' for stock in sector_stocks]
        
        # Sector aggregations
        features[f'{sector_name}_total_value'] = df[sector_close_cols].sum(axis=1)
        features[f'{sector_name}_avg_price'] = df[sector_close_cols].mean(axis=1) 
        features[f'{sector_name}_price_std'] = df[sector_close_cols].std(axis=1)
        features[f'{sector_name}_total_volume'] = df[sector_volume_cols].sum(axis=1)
        
        # Sector returns
        features[f'{sector_name}_return'] = features[f'{sector_name}_total_value'].pct_change()
        features[f'{sector_name}_return_lag1'] = features[f'{sector_name}_return'].shift(1)
        
        # Sector moving averages
        features[f'{sector_name}_ma_5'] = features[f'{sector_name}_total_value'].rolling(5).mean()
        features[f'{sector_name}_ma_10'] = features[f'{sector_name}_total_value'].rolling(10).mean()
        
        # Sector volatility
        features[f'{sector_name}_volatility_5d'] = features[f'{sector_name}_return'].rolling(5).std()
        
    # === MARKET COMPARISON FEATURES ===
    
    if 'Close_SP500' in df.columns:
        features['sp500_price'] = df['Close_SP500']
        features['sp500_return'] = df['Close_SP500'].pct_change()
        features['sp500_volume'] = df['Volume_SP500']
        
        # Portfolio vs Market
        features['portfolio_vs_sp500_return'] = features['portfolio_return'] - features['sp500_return']
        features['portfolio_vs_sp500_value'] = features['portfolio_total_value'] / df['Close_SP500']
        
        # Beta calculation (30-day rolling)
        features['portfolio_beta'] = features['portfolio_return'].rolling(30).cov(features['sp500_return']) / features['sp500_return'].rolling(30).var()
        features['portfolio_correlation'] = features['portfolio_return'].rolling(30).corr(features['sp500_return'])
    
    # === CROSS-SECTOR CORRELATIONS ===
    
    # Calculate rolling correlations between sectors
    sector_returns = {}
    for sector in sectors.keys():
        sector_returns[sector] = features[f'{sector}_return']
    
    # Correlation features (simplified)
    features['tech_finance_correlation'] = sector_returns['technology'].rolling(20).corr(sector_returns['finance'])
    features['healthcare_aviation_correlation'] = sector_returns['healthcare'].rolling(20).corr(sector_returns['aviation'])
    
    # === TECHNICAL INDICATORS (Portfolio Level) ===
    
    # RSI for portfolio
    delta = features['portfolio_total_value'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['portfolio_rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands for portfolio
    bb_window = 20
    bb_std = 2
    bb_middle = features['portfolio_total_value'].rolling(bb_window).mean()
    bb_std_dev = features['portfolio_total_value'].rolling(bb_window).std()
    features['portfolio_bb_upper'] = bb_middle + (bb_std_dev * bb_std)
    features['portfolio_bb_lower'] = bb_middle - (bb_std_dev * bb_std)
    features['portfolio_bb_position'] = (features['portfolio_total_value'] - features['portfolio_bb_lower']) / (features['portfolio_bb_upper'] - features['portfolio_bb_lower'])
    
    # MACD for portfolio
    ema_12 = features['portfolio_total_value'].ewm(span=12).mean()
    ema_26 = features['portfolio_total_value'].ewm(span=26).mean()
    features['portfolio_macd'] = ema_12 - ema_26
    features['portfolio_macd_signal'] = features['portfolio_macd'].ewm(span=9).mean()
    
    # === MOMENTUM FEATURES ===
    
    # Portfolio momentum
    for window in [5, 10, 20]:
        features[f'portfolio_momentum_{window}d'] = features['portfolio_total_value'] / features['portfolio_total_value'].shift(window) - 1
    
    # Sector momentum relative to portfolio
    for sector in sectors.keys():
        features[f'{sector}_relative_momentum'] = features[f'{sector}_return'] - features['portfolio_return']
    
    # === DIVERSITY FEATURES ===
    
    # Calculate how spread out the stocks are (diversity indicators)
    features['price_range'] = df[close_cols].max(axis=1) - df[close_cols].min(axis=1)
    features['price_range_normalized'] = features['price_range'] / features['portfolio_avg_price']
    
    # Count of stocks above/below average
    portfolio_avg = df[close_cols].mean(axis=1)
    features['stocks_above_avg'] = (df[close_cols].T > portfolio_avg).T.sum(axis=1)
    features['stocks_below_avg'] = 24 - features['stocks_above_avg']  # Total 24 stocks
    
    # Remove rows with NaN values
    features = features.dropna()
    
    print(f"✅ Portfolio features created: {features.shape[1]-2} features, {len(features)} samples")
    print(f"📅 Date range: {features['Date'].min()} to {features['Date'].max()}")
    print(f"💰 Target (portfolio value) range: ${features['target'].min():,.2f} - ${features['target'].max():,.2f}")
    
    return features

def prepare_portfolio_ml_data(features_df):
    """
    Prepare features and target for ML models
    """
    # Remove non-feature columns
    exclude_cols = ['Date', 'target', 'portfolio_total_value']  # Exclude current portfolio value to avoid data leakage
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_cols]
    y = features_df['target']  # Next day portfolio value
    dates = features_df['Date']
    
    # Remove any remaining NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    dates = dates[mask]
    
    print(f"🔧 ML data prepared:")
    print(f"   Features: {len(feature_cols)} columns")
    print(f"   Samples: {len(X)} rows")
    print(f"   Target: Next day portfolio value")
    
    return X, y, dates, feature_cols

def scale_features(X_train, X_test, feature_cols):
    """
    Scale features using StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )
    
    return X_train_scaled, X_test_scaled, scaler