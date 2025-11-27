import pandas as pd
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

sns.set_theme()
pd.set_option('display.max_columns', None)


def load_and_clean_data():
    tickers = [
        'AAL', 'ALGT', 'ALK', 'DAL', 'HA', 'LUV',
        'BCS', 'CS', 'DB', 'GS', 'MS', 'WFC',
        'BHC', 'JNJ', 'MRK', 'PFE', 'RHHBY', 'UNH',
        'AAPL', 'AMZN', 'FB', 'GOOG', 'IBM', 'MSFT',
        'SP500'
    ]

    dfs = {}
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset")
    
    for ticker in tickers:
        csv_path = os.path.join(dataset_path, f"{ticker}.csv")
        df = pd.read_csv(csv_path)
        df.dropna(how='all', inplace=True)
        df = df[['Date', 'Close', 'Volume']]
        df.columns = [col if col == 'Date' else f"{col}_{ticker}" for col in df.columns]
        dfs[ticker] = df

    return dfs


def merge_sector_data(dfs):
    # Aviation
    aviation = dfs['AAL'].merge(dfs['ALGT'], on='Date').merge(dfs['ALK'], on='Date') \
        .merge(dfs['DAL'], on='Date').merge(dfs['HA'], on='Date').merge(dfs['LUV'], on='Date')

    # Finance
    finance = dfs['BCS'].merge(dfs['CS'], on='Date').merge(dfs['DB'], on='Date') \
        .merge(dfs['GS'], on='Date').merge(dfs['MS'], on='Date').merge(dfs['WFC'], on='Date')

    # Healthcare
    healthcare = dfs['BHC'].merge(dfs['JNJ'], on='Date').merge(dfs['MRK'], on='Date') \
        .merge(dfs['PFE'], on='Date').merge(dfs['RHHBY'], on='Date').merge(dfs['UNH'], on='Date')

    # Technology
    tech = dfs['AAPL'].merge(dfs['AMZN'], on='Date').merge(dfs['FB'], on='Date') \
        .merge(dfs['GOOG'], on='Date').merge(dfs['IBM'], on='Date').merge(dfs['MSFT'], on='Date')

    # Merge all with SP500
    all_stocks = aviation.merge(finance, on='Date') \
        .merge(healthcare, on='Date') \
        .merge(tech, on='Date') \
        .merge(dfs['SP500'], on='Date')

    return all_stocks


def filter_last_five_years(df):
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
    df = df[(df['Date'] >= '2015-10-01') & (df['Date'] <= '2020-09-30')]
    return df


def preprocess_all():
    dfs = load_and_clean_data()
    all_stocks = merge_sector_data(dfs)
    all_stocks = filter_last_five_years(all_stocks)

    print("✅ Final shape:", all_stocks.shape)
    print("✅ No missing data" if all_stocks.isnull().sum().sum() == 0 else "❌ Missing values found")
    print(all_stocks.info())

    return all_stocks
