""" ML trading bot experiment """

import pandas as pd
import requests
import lightgbm as lgb
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time
import config
# --- Configuration ---
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA"]
FROM_DATE = "2018-01-01"
TO_DATE = "2023-12-31"
API_KEY = config.key
LOOK_FORWARD_DAYS = 5

def fetch_and_prepare_data(tickers):
    """
    Fetches data and creates a single DataFrame with a MultiIndex (Ticker, Date),
    including RSI, MACD, and ATR as features.
    """
    print("Fetching and preparing data...")
    all_dfs = []

    for ticker in tickers:
        print(f"  Fetching data for {ticker}...")
        link = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{FROM_DATE}/{TO_DATE}?adjusted=true&sort=asc&apiKey={API_KEY}"
        )
        response = requests.get(link)
        if response.status_code != 200 or not response.json().get('results'):
            print(f"  Could not fetch valid data for {ticker}.")
            continue

        df = pd.DataFrame(response.json()['results'])[['c', 'h', 'l', 't']]
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df['ticker'] = ticker

        all_dfs.append(df)
        if len(tickers) > 1:
            time.sleep(12)

    if not all_dfs:
        raise ValueError("No data could be fetched for any ticker. Check API key and ticker symbols.")

    combined_df = pd.concat(all_dfs).set_index(['ticker', 't'])

    # --- Feature Engineering ---
    grouped = combined_df.groupby(level='ticker')
    
    combined_df['pivot'] = (combined_df['h'] + combined_df['l'] + combined_df['c']) / 3
    combined_df['r1'] = combined_df['pivot'] + (0.382 * (combined_df['h'] - combined_df['l']))
    combined_df['s1'] = combined_df['pivot'] - (0.382 * (combined_df['h'] - combined_df['l']))
    combined_df['ema_10'] = grouped['c'].ewm(span=10, adjust=False).mean().values

    # 1. RSI (Relative Strength Index)
    delta = grouped['c'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    combined_df['rsi'] = (100 - (100 / (1 + rs))).values

    # 2. MACD (Moving Average Convergence Divergence)
    ema_12 = grouped['c'].ewm(span=12, adjust=False).mean()
    ema_26 = grouped['c'].ewm(span=26, adjust=False).mean()
    combined_df['macd'] = (ema_12 - ema_26).values
    combined_df['macd_signal'] = combined_df.groupby(level='ticker')['macd'].ewm(span=9, adjust=False).mean().values

    # 3. ATR (Average True Range)
    high_low = combined_df['h'] - combined_df['l']
    high_close = abs(combined_df['h'] - grouped['c'].shift())
    low_close = abs(combined_df['l'] - grouped['c'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    combined_df['atr'] = tr.rolling(window=14).mean().values
    
    # --- Target Variable Creation ---
    combined_df['future_price'] = grouped['c'].shift(-LOOK_FORWARD_DAYS)
    combined_df['target'] = (combined_df['future_price'] > combined_df['c']).astype(int)
    
    combined_df.dropna(inplace=True)

    features = ['c', 's1', 'pivot', 'r1', 'ema_10', 'rsi', 'macd', 'macd_signal', 'atr']
    target = 'target'
    
    model_data = combined_df[features + [target]].copy()
    
    print("Data preparation complete.")
    return model_data, features, target

def plot_backtest(test_df, signals, ticker):
    """Plots the model's performance against a buy-and-hold strategy."""
    plt.figure(figsize=(12, 6))
    
    plot_data = test_df.reset_index()

    buy_hold_returns = plot_data['c'].pct_change().cumsum().fillna(0)
    strategy_returns = plot_data['c'].pct_change() * signals.reset_index(drop=True).shift(1)
    strategy_cumulative_returns = strategy_returns.cumsum().fillna(0)

    plt.plot(plot_data['t'], buy_hold_returns, label='Buy and Hold')
    plt.plot(plot_data['t'], strategy_cumulative_returns, label='Model Strategy')
    plt.title(f'{ticker} Model Performance vs. Buy and Hold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    data, feature_names, target_name = fetch_and_prepare_data(TICKERS)
    
    X = data[feature_names]
    y = data[target_name]

    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []

    for ticker in X.index.get_level_values('ticker').unique():
        X_ticker = X[X.index.get_level_values('ticker') == ticker]
        y_ticker = y[y.index.get_level_values('ticker') == ticker]
        
        split_index = int(len(X_ticker) * 0.8)
        
        X_train_list.append(X_ticker.iloc[:split_index])
        X_test_list.append(X_ticker.iloc[split_index:])
        y_train_list.append(y_ticker.iloc[:split_index])
        y_test_list.append(y_ticker.iloc[split_index:])

    X_train = pd.concat(X_train_list)
    X_test = pd.concat(X_test_list)
    y_train = pd.concat(y_train_list)
    y_test = pd.concat(y_test_list)
    
    print(f"\nTotal training samples: {len(X_train)}")
    print(f"Total testing samples: {len(X_test)}")

    print("\nTraining LightGBM model...")
    model = lgb.LGBMClassifier(objective='binary', random_state=42)
    model.fit(X_train, y_train)

    print("\nEvaluating model on combined test data...")
    predictions = model.predict(X_test)
    print("\nOverall Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Price Down/Same', 'Price Up']))
    
    print("\nFeature Importances:")
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance_df)

    ticker_to_plot = TICKERS[0]
    print(f"\nPlotting backtest for: {ticker_to_plot}...")

    single_ticker_X_test = X_test.loc[ticker_to_plot]
    single_ticker_predictions = model.predict(single_ticker_X_test)
    signals = pd.Series(single_ticker_predictions, index=single_ticker_X_test.index)
    
    plot_backtest(single_ticker_X_test, signals, ticker_to_plot)
