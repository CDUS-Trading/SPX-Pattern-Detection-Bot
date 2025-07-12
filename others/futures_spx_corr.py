import pandas as pd
import numpy as np
from datetime import datetime
from yahooquery import Ticker
from tqdm import tqdm
import os

def get_premarket_futures(date):
    """Get ES futures performance from 6:30 AM to 9:30 AM ET"""
    # List of possible ES futures symbols
    symbols = ["ES=F", "ESH24=F", "ES.1=F", "ES1!", "/ES"]
    
    for symbol in symbols:
        try:
            # Get ES futures data using yahooquery
            es = Ticker(symbol)
            data = es.history(interval='1m',
                            start=date.strftime('%Y-%m-%d') + ' 06:30:00',
                            end=date.strftime('%Y-%m-%d') + ' 09:30:00')
            
            if not data.empty:
                start_price = data['open'].iloc[0]
                end_price = data['close'].iloc[-1]
                move_pct = ((end_price - start_price) / start_price) * 100
                
                # Get hourly moves
                hour1 = data[:60]['close'].iloc[-1]
                hour2 = data[60:120]['close'].iloc[-1]
                hour3 = data[120:]['close'].iloc[-1]
                
                hourly_moves = {
                    'hour1': ((hour1 - start_price) / start_price) * 100,
                    'hour2': ((hour2 - hour1) / hour1) * 100,
                    'hour3': ((hour3 - hour2) / hour2) * 100
                }
                
                return {
                    'total_move': move_pct,
                    'hourly_moves': hourly_moves,
                    'volume': data['volume'].sum()
                }
        except Exception as e:
            print(f"Failed with symbol {symbol}: {str(e)}")
            continue
    
    print(f"Could not get futures data for {date}")
    return None

def analyze_market_correlation(minute_file):
    """Analyze correlation between pre-market futures and SPX first 3 hours"""
    # Read minute data
    df = pd.read_csv(minute_file, header=None, 
                     names=['datetime', 'open', 'high', 'low', 'close'],
                     parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Filter for last year of data
    last_date = df.index.max()
    one_year_ago = last_date - pd.DateOffset(years=1)
    df = df[df.index >= one_year_ago]
    
    print(f"\nAnalyzing data from {one_year_ago.date()} to {last_date.date()}")
    
    # Group by trading day
    daily_data = []
    grouped = df.groupby(df.index.date)
    
    for date, day_df in tqdm(grouped, desc="Processing days"):
        date = pd.to_datetime(date)
        
        # Get pre-market futures data
        futures_data = get_premarket_futures(date)
        
        if futures_data is not None:
            # Get first 3 hours of SPX trading (9:30 AM - 12:30 PM)
            spx_3h = day_df.iloc[:180]  # First 180 minutes
            
            if not spx_3h.empty:
                spx_open = spx_3h['open'].iloc[0]
                
                # Calculate hourly returns for SPX
                hour1_close = spx_3h[:60]['close'].iloc[-1]
                hour2_close = spx_3h[60:120]['close'].iloc[-1]
                hour3_close = spx_3h[120:]['close'].iloc[-1]
                
                spx_hourly = {
                    'hour1': ((hour1_close - spx_open) / spx_open) * 100,
                    'hour2': ((hour2_close - hour1_close) / hour1_close) * 100,
                    'hour3': ((hour3_close - hour2_close) / hour2_close) * 100
                }
                
                daily_data.append({
                    'date': date,
                    'futures_move': futures_data['total_move'],
                    'futures_volume': futures_data['volume'],
                    'futures_h1': futures_data['hourly_moves']['hour1'],
                    'futures_h2': futures_data['hourly_moves']['hour2'],
                    'futures_h3': futures_data['hourly_moves']['hour3'],
                    'spx_h1': spx_hourly['hour1'],
                    'spx_h2': spx_hourly['hour2'],
                    'spx_h3': spx_hourly['hour3']
                })
    
    return pd.DataFrame(daily_data)

def analyze_patterns(df):
    """Analyze patterns between pre-market futures and SPX behavior"""
    print("\nOverall Correlation Analysis:")
    correlation = df['futures_move'].corr(df[['spx_h1', 'spx_h2', 'spx_h3']].sum(axis=1))
    print(f"Futures vs First 3 Hours SPX correlation: {correlation:.2f}")
    
    # Define futures move buckets
    buckets = [-np.inf, -0.5, -0.25, 0, 0.25, 0.5, np.inf]
    labels = ['< -0.5%', '-0.5% to -0.25%', '-0.25% to 0%',
              '0% to 0.25%', '0.25% to 0.5%', '> 0.5%']
    
    df['futures_category'] = pd.cut(df['futures_move'], bins=buckets, labels=labels)
    
    print("\nDetailed Analysis by Pre-market Futures Move:")
    for category in df['futures_category'].unique():
        if pd.isna(category):
            continue
            
        category_data = df[df['futures_category'] == category]
        if len(category_data) < 5:  # Skip if sample size too small
            continue
        
        print(f"\nWhen Futures move {category} (n={len(category_data)}):")
        
        # Calculate average SPX behavior
        for hour in ['h1', 'h2', 'h3']:
            avg_move = category_data[f'spx_{hour}'].mean()
            prob_positive = (category_data[f'spx_{hour}'] > 0).mean()
            print(f"Hour {hour}:")
            print(f"  Average move: {avg_move:.2f}%")
            print(f"  Probability of positive: {prob_positive:.1%}")

def main(filepath):
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use full path for input file
    full_filepath = os.path.join(script_dir, filepath)
    df = analyze_market_correlation(full_filepath)
    
    # Analyze patterns
    analyze_patterns(df)

main('SPX_full_1min.txt')
