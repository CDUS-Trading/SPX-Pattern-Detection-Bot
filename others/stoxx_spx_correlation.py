from datetime import datetime
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from requests import Session

def get_european_data(date):
    """Get STOXX50 performance using Eurex API with retry logic"""
    try:
        # Create session with retry logic
        session = Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504, 443],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        
        url = "https://api.deutsche-boerse.com/data/v1/tradingview"
        headers = {
            "X-DBP-APIKEY": "68cdafd2-c5c1-49be-8558-37244ab4f513",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Query for EURO STOXX 50 Index Futures
        params = {
            "resolution": "1",
            "symbol": "FESX",
            "from": int(datetime.strptime(f"{date.strftime('%Y-%m-%d')} 06:30:00", "%Y-%m-%d %H:%M:%S").timestamp()),
            "to": int(datetime.strptime(f"{date.strftime('%Y-%m-%d')} 09:30:00", "%Y-%m-%d %H:%M:%S").timestamp())
        }
        
        response = session.get(url, headers=headers, params=params, timeout=10)
        
        # Debug response
        print(f"\nAPI Response Status: {response.status_code}")
        print(f"URL: {response.url}")
        
        if response.status_code == 200:
            data = response.json()
            if 'c' in data and len(data['c']) > 0:
                prices = pd.Series(data['c'], index=pd.to_datetime(data['t'], unit='s'))
                
                start_price = prices.iloc[0]
                end_price = prices.iloc[-1]
                move_pct = ((end_price - start_price) / start_price) * 100
                
                # Get hourly moves
                hour1 = prices[:60].iloc[-1]
                hour2 = prices[60:120].iloc[-1]
                hour3 = prices[120:].iloc[-1]
                
                hourly_moves = {
                    'hour1': ((hour1 - start_price) / start_price) * 100,
                    'hour2': ((hour2 - hour1) / hour1) * 100,
                    'hour3': ((hour3 - hour2) / hour2) * 100
                }
                
                return {'total_move': move_pct, 'hourly_moves': hourly_moves}
            else:
                print(f"No price data available for {date}")
        else:
            print(f"API request failed with status {response.status_code}")
            
    except Exception as e:
        print(f"Error getting STOXX data: {e}")
        return None
    finally:
        session.close()
    
    return None

def analyze_first_hours_correlation(minute_file):
    """Analyze correlation between last 3 STOXX hours and first 3 SPX hours"""
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
        
        # Get STOXX50 performance
        stoxx_data = get_european_data(date)
        
        if stoxx_data is not None:
            # Get first 3 hours of SPX trading (9:30 AM - 12:30 PM)
            spx_3h = day_df.iloc[:180]  # First 180 minutes
            
            if not spx_3h.empty:
                us_open = spx_3h['open'].iloc[0]
                
                # Calculate hourly returns for SPX
                hour1_close = spx_3h[:60]['close'].iloc[-1]
                hour2_close = spx_3h[60:120]['close'].iloc[-1]
                hour3_close = spx_3h[120:]['close'].iloc[-1]
                
                spx_hourly = {
                    'hour1': ((hour1_close - us_open) / us_open) * 100,
                    'hour2': ((hour2_close - hour1_close) / hour1_close) * 100,
                    'hour3': ((hour3_close - hour2_close) / hour2_close) * 100
                }
                
                daily_data.append({
                    'date': date,
                    'stoxx_move': stoxx_data['total_move'],
                    'stoxx_h1': stoxx_data['hourly_moves']['hour1'],
                    'stoxx_h2': stoxx_data['hourly_moves']['hour2'],
                    'stoxx_h3': stoxx_data['hourly_moves']['hour3'],
                    'spx_h1': spx_hourly['hour1'],
                    'spx_h2': spx_hourly['hour2'],
                    'spx_h3': spx_hourly['hour3']
                })
    
    return pd.DataFrame(daily_data)

def analyze_hourly_patterns(df):
    """Analyze patterns between STOXX and SPX hourly moves"""
    print("\nHourly Correlation Analysis:")
    
    # Overall correlations
    correlations = {
        'STOXX Final 3h vs SPX First 3h': df['stoxx_move'].corr(df[['spx_h1', 'spx_h2', 'spx_h3']].sum(axis=1)),
        'STOXX H1 vs SPX H1': df['stoxx_h1'].corr(df['spx_h1']),
        'STOXX H2 vs SPX H1': df['stoxx_h2'].corr(df['spx_h1']),
        'STOXX H3 vs SPX H1': df['stoxx_h3'].corr(df['spx_h1'])
    }
    
    for desc, corr in correlations.items():
        print(f"{desc}: {corr:.2f}")
    
    # Define move buckets for STOXX
    buckets = [-np.inf, -0.5, -0.25, 0, 0.25, 0.5, np.inf]
    labels = ['< -0.5%', '-0.5% to -0.25%', '-0.25% to 0%', 
              '0% to 0.25%', '0.25% to 0.5%', '> 0.5%']
    
    print("\nAnalyzing SPX First Hour Behavior Based on STOXX Final Hour:")
    df['stoxx_h3_cat'] = pd.cut(df['stoxx_h3'], bins=buckets, labels=labels)
    
    for category in df['stoxx_h3_cat'].unique():
        if pd.isna(category):
            continue
            
        subset = df[df['stoxx_h3_cat'] == category]
        if len(subset) < 5:
            continue
            
        print(f"\nWhen STOXX final hour {category} (n={len(subset)}):")
        avg_spx_h1 = subset['spx_h1'].mean()
        prob_pos = (subset['spx_h1'] > 0).mean()
        
        print(f"Average SPX first hour: {avg_spx_h1:.2f}%")
        print(f"Probability of positive SPX first hour: {prob_pos:.1%}")

def main(filepath):
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use full path for input file
    full_filepath = os.path.join(script_dir, filepath)
    df = analyze_first_hours_correlation(full_filepath)
    
    # Analyze patterns
    analyze_hourly_patterns(df)

main('SPX_full_1min.txt')