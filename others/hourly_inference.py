import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from tqdm import tqdm
import json

def load_patterns(pattern_file='hourly_patterns.json'):
    """Load previously analyzed patterns"""
    with open(pattern_file, 'r') as f:
        return json.load(f)

def load_spx_data(file_path='SPX_week_1min.txt'):
    """Load SPX 1-minute data from text file"""
    print("Loading data file...")
    # Get total number of lines for progress bar
    total_lines = sum(1 for _ in open(file_path))
    
    # Read the data without headers and specify column names
    df = pd.read_csv(file_path, header=None, 
                    names=['datetime', 'open', 'high', 'low', 'close'],
                    chunksize=100000)  # Read in chunks
    
    # Process chunks with progress bar
    chunks = []
    for chunk in tqdm(df, total=total_lines//100000 + 1, desc="Loading data"):
        chunk['datetime'] = pd.to_datetime(chunk['datetime'])
        chunks.append(chunk)
    
    # Combine all chunks
    df = pd.concat(chunks, ignore_index=True)
    return df

def calculate_hourly_features(hour_data):
    """Calculate features for a given hour of data"""
    return {
        'open': hour_data.iloc[0]['open'],
        'high': hour_data['high'].max(),
        'low': hour_data['low'].min(),
        'close': hour_data.iloc[-1]['close'],
        'range': hour_data['high'].max() - hour_data['low'].min(),
        'avg_price': hour_data['close'].mean(),
        'volatility': hour_data['close'].std()
    }

def find_similar_patterns(current_hour_data, all_patterns):
    """Find patterns similar to the current hour's data"""
    current_features = calculate_hourly_features(current_hour_data)
    similar_patterns = []
    
    for pattern in all_patterns:
        # Compare current hour features with pattern's current hour features
        pattern_features = pattern['current_hour_features']
        
        # Calculate similarity score (simple Euclidean distance for now)
        similarity = np.sqrt(
            (current_features['range'] - pattern_features['range'])**2 +
            (current_features['volatility'] - pattern_features['volatility'])**2 +
            (current_features['avg_price'] - pattern_features['avg_price'])**2
        )
        
        if similarity < 10:  # Threshold for similarity
            similar_patterns.append(pattern)
    
    return similar_patterns

def analyze_trading_strategy(current_data, all_patterns, is_market_closed=False):
    """Analyze trading strategies based on current data"""
    strategies = []
    
    # Find similar patterns
    similar_patterns = find_similar_patterns(current_data, all_patterns)
    if not similar_patterns:
        return strategies
    
    # Calculate success rate and average TP/SL
    success_count = sum(1 for p in similar_patterns if p['direction'] == 'up' and p['price_change'] > 0 or 
                      p['direction'] == 'down' and p['price_change'] < 0)
    success_rate = (success_count / len(similar_patterns)) * 100
    
    avg_tp = np.mean([p['take_profit'] - p['current_hour_features']['close'] for p in similar_patterns])
    avg_sl = np.mean([p['current_hour_features']['close'] - p['stop_loss'] for p in similar_patterns])
    
    # Determine direction based on success rate
    direction = 'up' if success_rate > 50 else 'down'
    
    # Get current time and calculate next hour
    current_time = datetime.now()
    next_hour = current_time + timedelta(hours=1)
    
    # Format entry and exit times
    entry_time = current_time.strftime("%I:%M%p")
    exit_time = next_hour.strftime("%I:%M%p")
    
    strategies.append({
        'entry_time': entry_time,
        'exit_time': exit_time,
        'direction': direction,
        'take_profit': round(avg_tp, 2),
        'stop_loss': round(avg_sl, 2),
        'success_rate': round(success_rate, 2),
        'is_market_closed': is_market_closed
    })
    
    return strategies

def print_strategy(strategy, current_time):
    """Print trading strategy in formatted output"""
    if strategy['is_market_closed']:
        print(f"\n=== SPX Pattern Analysis for Tomorrow's First Hour ===")
        print(f"Based on today's ({current_time.strftime('%A, %Y-%m-%d')}) market data")
    else:
        print(f"\n=== SPX Pattern Analysis for Next Hour ===")
        print(f"Current Time: {current_time.strftime('%A, %Y-%m-%d %I:%M%p')} CT")
    
    print("\n===== Action Plan =====")
    print(f"Entry: {strategy['entry_time']} CT")
    print(f"Exit: {strategy['exit_time']} CT")
    print(f"Direction: {'Buy 📈' if strategy['direction'] == 'up' else 'Sell 📉'}")
    print(f"TP: {strategy['take_profit']} points")
    print(f"SL: {strategy['stop_loss']} points")
    print(f"Success Rate: {strategy['success_rate']}%")
    print("-" * 30)

def is_market_open():
    """Check if market is currently open"""
    current_time = datetime.now()
    # Market hours: 8:30 AM to 3:00 PM CT
    market_open = current_time.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=15, minute=0, second=0, microsecond=0)
    
    # Convert to CT (assuming script runs in CT)
    return market_open <= current_time <= market_close

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze SPX trading strategies')
    parser.add_argument('--date', type=str, help='Date to analyze (YYYY-MM-DD). If not provided, uses last available date.')
    args = parser.parse_args()
    
    # Load patterns and data
    print("Loading patterns and data...")
    patterns = load_patterns()
    df = load_spx_data()
    
    current_time = datetime.now()
    is_closed = not is_market_open()
    
    if args.date:
        # Use specified date
        target_date = pd.to_datetime(args.date).date()
        current_data = df[df['datetime'].dt.date == target_date]
        if len(current_data) == 0:
            print(f"No data found for date {target_date}")
            return
        print(f"Analyzing patterns for {target_date.strftime('%A, %Y-%m-%d')}")
        
        # For specified date, analyze each hour
        strategies = []
        for hour in range(8, 15):  # Market hours
            hour_data = current_data[current_data['datetime'].dt.hour == hour]
            if len(hour_data) > 0:
                hour_strategies = analyze_trading_strategy(hour_data, patterns, is_closed)
                strategies.extend(hour_strategies)
    else:
        if is_closed:
            # After market close: use last available date's data
            last_date = df['datetime'].dt.date.max()
            current_data = df[df['datetime'].dt.date == last_date]
            if len(current_data) == 0:
                print("No data found in dataset")
                return
            print(f"Using data from {last_date.strftime('%A, %Y-%m-%d')} for analysis")
        else:
            # During market hours: use last hour's data
            last_hour = current_time - timedelta(hours=1)
            current_data = df[
                (df['datetime'].dt.date == last_hour.date()) & 
                (df['datetime'].dt.hour == last_hour.hour)
            ]
            if len(current_data) == 0:
                # If no data for last hour, use last available hour from last date
                last_date = df['datetime'].dt.date.max()
                last_hour_data = df[df['datetime'].dt.date == last_date]
                if len(last_hour_data) == 0:
                    print("No data found in dataset")
                    return
                last_hour = last_hour_data['datetime'].max()
                current_data = last_hour_data[last_hour_data['datetime'].dt.hour == last_hour.hour]
                print(f"Using last available hour from {last_date.strftime('%A, %Y-%m-%d')} for analysis")
        
        # Analyze strategies
        strategies = analyze_trading_strategy(current_data, patterns, is_closed)
    
    if not strategies:
        print("No trading strategies found")
        return
    
    # Print strategies
    for strategy in strategies:
        print_strategy(strategy, current_time)

if __name__ == "__main__":
    main() 