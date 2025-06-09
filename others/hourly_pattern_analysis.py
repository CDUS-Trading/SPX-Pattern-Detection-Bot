import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
from tqdm import tqdm

def load_spx_data(file_path='SPX_full_1min.txt'):
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

def analyze_next_hour_movement(current_hour, next_hour):
    """Analyze the movement in the next hour based on current hour's data"""
    current_features = calculate_hourly_features(current_hour)
    next_features = calculate_hourly_features(next_hour)
    
    # Calculate price movement
    price_change = next_features['close'] - current_features['close']
    price_change_pct = (price_change / current_features['close']) * 100
    
    # Determine if price went up or down
    direction = 'up' if price_change > 0 else 'down'
    
    # Calculate potential take profit and stop loss levels
    # Using 0.5% for TP and 0.25% for SL as initial values
    tp_level = current_features['close'] * (1 + 0.005 if direction == 'up' else -0.005)
    sl_level = current_features['close'] * (1 - 0.0025 if direction == 'up' else -0.0025)
    
    return {
        'direction': direction,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'take_profit': tp_level,
        'stop_loss': sl_level,
        'current_hour_features': current_features,
        'next_hour_features': next_features
    }

def find_hourly_patterns(df):
    """Find patterns between consecutive hours of trading"""
    patterns = []
    
    # Group data by date and hour
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    
    # Get unique dates
    unique_dates = df['date'].unique()
    
    # Progress bar for dates
    for date in tqdm(unique_dates, desc="Analyzing patterns"):
        date_data = df[df['date'] == date]
        
        # Get unique hours for this date
        unique_hours = sorted(date_data['hour'].unique())
        
        for i in range(len(unique_hours) - 1):
            current_hour = unique_hours[i]
            next_hour = unique_hours[i + 1]
            
            current_hour_data = date_data[date_data['hour'] == current_hour]
            next_hour_data = date_data[date_data['hour'] == next_hour]
            
            if len(current_hour_data) > 0 and len(next_hour_data) > 0:
                pattern = analyze_next_hour_movement(current_hour_data, next_hour_data)
                pattern['date'] = str(date)
                pattern['current_hour'] = current_hour
                pattern['next_hour'] = next_hour
                patterns.append(pattern)
    
    return patterns

def save_patterns_to_json(patterns, output_file='hourly_patterns.json'):
    """Save patterns to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(patterns, f, indent=4, default=str)

def main():
    # Load SPX data
    print("Loading SPX data...")
    df = load_spx_data()
    
    # Find patterns
    print("Analyzing hourly patterns...")
    patterns = find_hourly_patterns(df)
    
    # Save patterns to JSON
    print(f"Saving {len(patterns)} patterns to JSON...")
    save_patterns_to_json(patterns)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 