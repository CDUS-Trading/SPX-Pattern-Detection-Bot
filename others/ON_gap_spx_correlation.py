import pandas as pd
import numpy as np
from datetime import datetime, time
from tqdm import tqdm
import requests
import os
from pathlib import Path
from collections import defaultdict

def get_daily_data():
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Read from CSV file using relative path
    daily_df = pd.read_csv(os.path.join(script_dir, 'SPX_daily_values.csv'))
    daily_df['timestamp'] = pd.to_datetime(daily_df['timestamp'])
    daily_df.set_index('timestamp', inplace=True)
    
    return daily_df

def calculate_overnight_moves(daily_df):
    """Calculate overnight price changes (previous close to current open)"""
    overnight_moves = pd.DataFrame()
    overnight_moves['prev_close'] = daily_df['close'].shift(1)
    overnight_moves['open'] = daily_df['open']
    overnight_moves['overnight_return'] = (overnight_moves['open'] - overnight_moves['prev_close'])
    
    return overnight_moves

def analyze_patterns(minute_df, daily_df):
    """Analyze intraday patterns using minute data combined with overnight moves"""
    # Merge overnight information with minute data
    minute_df['date'] = minute_df.index.date
    daily_df['date'] = daily_df.index.date
    
    # Add overnight move information to each minute
    merged_df = pd.merge(minute_df, 
                        calculate_overnight_moves(daily_df)[['overnight_return']], 
                        left_on='date',
                        right_index=True,
                        how='left')
    
    # Calculate intraday returns
    merged_df['intraday_return'] = (merged_df['close'] - merged_df['open'].groupby(merged_df['date']).transform('first')) / \
                                   merged_df['open'].groupby(merged_df['date']).transform('first') * 100
    
    return merged_df

def analyze_overnight_patterns(minute_file):
    # Read minute data
    df = pd.read_csv(minute_file, header=None, 
                     names=['datetime', 'open', 'high', 'low', 'close'],
                     parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Get daily data for overnight moves
    daily_df = get_daily_data()
    
    # Filter for last year of data
    last_date = df.index.max()
    one_year_ago = last_date - pd.DateOffset(years=1)
    df = df[df.index >= one_year_ago]
    daily_df = daily_df[daily_df.index >= one_year_ago]
    
    print(f"\nAnalyzing data from {one_year_ago.date()} to {last_date.date()}\n")
    
    # Group minute data by trading day
    daily_data = []
    grouped = df.groupby(df.index.date)
    
    # Add progress bar
    for date, day_df in tqdm(grouped, desc="Processing days"):
        date = pd.to_datetime(date)
        
        if date not in daily_df.index:
            continue
            
        # Calculate overnight move using daily data
        prev_day = daily_df.shift(1)
        overnight_move = daily_df.loc[date, 'open'] - prev_day.loc[date, 'close']
        
        # Find when the day's high and low occurred using minute data
        high_time = day_df['high'].idxmax().strftime('%H:%M')
        low_time = day_df['low'].idxmin().strftime('%H:%M')
        
        daily_data.append({
            'date': date,
            'overnight_move': overnight_move,
            'high_time': high_time,
            'low_time': low_time
        })
    
    return pd.DataFrame(daily_data)

def analyze_with_different_buckets(df):
    # Different bucket configurations to test
    bucket_configs = [
        # Tighter ranges around zero
        [-np.inf, -40, -30, -20, -10, 0, 10, 20, 30, 40, np.inf],
        # Very tight ranges
        [-np.inf, -25, -15, -10, -5, 0, 5, 10, 15, 25, np.inf],
        # Asymmetric ranges
        [-np.inf, -50, -30, -15, 0, 20, 40, 60, np.inf],
        # Original ranges
        [-np.inf, -80, -50, -20, 0, 20, 50, 80, np.inf]
    ]
    
    # Different time windows to analyze
    time_windows = [
        15,  # 15-minute blocks
        30,  # 30-minute blocks
        60,  # 1-hour blocks (original)
        120  # 2-hour blocks
    ]
    
    strong_patterns = []
    
    print("\nSearching for patterns (>= 40% probability)...")
    
    for buckets in tqdm(bucket_configs, desc="Testing bucket configurations"):
        labels = [f"{buckets[i]} to {buckets[i+1]}" for i in range(len(buckets)-1)]
        df['move_category'] = pd.cut(df['overnight_move'], bins=buckets, labels=labels)
        
        for window in time_windows:
            # Convert times to window blocks
            df['high_window'] = pd.to_datetime(df['high_time'], format='%H:%M').dt.hour * 60 + \
                               pd.to_datetime(df['high_time'], format='%H:%M').dt.minute
            df['high_window'] = (df['high_window'] // window) * window
            
            df['low_window'] = pd.to_datetime(df['low_time'], format='%H:%M').dt.hour * 60 + \
                              pd.to_datetime(df['low_time'], format='%H:%M').dt.minute
            df['low_window'] = (df['low_window'] // window) * window
            
            # Analyze each category
            for category in df['move_category'].unique():
                category_data = df[df['move_category'] == category]
                if len(category_data) < 30:  # Skip if sample size too small
                    continue
                
                # Check high time distributions for all windows
                high_dist = category_data['high_window'].value_counts(normalize=True)
                low_dist = category_data['low_window'].value_counts(normalize=True)
                
                # Record patterns for all time windows with prob >= 40%
                for time, prob in high_dist.items():
                    if prob >= 0.44:  # 40% or higher
                        strong_patterns.append({
                            'bucket_range': category,
                            'window_size': window,
                            'time': f"{time//60:02d}:{time%60:02d}",
                            'pattern': 'HIGH',
                            'probability': prob,
                            'sample_size': len(category_data)
                        })
                
                for time, prob in low_dist.items():
                    if prob >= 0.44:
                        strong_patterns.append({
                            'bucket_range': category,
                            'window_size': window,
                            'time': f"{time//60:02d}:{time%60:02d}",
                            'pattern': 'LOW',
                            'probability': prob,
                            'sample_size': len(category_data)
                        })
    
    return strong_patterns

def analyze_post_low_behavior(df):
    """Analyze when lows occur after big gap up days"""
    # Filter for big gap up days (>25 points)
    big_gaps = df[df['overnight_move'] > 25].copy()
    
    if len(big_gaps) == 0:
        print("No big gap up days found")
        return
    
    print(f"\nAnalyzing {len(big_gaps)} days with gaps > 25 points:")
    
    # Convert low times to minutes since market open
    low_times = pd.to_datetime(big_gaps['low_time'], format='%H:%M')
    minutes_since_open = (low_times.dt.hour - 9) * 60 + low_times.dt.minute
    
    # Analyze distribution across different time windows
    time_windows = [15, 30, 60]  # 15-min, 30-min, and 1-hour buckets
    
    for window in time_windows:
        print(f"\nAnalyzing {window}-minute buckets:")
        time_buckets = minutes_since_open // window
        bucket_counts = time_buckets.value_counts().sort_index()
        
        for bucket, count in bucket_counts.items():
            start_time = 9 * 60 + bucket * window  # minutes since midnight
            end_time = start_time + window
            bucket_start = f"{start_time//60:02d}:{start_time%60:02d}"
            bucket_end = f"{end_time//60:02d}:{end_time%60:02d}"
            probability = count / len(big_gaps)
            
            print(f"{bucket_start}-{bucket_end}: {count} lows ({probability:.1%})")

def analyze_gap_up_patterns(df):
    """Detailed analysis of big gap up days focusing on first and last hour"""
    # Filter for big gap up days (>25 points)
    big_gaps = df[df['overnight_move'] > 25].copy()
    
    if len(big_gaps) == 0:
        print("No big gap up days found")
        return
    
    # Convert times to minutes since market open
    low_times = pd.to_datetime(big_gaps['low_time'], format='%H:%M')
    high_times = pd.to_datetime(big_gaps['high_time'], format='%H:%M')
    
    # Categorize each day
    big_gaps['low_period'] = low_times.apply(lambda x: 
        'First Hour' if x.hour == 9 else
        'Last Hour' if x.hour == 15 else
        'Mid Day')
    
    big_gaps['high_period'] = high_times.apply(lambda x: 
        'First Hour' if x.hour == 9 else
        'Last Hour' if x.hour == 15 else
        'Mid Day')
    
    print(f"\nAnalyzing {len(big_gaps)} days with gaps > 25 points:")
    
    # Analyze low patterns
    print("\nLow Distribution:")
    low_dist = big_gaps['low_period'].value_counts()
    for period, count in low_dist.items():
        print(f"{period}: {count} days ({count/len(big_gaps):.1%})")
    
    # Analyze high patterns for each low period
    print("\nHigh Distribution by Low Period:")
    for low_period in ['First Hour', 'Last Hour', 'Mid Day']:
        subset = big_gaps[big_gaps['low_period'] == low_period]
        if len(subset) > 0:
            print(f"\nWhen low occurs in {low_period} (n={len(subset)}):")
            high_dist = subset['high_period'].value_counts()
            for period, count in high_dist.items():
                print(f"- High in {period}: {count} days ({count/len(subset):.1%})")
    
    # Analyze gap size impact
    print("\nGap Size Analysis:")
    gap_ranges = [(25, 35), (35, 50), (50, float('inf'))]
    for min_gap, max_gap in gap_ranges:
        subset = big_gaps[
            (big_gaps['overnight_move'] > min_gap) & 
            (big_gaps['overnight_move'] <= max_gap)
        ]
        if len(subset) > 0:
            print(f"\nGap {min_gap} to {max_gap} points (n={len(subset)}):")
            low_dist = subset['low_period'].value_counts()
            for period, count in low_dist.items():
                print(f"- Low in {period}: {count} days ({count/len(subset):.1%})")

def main(filepath):
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use full path for input file
    full_filepath = os.path.join(script_dir, filepath)
    df = analyze_overnight_patterns(full_filepath)
    
    # Analyze gap up patterns
    analyze_gap_up_patterns(df)
    
    print("\nPotential Trading Strategies:")
    print("1. First Hour Low Strategy:")
    print("   - Enter: After first hour low is made")
    print("   - Stop: Below the low")
    print("   - Target: Based on high distribution")
    print("\n2. Last Hour Low Strategy:")
    print("   - Enter: On late day weakness")
    print("   - Stop: Below the low")
    print("   - Hold: Consider overnight hold")
    print("\n3. Gap Size Specific Strategy:")
    print("   - Adjust approach based on gap size")
    print("   - Larger gaps may have different patterns")

main('SPX_full_1min.txt')
