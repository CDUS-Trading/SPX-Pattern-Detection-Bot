import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import os
import sys
import json
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Collect high/low timing patterns')
    parser.add_argument('--input-file', type=str, default='SPX_full_1min.txt',
                    help='Input file path (default: SPX_full_1min.txt)')
    return parser.parse_args()

def analyze_timing_patterns(minute_file):
    """Analyze historical data to find patterns in high/low timing"""
    # Read and prepare data
    df = pd.read_csv(minute_file, header=None, 
                    names=['datetime', 'open', 'high', 'low', 'close'],
                    parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Filter for last 3 years
    last_date = df.index.max()
    three_years_ago = last_date - pd.DateOffset(years=3)
    df = df[df.index >= three_years_ago]
    
    daily_data = []
    grouped = df.groupby(df.index.date)
    dates = sorted(grouped.groups.keys())
    
    for i in tqdm(range(1, len(dates)), desc="Processing days"):
        today_date = dates[i]
        yesterday_date = dates[i-1]
        
        yesterday_data = grouped.get_group(yesterday_date)
        today_data = grouped.get_group(today_date)
        
        # Skip if not enough data
        if len(yesterday_data) == 0 or len(today_data) == 0:
            continue
            
        if len(today_data) < 180:  # Less than 3 hours of data
            continue
        
        # Get yesterday's patterns
        yesterday_patterns = {
            'high_time': get_extreme_time(yesterday_data, 'high'),
            'low_time': get_extreme_time(yesterday_data, 'low'),
            'first_hour_range': get_first_hour_range(yesterday_data),
            'last_hour_range': get_last_hour_range(yesterday_data),
            'day_range': get_day_range(yesterday_data),
            'close_strength': get_closing_strength(yesterday_data),
            'volatility': get_volatility(yesterday_data)
        }
        
        # Get today's high/low times
        today_patterns = {
            'high_time': get_extreme_time(today_data, 'high'),
            'low_time': get_extreme_time(today_data, 'low'),
            'first_hour_range': get_first_hour_range(today_data),
            'last_hour_range': get_last_hour_range(today_data),
            'day_range': get_day_range(today_data)
        }
        
        daily_data.append({
            'date': today_date,
            **{f'yesterday_{k}': v for k, v in yesterday_patterns.items()},
            **{f'today_{k}': v for k, v in today_patterns.items()}
        })
    
    return pd.DataFrame(daily_data)

def get_extreme_time(data, extreme_type='high'):
    """Get the time when high/low occurred"""
    if extreme_type == 'high':
        extreme_idx = data['high'].idxmax()
    else:
        extreme_idx = data['low'].idxmin()
    return extreme_idx.time()

def get_first_hour_range(data):
    """Calculate range in first hour"""
    first_hour = data[:60]
    if len(first_hour) > 0:
        return (first_hour['high'].max() - first_hour['low'].min()) / first_hour['open'].iloc[0] * 100
    return 0

def get_last_hour_range(data):
    """Calculate range in last hour"""
    last_hour = data[-60:]
    if len(last_hour) > 0:
        return (last_hour['high'].max() - last_hour['low'].min()) / last_hour['open'].iloc[0] * 100
    return 0

def get_day_range(data):
    """Calculate day's trading range as a percentage"""
    high = data['high'].max()
    low = data['low'].min()
    return ((high - low) / low) * 100

def get_closing_strength(data):
    """Calculate closing price relative to day's range"""
    high = data['high'].max()
    low = data['low'].min()
    close = data['close'].iloc[-1]
    return (close - low) / (high - low) if (high - low) > 0 else 0

def get_volatility(data):
    """Calculate realized volatility"""
    returns = np.log(data['close'] / data['close'].shift(1))
    return np.std(returns.dropna()) * np.sqrt(252 * 390) * 100

def analyze_patterns(df):
    """Analyze patterns in high/low timing"""
    output_file = 'timing_patterns.json'
    all_patterns = []
    
    # Define time buckets for analysis
    time_buckets = [
        ('early_morning', '09:30:00', '10:30:00'),
        ('mid_morning', '10:30:00', '11:30:00'),
        ('lunch', '11:30:00', '13:00:00'),
        ('early_afternoon', '13:00:00', '14:30:00'),
        ('late_afternoon', '14:30:00', '16:00:00')
    ]
    
    # Analyze patterns for each time bucket
    for bucket_name, start_time, end_time in time_buckets:
        # High patterns
        high_patterns = analyze_time_bucket_patterns(
            df, 'high_time', bucket_name, start_time, end_time
        )
        all_patterns.extend(high_patterns)
        
        # Low patterns
        low_patterns = analyze_time_bucket_patterns(
            df, 'low_time', bucket_name, start_time, end_time
        )
        all_patterns.extend(low_patterns)
    
    # Save patterns to JSON
    save_patterns_to_json(all_patterns, output_file)
    
    return all_patterns

def analyze_time_bucket_patterns(df, time_type, bucket_name, start_time, end_time):
    """Analyze patterns for a specific time bucket"""
    patterns = []
    
    # Convert time strings to datetime.time objects
    start_time = datetime.strptime(start_time, '%H:%M:%S').time()
    end_time = datetime.strptime(end_time, '%H:%M:%S').time()
    
    # Define ranges for yesterday's metrics
    range_buckets = {
        'first_hour_range': [-np.inf, 0.1, 0.2, 0.3, 0.4, 0.5, np.inf],
        'last_hour_range': [-np.inf, 0.1, 0.2, 0.3, 0.4, 0.5, np.inf],
        'day_range': [-np.inf, 0.5, 1.0, 1.5, 2.0, 2.5, np.inf],
        'close_strength': [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
        'volatility': [-np.inf, 10, 15, 20, 25, np.inf]
    }
    
    # Create categorical columns
    for metric, buckets in range_buckets.items():
        df[f'yesterday_{metric}_cat'] = pd.cut(
            df[f'yesterday_{metric}'], 
            bins=buckets
        )
    
    # Analyze patterns for each combination of metrics
    for metric1 in range_buckets.keys():
        for metric2 in range_buckets.keys():
            if metric1 >= metric2:  # Avoid duplicate combinations
                continue
                
            for cat1 in df[f'yesterday_{metric1}_cat'].unique():
                if pd.isna(cat1):
                    continue
                    
                for cat2 in df[f'yesterday_{metric2}_cat'].unique():
                    if pd.isna(cat2):
                        continue
                    
                    mask = (
                        (df[f'yesterday_{metric1}_cat'] == cat1) & 
                        (df[f'yesterday_{metric2}_cat'] == cat2)
                    )
                    subset = df[mask]
                    
                    if len(subset) > 0:
                        # Check if time falls within bucket
                        time_in_bucket = subset[f'today_{time_type}'].apply(
                            lambda x: start_time <= x <= end_time
                        )
                        
                        success_rate = time_in_bucket.mean()
                        sample_size = len(subset)
                        
                        if sample_size >= 10:  # Minimum sample size
                            patterns.append({
                                'time_type': time_type,
                                'bucket_name': bucket_name,
                                'metric1': metric1,
                                'metric1_range': str(cat1),
                                'metric2': metric2,
                                'metric2_range': str(cat2),
                                'success_rate': float(success_rate * 100),
                                'sample_size': sample_size,
                                'start_time': start_time.strftime('%H:%M:%S'),
                                'end_time': end_time.strftime('%H:%M:%S')
                            })
    
    return patterns

def save_patterns_to_json(patterns, output_file):
    """Save patterns to JSON file"""
    # Convert patterns to JSON-serializable format
    patterns_json = []
    for pattern in patterns:
        pattern_json = {
            'time_type': pattern['time_type'],
            'bucket_name': pattern['bucket_name'],
            'metric1': pattern['metric1'],
            'metric1_range': pattern['metric1_range'],
            'metric2': pattern['metric2'],
            'metric2_range': pattern['metric2_range'],
            'success_rate': pattern['success_rate'],
            'sample_size': pattern['sample_size'],
            'start_time': pattern['start_time'],
            'end_time': pattern['end_time']
        }
        patterns_json.append(pattern_json)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(patterns_json, f, indent=2)
    
    print(f"\nSaved {len(patterns_json)} timing patterns to {output_file}")

def main():
    args = parse_arguments()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_filepath = os.path.join(script_dir, args.input_file)
    
    print(f"\nUsing input file: {args.input_file}")
    
    # First function call processes the minute data
    df = analyze_timing_patterns(full_filepath)
    # Second function call analyzes the patterns
    patterns = analyze_patterns(df)
    
    print("\nTiming Pattern Analysis Complete")
    print(f"Total patterns found: {len(patterns)}")

if __name__ == "__main__":
    main() 