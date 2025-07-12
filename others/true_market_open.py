# Same as spx_yday_today_corr.py but using market open price for calculations
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import os
import sys

def get_last_n_minutes_pattern(data, n):
    """Calculate return for last n minutes"""
    last_n = data.iloc[-n:]
    return ((last_n['close'].iloc[-1] - last_n['open'].iloc[0]) / 
            last_n['open'].iloc[0] * 100)

def get_closing_strength(data):
    """Calculate closing price relative to day's range"""
    high = data['high'].max()
    low = data['low'].min()
    close = data['close'].iloc[-1]
    return (close - low) / (high - low) if (high - low) > 0 else 0

def get_last_hour_volatility(data):
    """Calculate realized volatility in the last hour"""
    last_hour = data.iloc[-60:]
    returns = np.log(last_hour['close'] / last_hour['close'].shift(1))
    return np.std(returns.dropna()) * np.sqrt(252 * 390) * 100

def get_return(data):
    """Calculate return for a period"""
    if len(data) > 0:
        return ((data['close'].iloc[-1] - data['open'].iloc[0]) / 
                data['open'].iloc[0] * 100)
    return 0

def get_volatility(data):
    """Calculate realized volatility for a period"""
    returns = np.log(data['close'] / data['close'].shift(1))
    return np.std(returns.dropna()) * np.sqrt(252 * 390) * 100

def get_last_hour_pattern(data):
    """Calculate return for last hour"""
    return get_last_n_minutes_pattern(data, 60)

def get_closing_volatility(data):
    """Calculate volatility in the last hour"""
    last_hour = data.iloc[-60:]
    return (last_hour['high'].max() - last_hour['low'].min()) / last_hour['low'].min() * 100

def get_day_range(data):
    """Calculate day's trading range as a percentage"""
    high = data['high'].max()
    low = data['low'].min()
    return ((high - low) / low) * 100

def get_win_sequence(returns, is_bearish):
    """Convert returns into W/L sequence based on pattern direction"""
    if is_bearish:
        return ''.join(['W' if ret < 0 else 'L' for ret in returns])
    else:
        return ''.join(['W' if ret > 0 else 'L' for ret in returns])

def format_range(range_str):
    """Convert range string to more readable format"""
    if not range_str:
        return ""
    range_str = range_str.replace('(', '').replace(']', '').replace('[', '')
    start, end = range_str.split(',')
    
    if start == '-inf':
        return f"less than {end}%"
    elif end == 'inf':
        return f"more than {start}%"
    else:
        return f"between {start}% and {end}%"

def analyze_market_correlation(minute_file):
    """Analysis using market open price instead of first minute close"""
    df = pd.read_csv(minute_file, header=None, 
                     names=['datetime', 'open', 'high', 'low', 'close'],
                     parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    
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
        
        if len(yesterday_data) == 0 or len(today_data) == 0:
            continue
            
        if len(today_data) < 180:  # Less than 3 hours of data
            continue
        
        yesterday_patterns = {
            'last_hour': get_last_hour_pattern(yesterday_data),
            'last_30min': get_last_n_minutes_pattern(yesterday_data, 30),
            'last_15min': get_last_n_minutes_pattern(yesterday_data, 15),
            'close_vol': get_closing_volatility(yesterday_data),
            'last_hour_vol': get_last_hour_volatility(yesterday_data),
            'close_strength': get_closing_strength(yesterday_data),
            'day_range': get_day_range(yesterday_data)
        }
        
        today_3h = today_data.iloc[:180]
        if not today_3h.empty:
            market_open = today_3h['open'].iloc[0]  # Using market open price
            
            first_hour = today_3h[:60]
            first_30min = today_3h[:30]
            first_15min = today_3h[:15]
            
            today_patterns = {
                'first_15min': ((first_15min['close'].iloc[-1] - market_open) / 
                               market_open * 100),
                'first_30min': ((first_30min['close'].iloc[-1] - market_open) / 
                               market_open * 100),
                'first_hour': ((first_hour['close'].iloc[-1] - market_open) / 
                              market_open * 100),
                'hour2': get_return(today_3h[60:120]),
                'hour3': get_return(today_3h[120:180]),
                'first_hour_vol': get_volatility(first_hour),
                'gap': ((market_open - yesterday_data['close'].iloc[-1]) / 
                       yesterday_data['close'].iloc[-1] * 100)
            }
            
            daily_data.append({
                'date': today_date,
                **{f'yesterday_{k}': v for k, v in yesterday_patterns.items()},
                **{f'today_{k}': v for k, v in today_patterns.items()}
            })
    
    return pd.DataFrame(daily_data)

def analyze_combined_pattern(df, pattern1, pattern2):
    """Analyze combinations of patterns with stricter criteria"""
    buckets = [-np.inf, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, np.inf]
    
    df['pattern1_cat'] = pd.cut(df[f'yesterday_{pattern1}'], bins=buckets)
    df['pattern2_cat'] = pd.cut(df[f'yesterday_{pattern2}'], bins=buckets)
    
    high_prob_patterns = []
    
    for cat1 in df['pattern1_cat'].unique():
        if pd.isna(cat1):
            continue
            
        for cat2 in df['pattern2_cat'].unique():
            if pd.isna(cat2):
                continue
                
            mask = (df['pattern1_cat'] == cat1) & (df['pattern2_cat'] == cat2)
            subset = df[mask]
            
            if len(subset) >= 10:
                subset = subset.sort_values('date')
                returns = subset['today_first_hour']
                
                prob_up = (returns > 0).mean()
                prob_down = (returns < 0).mean()
                avg_move = returns.mean()
                
                win_sequence = get_win_sequence(returns, prob_down > prob_up)
                recent_n = len(subset) // 3
                recent_subset = subset.iloc[-recent_n:]
                recent_accuracy = (recent_subset['today_first_hour'] > 0).mean()
                
                avg_win = subset[subset['today_first_hour'] > 0]['today_first_hour'].mean() or 0
                avg_loss = subset[subset['today_first_hour'] < 0]['today_first_hour'].mean() or 0
                risk_reward = abs(avg_win/avg_loss) if avg_loss != 0 else float('inf')
                
                # Stricter criteria
                if ((prob_up > 0.67 or prob_down > 0.67) and
                    len(subset) >= 10):
                    
                    high_prob_patterns.append({
                        'pattern1': pattern1,
                        'cat1': cat1,
                        'pattern2': pattern2,
                        'cat2': cat2,
                        'prob_up': prob_up,
                        'prob_down': prob_down,
                        'avg_move': avg_move,
                        'n_samples': len(subset),
                        'win_sequence': win_sequence,
                        'risk_reward': risk_reward,
                        'recent_accuracy': recent_accuracy,
                        'recent_n': recent_n,
                        'avg_win': avg_win,
                        'avg_loss': avg_loss,
                        'subset': subset
                    })
    
    return high_prob_patterns

def analyze_patterns(df):
    """Pattern analysis with output to market open specific files"""
    output_file = 'pattern_analysis_market_open.txt'
    dates_file = 'pattern_dates_market_open.txt'
    pattern_dates = set()
    
    pattern_combinations = [
        ('last_15min', 'close_vol'),
        ('last_30min', 'close_vol'),
        ('last_hour', 'close_vol'),
        ('last_15min', 'close_strength'),
        ('last_30min', 'close_strength'),
        ('last_hour', 'close_strength'),
        ('last_15min', 'last_hour_vol'),
        ('last_30min', 'last_hour_vol'),
        ('last_hour', 'last_hour_vol')
    ]
    
    with open(output_file, 'w') as f:
        sys.stdout = f
        
        print("\nAnalyzing Patterns Using Market Open Price:")
        for pattern1, pattern2 in pattern_combinations:
            patterns = analyze_combined_pattern(df, pattern1, pattern2)
            for pattern in patterns:
                if isinstance(pattern['subset']['date'].iloc[0], pd.Timestamp):
                    dates = pattern['subset']['date'].dt.strftime('%Y-%m-%d')
                else:
                    dates = pattern['subset']['date']
                pattern_dates.update(dates)
                
                # Print pattern details
                print(f"\nPattern Found ({pattern['n_samples']} instances):")
                print(f"When yesterday's conditions were:")
                print(f"  • {pattern['pattern1']} {format_range(str(pattern['cat1']))}")
                print(f"  • {pattern['pattern2']} {format_range(str(pattern['cat2']))}")
                print(f"\nResults for next day's first hour:")
                is_bearish = pattern['prob_down'] > pattern['prob_up']
                print(f"  Direction: {'Bearish 📉' if is_bearish else 'Bullish 📈'}")
                print(f"  Success Rate: {max(pattern['prob_up'], pattern['prob_down']):.1%}")
                print(f"  Average Move: {abs(pattern['avg_move']):.2f}%")
                print(f"  Risk/Reward: {pattern['risk_reward']:.2f}")
                print(f"  Win/Loss: {pattern['avg_win']:.2f}% / {abs(pattern['avg_loss']):.2f}%")
                print(f"\nRecent Performance:")
                print(f"  Last {pattern['recent_n']} trades: {pattern['recent_accuracy']:.1%}")
                
                # Print trade history
                print("\nTrade History:")
                for _, row in pattern['subset'].iterrows():
                    print(f"\nDate: {row['date'].strftime('%Y-%m-%d')}")
                    print(f"  Yesterday's {pattern1}: {row[f'yesterday_{pattern1}']:.2f}%")
                    print(f"  Yesterday's {pattern2}: {row[f'yesterday_{pattern2}']:.2f}%")
                    print(f"  Today's Gap: {row['today_gap']:.2f}%")
                    print(f"  First Hour: {row['today_first_hour']:.2f}%")
                    print(f"  Second Hour: {row['today_hour2']:.2f}%")
                    print(f"  Third Hour: {row['today_hour3']:.2f}%")
    
    sys.stdout = sys.__stdout__
    
    with open(dates_file, 'w') as f:
        for date in sorted(pattern_dates):
            f.write(f"{date}\n")
    
    total_days = len(df['date'].unique())
    pattern_days = len(pattern_dates)
    print(f"\nAnalysis complete:")
    print(f"Total days analyzed: {total_days}")
    print(f"Days with patterns: {pattern_days} ({pattern_days/total_days:.1%})")
    print(f"Results written to {output_file}")
    print(f"Pattern dates written to {dates_file}")

def main(filepath):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_filepath = os.path.join(script_dir, filepath)
    df = analyze_market_correlation(full_filepath)
    analyze_patterns(df)

if __name__ == "__main__":
    main('SPX_full_1min.txt')
