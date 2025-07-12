import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import glob
import shutil
import re

# Set up module-level logger
logger = logging.getLogger(__name__)

class CollectionError(Exception):
    """Custom exception for pattern collection errors"""
    pass

class DataError(Exception):
    """Custom exception for data processing errors"""
    pass

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for the module.
    
    Args:
        level (int): Logging level to use (default: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the pattern collector.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
        
    Raises:
        CollectionError: If there is an error parsing the arguments
    """
    try:
        parser = argparse.ArgumentParser(description='Collect market patterns')
        parser.add_argument(
            '--input-file', 
            type=str,
            default='data/SPX_full_1min.txt',
            help='Input file path (default: data/SPX_full_1min.txt)'
        )
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug logging'
        )
        parser.add_argument(
            '--output-dir',
            type=str,
            default='data/processed',
            help='Directory to save output files (default: data/processed)'
        )
        
        args = parser.parse_args()
        logger.debug(f"Parsed arguments: {args}")
        return args
        
    except Exception as e:
        logger.error(f"Error parsing arguments: {str(e)}")
        raise CollectionError(f"Failed to parse command line arguments: {str(e)}")

def analyze_market_correlation(minute_file: str) -> pd.DataFrame:
    """
    Analyze market correlation patterns from minute data.
    
    Args:
        minute_file (str): Path to the minute data file
        
    Returns:
        pd.DataFrame: DataFrame containing analyzed market patterns
        
    Raises:
        DataError: If there is an error processing the data
    """
    try:
        logger.info(f"Starting market correlation analysis for file: {minute_file}")
        
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
            
            # Skip if data is insufficient
            if len(yesterday_data) == 0 or len(today_data) == 0:
                logger.warning(f"Skipping {today_date} - insufficient data")
                continue
                
            if len(today_data) < 180:  # Less than 3 hours of data
                logger.warning(f"Skipping {today_date} - less than 3 hours of data")
                continue
            
            # Calculate yesterday's patterns
            yesterday_patterns = {
                'last_hour': get_last_hour_pattern(yesterday_data),
                'last_30min': get_last_n_minutes_pattern(yesterday_data, 30),
                'last_15min': get_last_n_minutes_pattern(yesterday_data, 15),
                'close_vol': get_closing_volatility(yesterday_data),
                'last_hour_vol': get_last_hour_volatility(yesterday_data),
                'close_strength': get_closing_strength(yesterday_data),
                'day_range': get_day_range(yesterday_data)
            }
            
            # Add time-based metrics for yesterday
            first_5_min = get_first_5_minutes_behavior(yesterday_data)
            if first_5_min:
                yesterday_patterns.update({
                    'first_5min_return': first_5_min['return'],
                    'first_5min_range': first_5_min['range'],
                    'first_5min_high_test': first_5_min['high_test'],
                    'first_5min_low_test': first_5_min['low_test'],
                    'first_5min_vol': first_5_min['volatility']
                })
            
            power_hour = get_power_hour_characteristics(yesterday_data)
            if power_hour:
                yesterday_patterns.update({
                    'power_hour_return': power_hour['return'],
                    'power_hour_range': power_hour['range'],
                    'power_hour_trend_changes': power_hour['trend_changes'],
                    'power_hour_vol': power_hour['volatility'],
                    'power_hour_momentum': power_hour['momentum_consistency']
                })
            
            lunch_hour = get_lunch_hour_behavior(yesterday_data)
            if lunch_hour:
                yesterday_patterns.update({
                    'lunch_hour_range': lunch_hour['range'],
                    'lunch_hour_return': lunch_hour['return'],
                    'lunch_hour_direction_changes': lunch_hour['direction_changes'],
                    'lunch_hour_vol': lunch_hour['volatility'],
                    'lunch_hour_range_contraction': lunch_hour['range_contraction']
                })
            
            pre_close = get_pre_close_hour_momentum(yesterday_data)
            if pre_close:
                yesterday_patterns.update({
                    'pre_close_return': pre_close['return'],
                    'pre_close_range': pre_close['range'],
                    'pre_close_tendency': pre_close['closing_tendency'],
                    'pre_close_vol': pre_close['volatility'],
                    'pre_close_momentum': pre_close['momentum_strength']
                })
            
            strongest_moves = get_strongest_move_periods(yesterday_data)
            if strongest_moves:
                if strongest_moves['strongest_hour']:
                    yesterday_patterns.update({
                        'strongest_hour': strongest_moves['strongest_hour']['hour'],
                        'strongest_hour_range': strongest_moves['strongest_hour']['range']
                    })
                if strongest_moves['strongest_30min']:
                    yesterday_patterns.update({
                        'strongest_30min_period': strongest_moves['strongest_30min']['period'],
                        'strongest_30min_range': strongest_moves['strongest_30min']['range']
                    })
                if strongest_moves['strongest_15min']:
                    yesterday_patterns.update({
                        'strongest_15min_period': strongest_moves['strongest_15min']['period'],
                        'strongest_15min_range': strongest_moves['strongest_15min']['range']
                    })
            
            # Add early day volatility metrics for yesterday
            for minutes in [15, 30, 60]:
                period_data = yesterday_data[:minutes]
                if len(period_data) >= minutes:
                    yesterday_patterns.update({
                        f'first_{minutes}min_vol': (period_data['high'].max() - period_data['low'].min()) / 
                                                 period_data['open'].iloc[0] * 100
                    })
            
            # Calculate yesterday's momentum and trend metrics
            yesterday_hours = {}
            for hour in range(1, 5):
                start_idx = (hour - 1) * 60
                end_idx = hour * 60
                hour_data = yesterday_data[start_idx:end_idx]
                if len(hour_data) >= 60:
                    yesterday_hours[f'hour{hour}'] = get_return(hour_data)
            
            # Add momentum and trend strength for yesterday
            if len(yesterday_hours) >= 4:
                weights = [0.4, 0.3, 0.2, 0.1]
                momentum_score = sum(yesterday_hours[f'hour{i+1}'] * w for i, w in enumerate(weights))
                
                # Calculate trend strength
                moves = [yesterday_hours[f'hour{i}'] for i in range(1, 5)]
                consecutive = 1
                for i in range(1, len(moves)):
                    if (moves[i] > 0 and moves[i-1] > 0) or (moves[i] < 0 and moves[i-1] < 0):
                        consecutive += 1
                
                yesterday_patterns.update({
                    'pre_lunch_momentum': momentum_score,
                    'morning_trend_strength': consecutive
                })
            
            # Enhanced today analysis
            today_full = today_data
            if not today_full.empty:
                first_minute_close = today_full['close'].iloc[0]
                
                # Add period-specific high/low values for volatility calculation
                period_metrics = {}
                for minutes in [15, 30, 60]:
                    period_data = today_full[:minutes]
                    if len(period_data) >= minutes:
                        period_metrics.update({
                            f'high_{minutes}min': period_data['high'].max(),
                            f'low_{minutes}min': period_data['low'].min(),
                            f'open_{minutes}min': first_minute_close
                        })
                
                # Individual hours analysis
                hours_analysis = {}
                for hour in range(1, 7):
                    start_idx = (hour - 1) * 60
                    end_idx = hour * 60
                    hour_data = today_full[start_idx:end_idx]
                    if len(hour_data) >= 60:
                        hours_analysis[f'hour{hour}'] = get_return(hour_data)
                
                # Add 15-minute interval analysis for first two hours
                for hour in range(2):
                    for quarter in range(4):
                        start_idx = (hour * 60) + (quarter * 15)
                        end_idx = start_idx + 15
                        interval_data = today_full[start_idx:end_idx]
                        if len(interval_data) >= 15:
                            interval_name = f'hour{hour+1}_q{quarter+1}'
                            hours_analysis[interval_name] = get_return(interval_data)
                
                # Add 15-minute interval transitions
                for i in range(7):
                    current = f'hour{i//4+1}_q{(i%4)+1}'
                    next_interval = f'hour{(i+1)//4+1}_q{((i+1)%4)+1}'
                    if current in hours_analysis and next_interval in hours_analysis:
                        hours_analysis[f'momentum_{current}_to_{next_interval}'] = (
                            hours_analysis[next_interval] - hours_analysis[current]
                        )
                
                # Add 30-minute interval analysis for first two hours
                for hour in range(2):
                    for half in range(2):
                        start_idx = (hour * 60) + (half * 30)
                        end_idx = start_idx + 30
                        interval_data = today_full[start_idx:end_idx]
                        if len(interval_data) >= 30:
                            interval_name = f'hour{hour+1}_h{half+1}'
                            hours_analysis[interval_name] = get_return(interval_data)
                
                # Add 30-minute interval transitions
                for i in range(3):
                    current = f'hour{i//2+1}_h{(i%2)+1}'
                    next_interval = f'hour{(i+1)//2+1}_h{((i+1)%2)+1}'
                    if current in hours_analysis and next_interval in hours_analysis:
                        hours_analysis[f'momentum_{current}_to_{next_interval}'] = (
                            hours_analysis[next_interval] - hours_analysis[current]
                        )
                
                # Last 30 minutes
                last_30_min = today_full[360:390]
                if len(last_30_min) > 0:
                    hours_analysis['last_30_min'] = get_return(last_30_min)
                
                # Multi-hour periods
                hour_combinations = [
                    ('hours_1_2', 0, 120),
                    ('hours_2_3', 60, 180),
                    ('hours_3_4', 120, 240),
                    ('hours_4_5', 180, 300),
                    ('hours_5_6', 240, 360),
                    ('hours_5_6_30', 240, 390)
                ]
                
                # Calculate returns for all periods
                today_patterns = {
                    'first_15min': ((today_full[:15]['close'].iloc[-1] - first_minute_close) / 
                                   first_minute_close * 100),
                    'first_30min': ((today_full[:30]['close'].iloc[-1] - first_minute_close) / 
                                   first_minute_close * 100),
                    'first_hour': ((today_full[:60]['close'].iloc[-1] - first_minute_close) / 
                                  first_minute_close * 100),
                    **hours_analysis,
                    **{f'today_{k}': v for k, v in period_metrics.items()}
                }
                
                # Add multi-hour periods
                for period_name, start, end in hour_combinations:
                    period_data = today_full[start:end]
                    if len(period_data) > 0:
                        today_patterns[period_name] = ((period_data['close'].iloc[-1] - 
                                                      period_data['open'].iloc[0]) / 
                                                   period_data['open'].iloc[0] * 100)
                
                # Add gaps
                today_patterns.update({
                    'official_gap': ((first_minute_close - yesterday_data['close'].iloc[-1]) / 
                                   yesterday_data['close'].iloc[-1] * 100),
                    'effective_gap': ((first_minute_close - yesterday_data['close'].iloc[-1]) / 
                                    yesterday_data['close'].iloc[-1] * 100)
                })
                
                daily_data.append({
                    'date': today_date,
                    **{f'yesterday_{k}': v for k, v in yesterday_patterns.items()},
                    **{f'today_{k}': v for k, v in today_patterns.items()}
                })
        
        logger.info(f"Completed market correlation analysis. Processed {len(daily_data)} days")
        return pd.DataFrame(daily_data)
        
    except Exception as e:
        logger.error(f"Error in market correlation analysis: {str(e)}")
        raise DataError(f"Failed to analyze market correlation: {str(e)}")

def get_last_n_minutes_pattern(data: pd.DataFrame, n: int) -> float:
    """
    Calculate return for the last n minutes of trading.
    
    Args:
        data (pd.DataFrame): DataFrame containing minute data with 'open' and 'close' columns
        n (int): Number of minutes to analyze from the end
        
    Returns:
        float: Percentage return for the specified period
        
    Raises:
        DataError: If the data is invalid or insufficient
    """
    try:
        if len(data) < n:
            logger.warning(f"Insufficient data points for {n}-minute pattern")
            return 0.0
            
        last_n = data.iloc[-n:]
        return ((last_n['close'].iloc[-1] - last_n['open'].iloc[0]) / 
                last_n['open'].iloc[0] * 100)
    except Exception as e:
        logger.error(f"Error calculating {n}-minute pattern: {str(e)}")
        raise DataError(f"Failed to calculate {n}-minute pattern: {str(e)}")

def get_closing_strength(data: pd.DataFrame) -> float:
    """
    Calculate closing price relative to day's range.
    
    Args:
        data (pd.DataFrame): DataFrame containing minute data with 'high', 'low', and 'close' columns
        
    Returns:
        float: Closing strength as a ratio between 0 and 1
        
    Raises:
        DataError: If the data is invalid or insufficient
    """
    try:
        high = data['high'].max()
        low = data['low'].min()
        close = data['close'].iloc[-1]
        
        if high == low:
            logger.warning("High and low prices are equal, returning 0 for closing strength")
            return 0.0
            
        return (close - low) / (high - low)
    except Exception as e:
        logger.error(f"Error calculating closing strength: {str(e)}")
        raise DataError(f"Failed to calculate closing strength: {str(e)}")

def get_last_hour_volatility(data: pd.DataFrame) -> float:
    """
    Calculate realized volatility in the last hour.
    
    Args:
        data (pd.DataFrame): DataFrame containing minute data with 'close' column
        
    Returns:
        float: Annualized volatility percentage
        
    Raises:
        DataError: If the data is invalid or insufficient
    """
    try:
        if len(data) < 60:
            logger.warning("Insufficient data points for last hour volatility")
            return 0.0
            
        last_hour = data.iloc[-60:]
        returns = np.log(last_hour['close'] / last_hour['close'].shift(1))
        return np.std(returns.dropna()) * np.sqrt(252 * 390) * 100
    except Exception as e:
        logger.error(f"Error calculating last hour volatility: {str(e)}")
        raise DataError(f"Failed to calculate last hour volatility: {str(e)}")

def get_return(data: pd.DataFrame) -> float:
    """
    Calculate return for a period.
    
    Args:
        data (pd.DataFrame): DataFrame containing minute data with 'open' and 'close' columns
        
    Returns:
        float: Percentage return for the period
        
    Raises:
        DataError: If the data is invalid or insufficient
    """
    try:
        if len(data) == 0:
            logger.warning("Empty data for return calculation")
            return 0.0
            
        return ((data['close'].iloc[-1] - data['open'].iloc[0]) / 
                data['open'].iloc[0] * 100)
    except Exception as e:
        logger.error(f"Error calculating return: {str(e)}")
        raise DataError(f"Failed to calculate return: {str(e)}")

def get_volatility(data: pd.DataFrame) -> float:
    """
    Calculate realized volatility for a period.
    
    Args:
        data (pd.DataFrame): DataFrame containing minute data with 'close' column
        
    Returns:
        float: Annualized volatility percentage
        
    Raises:
        DataError: If the data is invalid or insufficient
    """
    try:
        if len(data) < 2:
            logger.warning("Insufficient data points for volatility calculation")
            return 0.0
            
        returns = np.log(data['close'] / data['close'].shift(1))
        return np.std(returns.dropna()) * np.sqrt(252 * 390) * 100
    except Exception as e:
        logger.error(f"Error calculating volatility: {str(e)}")
        raise DataError(f"Failed to calculate volatility: {str(e)}")

def get_last_hour_pattern(data: pd.DataFrame) -> float:
    """
    Calculate return for the last hour.
    
    Args:
        data (pd.DataFrame): DataFrame containing minute data with 'open' and 'close' columns
        
    Returns:
        float: Percentage return for the last hour
        
    Raises:
        DataError: If the data is invalid or insufficient
    """
    try:
        return get_last_n_minutes_pattern(data, 60)
    except Exception as e:
        logger.error(f"Error calculating last hour pattern: {str(e)}")
        raise DataError(f"Failed to calculate last hour pattern: {str(e)}")

def get_closing_volatility(data: pd.DataFrame) -> float:
    """
    Calculate volatility in the last hour.
    
    Args:
        data (pd.DataFrame): DataFrame containing minute data with 'high', 'low', and 'open' columns
        
    Returns:
        float: Volatility percentage for the last hour
        
    Raises:
        DataError: If the data is invalid or insufficient
    """
    try:
        if len(data) < 60:
            logger.warning("Insufficient data points for closing volatility")
            return 0.0
            
        last_hour = data.iloc[-60:]
        return (last_hour['high'].max() - last_hour['low'].min()) / last_hour['low'].min() * 100
    except Exception as e:
        logger.error(f"Error calculating closing volatility: {str(e)}")
        raise DataError(f"Failed to calculate closing volatility: {str(e)}")

def get_day_range(data: pd.DataFrame) -> float:
    """
    Calculate day's trading range as a percentage.
    
    Args:
        data (pd.DataFrame): DataFrame containing minute data with 'high' and 'low' columns
        
    Returns:
        float: Day's range as a percentage
        
    Raises:
        DataError: If the data is invalid or insufficient
    """
    try:
        if len(data) == 0:
            logger.warning("Empty data for day range calculation")
            return 0.0
            
        high = data['high'].max()
        low = data['low'].min()
        return ((high - low) / low) * 100
    except Exception as e:
        logger.error(f"Error calculating day range: {str(e)}")
        raise DataError(f"Failed to calculate day range: {str(e)}")

def get_win_sequence(returns: pd.Series, is_bearish: bool) -> str:
    """
    Convert returns into W/L sequence based on pattern direction.
    
    Args:
        returns (pd.Series): Series of returns to analyze
        is_bearish (bool): Whether the pattern is bearish (True) or bullish (False)
        
    Returns:
        str: Sequence of 'W' (win) and 'L' (loss) characters
    """
    if is_bearish:
        return ''.join(['W' if ret < 0 else 'L' for ret in returns])
    else:
        return ''.join(['W' if ret > 0 else 'L' for ret in returns])

def format_range(range_str: str) -> str:
    """
    Convert range string to more readable format.
    
    Args:
        range_str (str): Range string in format '(start,end]'
        
    Returns:
        str: Formatted range description
    """
    if not range_str:
        return ""
    # Remove parentheses and split
    range_str = range_str.replace('(', '').replace(']', '').replace('[', '')
    start, end = range_str.split(',')
    
    # Handle infinities
    if start == '-inf':
        return f"less than {end}%"
    elif end == 'inf':
        return f"more than {start}%"
    else:
        return f"between {start}% and {end}%"

def analyze_patterns(df: pd.DataFrame) -> List[Dict]:
    """
    Collect and analyze market patterns from the input DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing market data with pattern metrics
        
    Returns:
        List[Dict]: List of pattern dictionaries containing pattern definitions and statistics
        
    Note:
        This function redirects stdout to a file during pattern analysis to capture detailed output.
    """
    output_file = 'pattern_database.json'
    pattern_dates = set()
    all_patterns = []
    
    # Redirect stdout to file
    original_stdout = sys.stdout
    with open(output_file, 'w') as f:
        sys.stdout = f
        
        print("\nCollecting Market Patterns:")
        
        # Expanded pattern combinations focusing on early day
        pattern_combinations = [
            # Original combinations
            ('last_15min', 'close_vol'),
            ('last_30min', 'close_vol'),
            ('last_hour', 'close_vol'),
            ('last_15min', 'close_strength'),
            ('last_30min', 'close_strength'),
            ('last_hour', 'close_strength'),
            ('last_15min', 'last_hour_vol'),
            ('last_30min', 'last_hour_vol'),
            ('last_hour', 'last_hour_vol'),
            
            # Early day volatility combinations
            ('first_15min_vol', 'first_30min_vol'),
            ('first_30min_vol', 'first_60min_vol'),
            ('first_15min_vol', 'close_strength'),
            ('first_30min_vol', 'close_strength'),
            ('first_60min_vol', 'close_strength'),
            
            # Momentum and trend combinations
            ('pre_lunch_momentum', 'morning_trend_strength'),
            ('pre_lunch_momentum', 'close_strength'),
            ('morning_trend_strength', 'close_strength'),
            ('pre_lunch_momentum', 'last_hour'),
            ('morning_trend_strength', 'last_hour'),
            
            # Additional combinations using available metrics
            ('first_15min_vol', 'last_hour'),
            ('first_30min_vol', 'last_hour'),
            ('first_60min_vol', 'last_hour_vol'),
            ('day_range', 'first_15min_vol'),
            ('day_range', 'first_30min_vol'),
            
            # New combinations specifically for shorter timeframes
            ('last_15min', 'first_15min_vol'),
            ('last_30min', 'first_30min_vol'),
            ('close_vol', 'first_15min_vol'),
            ('close_vol', 'first_30min_vol'),
            ('close_strength', 'first_15min_vol'),
            ('close_strength', 'first_30min_vol'),
            ('last_hour_vol', 'first_15min_vol'),
            ('last_hour_vol', 'first_30min_vol'),

            # New time-based pattern combinations
            ('first_5min_return', 'first_5min_vol'),
            ('first_5min_range', 'first_5min_high_test'),
            ('first_5min_range', 'first_5min_low_test'),
            ('power_hour_return', 'power_hour_trend_changes'),
            ('power_hour_range', 'power_hour_momentum'),
            ('lunch_hour_range', 'lunch_hour_direction_changes'),
            ('lunch_hour_range_contraction', 'lunch_hour_vol'),
            ('pre_close_return', 'pre_close_tendency'),
            ('pre_close_range', 'pre_close_momentum'),
            ('strongest_hour_range', 'strongest_30min_range'),
            ('strongest_15min_range', 'strongest_30min_range'),
            
            # Cross-time period combinations
            ('first_5min_return', 'power_hour_return'),
            ('power_hour_return', 'lunch_hour_return'),
            ('lunch_hour_return', 'pre_close_return'),
            ('first_5min_vol', 'power_hour_vol'),
            ('power_hour_vol', 'lunch_hour_vol'),
            ('lunch_hour_vol', 'pre_close_vol'),
            
            # Range-based combinations
            ('first_5min_range', 'power_hour_range'),
            ('power_hour_range', 'lunch_hour_range'),
            ('lunch_hour_range', 'pre_close_range'),
            ('strongest_hour_range', 'day_range'),
            ('strongest_30min_range', 'day_range'),
            ('strongest_15min_range', 'day_range')
        ]
        
        # Track unique dates where patterns are found
        dates_with_patterns = set()
        
        # Add progress bar for pattern analysis
        for pattern1, pattern2 in tqdm(pattern_combinations, desc="Analyzing pattern combinations"):
            # Get patterns and their dates
            patterns = analyze_combined_pattern(df, pattern1, pattern2)
            
            # Add all patterns to our collection
            all_patterns.extend(patterns)
            
            # Add dates from each pattern
            for pattern in patterns:
                if isinstance(pattern['subset']['date'].iloc[0], pd.Timestamp):
                    dates = pattern['subset']['date'].dt.date
                else:
                    dates = pattern['subset']['date']
                dates_with_patterns.update(dates)
    
    # Restore stdout
    sys.stdout = original_stdout
    
    # Calculate percentage of days with patterns
    total_days = len(df['date'].unique())
    pattern_days = len(dates_with_patterns)
    pattern_percentage = (pattern_days / total_days) * 100
    
    print(f"\nPattern Collection Statistics:")
    print(f"Total trading days in dataset: {total_days}")
    print(f"Days with patterns: {pattern_days}")
    print(f"Percentage of days with patterns: {pattern_percentage:.2f}%")
    print(f"Total unique patterns found: {len(all_patterns)}")
    
    return all_patterns

def analyze_combined_pattern(df: pd.DataFrame, pattern1: str, pattern2: str) -> List[Dict]:
    """
    Analyze combinations of patterns without filtering.
    
    Args:
        df (pd.DataFrame): DataFrame containing market data with pattern metrics
        pattern1 (str): First pattern metric to analyze
        pattern2 (str): Second pattern metric to analyze
        
    Returns:
        List[Dict]: List of pattern dictionaries containing pattern definitions and statistics
    """
    print(f"\nAnalyzing patterns between {pattern1} and {pattern2}")
    
    # Print available columns for debugging
    print("\nAvailable columns in DataFrame:")
    print(df.columns.tolist())
    
    # Enhanced buckets with more granular divisions for early day patterns
    buckets1 = [-np.inf, -1.0, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1.0, np.inf]
    buckets2 = [-np.inf, -1.0, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1.0, np.inf]
    
    # Add new early-day specific columns
    df['first_15min_vol'] = df.apply(lambda x: calculate_period_volatility(x, 15), axis=1)
    df['first_30min_vol'] = df.apply(lambda x: calculate_period_volatility(x, 30), axis=1)
    df['first_hour_vol'] = df.apply(lambda x: calculate_period_volatility(x, 60), axis=1)
    
    # Add pre-lunch momentum indicators
    df['pre_lunch_momentum'] = df.apply(lambda x: calculate_momentum_score(x), axis=1)
    df['morning_trend_strength'] = df.apply(lambda x: calculate_trend_strength(x), axis=1)
    
    df['pattern1_cat'] = pd.cut(df[f'yesterday_{pattern1}'], bins=buckets1)
    df['pattern2_cat'] = pd.cut(df[f'yesterday_{pattern2}'], bins=buckets2)
    
    patterns = []
    
    for cat1 in df['pattern1_cat'].unique():
        if pd.isna(cat1):
            continue
            
        for cat2 in df['pattern2_cat'].unique():
            if pd.isna(cat2):
                continue
                
            mask = (df['pattern1_cat'] == cat1) & (df['pattern2_cat'] == cat2)
            subset = df[mask]
            
            if len(subset) > 0:
                subset = subset.sort_values('date')
                
                # Define all timeframes to analyze
                timeframes = [
                    # Original timeframes
                    'first_hour', 'hour2', 'hour3', 'hour4', 'hour5', 'hour6', 
                    'last_30_min', 'hours_2_3', 'hours_3_4', 'hours_4_5', 
                    'hours_5_6', 'hours_5_6_30',
                    
                    # 15-minute intervals for first two hours
                    'hour1_q1', 'hour1_q2', 'hour1_q3', 'hour1_q4',
                    'hour2_q1', 'hour2_q2', 'hour2_q3', 'hour2_q4',
                    
                    # 30-minute intervals for first two hours
                    'hour1_h1', 'hour1_h2',
                    'hour2_h1', 'hour2_h2',
                    
                    # 15-minute interval transitions
                    'momentum_hour1_q1_to_hour1_q2',
                    'momentum_hour1_q2_to_hour1_q3',
                    'momentum_hour1_q3_to_hour1_q4',
                    'momentum_hour1_q4_to_hour2_q1',
                    'momentum_hour2_q1_to_hour2_q2',
                    'momentum_hour2_q2_to_hour2_q3',
                    'momentum_hour2_q3_to_hour2_q4',
                    
                    # 30-minute interval transitions
                    'momentum_hour1_h1_to_hour1_h2',
                    'momentum_hour1_h2_to_hour2_h1',
                    'momentum_hour2_h1_to_hour2_h2'
                ]
                
                for timeframe in timeframes:
                    col = f'today_{timeframe}'
                    if col not in subset.columns:
                        continue
                        
                    returns = subset[col]
                    
                    # Calculate probabilities and moves
                    prob_up = (returns > 0).mean()
                    prob_down = (returns < 0).mean()
                    avg_move = returns.mean()
                    
                    # Determine pattern direction
                    is_bearish = prob_down > prob_up
                    
                    # Calculate moves in expected direction
                    if is_bearish:
                        expected_moves = returns[returns < 0]  # Down moves for bearish pattern
                        opposite_moves = returns[returns > 0]  # Up moves for bullish pattern
                    else:
                        expected_moves = returns[returns > 0]  # Up moves for bullish pattern
                        opposite_moves = returns[returns < 0]  # Down moves for bearish pattern
                    
                    # Calculate average moves
                    avg_expected_move = abs(expected_moves.mean()) if len(expected_moves) > 0 else 0
                    avg_opposite_move = abs(opposite_moves.mean()) if len(opposite_moves) > 0 else 0
                    risk_reward = avg_expected_move / avg_opposite_move if avg_opposite_move > 0 else 0
                    
                    # Extract range values from category objects
                    cat1_str = str(cat1)
                    cat2_str = str(cat2)
                    
                    # Parse the category strings to extract range values
                    cat1_range = cat1_str.strip('()[]').split(',')
                    cat2_range = cat2_str.strip('()[]').split(',')
                    
                    # Calculate overall risk/reward ratio for this pattern
                    overall_returns = subset[[f'today_{tf}' for tf in timeframes if f'today_{tf}' in subset.columns]].mean(axis=1)
                    overall_avg_up = overall_returns[overall_returns > 0].mean() if len(overall_returns[overall_returns > 0]) > 0 else 0
                    overall_avg_down = abs(overall_returns[overall_returns < 0].mean()) if len(overall_returns[overall_returns < 0]) > 0 else 0
                    
                    if overall_avg_down > 0:
                        overall_risk_reward = overall_avg_up / overall_avg_down
                    else:
                        overall_risk_reward = float('inf')
                    
                    patterns.append({
                        'pattern1': pattern1,
                        'cat1': cat1,
                        'pattern2': pattern2,
                        'cat2': cat2,
                        'subset': subset,
                        'range1': cat1_range,
                        'range2': cat2_range,
                        'overall_risk_reward': overall_risk_reward,
                        'timeframe': timeframe,
                        'direction': 'bearish' if is_bearish else 'bullish',
                        'success_rate': max(prob_up, prob_down) * 100,
                        'avg_move': avg_move,
                        'risk_reward': risk_reward,
                        'avg_win': avg_expected_move,
                        'avg_loss': avg_opposite_move,
                        'sample_size': len(subset)
                    })
    
    return patterns

def calculate_period_volatility(row: pd.Series, minutes: int) -> float:
    """
    Calculate volatility for a specific time period.
    
    Args:
        row (pd.Series): Row of data containing high, low, and open prices for the period
        minutes (int): Number of minutes in the period
        
    Returns:
        float: Volatility percentage for the specified period, or 0 if data is unavailable
    """
    try:
        high = row[f'today_high_{minutes}min']
        low = row[f'today_low_{minutes}min']
        open_price = row[f'today_open_{minutes}min']
        return (high - low) / open_price * 100
    except KeyError:
        # Return 0 if the period data isn't available
        return 0

def calculate_momentum_score(row: pd.Series) -> float:
    """
    Calculate momentum score based on first 4 hours of trading.
    
    Args:
        row (pd.Series): Row of data containing hourly returns
        
    Returns:
        float: Weighted momentum score, with more weight given to earlier hours
    """
    weights = [0.4, 0.3, 0.2, 0.1]  # More weight to earlier hours
    hours = ['hour1', 'hour2', 'hour3', 'hour4']
    
    score = 0
    for hour, weight in zip(hours, weights):
        score += row[f'today_{hour}'] * weight
    return score

def calculate_trend_strength(row: pd.Series) -> int:
    """
    Calculate trend strength in the morning session.
    
    Args:
        row (pd.Series): Row of data containing hourly returns
        
    Returns:
        int: Number of consecutive moves in the same direction during morning hours
    """
    morning_hours = ['hour1', 'hour2', 'hour3', 'hour4']
    moves = [row[f'today_{hour}'] for hour in morning_hours]
    
    # Count consecutive moves in same direction
    consecutive = 1
    for i in range(1, len(moves)):
        if (moves[i] > 0 and moves[i-1] > 0) or (moves[i] < 0 and moves[i-1] < 0):
            consecutive += 1
    return consecutive

def save_patterns_to_json(patterns: List[Dict], output_file: str) -> List[Dict]:
    """
    Save patterns to a JSON file for pattern_detection.py to use.
    
    Args:
        patterns (List[Dict]): List of pattern dictionaries to save
        output_file (str): Path to the output JSON file
        
    Returns:
        List[Dict]: List of processed patterns that were saved
        
    Note:
        This function handles infinity values and ensures JSON compatibility.
    """
    patterns_for_json = []
    
    # Check if patterns is None or empty
    if not patterns:
        print(f"No patterns found to save.")
        # Create empty JSON file
        with open(output_file, 'w') as f:
            json.dump([], f)
        return []
    
    for pattern in patterns:
        pattern1 = pattern['pattern1']
        pattern2 = pattern['pattern2']
        
        # Handle range values more safely
        try:
            range1 = [float(x) for x in pattern['range1'] if x and x.strip()]
            range2 = [float(x) for x in pattern['range2'] if x and x.strip()]
        except (ValueError, TypeError):
            print(f"Warning: Could not convert range values for pattern {pattern1}/{pattern2}")
            continue
        
        # Handle infinity values for JSON serialization
        range1_min = range1[0] if len(range1) > 0 else -999999
        range1_max = range1[1] if len(range1) > 1 else 999999
        range2_min = range2[0] if len(range2) > 0 else -999999
        range2_max = range2[1] if len(range2) > 1 else 999999
        
        # Replace infinity with large numbers for JSON compatibility
        if range1_min == float('-inf'):
            range1_min = -999999
        if range1_max == float('inf'):
            range1_max = 999999
        if range2_min == float('-inf'):
            range2_min = -999999
        if range2_max == float('inf'):
            range2_max = 999999
        
        # Create pattern entry
        pattern_entry = {
            'pattern1': pattern1,
            'range1_min': range1_min,
            'range1_max': range1_max,
            'pattern2': pattern2,
            'range2_min': range2_min,
            'range2_max': range2_max,
            'timeframe': pattern['timeframe'],
            'direction': pattern['direction'],
            'success_rate': float(pattern['success_rate']),
            'avg_move': float(pattern['avg_move']),
            'risk_reward': float(pattern['risk_reward']),
            'avg_win': float(pattern['avg_win']),
            'avg_loss': float(pattern['avg_loss']),
            'sample_size': int(pattern['sample_size'])
        }
        
        patterns_for_json.append(pattern_entry)
    
    # Custom JSON encoder to handle infinity values
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if obj == float('inf'):
                return 999999
            if obj == float('-inf'):
                return -999999
            return json.JSONEncoder.default(self, obj)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(patterns_for_json, f, indent=2, cls=CustomEncoder)
    
    print(f"\nSaved {len(patterns_for_json)} patterns to {output_file}")
    return patterns_for_json

def get_first_5_minutes_behavior(data: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Analyze first 5 minutes of trading.
    
    Args:
        data (pd.DataFrame): DataFrame containing minute data with OHLC columns
        
    Returns:
        Optional[Dict[str, float]]: Dictionary containing first 5 minutes metrics or None if insufficient data
    """
    if len(data) < 5:
        return None
    
    first_5 = data.iloc[:5]
    first_minute_close = data['close'].iloc[0]  # Use first minute close as reference
    
    return {
        'return': ((first_5['close'].iloc[-1] - first_minute_close) / first_minute_close * 100),
        'range': ((first_5['high'].max() - first_5['low'].min()) / first_minute_close * 100),
        'high_test': (first_5['high'].max() - first_minute_close) / first_minute_close * 100,
        'low_test': (first_minute_close - first_5['low'].min()) / first_minute_close * 100,
        'volatility': np.std(np.log(first_5['close'] / first_5['close'].shift(1)).dropna()) * np.sqrt(252 * 390) * 100
    }

def get_power_hour_characteristics(data: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Analyze 9:30-10:30 characteristics.
    
    Args:
        data (pd.DataFrame): DataFrame containing minute data with OHLC columns
        
    Returns:
        Optional[Dict[str, float]]: Dictionary containing power hour metrics or None if insufficient data
    """
    if len(data) < 60:
        return None
    
    power_hour = data.iloc[:60]
    returns = np.log(power_hour['close'] / power_hour['close'].shift(1)).dropna()
    
    # Count trend changes (when direction changes)
    direction_changes = 0
    for i in range(1, len(returns)):
        if (returns.iloc[i] > 0 and returns.iloc[i-1] < 0) or (returns.iloc[i] < 0 and returns.iloc[i-1] > 0):
            direction_changes += 1
    
    return {
        'return': ((power_hour['close'].iloc[-1] - power_hour['open'].iloc[0]) / power_hour['open'].iloc[0] * 100),
        'range': ((power_hour['high'].max() - power_hour['low'].min()) / power_hour['open'].iloc[0] * 100),
        'trend_changes': direction_changes,
        'volatility': np.std(returns) * np.sqrt(252 * 390) * 100,
        'momentum_consistency': len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
    }

def get_lunch_hour_behavior(data: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Analyze 12:00-1:00 behavior.
    
    Args:
        data (pd.DataFrame): DataFrame containing minute data with OHLC columns
        
    Returns:
        Optional[Dict[str, float]]: Dictionary containing lunch hour metrics or None if insufficient data
    """
    if len(data) < 390:  # Need at least 6.5 hours of data
        return None
    
    lunch_hour = data.iloc[150:210]  # 12:00-1:00 (assuming 9:30 start)
    returns = np.log(lunch_hour['close'] / lunch_hour['close'].shift(1)).dropna()
    
    return {
        'range': ((lunch_hour['high'].max() - lunch_hour['low'].min()) / lunch_hour['open'].iloc[0] * 100),
        'return': ((lunch_hour['close'].iloc[-1] - lunch_hour['open'].iloc[0]) / lunch_hour['open'].iloc[0] * 100),
        'direction_changes': sum(1 for i in range(1, len(returns)) if (returns.iloc[i] > 0 and returns.iloc[i-1] < 0) or (returns.iloc[i] < 0 and returns.iloc[i-1] > 0)),
        'volatility': np.std(returns) * np.sqrt(252 * 390) * 100,
        'range_contraction': ((lunch_hour['high'].max() - lunch_hour['low'].min()) / (data.iloc[120:150]['high'].max() - data.iloc[120:150]['low'].min())) if len(data) >= 150 else 0
    }

def get_pre_close_hour_momentum(data: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Analyze last hour of trading.
    
    Args:
        data (pd.DataFrame): DataFrame containing minute data with OHLC columns
        
    Returns:
        Optional[Dict[str, float]]: Dictionary containing pre-close metrics or None if insufficient data
    """
    if len(data) < 390:  # Need full day data
        return None
    
    last_hour = data.iloc[-60:]
    returns = np.log(last_hour['close'] / last_hour['close'].shift(1)).dropna()
    
    # Calculate closing tendency
    last_15 = data.iloc[-15:]
    closing_tendency = ((last_15['close'].iloc[-1] - last_15['open'].iloc[0]) / last_15['open'].iloc[0] * 100)
    
    return {
        'return': ((last_hour['close'].iloc[-1] - last_hour['open'].iloc[0]) / last_hour['open'].iloc[0] * 100),
        'range': ((last_hour['high'].max() - last_hour['low'].min()) / last_hour['open'].iloc[0] * 100),
        'closing_tendency': closing_tendency,
        'volatility': np.std(returns) * np.sqrt(252 * 390) * 100,
        'momentum_strength': abs(returns.mean()) / np.std(returns) if np.std(returns) > 0 else 0
    }

def get_strongest_move_periods(data: pd.DataFrame) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Identify periods with strongest moves.
    
    Args:
        data (pd.DataFrame): DataFrame containing minute data with OHLC columns
        
    Returns:
        Optional[Dict[str, Dict[str, float]]]: Dictionary containing strongest move periods or None if insufficient data
    """
    if len(data) < 390:  # Need full day data
        return None
    
    # Calculate hourly ranges
    hourly_ranges = []
    for i in range(6):  # First 6 hours
        start_idx = i * 60
        end_idx = (i + 1) * 60
        if end_idx <= len(data):
            hour_data = data.iloc[start_idx:end_idx]
            hourly_ranges.append({
                'hour': i + 1,
                'range': ((hour_data['high'].max() - hour_data['low'].min()) / hour_data['open'].iloc[0] * 100)
            })
    
    # Calculate 30-min ranges
    thirty_min_ranges = []
    for i in range(12):  # 12 thirty-minute periods
        start_idx = i * 30
        end_idx = (i + 1) * 30
        if end_idx <= len(data):
            period_data = data.iloc[start_idx:end_idx]
            thirty_min_ranges.append({
                'period': i + 1,
                'range': ((period_data['high'].max() - period_data['low'].min()) / period_data['open'].iloc[0] * 100)
            })
    
    # Calculate 15-min ranges
    fifteen_min_ranges = []
    for i in range(24):  # 24 fifteen-minute periods
        start_idx = i * 15
        end_idx = (i + 1) * 15
        if end_idx <= len(data):
            period_data = data.iloc[start_idx:end_idx]
            fifteen_min_ranges.append({
                'period': i + 1,
                'range': ((period_data['high'].max() - period_data['low'].min()) / period_data['open'].iloc[0] * 100)
            })
    
    return {
        'strongest_hour': max(hourly_ranges, key=lambda x: x['range']) if hourly_ranges else None,
        'strongest_30min': max(thirty_min_ranges, key=lambda x: x['range']) if thirty_min_ranges else None,
        'strongest_15min': max(fifteen_min_ranges, key=lambda x: x['range']) if fifteen_min_ranges else None
    }

def main() -> None:
    """
    Main entry point for the pattern collector script.
    
    This function:
    1. Parses command line arguments
    2. Analyzes market correlation patterns
    3. Saves the patterns to a JSON file with versioning
    
    The script expects an input file with minute data and will output
    a pattern database JSON file to the specified output directory.
    """
    args = parse_arguments()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Construct the full filepath relative to project root
    full_filepath = os.path.join(project_root, args.input_file)
    
    print(f"\nUsing input file: {full_filepath}")
    
    df = analyze_market_correlation(full_filepath)
    patterns = analyze_patterns(df)
    
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'master_pattern_database.json')

    # --- Versioning logic ---
    # Find all versioned files
    versioned_files = glob.glob(os.path.join(output_dir, 'master_pattern_database_v*_*.json'))
    version_numbers = []
    version_pattern = re.compile(r'master_pattern_database_v(\d+)_')
    for f in versioned_files:
        base = os.path.basename(f)
        match = version_pattern.search(base)
        if match:
            try:
                v_num = int(match.group(1))
                version_numbers.append(v_num)
            except ValueError:
                continue
    if version_numbers:
        next_version = max(version_numbers) + 1
    else:
        next_version = 2  # v1 is already created manually
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # If master_pattern_database.json exists, back it up
    if os.path.exists(output_file):
        backup_file = os.path.join(output_dir, f"master_pattern_database_v{next_version}_{timestamp}.json")
        print(f"Backing up previous master pattern database to {backup_file}")
        shutil.move(output_file, backup_file)

    # Save the new database as master_pattern_database.json
    if patterns is not None:
        save_patterns_to_json(patterns, output_file)
        print("\nPattern Database Information:")
        print("- All patterns have been collected and saved")
    else:
        print("No patterns were found to save.")
        with open(output_file, 'w') as f:
            json.dump([], f)

if __name__ == "__main__":
    main()