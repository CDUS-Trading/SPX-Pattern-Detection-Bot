"""
Market Metrics
------------
Functions for calculating various market metrics and indicators.
"""

import logging
from typing import Dict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def calculate_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate various market metrics needed for pattern detection.
    
    Args:
        data (pd.DataFrame): DataFrame containing market data with columns:
                           ['open', 'high', 'low', 'close', 'date']
    
    Returns:
        Dict[str, float]: Dictionary of calculated metrics
    """
    try:
        # Validate input data
        required_columns = ['open', 'high', 'low', 'close', 'date']
        if not all(col in data.columns for col in required_columns):
            error_msg = f"Input data must contain columns: {required_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if data.empty:
            error_msg = "Input data is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        metrics: Dict[str, float] = {}
        
        # Calculate last hour, last 30min, last 15min returns
        last_hour = data.tail(60)
        last_30min = data.tail(30)
        last_15min = data.tail(15)
        
        # Calculate returns
        metrics['last_hour'] = calculate_return(last_hour, 60)
        metrics['last_30min'] = calculate_return(last_30min, 30)
        metrics['last_15min'] = calculate_return(last_15min, 15)
        
        # Calculate volatility metrics
        metrics['close_vol'] = calculate_volatility(last_30min)
        metrics['last_hour_vol'] = calculate_volatility(last_hour)
        
        # Calculate close strength and day range
        if len(data) > 0:
            day_high = data['high'].max()
            day_low = data['low'].min()
            last_close = data['close'].iloc[-1]
            
            metrics['close_strength'] = calculate_close_strength(last_close, day_high, day_low)
            metrics['day_range'] = calculate_day_range(day_high, day_low, data['open'].iloc[0])
        else:
            metrics['close_strength'] = 0
            metrics['day_range'] = 0
        
        # Calculate early day volatility metrics
        for minutes in [15, 30, 60]:
            period_data = data[:minutes]
            metrics[f'first_{minutes}min_vol'] = calculate_volatility(period_data)
        
        # Calculate pre-lunch momentum and trend strength
        if len(data) >= 240:  # Need at least 4 hours of data
            metrics['pre_lunch_momentum'] = calculate_pre_lunch_momentum(data)
            metrics['morning_trend_strength'] = calculate_morning_trend_strength(data)
        else:
            metrics['pre_lunch_momentum'] = 0
            metrics['morning_trend_strength'] = 0
        
        # Calculate first 5 minutes metrics
        if len(data) >= 5:
            first_5 = data.iloc[:5]
            metrics.update(calculate_first_5_minutes_metrics(first_5))
        else:
            for metric in ['first_5min_return', 'first_5min_range', 'first_5min_high_test', 
                         'first_5min_low_test', 'first_5min_vol']:
                metrics[metric] = 0
        
        # Calculate power hour metrics
        if len(data) >= 60:
            power_hour = data.iloc[:60]
            metrics.update(calculate_power_hour_metrics(power_hour))
        else:
            for metric in ['power_hour_return', 'power_hour_range', 'power_hour_trend_changes', 
                         'power_hour_vol', 'power_hour_momentum']:
                metrics[metric] = 0
        
        # Calculate lunch hour metrics
        if len(data) >= 210:  # Need at least 3.5 hours of data
            lunch_hour = data.iloc[150:210]  # 12:00-1:00
            metrics.update(calculate_lunch_hour_metrics(lunch_hour, data))
        else:
            for metric in ['lunch_hour_range', 'lunch_hour_return', 'lunch_hour_direction_changes', 
                         'lunch_hour_vol', 'lunch_hour_range_contraction']:
                metrics[metric] = 0
        
        logger.debug(f"Calculated {len(metrics)} metrics")
        return metrics
        
    except Exception as e:
        error_msg = f"Error calculating metrics: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def calculate_return(data: pd.DataFrame, min_length: int) -> float:
    """
    Calculate return for a period of data.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data
        min_length (int): Minimum number of periods required
        
    Returns:
        float: Return percentage
    """
    if len(data) >= min_length:
        return ((data['close'].iloc[-1] - data['open'].iloc[0]) / 
                data['open'].iloc[0] * 100)
    return 0

def calculate_volatility(data: pd.DataFrame) -> float:
    """
    Calculate volatility for a period of data.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data
        
    Returns:
        float: Volatility percentage
    """
    if len(data) > 0:
        return ((data['high'].max() - data['low'].min()) / 
                data['open'].iloc[0] * 100)
    return 0

def calculate_close_strength(close: float, high: float, low: float) -> float:
    """
    Calculate closing price relative to day's range.
    
    Args:
        close (float): Closing price
        high (float): Day's high
        low (float): Day's low
        
    Returns:
        float: Close strength indicator (-1 to 1)
    """
    if high > low:
        return ((close - low) / (high - low) - 0.5) * 2
    return 0

def calculate_day_range(high: float, low: float, open_price: float) -> float:
    """
    Calculate day's trading range as a percentage.
    
    Args:
        high (float): Day's high
        low (float): Day's low
        open_price (float): Opening price
        
    Returns:
        float: Day range percentage
    """
    return ((high - low) / open_price * 100)

def calculate_pre_lunch_momentum(data: pd.DataFrame) -> float:
    """
    Calculate momentum score for the first 4 hours.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data
        
    Returns:
        float: Momentum score
    """
    weights = [0.4, 0.3, 0.2, 0.1]
    momentum_score = 0
    for hour in range(4):
        start_idx = hour * 60
        end_idx = (hour + 1) * 60
        hour_data = data[start_idx:end_idx]
        if len(hour_data) >= 60:
            hour_return = ((hour_data['close'].iloc[-1] - hour_data['open'].iloc[0]) / 
                        hour_data['open'].iloc[0] * 100)
            momentum_score += hour_return * weights[hour]
    return momentum_score

def calculate_morning_trend_strength(data: pd.DataFrame) -> int:
    """
    Calculate trend strength in the morning session.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data
        
    Returns:
        int: Number of consecutive moves in the same direction
    """
    moves = []
    for hour in range(4):
        start_idx = hour * 60
        end_idx = (hour + 1) * 60
        hour_data = data[start_idx:end_idx]
        if len(hour_data) >= 60:
            hour_return = ((hour_data['close'].iloc[-1] - hour_data['open'].iloc[0]) / 
                          hour_data['open'].iloc[0] * 100)
            moves.append(hour_return)
    
    consecutive = 1
    for i in range(1, len(moves)):
        if (moves[i] > 0 and moves[i-1] > 0) or (moves[i] < 0 and moves[i-1] < 0):
            consecutive += 1
    return consecutive

def calculate_first_5_minutes_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate metrics for the first 5 minutes of trading.
    
    Args:
        data (pd.DataFrame): DataFrame containing price data
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    return {
        'first_5min_return': ((data['close'].iloc[-1] - data['open'].iloc[0]) / 
                             data['open'].iloc[0] * 100),
        'first_5min_range': ((data['high'].max() - data['low'].min()) / 
                            data['open'].iloc[0] * 100),
        'first_5min_high_test': ((data['high'].max() - data['open'].iloc[0]) / 
                                data['open'].iloc[0] * 100),
        'first_5min_low_test': ((data['open'].iloc[0] - data['low'].min()) / 
                               data['open'].iloc[0] * 100),
        'first_5min_vol': np.std(np.log(data['close'] / data['close'].shift(1)).dropna()) * 
                        np.sqrt(252 * 390) * 100
    }

def calculate_power_hour_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate metrics for the power hour (9:30-10:30).
    
    Args:
        data (pd.DataFrame): DataFrame containing price data
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    returns = np.log(data['close'] / data['close'].shift(1)).dropna()
    
    # Count trend changes
    direction_changes = sum(1 for i in range(1, len(returns))
                          if (returns.iloc[i] > 0 and returns.iloc[i-1] < 0) or 
                             (returns.iloc[i] < 0 and returns.iloc[i-1] > 0))
    
    return {
        'power_hour_return': ((data['close'].iloc[-1] - data['open'].iloc[0]) / 
                             data['open'].iloc[0] * 100),
        'power_hour_range': ((data['high'].max() - data['low'].min()) / 
                            data['open'].iloc[0] * 100),
        'power_hour_trend_changes': direction_changes,
        'power_hour_vol': np.std(returns) * np.sqrt(252 * 390) * 100,
        'power_hour_momentum': len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
    }

def calculate_lunch_hour_metrics(data: pd.DataFrame, full_data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate metrics for the lunch hour (12:00-1:00).
    
    Args:
        data (pd.DataFrame): DataFrame containing price data
        full_data (pd.DataFrame): Complete day's data
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    returns = np.log(data['close'] / data['close'].shift(1)).dropna()
    
    metrics = {
        'lunch_hour_range': ((data['high'].max() - data['low'].min()) / 
                            data['open'].iloc[0] * 100),
        'lunch_hour_return': ((data['close'].iloc[-1] - data['open'].iloc[0]) / 
                             data['open'].iloc[0] * 100),
        'lunch_hour_direction_changes': sum(1 for i in range(1, len(returns))
                                          if (returns.iloc[i] > 0 and returns.iloc[i-1] < 0) or 
                                             (returns.iloc[i] < 0 and returns.iloc[i-1] > 0)),
        'lunch_hour_vol': np.std(returns) * np.sqrt(252 * 390) * 100
    }
    
    # Calculate range contraction
    if len(full_data) >= 150:
        prev_hour = full_data.iloc[120:150]
        metrics['lunch_hour_range_contraction'] = (
            (data['high'].max() - data['low'].min()) / 
            (prev_hour['high'].max() - prev_hour['low'].min())
        )
    else:
        metrics['lunch_hour_range_contraction'] = 0
        
    return metrics 