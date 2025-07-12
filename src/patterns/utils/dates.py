"""
Date Utilities
------------
Functions for handling dates and trading days.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Union, Tuple, List
import pandas as pd
import pandas_market_calendars as mcal
import os

logger = logging.getLogger(__name__)

def get_next_trading_day(current_date: Union[str, datetime.date]) -> datetime.date:
    """
    Calculate the next trading day, accounting for weekends and holidays.
    
    Args:
        current_date (Union[str, datetime.date]): Current date as string (YYYY-MM-DD) or datetime.date object
        
    Returns:
        datetime.date: The next trading day
        
    Raises:
        ValueError: If there's an error calculating the next trading day
    """
    try:
        if isinstance(current_date, str):
            current_date = datetime.strptime(current_date, '%Y-%m-%d').date()
        
        # Get NYSE calendar
        nyse = mcal.get_calendar('NYSE')
        
        # Get next trading day
        next_day = current_date + timedelta(days=1)
        max_attempts = 10  # Prevent infinite loop
        attempts = 0
        
        while attempts < max_attempts:
            schedule = nyse.schedule(start_date=next_day, end_date=next_day)
            if not schedule.empty:
                logger.debug(f"Next trading day found: {next_day}")
                return next_day
            next_day += timedelta(days=1)
            attempts += 1
            
        error_msg = f"Could not find next trading day after {max_attempts} attempts"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    except Exception as e:
        error_msg = f"Error calculating next trading day: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def get_most_recent_trading_day_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Get the most recent trading day's data from a DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame containing market data with a 'date' column
        
    Returns:
        pd.DataFrame: DataFrame containing only the most recent trading day's data
        
    Raises:
        ValueError: If the input data is invalid or empty
    """
    try:
        if data.empty:
            error_msg = "Input DataFrame is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if 'date' not in data.columns:
            error_msg = "Input DataFrame must have a 'date' column"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Get the latest date in the dataset
        latest_date = data['date'].dt.date.max()
        logger.debug(f"Most recent trading day: {latest_date}")
        
        # Filter data for the most recent trading day
        recent_data = data[data['date'].dt.date == latest_date].copy()
        
        if recent_data.empty:
            error_msg = f"No data found for the most recent trading day: {latest_date}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        return recent_data
        
    except Exception as e:
        error_msg = f"Error getting most recent trading day data: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def get_data_for_date(data: pd.DataFrame, target_date: Optional[Union[str, datetime.date]] = None) -> Tuple[Optional[pd.DataFrame], datetime.date]:
    """
    Get data for a specific date or the most recent date if none specified.
    
    Args:
        data (pd.DataFrame): DataFrame containing market data
        target_date (Optional[Union[str, datetime.date]]): Target date as string (YYYY-MM-DD) 
                                                         or datetime.date object
        
    Returns:
        Tuple[Optional[pd.DataFrame], datetime.date]: Tuple containing:
            - DataFrame with data for the target date (or None if not found)
            - The target date as datetime.date object
            
    Raises:
        ValueError: If there's an error processing the data
    """
    try:
        if target_date is None:
            # Use the last date in the dataset
            target_date = data['date'].dt.date.max()
            logger.info(f"No target date specified, using most recent date: {target_date}")
        elif isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
            logger.info(f"Using specified target date: {target_date}")
        
        # Get data for the target date
        day_data = data[data['date'].dt.date == target_date].copy()
        if len(day_data) == 0:
            logger.warning(f"No data found for target date: {target_date}")
            return None, target_date
            
        logger.info(f"Found data for target date: {target_date}")
        return day_data, target_date
        
    except Exception as e:
        error_msg = f"Error getting data for date: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def get_previous_close(data: pd.DataFrame, target_date: datetime.date) -> Optional[float]:
    """
    Get the closing price from the latest day in the dataset.
    
    Args:
        data (pd.DataFrame): DataFrame containing market data
        target_date (datetime.date): Target date to find close for
        
    Returns:
        Optional[float]: Latest day's closing price, or None if not found
        
    Raises:
        ValueError: If there's an error processing the data
    """
    try:
        # Get data for the target date
        day_data = data[data['date'].dt.date == target_date].copy()
        if len(day_data) > 0:
            close_price = day_data['close'].iloc[-1]
            logger.info(f"Found closing price {close_price:.2f} for {target_date}")
            return close_price
            
        logger.warning(f"No data found for target date: {target_date}")
        return None
        
    except Exception as e:
        error_msg = f"Error getting closing price: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def get_first_trading_day() -> str:
    """
    Get the first trading day in our dataset.
    
    Returns:
        str: First trading day in YYYY-MM-DD format
    """
    try:
        # Read the minute data file to get the first date
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        data_path = os.path.join(project_root, 'data', 'SPX_full_1min_CT.txt')
        
        df = pd.read_csv(data_path, names=['datetime', 'open', 'high', 'low', 'close'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        first_date = df['datetime'].dt.date.min()
        
        logger.info(f"First trading day: {first_date}")
        return first_date.strftime('%Y-%m-%d')
        
    except Exception as e:
        error_msg = f"Error getting first trading day: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def get_last_trading_day() -> str:
    """
    Get the last trading day in our dataset.
    
    Returns:
        str: Last trading day in YYYY-MM-DD format
    """
    try:
        # Read the minute data file to get the last date
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        data_path = os.path.join(project_root, 'data', 'SPX_full_1min_CT.txt')
        
        df = pd.read_csv(data_path, names=['datetime', 'open', 'high', 'low', 'close'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        last_date = df['datetime'].dt.date.max()
        
        logger.info(f"Last trading day: {last_date}")
        return last_date.strftime('%Y-%m-%d')
        
    except Exception as e:
        error_msg = f"Error getting last trading day: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def get_trading_days(start_date: datetime.date, end_date: datetime.date) -> List[datetime.date]:
    """
    Get list of trading days between start and end dates.
    
    Args:
        start_date (datetime.date): Start date
        end_date (datetime.date): End date
        
    Returns:
        List[datetime.date]: List of trading days
    """
    try:
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        trading_days = schedule.index.date.tolist()
        logger.info(f"Found {len(trading_days)} trading days between {start_date} and {end_date}")
        return trading_days
    except Exception as e:
        error_msg = f"Error getting trading days: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def get_analysis_data(target_date: datetime.date) -> Optional[pd.DataFrame]:
    """
    Get minute data for analysis for a specific date.
    
    Args:
        target_date (datetime.date): Target date to get data for
        
    Returns:
        Optional[pd.DataFrame]: DataFrame with minute data for the target date, or None if not found
    """
    try:
        # Read the minute data file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        data_path = os.path.join(project_root, 'data', 'SPX_full_1min_CT.txt')
        
        df = pd.read_csv(data_path, names=['datetime', 'open', 'high', 'low', 'close'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date
        
        # Filter for target date
        day_data = df[df['date'] == target_date].copy()
        if day_data.empty:
            logger.warning(f"No data found for {target_date}")
            return None
            
        logger.info(f"Found {len(day_data)} minute bars for {target_date}")
        return day_data
        
    except Exception as e:
        error_msg = f"Error getting analysis data: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) 