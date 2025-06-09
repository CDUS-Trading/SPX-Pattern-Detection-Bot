#!/usr/bin/env python3

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

# Timezone constants
ET = ZoneInfo('America/New_York')
CT = ZoneInfo('America/Chicago')

def convert_et_to_ct(time_str: str) -> str:
    """
    Convert time string from ET to CT.
    
    Args:
        time_str (str): Time string in format "HH:MMAM/PM ET"
        
    Returns:
        str: Time string in format "HH:MMAM/PM CT"
    """
    # Remove timezone and parse time
    time_str = time_str.replace(' ET', '')
    time_obj = datetime.strptime(time_str, '%I:%M%p')
    
    # Create datetime with ET timezone
    et_time = time_obj.replace(tzinfo=ET)
    
    # Convert to CT
    ct_time = et_time.astimezone(CT)
    
    # Format back to string
    return ct_time.strftime('%I:%M%p') + ' CT'

def convert_ct_to_et(time_str: str) -> str:
    """
    Convert time string from CT to ET.
    
    Args:
        time_str (str): Time string in format "HH:MMAM/PM CT"
        
    Returns:
        str: Time string in format "HH:MMAM/PM ET"
    """
    # Remove timezone and parse time
    time_str = time_str.replace(' CT', '')
    time_obj = datetime.strptime(time_str, '%I:%M%p')
    
    # Create datetime with CT timezone
    ct_time = time_obj.replace(tzinfo=CT)
    
    # Convert to ET
    et_time = ct_time.astimezone(ET)
    
    # Format back to string
    return et_time.strftime('%I:%M%p') + ' ET'

def parse_time_with_tz(time_str: str, target_tz: str = 'ET') -> datetime:
    """
    Parse time string with timezone handling.
    
    Args:
        time_str (str): Time string in format "HH:MMAM/PM ET" or "HH:MMAM/PM CT"
        target_tz (str): Target timezone ('ET' or 'CT')
        
    Returns:
        datetime: Parsed datetime object in target timezone
    """
    # Remove timezone and parse time
    if ' ET' in time_str:
        time_str = time_str.replace(' ET', '')
        source_tz = ET
    elif ' CT' in time_str:
        time_str = time_str.replace(' CT', '')
        source_tz = CT
    else:
        raise ValueError("Time string must include timezone (ET or CT)")
    
    time_obj = datetime.strptime(time_str, '%I:%M%p')
    time_obj = time_obj.replace(tzinfo=source_tz)
    
    # Convert to target timezone
    if target_tz == 'ET':
        return time_obj.astimezone(ET)
    elif target_tz == 'CT':
        return time_obj.astimezone(CT)
    else:
        raise ValueError("Target timezone must be 'ET' or 'CT'")

def format_time_with_tz(time_obj: datetime, target_tz: str = 'ET') -> str:
    """
    Format datetime object with timezone.
    
    Args:
        time_obj (datetime): Datetime object
        target_tz (str): Target timezone ('ET' or 'CT')
        
    Returns:
        str: Formatted time string with timezone
    """
    if target_tz == 'ET':
        time_obj = time_obj.astimezone(ET)
        return time_obj.strftime('%I:%M%p') + ' ET'
    elif target_tz == 'CT':
        time_obj = time_obj.astimezone(CT)
        return time_obj.strftime('%I:%M%p') + ' CT'
    else:
        raise ValueError("Target timezone must be 'ET' or 'CT'")

def is_market_open(time_obj: datetime) -> bool:
    """
    Check if the market is open at the given time.
    
    Args:
        time_obj (datetime): Datetime object in any timezone
        
    Returns:
        bool: True if market is open, False otherwise
    """
    # Convert to ET
    et_time = time_obj.astimezone(ET)
    
    # Market hours in ET
    market_open = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Check if it's a weekday
    if et_time.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check if time is within market hours
    return market_open <= et_time <= market_close 