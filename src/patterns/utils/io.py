"""
Input/Output Utilities
--------------------
Functions for handling file operations and data loading/saving.
"""

import logging
import json
import os
import glob
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
import datetime

__all__ = [
    'load_minute_data',
    'load_patterns',
    'save_patterns',
    'get_filter_levels',
    'get_latest_pattern_database',
    'get_pattern_database'
]

logger = logging.getLogger(__name__)

def load_minute_data(file_path: str) -> pd.DataFrame:
    """
    Load minute data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing minute data
        
    Returns:
        pd.DataFrame: DataFrame containing the loaded data with columns:
                     ['datetime', 'open', 'high', 'low', 'close', 'date']
        
    Raises:
        ValueError: If there's an error loading or processing the data
    """
    try:
        logger.info(f"Loading minute data from {os.path.relpath(file_path, os.getcwd())}")
        
        # Read the data without headers and assign column names
        df = pd.read_csv(file_path, header=None, 
                        names=['datetime', 'open', 'high', 'low', 'close'])
        
        # Convert datetime column to pandas datetime
        df['date'] = pd.to_datetime(df['datetime'])
        
        # Convert price columns to numeric
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Validate the data
        if df.empty:
            error_msg = "Loaded DataFrame is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if df['date'].isna().any():
            error_msg = "Invalid dates found in the data"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Successfully loaded {len(df)} rows of minute data")
        return df
        
    except FileNotFoundError as e:
        error_msg = f"Data file not found: {file_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except pd.errors.EmptyDataError as e:
        error_msg = f"Data file is empty: {file_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error loading minute data: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def load_patterns(file_path: str) -> List[Dict]:
    """
    Load patterns from a JSON file.
    
    Args:
        file_path (str): Path to the pattern database JSON file
        
    Returns:
        List[Dict]: List of loaded patterns
        
    Raises:
        ValueError: If there's an error loading the pattern database
    """
    try:
        logger.info(f"Loading patterns from {os.path.relpath(file_path, os.getcwd())}")
        with open(file_path, 'r') as f:
            patterns = json.load(f)
        logger.info(f"Successfully loaded {len(patterns)} patterns")
        return patterns
        
    except json.JSONDecodeError as e:
        error_msg = f"Error decoding pattern database JSON: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error loading pattern database: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def save_patterns(patterns: List[Dict], file_path: str) -> None:
    """
    Save patterns to a JSON file.
    
    Args:
        patterns (List[Dict]): List of patterns to save
        file_path (str): Path to save the patterns
        
    Raises:
        ValueError: If there's an error saving the pattern database
    """
    try:
        logger.info(f"Saving {len(patterns)} patterns to {file_path}")
        with open(file_path, 'w') as f:
            json.dump(patterns, f, indent=2)
        logger.info("Successfully saved patterns")
        
    except Exception as e:
        error_msg = f"Error saving pattern database: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def get_filter_levels(project_root: str) -> List[Dict]:
    """
    Load filter levels from JSON file.
    
    Args:
        project_root (str): Path to the project root directory
        
    Returns:
        List[Dict]: List of filter level configurations
        
    Raises:
        ValueError: If there's an error loading the filter levels
    """
    try:
        # Construct the full path to the config file
        filter_levels_path = os.path.join(project_root, 'config', 'filter_levels.json')
        
        logger.info(f"Loading filter levels from {os.path.relpath(filter_levels_path, project_root)}")
        with open(filter_levels_path, 'r') as f:
            config = json.load(f)
            return config['filter_levels']
            
    except FileNotFoundError as e:
        error_msg = f"Filter levels config file not found: {filter_levels_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Error decoding filter levels JSON: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error loading filter levels: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def get_latest_pattern_database(project_root: str) -> Tuple[str, int]:
    """
    Get the latest version of the pattern database.
    
    Args:
        project_root (str): Path to the project root directory
        
    Returns:
        Tuple[str, int]: Path to the latest pattern database and its version number
        
    Raises:
        ValueError: If no pattern database is found
    """
    try:
        processed_dir = os.path.join(project_root, 'data', 'processed')
        versioned_files = glob.glob(os.path.join(processed_dir, "master_pattern_database_v*_*.json"))
        
        if not versioned_files:
            # Fallback to non-versioned database
            default_db = os.path.join(processed_dir, "master_pattern_database.json")
            if os.path.exists(default_db):
                logger.info("Using default pattern database")
                return default_db, 0
            raise ValueError("No pattern database found")
            
        # Extract versions and timestamps
        def extract_info(f):
            base = os.path.basename(f)
            m = re.match(r"master_pattern_database_v(\d+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.json", base)
            if m:
                return int(m.group(1)), m.group(2)
            return 0, ''
            
        # Sort by version and timestamp
        versioned_files.sort(key=extract_info, reverse=True)
        latest_file = versioned_files[0]
        version = extract_info(latest_file)[0]
        
        logger.info(f"Using latest pattern database version v{version}: {os.path.relpath(latest_file, os.getcwd())}")
        return latest_file, version
        
    except Exception as e:
        error_msg = f"Error finding latest pattern database: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def get_pattern_database(project_root: str, version: Optional[int] = None, target_date: Optional[object] = None) -> Tuple[str, int]:
    """
    Get the pattern database for the specified version and/or date, or latest if not specified.
    
    Args:
        project_root (str): Path to the project root directory
        version (Optional[int]): Specific version number to use
        target_date (Optional[object]): Only use databases with date <= this (datetime.date, datetime, or str)
        
    Returns:
        Tuple[str, int]: Path to the pattern database and its version number
        
    Raises:
        ValueError: If the specified version/date is not found or no database exists
    """
    processed_dir = os.path.join(project_root, 'data', 'processed')
    versioned_files = glob.glob(os.path.join(processed_dir, "master_pattern_database_v*_*.json"))
    logger.debug(f"Pattern DB files found: {versioned_files}")

    def parse_file_info(f):
        base = os.path.basename(f)
        m = re.match(r"master_pattern_database_v(\d+)_((\d{4}-\d{2}-\d{2})(?:_(\d{2}-\d{2}-\d{2}))?)\.json", base)
        if not m:
            return None
        v = int(m.group(1))
        date_str = m.group(3)
        time_str = m.group(4)
        if time_str:
            dt_str = f"{date_str}_{time_str}"
            dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d_%H-%M-%S")
        else:
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return {
            'file': f,
            'version': v,
            'datetime': dt
        }

    # Parse all files
    file_infos = [parse_file_info(f) for f in versioned_files]
    file_infos = [fi for fi in file_infos if fi]
    logger.debug(f"Pattern DB files parsed: {[fi['file'] for fi in file_infos]}")

    # If no files, fallback to default
    if not file_infos:
        default_db = os.path.join(processed_dir, "master_pattern_database.json")
        if os.path.exists(default_db):
            logger.info("Using default pattern database")
            return default_db, 0
        raise ValueError("No pattern database found")

    # Parse target_date
    if target_date is not None:
        if isinstance(target_date, str):
            try:
                # Try full datetime first
                dt = datetime.datetime.strptime(target_date, "%Y-%m-%d_%H-%M-%S")
            except ValueError:
                dt = datetime.datetime.strptime(target_date, "%Y-%m-%d")
        elif isinstance(target_date, datetime.datetime):
            dt = target_date
        elif hasattr(target_date, 'year') and hasattr(target_date, 'month') and hasattr(target_date, 'day'):
            # datetime.date
            dt = datetime.datetime(target_date.year, target_date.month, target_date.day)
        else:
            raise ValueError(f"Unrecognized target_date type: {type(target_date)}")
    else:
        dt = None

    # Filter by version if needed
    if version is not None:
        file_infos = [fi for fi in file_infos if fi['version'] == version]
        if not file_infos:
            raise ValueError(f"No pattern database found for version v{version}")

    # Filter by date if needed
    if dt is not None:
        file_infos = [fi for fi in file_infos if fi['datetime'] <= dt]
        if not file_infos:
            raise ValueError(f"No pattern database found for the given date/version (<= {dt.strftime('%Y-%m-%d %H:%M:%S')})")

    # Pick the latest (by datetime)
    file_infos.sort(key=lambda fi: (fi['datetime'], fi['version']), reverse=True)
    chosen = file_infos[0]
    return chosen['file'], chosen['version'] 