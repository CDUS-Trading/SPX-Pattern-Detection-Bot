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
    'get_pattern_database',
    'load_runs_database',
    'extract_slot_runs',
    'get_latest_spx_date',
    'create_and_save_copy_days',
    'get_constraint_levels'
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
        
        # Read the data with headers
        df = pd.read_csv(file_path)
        
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

def load_runs_database(file_path: str) -> List[Dict]:
    """
    Load the runs database from JSON file.
    
    Args:
        file_path (str): Path to the runs database JSON file
        
    Returns:
        List[Dict]: List of daily run data
        
    Raises:
        FileNotFoundError: If the runs database file is not found
        ValueError: If there's an error loading or parsing the runs database
    """
    try:
        logger.info(f"Loading runs database from {os.path.relpath(file_path, os.getcwd())}")
        with open(file_path, 'r') as f:
            runs_data = json.load(f)
        logger.info(f"Successfully loaded runs database with {len(runs_data)} days")
        return runs_data
    except FileNotFoundError:
        error_msg = f"Runs database not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in runs database: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error loading runs database: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def extract_slot_runs(runs_data: List[Dict], target_date: str, slot_number: int) -> Optional[List[Dict]]:
    """
    Extract runs for a specific slot and date from the runs database.
    
    Args:
        runs_data (List[Dict]): The runs database data
        target_date (str): Target date in YYYY-MM-DD format
        slot_number (int): Slot number (1-6)
        
    Returns:
        Optional[List[Dict]]: List of runs for the specified slot and date, or None if not found
    """
    slot_name = f'slot_{slot_number}'
    for day_data in runs_data:
        if day_data.get('date') == target_date:
            time_buckets = day_data.get('time_buckets', {})
            slot_data = time_buckets.get(slot_name, {})
            runs = slot_data.get('runs', [])
            logger.debug(f"Found {len(runs)} runs for {target_date} slot_{slot_number}")
            return runs
    
    logger.debug(f"No runs found for {target_date} slot_{slot_number}")
    return None

def get_latest_spx_date(data_file: str) -> str:
    """
    Get the latest date from the SPX data file.
    
    Args:
        data_file (str): Path to the SPX data file
        
    Returns:
        str: Latest date in YYYY-MM-DD format
        
    Raises:
        FileNotFoundError: If the SPX data file is not found
        ValueError: If there's an error reading the SPX data file
    """
    try:
        logger.info(f"Getting latest date from SPX data file: {os.path.relpath(data_file, os.getcwd())}")
        last_date = None
        with open(data_file, 'r') as f:
            next(f)  # Skip header line
            for line in f:
                date_str = line.split(',')[0][:10]
                last_date = date_str
        
        if last_date is None:
            raise ValueError("No data found in SPX data file")
        
        logger.info(f"Latest date in SPX data: {last_date}")
        return last_date
    except FileNotFoundError:
        error_msg = f"SPX data file not found: {data_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"Error reading SPX data file: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def create_and_save_copy_days(
    target_date: str,
    input_slot: int,
    target_slot: int,
    target_runs: List[Dict],
    historical_matches: List[Tuple[str, List[Dict]]],
    constraint_info: Dict,
    all_runs_mode: bool,
    runs_data: List[Dict],
    file_path: str = 'copy_days.json'
) -> Dict:
    """
    Create and save comprehensive copy days data with slot information.
    
    This function handles the entire copy days workflow:
    1. Creates the copy days data structure
    2. Adds slot information for each copy day
    3. Saves the data to JSON file
    
    Args:
        target_date (str): Target date being analyzed
        input_slot (int): Input slot number
        target_slot (int): Target slot number
        target_runs (List[Dict]): Target runs from input slot
        historical_matches (List[Tuple[str, List[Dict]]]): Historical matches found
        constraint_info (Dict): Constraint information used for matching
        all_runs_mode (bool): Whether all runs mode was used
        runs_data (List[Dict]): Full runs database data
        file_path (str): Path to save the copy days data (default: 'copy_days.json')
        
    Returns:
        Dict: The created copy days data structure
        
    Raises:
        ValueError: If there's an error creating or saving the copy days data
    """
    try:
        logger.info(f"Creating copy days data for {target_date}")
        
        # Create enhanced copy days data with slot information
        copy_days_data = {
            "target_date": target_date,
            "input_slot": input_slot,
            "target_slot": target_slot,
            "mode": "all_runs",
            "constraints_used": constraint_info,
            "target_runs": [
                {
                    "direction": run['direction'],
                    "total_move": run['total_move'],
                    "start_time": run['start_time'],
                    "end_time": run['end_time'],
                    "duration": run['duration']
                } for run in target_runs
            ],
            "copy_days": []
        }
        
        # Add slot information for each copy day
        for date, matching_runs in historical_matches:
            # Get target slot runs for this copy day
            target_slot_runs = extract_slot_runs(runs_data, date, target_slot)
            
            # Calculate slot direction (net move for the entire slot)
            slot_direction = "neutral"
            slot_net_move = 0.0
            
            if target_slot_runs:
                # Calculate net move for the entire slot
                slot_net_move = sum(run['total_move'] for run in target_slot_runs)
                slot_direction = "bull" if slot_net_move > 0 else "bear" if slot_net_move < 0 else "neutral"
            
            copy_day_info = {
                "date": date,
                "matching_runs": [
                    {
                        "direction": run['direction'],
                        "total_move": run['total_move'],
                        "start_time": run['start_time'],
                        "end_time": run['end_time'],
                        "duration": run['duration']
                    } for run in matching_runs
                ],
                "target_slot": {
                    "direction": slot_direction,
                    "net_move": slot_net_move,
                    "all_runs": target_slot_runs
                }
            }
            copy_days_data["copy_days"].append(copy_day_info)
        
        # Save copy days to file
        try:
            logger.info(f"Saving copy days data to {os.path.relpath(file_path, os.getcwd())}")
            with open(file_path, 'w') as f:
                json.dump(copy_days_data, f, indent=2)
            logger.info("Successfully saved copy days data")
        except Exception as e:
            error_msg = f"Error saving copy days data: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Successfully created and saved copy days data with {len(copy_days_data['copy_days'])} copy days")
        return copy_days_data
        
    except Exception as e:
        error_msg = f"Error creating and saving copy days data: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def get_constraint_levels(project_root: str) -> List[Dict]:
    """
    Load constraint levels from JSON config file.
    
    Args:
        project_root (str): Path to the project root directory
        
    Returns:
        List[Dict]: List of constraint level configurations
        
    Raises:
        ValueError: If there's an error loading the constraint levels
    """
    try:
        # Construct the full path to the config file
        constraint_levels_path = os.path.join(project_root, 'config', 'constraint_levels.json')
        
        logger.info(f"Loading constraint levels from {os.path.relpath(constraint_levels_path, project_root)}")
        with open(constraint_levels_path, 'r') as f:
            config = json.load(f)
            return config['constraint_levels']
            
    except FileNotFoundError as e:
        error_msg = f"Constraint levels config file not found: {constraint_levels_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Error decoding constraint levels JSON: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error loading constraint levels: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) 