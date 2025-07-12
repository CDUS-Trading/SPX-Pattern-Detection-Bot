#!/usr/bin/env python3

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import json
from tqdm import tqdm
import contextlib

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.patterns.utils import metrics, dates
from src.patterns.utils.io import load_minute_data, get_pattern_database, get_filter_levels
from src.patterns.core.pattern_detector_class import PatternDetector, PatternError, PatternDatabaseError

# Set up module-level logger
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def suppress_logging():
    """Context manager to temporarily disable logging."""
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)

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
    Parse command line arguments for the 8:31 AM pattern collector.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Collect 8:31 AM patterns for analysis')
    
    # Data input/output arguments
    parser.add_argument('--data', type=str, help='Path to the minute data CSV file', 
                      default='data/SPX_full_1min.txt')
    parser.add_argument('--start-date', type=str, help='Start date for pattern collection (YYYY-MM-DD)',
                      default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    parser.add_argument('--end-date', type=str, help='End date for pattern collection (YYYY-MM-DD)',
                      default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--pattern-db', type=str, help='Path to pattern database file')
    parser.add_argument('--output', type=str, help='Path to save the 8:31 AM patterns',
                      default='data/831am_patterns.json')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    return parser.parse_args()

def collect_831am_patterns(detector: PatternDetector, data: pd.DataFrame, 
                          start_date: datetime.date, end_date: datetime.date) -> List[Dict]:
    """
    Collect 8:31 AM patterns using recursive filter approach.
    
    Args:
        detector (PatternDetector): Pattern detector instance
        data (pd.DataFrame): Minute data DataFrame
        start_date (datetime.date): Start date for pattern collection
        end_date (datetime.date): End date for pattern collection
        
    Returns:
        List[Dict]: List of 8:31 AM patterns found
    """
    patterns = []
    trading_days = dates.get_trading_days(start_date, end_date)
    
    # Create progress bar for trading days
    pbar = tqdm(trading_days, desc="Processing trading days", unit="day")
    
    for date in pbar:
        pbar.set_description(f"Processing {date}")
        
        with suppress_logging():
            # Get data for analysis
            analysis_data, _ = dates.get_data_for_date(data, date.strftime('%Y-%m-%d'))
            if analysis_data is None:
                continue
            
            # Calculate metrics
            metrics_dict = metrics.calculate_metrics(analysis_data)
            
            # Get filter levels
            filter_levels = get_filter_levels(project_root)
            
            # Try each filter level until we find a pattern
            for level in filter_levels:
                matched_patterns = detector.detect_patterns(
                    metrics_dict,
                    min_tp=level['min_tp'],
                    min_success_rate=level['min_success_rate'],
                    min_occurrences=level['min_occurrences'],
                    min_risk_reward=level['min_risk_reward']
                )
                
                if matched_patterns:
                    # Filter for 8:31 AM patterns
                    for pattern in matched_patterns:
                        strategy = detector.generate_trading_strategy(pattern)
                        timeframe = strategy['timeframe']
                        
                        # Check if pattern starts at 8:31 AM
                        if timeframe.startswith('8:31AM'):
                            # Get previous day's close price
                            prev_close = dates.get_previous_close(data, date)
                            if prev_close is None:
                                continue
                            
                            # Create pattern data in the required format
                            pattern_data = {
                                "pattern_day": date.strftime('%A'),
                                "pattern_date": date.strftime('%Y-%m-%d'),
                                "based_on_day": (date - timedelta(days=1)).strftime('%A'),
                                "based_on_date": (date - timedelta(days=1)).strftime('%Y-%m-%d'),
                                "close_price": prev_close,
                                "filter_level": level['name'],
                                "patterns": {
                                    "sessions": {
                                        "morning": [],
                                        "mixed": [],
                                        "afternoon": []
                                    }
                                }
                            }
                            
                            # Add the pattern to the appropriate session
                            session = strategy['period']
                            if session not in pattern_data["patterns"]["sessions"]:
                                session = "morning"  # Default to morning for 8:31 AM patterns
                            
                            # Extract entry and exit times from timeframe
                            if '-' in timeframe:
                                entry_time, exit_time = timeframe.split('-')
                                entry_time = entry_time.strip()
                                exit_time = exit_time.strip()
                            else:
                                entry_time = timeframe.strip()
                                exit_time = timeframe.strip()
                            
                            # Ensure both times have CT
                            if 'CT' not in entry_time:
                                entry_time = f"{entry_time} CT"
                            if 'CT' not in exit_time:
                                exit_time = f"{exit_time} CT"
                            
                            # Calculate target and stop loss points
                            if strategy['direction'] == 'bullish':
                                tp_pct = float(strategy['target'].split('+')[1].split('%')[0])
                                sl_pct = float(strategy['stop_loss'].split('-')[1].split('%')[0])
                                tp_points = prev_close * (tp_pct / 100)
                                sl_points = prev_close * (sl_pct / 100)
                            else:  # bearish
                                tp_pct = float(strategy['target'].split('-')[1].split('%')[0])
                                sl_pct = float(strategy['stop_loss'].split('+')[1].split('%')[0])
                                tp_points = prev_close * (tp_pct / 100)
                                sl_points = prev_close * (sl_pct / 100)
                            
                            # Add pattern to the session
                            pattern_data["patterns"]["sessions"][session].append({
                                "entry_time": entry_time,
                                "exit_time": exit_time,
                                "direction": "Buy" if strategy['direction'] == 'bullish' else "Sell",
                                "target_points": round(tp_points, 2),
                                "stop_loss_points": round(sl_points, 2),
                                "success_rate": round(float(strategy['success_rate']), 2),
                                "historical_dates": pattern.get('pattern_dates', [])
                            })
                            
                            patterns.append(pattern_data)
                            pbar.set_postfix({"Patterns found": len(patterns)})
                            break  # Stop after finding first 8:31 AM pattern
                    
                    if any(p['pattern_date'] == date.strftime('%Y-%m-%d') for p in patterns):
                        break  # Stop if we found a pattern for this date
    
    return patterns

def main() -> None:
    """
    Main execution function for collecting 8:31 AM patterns.
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Set up logging
        setup_logging(logging.DEBUG if args.debug else logging.INFO)
        
        # Convert dates
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
        
        # Load data
        data_path = os.path.join(project_root, args.data)
        try:
            with suppress_logging():
                full_data = load_minute_data(data_path)
        except ValueError as e:
            if "Data file not found" in str(e):
                logger.error("Required minute data files are missing.")
                logger.error("Please run the following command to download the required data files:")
                logger.error("python src/data/downloader.py")
                return
            raise
        
        # Get pattern database path
        try:
            with suppress_logging():
                pattern_db_path, version = get_pattern_database(project_root)
                logger.info(f"Using pattern database version v{version}")
        except ValueError as e:
            logger.error(str(e))
            return
        
        # Initialize pattern detector
        with suppress_logging():
            detector = PatternDetector(pattern_db_path)
        
        # Collect 8:31 AM patterns
        patterns = collect_831am_patterns(detector, full_data, start_date, end_date)
        
        # Save patterns to JSON
        output_path = os.path.join(project_root, args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(patterns, f, indent=2)
        
        logger.info(f"Saved {len(patterns)} 8:31 AM patterns to {args.output}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise PatternError(f"Error in main execution: {str(e)}")

if __name__ == '__main__':
    main()
