#!/usr/bin/env python3

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
import contextlib

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.patterns.utils.io import load_minute_data

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

def get_market_direction(data: pd.DataFrame, date: str, entry_time: str, exit_time: str) -> Tuple[str, float]:
    """
    Get the actual market direction for a given time window.
    
    Args:
        data (pd.DataFrame): Minute data DataFrame
        date (str): Date to analyze
        entry_time (str): Entry time (e.g., "8:31AM CT")
        exit_time (str): Exit time (e.g., "9:30AM CT")
        
    Returns:
        Tuple[str, float]: (direction, points_moved)
    """
    try:
        # Convert times to datetime
        entry_dt = pd.to_datetime(f"{date} {entry_time.replace(' CT', '')}")
        exit_dt = pd.to_datetime(f"{date} {exit_time.replace(' CT', '')}")
        
        # Get data for the date
        day_data = data[data['date'].dt.date == pd.to_datetime(date).date()].copy()
        if day_data.empty:
            return None, 0
        
        # Ensure datetime column is in datetime format
        day_data['datetime'] = pd.to_datetime(day_data['datetime'])
        
        # Get entry and exit prices
        entry_mask = day_data['datetime'] >= entry_dt
        exit_mask = day_data['datetime'] <= exit_dt
        
        if not any(entry_mask) or not any(exit_mask):
            return None, 0
            
        entry_price = day_data[entry_mask]['open'].iloc[0]
        exit_price = day_data[exit_mask]['close'].iloc[-1]
        
        # Calculate movement
        points_moved = exit_price - entry_price
        
        # Determine direction
        direction = "bullish" if points_moved > 0 else "bearish"
        
        return direction, points_moved
        
    except Exception as e:
        logger.warning(f"Error getting market direction for {date}: {str(e)}")
        return None, 0

def analyze_direction_accuracy(patterns: List[Dict], data: pd.DataFrame) -> Dict:
    """
    Analyze the direction accuracy of 8:31 AM patterns.
    
    Args:
        patterns (List[Dict]): List of 8:31 AM patterns
        data (pd.DataFrame): Minute data DataFrame
        
    Returns:
        Dict: Analysis results
    """
    results = {
        "total_patterns": len(patterns),
        "analyzed_patterns": 0,
        "correct_directions": 0,
        "direction_accuracy": 0.0,
        "by_filter_level": {},
        "details": []
    }
    
    # Create progress bar
    pbar = tqdm(patterns, desc="Analyzing pattern directions", unit="pattern")
    
    for pattern in pbar:
        pattern_date = pattern["pattern_date"]
        filter_level = pattern["filter_level"]
        
        # Initialize filter level stats if not exists
        if filter_level not in results["by_filter_level"]:
            results["by_filter_level"][filter_level] = {
                "correct": 0,
                "total": 0,
                "accuracy": 0.0
            }
        
        # Get pattern details
        session_patterns = pattern["patterns"]["sessions"]["morning"]
        if not session_patterns:
            continue
            
        pattern_info = session_patterns[0]  # Take first pattern
        predicted_direction = "bullish" if pattern_info["direction"] == "Buy" else "bearish"
        
        # Get actual market direction
        actual_direction, points_moved = get_market_direction(
            data, pattern_date, pattern_info["entry_time"], pattern_info["exit_time"]
        )
        
        if actual_direction is None:
            continue
            
        # Update statistics
        results["analyzed_patterns"] += 1
        results["by_filter_level"][filter_level]["total"] += 1
        
        is_correct = predicted_direction == actual_direction
        if is_correct:
            results["correct_directions"] += 1
            results["by_filter_level"][filter_level]["correct"] += 1
        
        # Store pattern details
        pattern_detail = {
            "date": pattern_date,
            "filter_level": filter_level,
            "predicted": predicted_direction,
            "actual": actual_direction,
            "points_moved": points_moved,
            "is_correct": is_correct
        }
        results["details"].append(pattern_detail)
        
        # Update progress bar
        pbar.set_postfix({
            "Correct": results["correct_directions"],
            "Analyzed": results["analyzed_patterns"]
        })
    
    # Calculate overall accuracy
    if results["analyzed_patterns"] > 0:
        results["direction_accuracy"] = (results["correct_directions"] / results["analyzed_patterns"]) * 100
    
    # Calculate filter level accuracies
    for level in results["by_filter_level"]:
        stats = results["by_filter_level"][level]
        if stats["total"] > 0:
            stats["accuracy"] = (stats["correct"] / stats["total"]) * 100
    
    return results

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Analyze 8:31 AM pattern direction accuracy')
    
    # Add date range arguments
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for analysis (YYYY-MM-DD)',
        default=(datetime.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for analysis (YYYY-MM-DD)',
        default=datetime.now().strftime('%Y-%m-%d')
    )
    
    # Add debug flag
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()

def filter_patterns_by_date(patterns: List[Dict], start_date: str, end_date: str) -> List[Dict]:
    """
    Filter patterns to only include those within the specified date range.
    
    Args:
        patterns (List[Dict]): List of patterns to filter
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        List[Dict]: Filtered list of patterns
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    return [
        pattern for pattern in patterns
        if start_dt <= pd.to_datetime(pattern["pattern_date"]) <= end_dt
    ]

def main() -> None:
    """
    Main execution function for direction accuracy analysis.
    """
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Set up logging
        log_level = logging.DEBUG if args.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Load minute data
        data_path = os.path.join(project_root, 'data/SPX_full_1min.txt')
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
        
        # Load patterns
        patterns_path = os.path.join(project_root, 'data/831am_patterns.json')
        with open(patterns_path, 'r') as f:
            patterns = json.load(f)
        
        # Filter patterns by date range
        filtered_patterns = filter_patterns_by_date(patterns, args.start_date, args.end_date)
        logger.info(f"Found {len(filtered_patterns)} patterns between {args.start_date} and {args.end_date}")
        
        # Analyze direction accuracy
        results = analyze_direction_accuracy(filtered_patterns, full_data)
        
        # Save results
        output_path = os.path.join(project_root, 'data/o2c_analysis.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n8:31 AM Pattern Direction Accuracy Analysis")
        print("==========================================")
        print(f"Analysis Period: {args.start_date} to {args.end_date}")
        print(f"Total Patterns Found: {results['total_patterns']}")
        print(f"Patterns Analyzed: {results['analyzed_patterns']}")
        print(f"Overall Direction Accuracy: {results['direction_accuracy']:.2f}%")
        print("\nAccuracy by Filter Level:")
        for level, stats in sorted(results["by_filter_level"].items()):
            if stats["total"] > 0:
                print(f"{level}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main() 