#!/usr/bin/env python3

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import json

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.patterns.utils import metrics, dates
from src.patterns.utils.io import load_minute_data, get_pattern_database, get_filter_levels
from src.patterns.core import filter
from src.patterns.core.pattern_detector_class import PatternDetector, PatternError, PatternDatabaseError

# Set up module-level logger
logger = logging.getLogger(__name__)

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
    Parse command line arguments for the pattern detector.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
        
    Raises:
        PatternError: If there's an error parsing the arguments
    """
    try:
        parser = argparse.ArgumentParser(description='Detect market patterns and generate trading strategies')
        
        # Data input/output arguments
        parser.add_argument('--data', type=str, help='Path to the minute data CSV file', 
                          default='data/SPX_full_1min.txt')
        parser.add_argument('--date', type=str, help='Target date for pattern analysis (YYYY-MM-DD)')
        parser.add_argument('--save-patterns', type=str, help='Path to save the pattern database')
        parser.add_argument('--load-patterns', type=str, help='Path to load the pattern database')
        parser.add_argument('--output', type=str, help='Path to save the analysis output')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--filter', type=str, help='Specific filter level to use (e.g., "Minimum", "Strict", "Moderate")')
        parser.add_argument('--pattern-version', '-pv', type=int, help='Pattern database version number to use (e.g., 2 for v2). If not provided, uses the latest.')

        # Pattern filter arguments
        parser.add_argument('--min-tp', type=float, default=0.4, 
                          help='Minimum target profit (default: 0.4)')
        parser.add_argument('--min-success-rate', type=float, default=0.7,
                          help='Minimum success rate (default: 0.7)')
        parser.add_argument('--min-occurrences', type=int, default=10,
                          help='Minimum number of pattern occurrences (default: 10)')
        parser.add_argument('--min-risk-reward', type=float, default=0.5,
                          help='Minimum risk/reward ratio (default: 0.5)')
        
        args = parser.parse_args()
        logger.debug(f"Parsed command line arguments: {args}")
        return args
        
    except Exception as e:
        error_msg = f"Error parsing command line arguments: {str(e)}"
        logger.error(error_msg)
        raise PatternError(error_msg)

def print_analysis(metrics: Dict[str, float], matched_patterns: List[Dict], target_date: datetime.date,
                  next_trading_day: datetime.date, detector: PatternDetector, analysis_data: pd.DataFrame,
                  full_data: pd.DataFrame, args: argparse.Namespace, 
                  filter_level_name: Optional[str] = None) -> None:
    """
    Print the analysis results, either to stdout or to a file.
    Maintains a clean, readable format for sharing with non-technical stakeholders.
    
    Args:
        metrics (Dict[str, float]): Dictionary of market metrics
        matched_patterns (List[Dict]): List of matched patterns
        target_date (datetime.date): Date of the analysis
        next_trading_day (datetime.date): Next trading day
        detector (PatternDetector): Pattern detector instance
        analysis_data (pd.DataFrame): Data used for analysis
        full_data (pd.DataFrame): Complete market data
        args (argparse.Namespace): Command line arguments
        filter_level_name (Optional[str]): Name of the filter level used
        
    Raises:
        PatternError: If there's an error printing the analysis
    """
    try:
        # Get previous day's close
        prev_close = dates.get_previous_close(full_data, target_date)
        if prev_close is None:
            logger.error("Could not find previous day's closing price")
            return
            
        # Create output structure
        output_data = {
            "pattern_day": next_trading_day.strftime('%A'),
            "pattern_date": next_trading_day.strftime('%Y-%m-%d'),
            "based_on_day": target_date.strftime('%A'),
            "based_on_date": target_date.strftime('%Y-%m-%d'),
            "close_price": prev_close,
            "filter_level": filter_level_name,
            "patterns": {
                "sessions": {
                    "morning": [],
                    "mixed": [],
                    "afternoon": []
                }
            }
        }
        
        if matched_patterns:
            # Group patterns by period and timeframe
            patterns_by_period = {}
            for pattern in matched_patterns:
                strategy = detector.generate_trading_strategy(pattern)
                period = strategy['period']
                
                if period not in patterns_by_period:
                    patterns_by_period[period] = []
                patterns_by_period[period].append(strategy)
            
            # Process patterns by period
            for period in ['morning', 'mixed', 'afternoon']:
                if period in patterns_by_period:
                    # Filter to keep only highest probability patterns for each timeframe
                    highest_prob_patterns = filter.filter_by_highest_probability(patterns_by_period[period])
                    
                    # Sort patterns by entry time
                    sorted_patterns = filter.sort_patterns_by_entry_time(
                        list(highest_prob_patterns.values())
                    )
                    
                    # Filter out patterns that are contained within larger patterns
                    filtered_patterns = filter.filter_overlapping_patterns(sorted_patterns)
                    
                    for strategy in filtered_patterns:
                        try:
                            # Extract entry and exit times from timeframe
                            timeframe = strategy['timeframe']
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
                            
                            # Extract TP and SL percentages and convert to points
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
                            
                            pattern_data = {
                                "entry_time": entry_time,
                                "exit_time": exit_time,
                                "direction": "Buy" if strategy['direction'] == 'bullish' else "Sell",
                                "target_points": round(tp_points, 2),
                                "stop_loss_points": round(sl_points, 2),
                                "success_rate": round(float(strategy['success_rate']), 2)
                            }
                            
                            output_data["patterns"]["sessions"][period].append(pattern_data)
                            
                            # Log the same information for debugging
                            logger.debug(f"Pattern found: {entry_time} to {exit_time}, {strategy['direction']}, "
                                       f"TP: {tp_points:.2f}, SL: {sl_points:.2f}, "
                                       f"Success: {strategy['success_rate']:.2f}%")
                            
                        except Exception as e:
                            logger.warning(f"Skipping pattern due to error: {str(e)}")
                            continue
        else:
            logger.info("No patterns found matching the criteria")
        
        # Ensure logs directory exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        logs_dir = os.path.join(project_root, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Save current patterns to JSON
        current_patterns_file = os.path.join(logs_dir, "current_detected_patterns.json")
        with open(current_patterns_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Current patterns saved to {os.path.relpath(current_patterns_file, project_root)}")
        
        # Save current patterns to TXT in a readable format
        current_patterns_txt = os.path.join(logs_dir, "current_detected_patterns.txt")
        with open(current_patterns_txt, 'w') as f:
            f.write(f"=== SPX Pattern Analysis for {output_data['pattern_day']}, {output_data['pattern_date']} ===\n")
            f.write(f"Based on data from {output_data['based_on_day']}, {output_data['based_on_date']}\n")
            f.write(f"Close from {output_data['based_on_day']}: {output_data['close_price']}\n\n")
            f.write(f"Filter Level: {output_data['filter_level']}\n\n")
            
            for session in ['morning', 'mixed', 'afternoon']:
                patterns = output_data['patterns']['sessions'][session]
                if patterns:
                    f.write(f"{session.upper()} SESSION PATTERNS:\n")
                    f.write(f"{'=' * 50}\n\n")
                    
                    for pattern in patterns:
                        f.write(f"===== Action Plan =====\n")
                        f.write(f"Entry: {pattern['entry_time']}\n")
                        f.write(f"Exit: {pattern['exit_time']}\n")
                        f.write(f"Direction: {pattern['direction']} {'📈' if pattern['direction'] == 'Buy' else '📉'}\n")
                        f.write(f"TP: {pattern['target_points']} points\n")
                        f.write(f"SL: {pattern['stop_loss_points']} points\n")
                        f.write(f"Success Rate: {pattern['success_rate']}%\n")
                        f.write(f"{'-' * 30}\n\n")
        logger.info(f"Current patterns saved to {os.path.relpath(current_patterns_txt, project_root)}")
        
        # Append to all patterns JSON file
        all_patterns_file = os.path.join(logs_dir, 'all_live_detected_patterns.json')
        try:
            # Read existing patterns if file exists
            if os.path.exists(all_patterns_file):
                with open(all_patterns_file, 'r') as f:
                    all_patterns = json.load(f)
            else:
                all_patterns = []
            
            # Check if this pattern date already exists
            pattern_date = f"{output_data['pattern_day']}, {output_data['pattern_date']}"
            existing_dates = [p.get('pattern_day', '') + ', ' + p.get('pattern_date', '') for p in all_patterns]
            
            if pattern_date not in existing_dates:
                all_patterns.append(output_data)
                with open(all_patterns_file, 'w') as f:
                    json.dump(all_patterns, f, indent=2)
                logger.info(f"Pattern logged to {os.path.relpath(all_patterns_file, project_root)}")
            else:
                logger.info("Pattern already exists in the log, skipping logging.")
                
        except json.JSONDecodeError:
            logger.warning("Error reading existing patterns file, creating new file")
            with open(all_patterns_file, 'w') as f:
                json.dump([output_data], f, indent=2)
        
        # Also print to stdout or file if requested
        if args.output:
            output_path = os.path.join(project_root, args.output)
            logger.info(f"Redirecting output to {os.path.relpath(output_path, project_root)}")
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Analysis saved to {os.path.relpath(output_path, project_root)}")
        else:
            logger.info("Pattern analysis completed. Check the logs directory for detailed results.")
            
    except Exception as e:
        error_msg = f"Error printing analysis: {str(e)}"
        logger.error(error_msg)
        raise PatternError(error_msg)

def main() -> None:
    """
    Main execution function for the pattern detector.
    
    Raises:
        PatternError: If there's an error during execution
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Set up logging
        setup_logging(logging.DEBUG if args.debug else logging.INFO)
        
        # Get the project root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        
        # Load data
        data_path = os.path.join(project_root, args.data)
        try:
            full_data = load_minute_data(data_path)
        except ValueError as e:
            if "Data file not found" in str(e):
                logger.error("Required minute data files are missing.")
                logger.error("Please run the following command to download the required data files:")
                logger.error("python src/data/downloader.py")
                return
            raise
        
        # Get data for analysis based on target date
        analysis_data, target_date = dates.get_data_for_date(full_data, args.date)
        
        if analysis_data is None:
            if args.date:
                logger.error(f"Insufficient data for pattern analysis on {target_date}. Need previous trading day data.")
            else:
                logger.error("No data available for pattern analysis.")
            return
        
        # Calculate the next trading day
        next_trading_day = dates.get_next_trading_day(target_date)
        
        # Get pattern database path based on version
        try:
            pattern_db_path, version = get_pattern_database(project_root, args.pattern_version)
            logger.info(f"Using pattern database version v{version}")
        except ValueError as e:
            logger.error(str(e))
            return
        
        # Initialize pattern detector
        detector = PatternDetector(pattern_db_path)
        
        # Save patterns if requested
        if args.save_patterns:
            save_path = os.path.join(project_root, args.save_patterns)
            detector.save_patterns(save_path)
        
        # Calculate metrics using the metrics utility
        metrics_dict = metrics.calculate_metrics(analysis_data)
        
        # Get filter levels
        filter_levels = get_filter_levels(project_root)
        
        matched_patterns = []
        filter_level_name = None
        
        if args.filter:
            # Use specific filter level if provided
            selected_level = next((level for level in filter_levels if level['name'].lower() == args.filter.lower()), None)
            if selected_level:
                filter_level_name = selected_level['name']
                logger.info(f"Using filter level: {filter_level_name}")
                matched_patterns = detector.detect_patterns(
                    metrics_dict,
                    min_tp=selected_level['min_tp'],
                    min_success_rate=selected_level['min_success_rate'],
                    min_occurrences=selected_level['min_occurrences'],
                    min_risk_reward=selected_level['min_risk_reward']
                )
            else:
                logger.error(f"Filter level '{args.filter}' not found. Available levels: {[level['name'] for level in filter_levels]}")
                return
        else:
            # Use recursive filter approach starting from strictest
            for level in filter_levels:
                filter_level_name = level['name']
                logger.info(f"Trying filter level: {filter_level_name}")
                
                matched_patterns = detector.detect_patterns(
                    metrics_dict,
                    min_tp=level['min_tp'],
                    min_success_rate=level['min_success_rate'],
                    min_occurrences=level['min_occurrences'],
                    min_risk_reward=level['min_risk_reward']
                )
                
                if matched_patterns:
                    logger.info(f"Found {len(matched_patterns)} patterns using {filter_level_name} filters")
                    break
                else:
                    logger.warning(f"No patterns found with filter level: {filter_level_name}")
        
        if not matched_patterns:
            logger.warning("No patterns found with any filter level")
        
        # Print analysis (this will now also save to logs directory)
        print_analysis(metrics_dict, matched_patterns, target_date, next_trading_day, 
                     detector, analysis_data, full_data, args, filter_level_name)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise PatternError(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
