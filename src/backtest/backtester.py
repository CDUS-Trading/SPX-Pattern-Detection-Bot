#!/usr/bin/env python3

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import json
import pandas_market_calendars as mcal
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
import traceback

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import necessary functions from other modules
from src.patterns.utils.dates import get_data_for_date, get_next_trading_day, get_previous_close
from src.patterns.utils.io import load_minute_data, get_filter_levels, get_pattern_database
from src.patterns.core.pattern_detector_class import PatternDetector
from src.patterns.utils.metrics import calculate_metrics
from src.utils.convert_json_to_txt import convert_json_to_txt

# Import detect_patterns_with_levels and filter functions from filter
from src.patterns.core.filter import (
    detect_patterns_with_levels,
    filter_by_highest_probability,
    sort_patterns_by_entry_time,
    filter_overlapping_patterns
)

# Initialize logger at module level
logger = logging.getLogger(__name__)

def setup_logging(debug: bool = False):
    """
    Configure logging for the backtester.
    
    Args:
        debug (bool): Whether to enable debug logging
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up file handler for detailed logging
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(os.path.join("logs", "backtester_debug.log"))
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Set up console handler for minimal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set specific loggers to higher levels to reduce noise
    logging.getLogger('src.patterns.utils.dates').setLevel(logging.WARNING)
    logging.getLogger('src.patterns.utils.io').setLevel(logging.WARNING)
    logging.getLogger('src.patterns.core.pattern_detector_class').setLevel(logging.WARNING)
    logging.getLogger('src.patterns.utils.convert_json_to_txt').setLevel(logging.WARNING)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the backtester.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Backtest SPX pattern detection over a date range')
    
    # Date range arguments
    today = datetime.now().date()
    one_week_ago = today - timedelta(days=7)
    
    parser.add_argument('--start-date', 
                       type=str,
                       default=one_week_ago.strftime('%Y-%m-%d'),
                       help='Start date for backtesting (YYYY-MM-DD)')
    
    parser.add_argument('--end-date',
                       type=str,
                       default=today.strftime('%Y-%m-%d'),
                       help='End date for backtesting (YYYY-MM-DD)')
    
    # Data file argument
    parser.add_argument('--data',
                       type=str,
                       default='data/SPX_full_1min.txt',
                       help='Path to the minute data CSV file')
    
    # Pattern version argument
    parser.add_argument('--pattern-version', '-pv',
                       type=int,
                       help='Pattern database version number to use (e.g., 2 for v2). If not provided, uses the latest.')
    
    # Filter level argument
    parser.add_argument('--filter',
                       type=str,
                       choices=['Strict', 'Moderate', 'Minimum', 'Poor'],
                       help='Restrict pattern detection to a specific filter level') # No filter mentioned means recursive filter approach used
    
    # Add debug flag
    parser.add_argument('--debug',
                       action='store_true',
                       help='Enable debug mode for more verbose logging')
    
    # Add output argument
    parser.add_argument('--output',
                       type=str,
                       help='Path to save the analysis output')
    
    # Add table output flag
    parser.add_argument('--table',
                       action='store_true',
                       help='Generate a CSV table with pattern information')
    
    return parser.parse_args()

def get_trading_days(start_date: datetime.date, end_date: datetime.date) -> List[datetime.date]:
    """
    Get list of trading days between start and end dates.
    
    Args:
        start_date (datetime.date): Start date
        end_date (datetime.date): End date
        
    Returns:
        List[datetime.date]: List of trading days
    """
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index.date.tolist()
    logger.info(f"Found {len(trading_days)} trading days between {start_date} and {end_date}")
    return trading_days

def initialize_backtest(args: argparse.Namespace) -> Tuple[PatternDetector, pd.DataFrame, List[Dict]]:
    """
    Initialize backtest components.
    
    Args:
        args (argparse.Namespace): Command line arguments
        
    Returns:
        Tuple[PatternDetector, pd.DataFrame, List[Dict]]: Pattern detector, market data, and filter levels
    """
    # Get pattern database path based on version
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pattern_db_path, version = get_pattern_database(project_root, args.pattern_version)
        logger.info(f"Using pattern database version v{version}")
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Load pattern database
    detector = PatternDetector(pattern_db_path)
    
    # Load minute data
    logger.info(f"Loading minute data from: {os.path.relpath(args.data, os.getcwd())}")
    full_data = load_minute_data(args.data)
    logger.info(f"Loaded {len(full_data)} data points")
    
    # Get filter levels
    filter_levels = get_filter_levels(project_root)
    logger.info(f"Loaded {len(filter_levels)} filter levels")
    
    return detector, full_data, filter_levels

def analyze_trading_day(
    current_date: datetime.date,
    detector: PatternDetector,
    full_data: pd.DataFrame,
    filter_levels: List[Dict],
    args: argparse.Namespace,
    log_file: str
) -> Tuple[Dict, List[Dict]]:
    """
    Analyze patterns for a single trading day.
    
    Args:
        current_date (datetime.date): Date to analyze
        detector (PatternDetector): Pattern detector instance
        full_data (pd.DataFrame): Complete market data
        filter_levels (List[Dict]): List of filter level configurations
        args (argparse.Namespace): Command line arguments
        log_file (str): Path to log file
        
    Returns:
        Tuple[Dict, List[Dict]]: Analysis output and pattern information for table generation
    """
    logger.info(f"Analyzing trading day: {current_date}")
    pattern_info_list = []
    
    try:
        # Get data for this date
        analysis_data, target_date = get_data_for_date(full_data, current_date)
        
        if analysis_data is None:
            logger.warning(f"No data available for date: {current_date}")
            return {
                "date": current_date.strftime('%Y-%m-%d'),
                "status": "no_data",
                "message": f"No data available for date: {current_date}"
            }, pattern_info_list
        
        logger.info(f"Got {len(analysis_data)} data points for {current_date}")
        
        # Calculate metrics
        metrics = calculate_metrics(analysis_data)
        logger.info(f"Calculated metrics for {current_date}")
        
        if args.debug:
            logger.debug(f"Metrics: {metrics}")
        
        matched_patterns = []
        filter_level_name = None
        
        # If specific filter level is requested, use only that filter
        if args.filter:
            logger.info(f"Using specified filter level: {args.filter}")
            for level in filter_levels:
                if level['name'] == args.filter:
                    logger.info(f"Found filter level {args.filter}: {level}")
                    matched_patterns = detector.detect_patterns(
                        metrics,
                        min_tp=level['min_tp'],
                        min_success_rate=level['min_success_rate'],
                        min_occurrences=level['min_occurrences'],
                        min_risk_reward=level['min_risk_reward']
                    )
                    filter_level_name = level['name']
                    logger.info(f"Detected {len(matched_patterns)} patterns with {filter_level_name} filter")
                    break
        else:
            # Use recursive filter approach starting from strictest to weakest
            logger.info("Using automatic filter level progression")
            for level in filter_levels:
                filter_level_name = level['name']
                logger.info(f"Trying filter level: {filter_level_name}")
                
                matched_patterns = detector.detect_patterns(
                    metrics,
                    min_tp=level['min_tp'],
                    min_success_rate=level['min_success_rate'],
                    min_occurrences=level['min_occurrences'],
                    min_risk_reward=level['min_risk_reward']
                )
                
                if matched_patterns:
                    logger.info(f"Found {len(matched_patterns)} patterns using {filter_level_name} filters")
                    break
        
        # Calculate next trading day
        next_trading_day = get_next_trading_day(current_date)
        logger.info(f"Next trading day after {current_date} is {next_trading_day}")
        
        # Get previous close price
        prev_close = get_previous_close(full_data, current_date)
        if prev_close is None:
            logger.warning(f"Could not find previous day's closing price for {current_date}")
            return {
                "date": current_date.strftime('%Y-%m-%d'),
                "status": "error",
                "message": f"Could not find previous day's closing price for {current_date}"
            }, pattern_info_list
        
        # Create analysis output
        analysis_output = {
            "pattern_day": next_trading_day.strftime('%A'),
            "pattern_date": next_trading_day.strftime('%Y-%m-%d'),
            "based_on_day": current_date.strftime('%A'),
            "based_on_date": current_date.strftime('%Y-%m-%d'),
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
                    highest_prob_patterns = filter_by_highest_probability(patterns_by_period[period])
                    
                    # Sort patterns by entry time
                    sorted_patterns = sort_patterns_by_entry_time(
                        list(highest_prob_patterns.values())
                    )
                    
                    # Filter out patterns that are contained within larger patterns
                    filtered_patterns = filter_overlapping_patterns(sorted_patterns)
                    
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
                            
                            analysis_output["patterns"]["sessions"][period].append(pattern_data)
                            
                            # Add to pattern info list for table generation
                            if args.table:
                                pattern_info = {
                                    'date': current_date,
                                    'filter_level': filter_level_name,
                                    'entry_time': entry_time,
                                    'exit_time': exit_time,
                                    'direction': strategy['direction'],
                                    'tp_points': tp_points,
                                    'sl_points': sl_points,
                                    'success_rate': strategy['success_rate'],
                                    'occurrences': pattern.get('occurrences', pattern.get('total_occurrences', 0)),
                                    'risk_reward': strategy.get('risk_reward', 0),
                                    'exit_type': 'Unknown'  # Will be filled by simulator
                                }
                                pattern_info_list.append(pattern_info)
                            
                        except Exception as e:
                            logger.warning(f"Skipping pattern due to error: {str(e)}")
                            continue
        
        return analysis_output, pattern_info_list
            
    except Exception as e:
        error_msg = f"Error analyzing {current_date}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "date": current_date.strftime('%Y-%m-%d'),
            "status": "error",
            "message": error_msg,
            "traceback": traceback.format_exc()
        }, pattern_info_list
    
    

def run_backtest(
    start_date: datetime.date,
    end_date: datetime.date,
    detector: PatternDetector,
    full_data: pd.DataFrame,
    filter_levels: List[Dict],
    args: argparse.Namespace,
    log_file: str
) -> None:
    """
    Run backtest over a date range.
    
    Args:
        start_date (datetime.date): Start date
        end_date (datetime.date): End date
        detector (PatternDetector): Pattern detector instance
        full_data (pd.DataFrame): Complete market data
        filter_levels (List[Dict]): List of filter level configurations
        args (argparse.Namespace): Command line arguments
        log_file (str): Path to log file
    """
    # Print metadata to console
    print("\nBacktest Results -", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Filter Level: {args.filter if args.filter else 'recursive'}")
    
    # Get pattern database version info
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        _, version = get_pattern_database(project_root, args.pattern_version)
        print(f"Pattern Database: v{version}")
    except ValueError as e:
        print(f"Pattern Database: Error - {str(e)}")
    
    # Get trading days
    trading_days = get_trading_days(start_date, end_date)
    
    if not trading_days:
        logger.error("No trading days found in the specified date range")
        sys.exit(1)
    
    print(f"Total Trading Days: {len(trading_days)}")
    print("=" * 80 + "\n")
    
    # Initialize list to store pattern information for table
    all_pattern_info = []
    all_results = []
    
    # Track filter level statistics
    filter_stats = {}
    total_patterns = 0
    
    # Run backtest for each trading day
    for current_date in tqdm(trading_days, desc="Backtesting patterns"):
        result, pattern_info = analyze_trading_day(current_date, detector, full_data, filter_levels, args, log_file)
        if isinstance(result, dict):
            all_results.append(result)
            # Update filter statistics if we have patterns
            if "filter_level" in result and result["filter_level"]:
                filter_level = result["filter_level"]
                filter_stats[filter_level] = filter_stats.get(filter_level, 0) + 1
                total_patterns += 1
        all_pattern_info.extend(pattern_info)
    
    # Print filter level statistics
    if not args.filter:  # Only show stats when using recursive filtering
        print("\nFilter Level Distribution:")
        print("-" * 40)
        print(f"{'Filter Level':<15} {'Count':<10} {'Percentage':<10}")
        print("-" * 40)
        for level, count in sorted(filter_stats.items()):
            percentage = (count / total_patterns) * 100 if total_patterns > 0 else 0
            print(f"{level:<15} {count:<10} {percentage:>6.1f}%")
        print("-" * 40)
        print(f"{'Total':<15} {total_patterns:<10} {100:>6.1f}%")
        print()
    
    # Generate CSV table if requested
    if args.table and all_pattern_info:
        # Create a TradeSimulator instance to get exit types
        from src.simulator.trade_simulator import TradeSimulator
        simulator = TradeSimulator(
            backtest_dates_path=os.path.join("logs", "backtest_dates.json"),
            data_path=args.data,
            pattern_db_path=args.pattern_db,
            filter_type=args.filter
        )
        
        # Simulate trades for each pattern to get exit types
        for pattern in all_pattern_info:
            trade_result = simulator.simulate_trade(
                pattern['date'],
                pattern['entry_time'],
                pattern['exit_time'],
                pattern['direction'],
                pattern['tp_points'],
                pattern['sl_points']
            )
            if trade_result:
                pattern['exit_type'] = trade_result['exit_type']
            else:
                # Check if we have data for this date
                day_data = simulator.full_data[simulator.full_data['date'].dt.date == pattern['date']]
                if day_data.empty:
                    pattern['exit_type'] = 'NO_DATA'
                else:
                    # Check if we have data at entry time
                    entry_dt = simulator.parse_time(pattern['entry_time'])
                    entry_dt = datetime.combine(pattern['date'], entry_dt.time())
                    entry_data = day_data[day_data['date'] >= entry_dt]
                    if entry_data.empty:
                        pattern['exit_type'] = 'NO_ENTRY_DATA'
                    else:
                        # Check if we have data between entry and exit
                        exit_dt = simulator.parse_time(pattern['exit_time'])
                        exit_dt = datetime.combine(pattern['date'], exit_dt.time())
                        trade_data = day_data[(day_data['date'] >= entry_dt) & (day_data['date'] <= exit_dt)]
                        if trade_data.empty:
                            pattern['exit_type'] = 'NO_TRADE_DATA'
                        else:
                            pattern['exit_type'] = 'SIMULATION_ERROR'
        
        # Save to CSV
        table_file = os.path.join("logs", f"pattern_table_{args.filter if args.filter else 'recursive'}.csv")
        df = pd.DataFrame(all_pattern_info)
        df.to_csv(table_file, index=False)
        logger.info(f"Pattern table written to {table_file}")
    
    # Write results to JSON file
    json_file = log_file
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Write results to TXT file
    txt_file = log_file.replace('.json', '.txt')
    with open(txt_file, 'w') as f:
        for result in all_results:
            if "status" in result:
                # Write error or no-data message
                f.write(f"Date: {result['date']}\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Message: {result['message']}\n")
                f.write("-" * 50 + "\n\n")
            else:
                # Write pattern analysis
                f.write(f"=== SPX Pattern Analysis for {result['pattern_day']}, {result['pattern_date']} ===\n")
                f.write(f"Based on data from {result['based_on_day']}, {result['based_on_date']}\n")
                f.write(f"Close from {result['based_on_day']}: {result['close_price']}\n\n")
                f.write(f"Filter Level: {result['filter_level']}\n\n")
                
                for session in ['morning', 'mixed', 'afternoon']:
                    patterns = result['patterns']['sessions'][session]
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
    
    logger.info(f"Backtest completed. Results written to {json_file} and {txt_file}")
    print(f"\nBacktest completed. Results written to {json_file} and {txt_file}")

def main() -> None:
    """
    Main execution function for the backtester.
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Set up logging
        setup_logging(args.debug)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting backtest")
        
        # Convert date strings to datetime objects
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Initialize backtest components
        detector, full_data, filter_levels = initialize_backtest(args)
        
        # Set up log file
        filter_label = args.filter if args.filter else "recursive"
        log_file = os.path.join("logs", f"backtest_patterns_{filter_label}.json")
        logger.info(f"Log file: {log_file}")
        
        # Run backtest
        run_backtest(start_date, end_date, detector, full_data, filter_levels, args, log_file)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()