import argparse
from datetime import datetime
import os
import sys
import json
from collections import Counter
from tqdm import tqdm
import pandas as pd
try:
    import pandas_market_calendars as mcal
    NYSE_CAL = mcal.get_calendar('XNYS')
except ImportError:
    NYSE_CAL = None

# Add parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.patterns.utils.io import get_pattern_database, load_minute_data
from src.patterns.utils.metrics import calculate_metrics
from src.patterns.format_action_plan import format_action_plans_to_txt
from src.patterns.utils.dates import get_next_trading_day

def parse_arguments():
    parser = argparse.ArgumentParser(description='Detect patterns for the next trading day using the latest pattern database.')
    parser.add_argument('--date', type=str, default=None, help='Date (YYYY-MM-DD) to extract metrics for. If not provided, uses the latest date in the minute data.')
    parser.add_argument('--input-database', type=str, default=None, help='Path to pattern database JSON. If not provided, uses the latest versioned database.')
    parser.add_argument('--start-date', type=str, default=None, help='Start date (YYYY-MM-DD) for backtest. If provided, runs detection for each day in the range.')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD) for backtest. Defaults to latest date in the minute data.')
    parser.add_argument('--top-n', type=int, default=3, help='Number of top patterns to output (default: 3)')
    args = parser.parse_args()
    return args

def run_detection_for_date(target_date, minute_df, pattern_db_path, sort_keys=None):
    # Extract all minute bars for the target date
    day_minutes = minute_df[minute_df['date'].dt.date == target_date]
    if day_minutes.empty:
        print(f"[ERROR] No minute data found for {target_date}.")
        return []

    # Get previous day's close
    prev_date = (minute_df['date'].dt.date[minute_df['date'].dt.date < target_date]).max()
    prev_close = None
    if prev_date is not None:
        prev_close_row = minute_df[minute_df['date'].dt.date == prev_date].iloc[-1]
        prev_close = float(prev_close_row['close'])
    # Compute metrics
    metrics = calculate_metrics(day_minutes)
    # Load pattern db
    with open(pattern_db_path, 'r') as f:
        pattern_db = json.load(f)
    matched_patterns = []
    for pattern in pattern_db:
        p1 = pattern['pattern1']
        p2 = pattern['pattern2']
        m1 = metrics.get(p1)
        m2 = metrics.get(p2)
        if m1 is None or m2 is None:
            continue
        if pattern['range1_min'] <= m1 <= pattern['range1_max'] and pattern['range2_min'] <= m2 <= pattern['range2_max']:
            matched_patterns.append(pattern)
    # Count patterns per unique timeframe
    unique_timeframes = set(p['timeframe'] for p in matched_patterns)
    best_per_timeframe = []
    for timeframe in unique_timeframes:
        patterns_in_tf = [p for p in matched_patterns if p['timeframe'] == timeframe]
        if not patterns_in_tf:
            continue
        # Sort by tp_hit_rate, then fav_direction_rate as tiebreaker
        patterns_in_tf.sort(key=lambda p: (p.get('tp_hit_rate', 0), p.get('fav_direction_rate', 0)), reverse=True)
        best_per_timeframe.append(patterns_in_tf[0])
    # Now, sort all best_per_timeframe patterns by tp_hit_rate, then fav_direction_rate
    best_per_timeframe.sort(key=lambda p: (p.get('tp_hit_rate', 0), p.get('fav_direction_rate', 0)), reverse=True)
    return best_per_timeframe

def is_market_holiday_or_weekend(date):
    # date: datetime.date
    if date.weekday() >= 5:
        return True
    if NYSE_CAL is not None:
        schedule = NYSE_CAL.schedule(start_date=str(date), end_date=str(date))
        return schedule.empty
    return False

def main():
    args = parse_arguments()
    sort_keys = ['tp_hit_rate', 'fav_direction_rate']
    top_n = args.top_n
    print("[INFO] Loading pattern database...")
    if args.input_database:
        pattern_db_path = args.input_database
        version = 'custom'
        print(f"[INFO] Loaded pattern database from argument: {pattern_db_path}")
    else:
        pattern_db_path, version = get_pattern_database(project_root)
        print(f"[INFO] Loaded pattern database: {pattern_db_path} (version {version})")
    print("[INFO] Loading minute data and extracting analysis day...")
    minute_data_path = os.path.join(project_root, 'data', 'SPX_full_1min_CT.txt')
    minute_df = load_minute_data(minute_data_path)

    # Determine date range for backtest
    latest_date = minute_df['date'].dt.date.max()
    if args.start_date:
        start_date = pd.to_datetime(args.start_date).date()
        end_date = pd.to_datetime(args.end_date).date() if args.end_date else latest_date
        print(f"[INFO] Running backtest from {start_date} to {end_date}...")
        all_results = []
        logs_dir = os.path.join(project_root, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        backtest_txt_file = os.path.join(logs_dir, 'backtest_new_detector.txt')
        backtest_json_file = os.path.join(logs_dir, 'backtest_new_detector.json')
        # Clear TXT file at start
        with open(backtest_txt_file, 'w') as _:
            pass
        date_range = list(pd.date_range(start_date, end_date))
        progress_bar = tqdm(date_range, unit='day')
        for single_date in progress_bar:
            target_date = single_date.date()
            if is_market_holiday_or_weekend(target_date):
                progress_bar.set_description_str(f"{target_date} (Holiday/Weekend)")
                continue
            progress_bar.set_description_str(f"{target_date}")
            patterns = run_detection_for_date(target_date, minute_df, pattern_db_path)
            # Now, pick top N by tp_hit_count, then pattern_number as tiebreaker (same as new_pnl_sim.py)
            # First add temporary pattern_number for sorting
            for i, pattern in enumerate(patterns):
                pattern['temp_pattern_number'] = i + 1
            
            patterns_sorted = sorted(patterns, key=lambda p: (-p.get('tp_hit_count', 0), p.get('temp_pattern_number', 0)))
            
            # Add final pattern_number to each pattern (similar to new_pnl_sim.py)
            for i, pattern in enumerate(patterns_sorted):
                pattern['pattern_number'] = i + 1
                # Remove temporary field
                if 'temp_pattern_number' in pattern:
                    del pattern['temp_pattern_number']
            
            # Get previous day's close
            prev_date = (minute_df['date'].dt.date[minute_df['date'].dt.date < target_date]).max()
            prev_close = None
            if prev_date is not None:
                prev_close_row = minute_df[minute_df['date'].dt.date == prev_date].iloc[-1]
                prev_close = float(prev_close_row['close'])
            close_price = prev_close
            percent_fields = [
                'avg_win', 'avg_loss',
                'favorable_avg_move', 'favorable_median_move', 'favorable_std',
                'adverse_avg_move', 'adverse_median_move', 'adverse_std'
            ]
            abs_fields = [
                'favorable_avg_move', 'favorable_median_move', 'favorable_std',
                'adverse_avg_move', 'adverse_median_move', 'adverse_std'
            ]
            def convert_pattern_to_points(p):
                p_new = p.copy()
                for field in percent_fields:
                    if field in p_new and close_price is not None:
                        val = p_new[field] * close_price / 100
                        if field in abs_fields:
                            val = abs(val)
                        p_new[field] = val
                return p_new
            patterns_points_sorted = [convert_pattern_to_points(p) for p in patterns_sorted]
            # Save for JSON (top 3 patterns)
            all_results.append({
                'date': str(target_date),
                'patterns': patterns_points_sorted[:top_n]
            })
            # Use format_action_plans_to_txt for TXT output (append mode, top 3 patterns)
            next_trading_day = get_next_trading_day(target_date)
            analysis_day = next_trading_day.strftime('%A, %Y-%m-%d')
            based_on_day = target_date.strftime('%A')
            based_on_date = target_date.strftime('%Y-%m-%d')
            format_action_plans_to_txt(
                patterns_points_sorted[:top_n],
                backtest_txt_file,
                analysis_day=analysis_day,
                based_on_day=based_on_day,
                based_on_date=based_on_date,
                close_price=close_price,
                top_n=top_n,
                show_all_slots=False,
                append=True
            )
        # Save all results to JSON
        with open(backtest_json_file, 'w') as jf:
            json.dump(all_results, jf, indent=2)
        print(f"[SUCCESS] Backtest results written to:\n  {backtest_txt_file}\n  {backtest_json_file}")
        return

    # Single day mode (original behavior)
    if args.date:
        target_date = pd.to_datetime(args.date).date()
        print(f"[INFO] Using provided date: {args.date}")
    else:
        target_date = latest_date
        print("[INFO] No date provided, will use the latest date in the minute data.")
    print(f"[INFO] Analysis date in data: {target_date}")
    patterns = run_detection_for_date(target_date, minute_df, pattern_db_path)
    # Now, pick top N by tp_hit_count, then pattern_number as tiebreaker (same as new_pnl_sim.py)
    # First add temporary pattern_number for sorting
    for i, pattern in enumerate(patterns):
        pattern['temp_pattern_number'] = i + 1
    
    patterns_sorted = sorted(patterns, key=lambda p: (-p.get('tp_hit_count', 0), p.get('temp_pattern_number', 0)))
    
    # Add final pattern_number to each pattern (similar to new_pnl_sim.py)
    for i, pattern in enumerate(patterns_sorted):
        pattern['pattern_number'] = i + 1
        # Remove temporary field
        if 'temp_pattern_number' in pattern:
            del pattern['temp_pattern_number']
    
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    # Get previous day's close
    prev_date = (minute_df['date'].dt.date[minute_df['date'].dt.date < target_date]).max()
    prev_close = None
    if prev_date is not None:
        prev_close_row = minute_df['date'].dt.date == prev_date
        prev_close = float(minute_df[prev_close_row].iloc[-1]['close'])
    close_price = prev_close
    percent_fields = [
        'avg_win', 'avg_loss',
        'favorable_avg_move', 'favorable_median_move', 'favorable_std',
        'adverse_avg_move', 'adverse_median_move', 'adverse_std'
    ]
    abs_fields = [
        'favorable_avg_move', 'favorable_median_move', 'favorable_std',
        'adverse_avg_move', 'adverse_median_move', 'adverse_std'
    ]
    def convert_pattern_to_points(p):
        p_new = p.copy()
        for field in percent_fields:
            if field in p_new and close_price is not None:
                val = p_new[field] * close_price / 100
                if field in abs_fields:
                    val = abs(val)
                p_new[field] = val
        return p_new
    patterns_points_sorted = [convert_pattern_to_points(p) for p in patterns_sorted]
    current_detected_json = os.path.join(logs_dir, 'current_detected_patterns.json')
    with open(current_detected_json, 'w') as f:
        json.dump(patterns_points_sorted[:top_n], f, indent=2)
    print(f"[INFO] Saved top {min(top_n, len(patterns_points_sorted))} patterns to {current_detected_json}")
    print(f"[INFO] Sort keys used: ['tp_hit_count', 'pattern_number'] (same as new_pnl_sim.py)")
    print("[INFO] Formatting action plan and writing to logs...")
    next_trading_day = get_next_trading_day(target_date)
    analysis_day = next_trading_day.strftime('%A, %Y-%m-%d')
    based_on_day = target_date.strftime('%A')
    based_on_date = target_date.strftime('%Y-%m-%d')
    
    # Create current_detected_patterns.txt (top 3 patterns)
    txt_file = os.path.join(logs_dir, 'current_detected_patterns.txt')
    format_action_plans_to_txt(
        patterns_points_sorted[:top_n], txt_file,
        analysis_day=analysis_day,
        based_on_day=based_on_day,
        based_on_date=based_on_date,
        close_price=close_price,
        top_n=top_n
    )
    
    # Create all_detected_patterns.txt (all slots)
    all_txt_file = os.path.join(logs_dir, 'all_detected_patterns.txt')
    format_action_plans_to_txt(
        patterns_points_sorted, all_txt_file,
        analysis_day=analysis_day,
        based_on_day=based_on_day,
        based_on_date=based_on_date,
        close_price=close_price,
        top_n=top_n,  # Use all patterns
        show_all_slots=True
    )
    
    # Save all detected patterns to JSON (not just top 3)
    all_detected_json = os.path.join(logs_dir, 'all_detected_patterns.json')
    with open(all_detected_json, 'w') as f:
        json.dump(patterns_points_sorted, f, indent=2)
    
    print(f"[SUCCESS] Action plans for {next_trading_day.strftime('%Y-%m-%d')} (based on metrics from {target_date}) written to:\n  {txt_file}\n  {all_txt_file}")

if __name__ == "__main__":
    main() 