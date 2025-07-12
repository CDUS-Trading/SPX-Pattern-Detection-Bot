import os
import sys
import csv
import json
import argparse
from datetime import datetime
import pandas as pd
import logging
from itertools import product
import collections

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(handler)

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.utils.reversal_continuation import analyze_reversal_continuation, print_analysis_summary, write_analysis_to_csv
from src.patterns.utils.io import get_pattern_database
from trade_engine import TradeEngine

# Default pattern files
LIVE_PATTERN_FILE = 'logs/all_live_detected_patterns.json'
BACKTEST_PATTERN_FILE = 'logs/backtest_patterns_recursive.json'
MINUTE_DATA_FILE = 'data/SPX_full_1min_CT.txt'
OUTPUT_CSV = 'output/parsed_patterns_with_pnl.csv'
PATTERN_DB_PATH = 'data/processed/master_pattern_database.json'

# TP and SL values to generate combinations from
TP_VALUES = [7, 10, 15]
SL_VALUES = [20, 30, 45]

# Generate all TP/SL combinations
TP_SL_COMBINATIONS = list(product(TP_VALUES, SL_VALUES))

# Pattern files for different filter levels
FILTER_PATTERN_FILES = {
    'Minimum': 'logs/backtest_patterns_Minimum.json',
    'Moderate': 'logs/backtest_patterns_Moderate.json',
    'Strict': 'logs/backtest_patterns_Strict.json',
    'Poor': 'logs/backtest_patterns_Poor.json'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Simulate P&L for trading patterns')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD). If not provided, uses first date in pattern file.')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD). If not provided, uses last date in pattern file.')
    parser.add_argument('--tp-adjustment', '-atp', type=float, default=0, help='Adjustment for take profit levels')
    parser.add_argument('--sl-adjustment', '-asl', type=float, default=0, help='Adjustment for stop loss levels')
    parser.add_argument('--fixed-tp', '-ftp', type=float, help='Override pattern TP with fixed value (in points)')
    parser.add_argument('--fixed-sl', '-fsl', type=float, help='Override pattern SL with fixed value (in points)')
    parser.add_argument('--reversal-continuation', '-rc', action='store_true', help='Run reversal-continuation analysis')
    parser.add_argument('--continuous-analysis', '-ca', action='store_true', help='Run consecutive wins/losses analysis')
    parser.add_argument('--backtest', action='store_true', help='Use backtest pattern files instead of live patterns')
    parser.add_argument('--filter', type=str, help='Specific filter level to use for backtest patterns (e.g., "Minimum", "Strict", "Moderate")')
    parser.add_argument('--entry-time-slot-analysis', '-esta', action='store_true', help='Run entry time slot analysis')
    parser.add_argument('--pattern-version', '-pv', type=int, help='Pattern database version number to use (e.g., 2 for v2). If not provided, uses the latest.')
    parser.add_argument('--pattern-file', type=str, help='Custom pattern file (e.g., filtered_sentiment_based_patterns.json)')
    return parser.parse_args()

# Helper to parse time like '8:30AM CT' to datetime.time
def parse_ct_time(timestr):
    timestr = timestr.replace(' CT', '')
    return datetime.strptime(timestr, '%I:%M%p').time()

# Parse the pattern file
def parse_patterns(pattern_file, start_date=None, end_date=None):
    """
    Parse patterns from JSON file and filter by date range if specified.
    
    Args:
        pattern_file (str): Path to the pattern JSON file
        start_date (str): Optional start date in YYYY-MM-DD format
        end_date (str): Optional end date in YYYY-MM-DD format
        
    Returns:
        list: List of patterns within the date range
    """
    try:
        with open(pattern_file, 'r') as f:
            data = json.load(f)
            
        # Convert date strings to datetime objects if provided
        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
            
        # Track first and last dates in file
        first_date = None
        last_date = None
        total_patterns = 0
        
        # Process patterns
        patterns = []
        for day_data in data:
            try:
                # Handle both live and backtest pattern formats
                if 'pattern_date' in day_data:
                    current_date = datetime.strptime(day_data['pattern_date'], '%Y-%m-%d').date()
                elif 'date' in day_data:
                    current_date = datetime.strptime(day_data['date'], '%Y-%m-%d').date()
                else:
                    logger.warning(f"Skipping entry with no date: {day_data}")
                    continue
                
                # Update first and last dates
                if first_date is None or current_date < first_date:
                    first_date = current_date
                if last_date is None or current_date > last_date:
                    last_date = current_date
                
                # Skip if outside date range
                if start_date and current_date < start_date:
                    continue
                if end_date and current_date > end_date:
                    continue
                
                # Process patterns for this day
                if 'patterns' in day_data and 'sessions' in day_data['patterns']:
                    signal_num = 0
                    for session in ['morning', 'mixed', 'afternoon']:
                        session_patterns = day_data['patterns']['sessions'].get(session, [])
                        for pattern in session_patterns:
                            signal_num += 1
                            pattern_data = {
                                'date': current_date,
                                'signal_num': signal_num,
                                'entry_time': pattern['entry_time'],
                                'exit_time': pattern['exit_time'],
                                'direction': pattern['direction'],
                                'tp': pattern['target_points'],
                                'sl': pattern['stop_loss_points'],
                                'success_rate': pattern['success_rate'],
                                'filter_level': day_data.get('filter_level', 'Unknown')
                            }
                            patterns.append(pattern_data)
                            total_patterns += 1
                
            except Exception as e:
                logger.warning(f"Error processing day data: {str(e)}")
                continue
        
        # Print date range information
        if first_date and last_date:
            print(f"\nPattern file date range: {first_date} to {last_date}")
            if start_date or end_date:
                print(f"Filtered date range: {start_date or first_date} to {end_date or last_date}")
            print(f"Total patterns found: {total_patterns}")
            print(f"Patterns in date range: {len(patterns)}\n")
        
        return patterns
        
    except Exception as e:
        logger.error(f"Error parsing patterns file: {str(e)}")
        return []

# Parse minute data into a DataFrame
def load_minute_data(minute_file):
    df = pd.read_csv(minute_file, names=['datetime', 'open', 'high', 'low', 'close'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time
    return df

# Simulate P&L for a single pattern
def simulate_pnl(pattern, minute_df, tp_adjustment=0, sl_adjustment=0, fixed_tp=None, fixed_sl=None):
    date = pattern['date']
    entry_time = parse_ct_time(pattern['entry_time'])
    exit_time = parse_ct_time(pattern['exit_time'])
    direction = pattern['direction']
    
    # Use fixed values if provided, otherwise use pattern values with adjustments
    tp = fixed_tp if fixed_tp is not None else pattern['tp'] + tp_adjustment
    sl = fixed_sl if fixed_sl is not None else pattern['sl'] + sl_adjustment

    # Get all minute bars for this date
    day_df = minute_df[minute_df['date'] == date]
    if day_df.empty:
        return None, None
    # Find entry row: first row at or after entry_time
    entry_row = day_df[day_df['time'] >= entry_time].head(1)
    if entry_row.empty:
        return None, None
    entry_idx = entry_row.index[0]
    entry_price = entry_row.iloc[0]['open']
    # Find all rows between entry and exit time (inclusive)
    trade_df = day_df[(day_df.index >= entry_idx) & (day_df['time'] <= exit_time)]
    if trade_df.empty:
        return None, None
    # Simulate trade minute by minute
    for _, row in trade_df.iterrows():
        if direction == 'Buy':
            if row['high'] >= entry_price + tp:
                return tp, 'TP'  # TP hit
            if row['low'] <= entry_price - sl:
                return -sl, 'SL'  # SL hit
        else:  # Sell
            if row['low'] <= entry_price - tp:
                return tp, 'TP'  # TP hit
            if row['high'] >= entry_price + sl:
                return -sl, 'SL'  # SL hit
    # If neither TP nor SL hit, exit at last close in trade_df
    exit_price = trade_df.iloc[-1]['close']
    if direction == 'Buy':
        return exit_price - entry_price, 'Time'
    else:
        return entry_price - exit_price, 'Time'

def analyze_entry_time_stats(results):
    """
    Analyze trading statistics grouped by entry time.
    
    Args:
        results (list): List of trade results from simulation
    """
    # Group trades by entry time
    entry_time_stats = {}
    
    for trade in results:
        entry_time = trade['entry_time']
        if entry_time not in entry_time_stats:
            entry_time_stats[entry_time] = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0,
                'max_pnl': float('-inf'),
                'min_pnl': float('inf'),
                'exit_types': {},
                'avg_tp': 0,
                'avg_sl': 0,
                'avg_success_rate': 0
            }
        
        stats = entry_time_stats[entry_time]
        stats['total_trades'] += 1
        stats['total_pnl'] += trade['original_tp_sl_pnl']
        stats['max_pnl'] = max(stats['max_pnl'], trade['original_tp_sl_pnl'])
        stats['min_pnl'] = min(stats['min_pnl'], trade['original_tp_sl_pnl'])
        stats['avg_tp'] += trade['tp']
        stats['avg_sl'] += trade['sl']
        stats['avg_success_rate'] += trade['success_rate']
        
        # Track exit types
        exit_type = trade['exit_type']
        stats['exit_types'][exit_type] = stats['exit_types'].get(exit_type, 0) + 1
        
        # Track wins/losses
        if trade['original_tp_sl_pnl'] > 0:
            stats['winning_trades'] += 1
        else:
            stats['losing_trades'] += 1
    
    # Calculate averages and percentages
    for entry_time, stats in entry_time_stats.items():
        total = stats['total_trades']
        if total > 0:
            stats['win_rate'] = (stats['winning_trades'] / total) * 100
            stats['avg_pnl'] = stats['total_pnl'] / total
            stats['avg_tp'] = stats['avg_tp'] / total
            stats['avg_sl'] = stats['avg_sl'] / total
            stats['avg_success_rate'] = stats['avg_success_rate'] / total
            
            # Calculate exit type percentages
            for exit_type in stats['exit_types']:
                stats['exit_types'][exit_type] = {
                    'count': stats['exit_types'][exit_type],
                    'percentage': (stats['exit_types'][exit_type] / total) * 100
                }
    
    # Print analysis
    print("\nEntry Time Analysis")
    print("=" * 100)
    
    # Sort entry times by converting to datetime for proper time-based sorting
    def parse_time_for_sorting(time_str):
        time_str = time_str.replace(' CT', '')
        return datetime.strptime(time_str, '%I:%M%p')
    
    sorted_times = sorted(entry_time_stats.keys(), key=parse_time_for_sorting)
    
    for entry_time in sorted_times:
        stats = entry_time_stats[entry_time]
        print(f"\nEntry Time: {entry_time}")
        print("-" * 50)
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Total P&L: {stats['total_pnl']:.2f} points")
        print(f"Average P&L: {stats['avg_pnl']:.2f} points")
        print(f"Max P&L: {stats['max_pnl']:.2f} points")
        print(f"Min P&L: {stats['min_pnl']:.2f} points")
        print(f"Average TP: {stats['avg_tp']:.2f} points")
        print(f"Average SL: {stats['avg_sl']:.2f} points")
        print(f"Average Success Rate: {stats['avg_success_rate']:.1f}%")
        
        print("\nExit Type Distribution:")
        for exit_type, data in sorted(stats['exit_types'].items()):
            print(f"  {exit_type}: {data['count']} ({data['percentage']:.1f}%)")
    
    print("\n" + "=" * 100)

def parse_filtered_sentiment_patterns(pattern_file, start_date=None, end_date=None):
    """
    Parse patterns from filtered sentiment-based pattern file and flatten them.
    Returns a list of patterns in the same format as parse_patterns().
    """
    with open(pattern_file, 'r') as f:
        data = json.load(f)
    patterns = []
    for day_data in data:
        if 'pattern_date' not in day_data or 'patterns' not in day_data or 'sessions' not in day_data['patterns']:
            continue
        current_date = datetime.strptime(day_data['pattern_date'], '%Y-%m-%d').date()
        if start_date and current_date < start_date:
            continue
        if end_date and current_date > end_date:
            continue
        signal_num = 0
        for session in ['morning', 'afternoon']:
            session_patterns = day_data['patterns']['sessions'].get(session, [])
            for pattern in session_patterns:
                signal_num += 1
                patterns.append({
                    'date': current_date,
                    'signal_num': signal_num,
                    'entry_time': pattern['entry_time'],
                    'exit_time': pattern['exit_time'],
                    'direction': pattern['direction'],
                    'tp': pattern['target_points'],
                    'sl': pattern['stop_loss_points'],
                    'success_rate': pattern.get('success_rate', 0),
                    'filter_level': day_data.get('filter_level', 'Unknown')
                })
    return patterns

def parse_new_detector_patterns(pattern_file, translatory_constants_path, start_date=None, end_date=None):
    """
    Parse logs/backtest_new_detector.json and deduplicate by (date, timeframe) keeping highest success_rate.
    Returns a list of dicts: date, signal_num, entry_time, exit_time, direction, success_rate, timeframe
    """
    with open(pattern_file, 'r') as f:
        data = json.load(f)
    with open(translatory_constants_path, 'r') as f:
        translatory = json.load(f)
    timeframe_labels = translatory['timeframe_labels']
    # Map timeframe label to (entry_time, exit_time)
    timeframe_to_times = {}
    for tf, label in timeframe_labels.items():
        # label is like '8:30am CT - 10:30am CT'
        entry, exit = label.split(' - ')
        entry = entry.strip()
        exit = exit.strip()
        timeframe_to_times[tf] = (entry, exit)
    # Build (date, timeframe) -> pattern with highest success_rate
    best_patterns = {}
    for patterns in data.values():
        for pattern in patterns:
            tf = pattern['timeframe']
            direction = pattern['direction']
            success_rate = pattern['success_rate']
            for date_str in pattern['pattern_dates']:
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
                if start_date and date < datetime.strptime(start_date, '%Y-%m-%d').date():
                    continue
                if end_date and date > datetime.strptime(end_date, '%Y-%m-%d').date():
                    continue
                key = (date, tf)
                if key not in best_patterns or success_rate > best_patterns[key]['success_rate']:
                    best_patterns[key] = {
                        'date': date,
                        'timeframe': tf,
                        'direction': direction,
                        'success_rate': success_rate
                    }
    # Assign signal_num per date (sorted by entry_time)
    patterns_by_date = collections.defaultdict(list)
    for (date, tf), pat in best_patterns.items():
        entry_time, exit_time = timeframe_to_times[tf]
        patterns_by_date[date].append({
            'date': date,
            'timeframe': tf,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': pat['direction'],
            'success_rate': pat['success_rate']
        })
    # Sort patterns for each date by entry_time and assign signal_num
    all_patterns = []
    for date, pats in patterns_by_date.items():
        pats_sorted = sorted(pats, key=lambda x: datetime.strptime(x['entry_time'].replace(' CT',''), '%I:%M%p'))
        for i, pat in enumerate(pats_sorted, 1):
            pat['signal_num'] = i
            all_patterns.append(pat)
    return all_patterns

def main(tp_adjustment=0, sl_adjustment=0, start_date=None, end_date=None, fixed_tp=None, fixed_sl=None, 
         reversal_continuation=False, continuous_analysis=False, backtest=False, filter_level=None, entry_time_slot_analysis=False,
         pattern_version=None, pattern_file=None):
    # Special logic for logs/backtest_new_detector.json
    if pattern_file == 'logs/backtest_new_detector.json':
        print(f"\nUsing special logic for {pattern_file}")
        patterns = parse_new_detector_patterns(pattern_file, 'config/translatory_constants.json', start_date, end_date)
        if not patterns:
            print("No patterns found in the specified date range")
            return
        # Simulate trades with fixed TP=10, SL=20
        simulator = TradeEngine(MINUTE_DATA_FILE, PATTERN_DB_PATH)
        results = []
        for pat in patterns:
            trade = simulator.simulate_trade(
                pat['date'],
                pat['entry_time'],
                pat['exit_time'],
                pat['direction'],
                10,  # fixed TP
                20   # fixed SL
            )
            if trade:
                results.append({
                    'date': pat['date'],
                    'signal_num': pat['signal_num'],
                    'entry_time': pat['entry_time'],
                    'exit_time': pat['exit_time'],
                    'direction': pat['direction'].capitalize(),
                    'tp': 10,
                    'sl': 20,
                    'original_tp_sl_pnl': trade['profit_loss'],
                    'success_rate': pat['success_rate'],
                    'exit_type': trade['exit_type'],
                    'filter_level': ''  # Not present in this file
                })
        # Write to CSV
        with open(OUTPUT_CSV, 'w', newline='') as f:
            fieldnames = [
                'date', 'signal_num', 'entry_time', 'exit_time', 'direction',
                'tp', 'sl', 'original_tp_sl_pnl', 'success_rate', 'exit_type', 'filter_level'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Wrote {len(results)} rows to {OUTPUT_CSV}")
        calculate_metrics_from_csv(OUTPUT_CSV)
        return
    # Add warning for TP/SL settings
    if fixed_tp is not None or fixed_sl is not None:
        print("\n⚠️  WARNING: Using fixed values:")
        if fixed_tp is not None:
            print(f"   Fixed TP: {fixed_tp:+.2f} points")
        if fixed_sl is not None:
            print(f"   Fixed SL: {fixed_sl:+.2f} points")
        print()
    elif tp_adjustment != 0 or sl_adjustment != 0:
        print("\n⚠️  WARNING: Using adjusted values:")
        print(f"   TP Adjustment: {tp_adjustment:+.2f} points")
        print(f"   SL Adjustment: {sl_adjustment:+.2f} points\n")
    else:
        print("\nRunning with real performance values (no TP/SL adjustments)\n")

    # If a custom pattern file is provided, use it and parse as filtered sentiment patterns
    if pattern_file:
        print(f"\nUsing custom pattern file: {pattern_file}")
        # Convert date strings to datetime.date if provided
        s_date = datetime.strptime(start_date, '%Y-%m-%d').date() if start_date else None
        e_date = datetime.strptime(end_date, '%Y-%m-%d').date() if end_date else None
        patterns = parse_filtered_sentiment_patterns(pattern_file, s_date, e_date)
        if not patterns:
            print("No patterns found in the specified date range")
            return
        # Use default pattern db path
        pattern_db_path = PATTERN_DB_PATH
    else:
        # Select pattern file based on mode and filter level
        if backtest and filter_level:
            if filter_level not in FILTER_PATTERN_FILES:
                print(f"Error: Invalid filter level '{filter_level}'. Must be one of: {', '.join(FILTER_PATTERN_FILES.keys())}")
                return
            pattern_file = FILTER_PATTERN_FILES[filter_level]
            print(f"\nUsing pattern file: {pattern_file}")
            print(f"Filter level: {filter_level}")
        else:
            pattern_file = BACKTEST_PATTERN_FILE if backtest else LIVE_PATTERN_FILE
            print(f"\nUsing pattern file: {pattern_file}")
        # Load patterns
        patterns = parse_patterns(pattern_file, start_date, end_date)
        if not patterns:
            print("No patterns found in the specified date range")
            return
        # Get pattern database path based on version
        try:
            pattern_db_path, version = get_pattern_database(project_root, pattern_version)
            print(f"\nUsing pattern database version v{version}")
        except ValueError as e:
            print(f"Error: {str(e)}")
            return

    # Initialize trade simulator
    simulator = TradeEngine(MINUTE_DATA_FILE, pattern_db_path)

    # Convert patterns to simulator format
    patterns_by_date = {}
    for pattern in patterns:
        date = pattern['date']
        if date not in patterns_by_date:
            patterns_by_date[date] = []
        # Apply TP/SL adjustments
        tp = fixed_tp if fixed_tp is not None else pattern['tp'] + tp_adjustment
        sl = fixed_sl if fixed_sl is not None else pattern['sl'] + sl_adjustment
        patterns_by_date[date].append({
            'entry_time': pattern['entry_time'],
            'exit_time': pattern['exit_time'],
            'direction': 'bullish' if pattern['direction'].lower() == 'buy' else 'bearish',
            'tp_points': tp,
            'sl_points': sl,
            'success_rate': pattern['success_rate']
        })

    # Run simulation with original TP/SL values
    start_date_sim = min(patterns_by_date.keys())
    end_date_sim = max(patterns_by_date.keys())
    trades = simulator.run_simulation(patterns_by_date, start_date_sim, end_date_sim)

    # Convert trades back to original format for analysis
    results = []
    for trade in trades:
        # Get the original pattern to access TP/SL values
        date = trade['date']
        entry_time = trade['entry_time']
        original_pattern = next((p for p in patterns if p['date'] == date and p['entry_time'] == entry_time), None)
        
        if original_pattern:
            results.append({
                'date': trade['date'],
                'signal_num': original_pattern.get('signal_num', 0),  # Use get() with default value
                'entry_time': trade['entry_time'],
                'exit_time': trade['exit_time'],
                'direction': 'Buy' if trade['direction'] == 'bullish' else 'Sell',
                'tp': original_pattern['tp'],
                'sl': original_pattern['sl'],
                'original_tp_sl_pnl': trade['profit_loss'],
                'exit_type': trade['exit_type'],
                'success_rate': original_pattern['success_rate'],
                'filter_level': original_pattern['filter_level']
            })

    # Run simulations for each TP/SL combination
    print(f"\nRunning simulations for {len(TP_SL_COMBINATIONS)} TP/SL combinations...")
    for tp, sl in TP_SL_COMBINATIONS:
        print(f"  Simulating TP={tp}, SL={sl}...")
        
        # Create patterns with fixed TP/SL values
        fixed_patterns_by_date = {}
        for date, date_patterns in patterns_by_date.items():
            fixed_patterns_by_date[date] = []
            for pattern in date_patterns:
                fixed_pattern = pattern.copy()
                fixed_pattern['tp_points'] = tp
                fixed_pattern['sl_points'] = sl
                fixed_patterns_by_date[date].append(fixed_pattern)
        
        # Reset simulator before each run
        simulator.reset()
        # Run simulation with fixed TP/SL
        fixed_trades = simulator.run_simulation(fixed_patterns_by_date, start_date_sim, end_date_sim)
        
        # Build a lookup for (date, entry_time) -> profit_loss
        trade_lookup = {(trade['date'], trade['entry_time']): trade['profit_loss'] for trade in fixed_trades}
        
        # Assign results to the correct row by (date, entry_time)
        for row in results:
            key = (row['date'], row['entry_time'])
            row[f'{tp}/{sl}'] = trade_lookup.get(key, None)

    # Write to CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        # Define fieldnames including all TP/SL combinations
        fieldnames = [
            'date', 'signal_num', 'entry_time', 'exit_time', 'direction', 
            'tp', 'sl', 'original_tp_sl_pnl', 'success_rate', 'exit_type', 'filter_level'
        ]
        
        # Add TP/SL combination columns
        for tp, sl in TP_SL_COMBINATIONS:
            fieldnames.append(f'{tp}/{sl}')
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in results:
            # Create new row with only the fields we want
            new_row = {
                'date': row['date'],
                'signal_num': row['signal_num'],
                'entry_time': row['entry_time'],
                'exit_time': row['exit_time'],
                'direction': row['direction'],
                'tp': row['tp'],
                'sl': row['sl'],
                'original_tp_sl_pnl': row['original_tp_sl_pnl'],
                'success_rate': row['success_rate'],
                'exit_type': row['exit_type'],
                'filter_level': row['filter_level']
            }
            
            # Add TP/SL combination results
            for tp, sl in TP_SL_COMBINATIONS:
                column_name = f'{tp}/{sl}'
                new_row[column_name] = row.get(column_name, None)
            
            writer.writerow(new_row)

    print(f"Wrote {len(results)} rows to {OUTPUT_CSV}")
    calculate_metrics_from_csv(OUTPUT_CSV)
    
    # Add entry time analysis
    if entry_time_slot_analysis:
        analyze_entry_time_stats(results)

def calculate_metrics_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    total_pnl = df['original_tp_sl_pnl'].sum()
    total_pnl_usd = total_pnl * 5  # $5 per point
    total_trades = len(df)
    tp_count = len(df[df['exit_type'] == 'TP'])
    sl_count = len(df[df['exit_type'] == 'SL'])
    time_count = len(df[df['exit_type'] == 'TIME'])
    tp_percentage = (tp_count / total_trades) * 100 if total_trades > 0 else 0
    sl_percentage = (sl_count / total_trades) * 100 if total_trades > 0 else 0
    time_percentage = (time_count / total_trades) * 100 if total_trades > 0 else 0
    
    # Calculate pattern success rate
    successful_patterns = len(df[df['original_tp_sl_pnl'] > 0])
    pattern_success_rate = (successful_patterns / total_trades) * 100 if total_trades > 0 else 0
    
    print(f"Total P&L (Original TP/SL): {total_pnl:.2f} points (${total_pnl_usd:.2f})")
    print(f"TP Hit: {tp_count}/{total_trades} ({tp_percentage:.2f}%)")
    print(f"SL Hit: {sl_count}/{total_trades} ({sl_percentage:.2f}%)")
    print(f"Time Exit: {time_count}/{total_trades} ({time_percentage:.2f}%)")
    print(f"Pattern Success Rate: {successful_patterns}/{total_trades} ({pattern_success_rate:.2f}%)")

    # Add filter level distribution analysis with P&L contribution
    print("\nFilter Level Distribution:")
    filter_stats = df.groupby('filter_level').agg({
        'original_tp_sl_pnl': ['count', 'sum', 'mean'],
        'exit_type': lambda x: (x == 'TP').mean() * 100  # Calculate TP hit rate
    }).round(2)
    
    filter_stats.columns = ['trades', 'total_pnl', 'avg_pnl', 'tp_rate']
    
    for filter_level, stats in filter_stats.iterrows():
        pnl_percentage = (stats['total_pnl'] / total_pnl) * 100 if total_pnl != 0 else 0
        print(f"  {filter_level}:")
        print(f"    Trades: {int(stats['trades'])} ({stats['trades']/total_trades*100:.1f}%)")
        print(f"    P&L: {stats['total_pnl']:.2f} points (${stats['total_pnl']*5:.2f})")
        print(f"    P&L Contribution: {pnl_percentage:.1f}%")
        print(f"    Avg P&L per Trade: {stats['avg_pnl']:.2f} points")
        print(f"    TP Hit Rate: {stats['tp_rate']:.1f}%")

    # Analyze TP/SL combinations
    print("\nTP/SL Combination Analysis:")
    for tp, sl in TP_SL_COMBINATIONS:
        column_name = f'{tp}/{sl}'
        if column_name in df.columns:
            combination_pnl = df[column_name].sum()
            combination_pnl_usd = combination_pnl * 5
            successful_trades = len(df[df[column_name] > 0])
            success_rate = (successful_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Count TP hits (exact TP value), SL hits (exact negative SL value), and timeouts
            tp_hits = len(df[df[column_name] == tp])
            sl_hits = len(df[df[column_name] == -sl])
            timeouts = len(df[(df[column_name] != tp) & (df[column_name] != -sl) & (df[column_name].notna())])
            
            print(f"  {column_name}:")
            print(f"    Total P&L: {combination_pnl:.2f} points (${combination_pnl_usd:.2f})")
            print(f"    Success Rate: {successful_trades}/{total_trades} ({success_rate:.1f}%)")
            print(f"    TP Hits: {tp_hits}/{total_trades} ({tp_hits/total_trades*100:.1f}%)")
            print(f"    SL Hits: {sl_hits}/{total_trades} ({sl_hits/total_trades*100:.1f}%)")
            print(f"    Timeouts: {timeouts}/{total_trades} ({timeouts/total_trades*100:.1f}%)")

if __name__ == '__main__':
    args = parse_args()
    main(
        tp_adjustment=args.tp_adjustment,
        sl_adjustment=args.sl_adjustment,
        start_date=args.start_date,
        end_date=args.end_date,
        fixed_tp=args.fixed_tp,
        fixed_sl=args.fixed_sl,
        reversal_continuation=args.reversal_continuation,
        continuous_analysis=args.continuous_analysis,
        backtest=args.backtest,
        filter_level=args.filter,
        entry_time_slot_analysis=args.entry_time_slot_analysis,
        pattern_version=args.pattern_version,
        pattern_file=args.pattern_file
    ) 