import re
import csv
import json
import argparse
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

PATTERN_FILE = 'logs/all_live_detected_patterns.json'
MINUTE_DATA_FILE = 'data/SPX_full_1min_CT.txt'  # or SPX_week_1min_CT.txt for testing
OUTPUT_CSV = 'output/parsed_patterns_with_pnl.csv'

def parse_args():
    parser = argparse.ArgumentParser(description='Simulate P&L for trading patterns')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD). If not provided, uses first date in pattern file.')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD). If not provided, uses last date in pattern file.')
    parser.add_argument('--tp-adjustment', '-atp', type=float, default=0, help='Adjustment for take profit levels')
    parser.add_argument('--sl-adjustment', '-asl', type=float, default=0, help='Adjustment for stop loss levels')
    parser.add_argument('--fixed-tp', '-ftp', type=float, help='Override pattern TP with fixed value (in points)')
    parser.add_argument('--fixed-sl', '-fsl', type=float, help='Override pattern SL with fixed value (in points)')
    return parser.parse_args()

# Helper to parse time like '8:30AM CT' to datetime.time
def parse_ct_time(timestr):
    timestr = timestr.replace(' CT', '')
    return datetime.strptime(timestr, '%I:%M%p').time()

# Parse the pattern file
def parse_patterns(pattern_file, start_date=None, end_date=None):
    patterns = []
    
    # Load JSON file
    with open(pattern_file, 'r') as f:
        pattern_data = json.load(f)
    
    # Convert string dates to datetime.date objects if provided
    if start_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if end_date:
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # Track date range in file
    first_date = None
    last_date = None
    
    # Process each day's patterns
    for day_data in pattern_data:
        current_date = datetime.strptime(day_data['pattern_date'], '%Y-%m-%d').date()
        
        # Track first and last dates in file
        if first_date is None:
            first_date = current_date
        last_date = current_date
        
        # Skip if date is outside our range
        if (start_date and current_date < start_date) or (end_date and current_date > end_date):
            continue
        
        signal_num = 0
        
        # Process patterns from each session
        for session in ['morning', 'mixed', 'afternoon']:
            for pattern in day_data['patterns']['sessions'][session]:
                signal_num += 1
                
                patterns.append({
                    'date': current_date,
                    'signal_num': signal_num,
                    'entry_time': pattern['entry_time'],
                    'exit_time': pattern['exit_time'],
                    'direction': pattern['direction'],
                    'tp': pattern['target_points'],
                    'sl': pattern['stop_loss_points'],
                    'success_rate': pattern['success_rate'],
                    'filter_level': day_data['filter_level']
                })
    
    # Print date range information
    print(f"\nPattern file date range: {first_date} to {last_date}")
    if start_date or end_date:
        print(f"Filtered date range: {start_date or first_date} to {end_date or last_date}")
    print(f"Total patterns found: {len(patterns)}")
    
    return patterns

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

def main(tp_adjustment=0, sl_adjustment=0, start_date=None, end_date=None, fixed_tp=None, fixed_sl=None):
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
    
    patterns = parse_patterns(PATTERN_FILE, start_date, end_date)
    if not patterns:
        print("No patterns found in the specified date range")
        return
        
    minute_df = load_minute_data(MINUTE_DATA_FILE)
    results = []
    for pat in tqdm(patterns, desc="Processing patterns"):
        pnl, exit_type = simulate_pnl(pat, minute_df, tp_adjustment, sl_adjustment, fixed_tp, fixed_sl)
        results.append({**pat, 'pnl': pnl, 'exit_type': exit_type})
    # Write to CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        # Create column names with adjustment info
        if fixed_tp is not None:
            tp_col = f"TP (fixed at {fixed_tp:+.2f})"
        else:
            tp_col = f"TP (adjusted by {tp_adjustment:+.2f})" if tp_adjustment != 0 else "TP (unadjusted)"
            
        if fixed_sl is not None:
            sl_col = f"SL (fixed at {fixed_sl:+.2f})"
        else:
            sl_col = f"SL (adjusted by {sl_adjustment:+.2f})" if sl_adjustment != 0 else "SL (unadjusted)"
        
        fieldnames = [
            'date', 'signal_num', 'entry_time', 'exit_time', 'direction', 
            tp_col, sl_col, 'success_rate', 'pnl', 'exit_type', 'filter_level'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Create a new row with the correct column names
            new_row = {k: v for k, v in row.items() if k not in ['tp', 'sl']}
            new_row[tp_col] = fixed_tp if fixed_tp is not None else row['tp']
            new_row[sl_col] = fixed_sl if fixed_sl is not None else row['sl']
            writer.writerow(new_row)
    print(f"Wrote {len(results)} rows to {OUTPUT_CSV}")

def calculate_metrics_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    total_pnl = df['pnl'].sum()
    total_trades = len(df)
    tp_count = len(df[df['exit_type'] == 'TP'])
    sl_count = len(df[df['exit_type'] == 'SL'])
    time_count = len(df[df['exit_type'] == 'Time'])
    tp_percentage = (tp_count / total_trades) * 100 if total_trades > 0 else 0
    sl_percentage = (sl_count / total_trades) * 100 if total_trades > 0 else 0
    time_percentage = (time_count / total_trades) * 100 if total_trades > 0 else 0
    print(f"Total P&L: {total_pnl}")
    print(f"TP Hit Percentage: {tp_percentage:.2f}%")
    print(f"SL Hit Percentage: {sl_percentage:.2f}%")
    print(f"Time Exit Percentage: {time_percentage:.2f}%")

if __name__ == '__main__':
    args = parse_args()
    main(
        tp_adjustment=args.tp_adjustment,
        sl_adjustment=args.sl_adjustment,
        start_date=args.start_date,
        end_date=args.end_date,
        fixed_tp=args.fixed_tp,
        fixed_sl=args.fixed_sl
    )
    calculate_metrics_from_csv(OUTPUT_CSV) 