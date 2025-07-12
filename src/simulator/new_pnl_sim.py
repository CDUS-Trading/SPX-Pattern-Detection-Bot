"""
new_pnl_sim.py

Simulates trades based on detected patterns and minute data, with fixed TP/SL.
Outputs a CSV with trade results.

Usage:
    python new_pnl_sim.py --start YYYY-MM-DD --end YYYY-MM-DD --output output.csv
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import csv
import argparse
from datetime import datetime
from tqdm import tqdm
from src.utils.analysis_utils import analyze_consecutive_patterns

CONFIG_PATH = "config/translatory_constants.json"
PATTERNS_PATH = "logs/backtest_new_detector.json"
MINUTE_DATA_PATH = "data/SPX_full_1min_CT.txt"

POINT_TO_DOLLAR = 5

def load_timeframe_mapping(config_path):
    """Load timeframe indices and labels from config."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["timeframe_indices"], config["timeframe_labels"]

def load_patterns(patterns_path, start_date, end_date):
    """Load and filter patterns by date."""
    with open(patterns_path, "r") as f:
        all_days = json.load(f)
    filtered = []
    for day in all_days:
        date = day["date"]
        if start_date <= date <= end_date:
            for i, pattern in enumerate(day["patterns"]):
                filtered.append({
                    "date": date,
                    "pattern": pattern,
                    "pattern_number": i + 1
                })
    return filtered

def minute_data_generator(minute_data_path, needed_dates):
    """
    Yields (date, [minute rows]) for each needed date.
    Each row: (datetime_str, open, high, low, close)
    """
    current_date = None
    current_rows = []
    with open(minute_data_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 5:
                continue
            dt_str = parts[0]
            date_str = dt_str.split(" ")[0]
            if date_str != current_date:
                if current_date and current_date in needed_dates:
                    yield current_date, current_rows
                current_date = date_str
                current_rows = []
            current_rows.append(parts)
        # Last date
        if current_date and current_date in needed_dates:
            yield current_date, current_rows

def simulate_trade(pattern, minute_rows, timeframe_indices, pattern_number, tp_points, sl_points):
    """
    Simulate a single trade for the given pattern and minute data.
    Returns a dict with output columns, or None if data is missing.
    """
    timeframe = pattern.get("timeframe")
    direction = pattern.get("direction")
    date = pattern.get("date") if "date" in pattern else None  # fallback if needed

    # Get the minute indices for the timeframe
    if timeframe not in timeframe_indices:
        return None
    start_idx, end_idx = timeframe_indices[timeframe]
    # end_idx is exclusive, so we want minute_rows[start_idx:end_idx]
    trade_minutes = minute_rows[start_idx:end_idx]
    if len(trade_minutes) != (end_idx - start_idx):
        # Missing minute data, skip trade
        return None

    entry_row = trade_minutes[0]
    exit_row = trade_minutes[-1]
    entry_time = entry_row[0]
    exit_time = exit_row[0]
    entry_price = float(entry_row[1])  # open price
    exit_price = float(exit_row[4])    # close price

    tp_hit = False
    sl_hit = False
    exit_type = "TIME"
    pnl_points = None

    for row in trade_minutes:
        minute_high = float(row[2])
        minute_low = float(row[3])
        # For bullish
        if direction == "bullish":
            if minute_high >= entry_price + tp_points:
                exit_type = "TP"
                exit_time = row[0]
                exit_price = entry_price + tp_points
                tp_hit = True
                break
            if minute_low <= entry_price - sl_points:
                exit_type = "SL"
                exit_time = row[0]
                exit_price = entry_price - sl_points
                sl_hit = True
                break
        # For bearish
        elif direction == "bearish":
            if minute_low <= entry_price - tp_points:
                exit_type = "TP"
                exit_time = row[0]
                exit_price = entry_price - tp_points
                tp_hit = True
                break
            if minute_high >= entry_price + sl_points:
                exit_type = "SL"
                exit_time = row[0]
                exit_price = entry_price + sl_points
                sl_hit = True
                break
        else:
            # Unknown direction
            return None

    # Calculate PnL
    if direction == "bullish":
        pnl_points = exit_price - entry_price
    else:
        pnl_points = entry_price - exit_price
    pnl_dollars = pnl_points * POINT_TO_DOLLAR

    return {
        "date": pattern.get("date", date),
        "pattern_number_of_day": pattern_number,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2),
        "pnl_points": round(pnl_points, 2),
        "pnl_dollars": round(pnl_dollars, 2),
        "exit_type": exit_type
    }

def main(start_date, end_date, output_csv, top_n_patterns=None, tp_points=10, sl_points=20, show_tf_stats=False):
    timeframe_indices, timeframe_labels = load_timeframe_mapping(CONFIG_PATH)
    patterns = load_patterns(PATTERNS_PATH, start_date, end_date)
    needed_dates = set(p["date"] for p in patterns)

    # Prepare output
    output_rows = []
    minute_data = {date: rows for date, rows in minute_data_generator(MINUTE_DATA_PATH, needed_dates)}

    # If top_n_patterns is set, for each day, select top N patterns by tp_hit_count
    if top_n_patterns is not None:
        from collections import defaultdict
        patterns_by_date = defaultdict(list)
        for p in patterns:
            patterns_by_date[p["date"]].append(p)
        filtered_patterns = []
        for date, pats in patterns_by_date.items():
            # Sort by tp_hit_count descending, then by pattern_number ascending for tie-breaker
            sorted_pats = sorted(pats, key=lambda x: (-x["pattern"].get("tp_hit_count", 0), x["pattern_number"]))
            filtered_patterns.extend(sorted_pats[:top_n_patterns])
        patterns = filtered_patterns
        print(f"Filtered to {len(patterns)} patterns for top {top_n_patterns} patterns per day by tp_hit_count.")

    # Now simulate trades as before
    output_rows = []
    for p in tqdm(patterns, desc="Simulating trades"):
        date = p["date"]
        pattern = p["pattern"]
        pattern_number = p["pattern_number"]
        minute_rows = minute_data.get(date)
        if not minute_rows:
            continue
        result = simulate_trade(pattern, minute_rows, timeframe_indices, pattern_number, tp_points, sl_points)
        if result:
            # Ensure date is present in each row
            result["date"] = date
            # Add timeframe for stats
            result["timeframe"] = pattern.get("timeframe", "unknown")
            output_rows.append(result)

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date", "pattern_number_of_day", "entry_time", "exit_time",
            "entry_price", "exit_price", "pnl_points", "pnl_dollars", "exit_type"
        ])
        writer.writeheader()
        for row in output_rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    # Print trading stats to console
    if not output_rows:
        print("No trades to report.")
        return

    # Always print overall trading stats
    dates = [row["date"] for row in output_rows]
    min_date = min(dates)
    max_date = max(dates)
    print(f"Simulated trades from {min_date} to {max_date}")

    total_pnl = sum(row["pnl_dollars"] for row in output_rows)
    exit_type_counts = {}
    exit_type_pnl = {}
    for row in output_rows:
        exit_type = row["exit_type"]
        exit_type_counts[exit_type] = exit_type_counts.get(exit_type, 0) + 1
        exit_type_pnl[exit_type] = exit_type_pnl.get(exit_type, 0) + row["pnl_dollars"]
    total_trades = len(output_rows)
    print(f"\n=== Overall Trading Stats ===")
    print(f"Total trades: {total_trades}")
    print(f"Total PnL ($): {round(total_pnl, 2)}")
    for etype, count in exit_type_counts.items():
        pct = 100 * count / total_trades
        print(f"{etype}: {count} trades ({pct:.2f}%)")
    print("Exit type PnL contribution:")
    for etype, pnl in exit_type_pnl.items():
        print(f"{etype}: {round(pnl, 2)}$")

    # Only print per-timeframe stats if show_tf_stats is True
    if show_tf_stats:
        print(f"\n=== Per-Timeframe Stats ===")
        from collections import defaultdict
        tf_stats = defaultdict(list)
        for row in output_rows:
            tf_stats[row["timeframe"]].append(row)
        tf_order = [f"hour{i}" for i in range(1, 7)]
        for tf in tf_order:
            if tf not in tf_stats:
                continue
            rows = tf_stats[tf]
            tf_pnl = sum(r["pnl_dollars"] for r in rows)
            tf_total = len(rows)
            tf_exit_types = {}
            tf_exit_type_pnl = {}
            for r in rows:
                tf_exit_types[r["exit_type"]] = tf_exit_types.get(r["exit_type"], 0) + 1
                tf_exit_type_pnl[r["exit_type"]] = tf_exit_type_pnl.get(r["exit_type"], 0) + r["pnl_dollars"]
            print(f"Timeframe: {tf}")
            print(f"  Trades: {tf_total}")
            print(f"  Total PnL ($): {round(tf_pnl, 2)}")
            for etype, count in tf_exit_types.items():
                pct = 100 * count / tf_total
                print(f"    {etype}: {count} trades ({pct:.2f}%)")
            print(f"  Exit type PnL contribution:")
            for etype, pnl in tf_exit_type_pnl.items():
                print(f"    {etype}: {round(pnl, 2)}$")
            print("")

def get_date_range_from_patterns(patterns_path):
    """Return (min_date, max_date) as strings from the patterns JSON."""
    with open(patterns_path, "r") as f:
        all_days = json.load(f)
    dates = [day["date"] for day in all_days if "date" in day]
    if not dates:
        raise ValueError("No dates found in patterns file.")
    return min(dates), max(dates)

if __name__ == "__main__":
    # Get default date range from patterns file
    default_start, default_end = get_date_range_from_patterns(PATTERNS_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=default_start, help=f"Start date (YYYY-MM-DD), default: {default_start}")
    parser.add_argument("--end-date", default=default_end, help=f"End date (YYYY-MM-DD), default: {default_end}")
    parser.add_argument("--output", default="logs/tp_hit_rate_patterns_pnl.csv", help="Output CSV file (default: logs/tp_hit_rate_patterns_pnl.csv)")
    parser.add_argument("--top-n-patterns", type=int, default=None, help="Simulate trades only for top N patterns per day by tp_hit_count")
    parser.add_argument("--tp", type=float, default=10, help="Take profit points (default: 10)")
    parser.add_argument("--sl", type=float, default=20, help="Stop loss points (default: 20)")
    parser.add_argument("--show-tf-stats", action="store_true", help="Show trading stats in console output")
    parser.add_argument("--continuous-analysis", '-ca', action="store_true", help="Analyze consecutive win/loss streaks after simulation")
    args = parser.parse_args()
    main(args.start_date, args.end_date, args.output, args.top_n_patterns, args.tp, args.sl, show_tf_stats=args.show_tf_stats)

    # If requested, analyze consecutive win/loss streaks
    if args.continuous_analysis:
        # Read the output CSV and convert to the expected format
        import csv
        results = []
        with open(args.output, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert pnl_points or pnl_dollars to float, use as 'original_tp_sl_pnl'
                row['original_tp_sl_pnl'] = float(row.get('pnl_points', 0))
                results.append(row)
        analyze_consecutive_patterns(results)
