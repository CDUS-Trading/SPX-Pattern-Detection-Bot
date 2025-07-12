import sys
import os
# Add project root to Python path BEFORE any src imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

import argparse
import json
from datetime import datetime
import pandas as pd
from src.patterns.utils.metrics import calculate_metrics
from src.patterns.utils.io import get_pattern_database
from src.patterns.core.pattern_detector_class import PatternDetector

# Hardcoded metric pairs (replace/add as needed)
HARDCODED_METRIC_PAIRS = [
    {
        "power_hour_return": 0.2333365625198235,
        "lunch_hour_return": -0.18310314810675954
    },
    {
        "pre_lunch_momentum": 0.0727700573705026,
        "last_hour": -0.4413762481433157
    },
    {
        "last_30min": -0.3091091938813234,
        "close_vol": 0.3923851867534675
    },
    {
        "first_5min_range": 0.30219333300326323,
        "first_5min_high_test": 0.0
    },
    {
        "last_hour": -0.6111845187645901,
        "close_vol": 0.6652732955333129
    },
    {
        "first_60min_vol": 0.8580237950855302,
        "close_strength": 0.34793898085036745
    },
    {
        "last_30min": 0.10175670464564004,
        "close_strength": 0.18873403019744206
    },
    {
        "first_5min_return": 0.13159813727585443,
        "power_hour_return": -0.09479814270258051
    },
    {
        "first_5min_range": 0.33587633109588855,
        "first_5min_low_test": 0.2504371407965297
    },
    {
        "power_hour_return": -0.18934983140777473,
        "lunch_hour_return": 0.15503061642883392
    },
    {
        "first_5min_return": 0.11138225376928286,
        "power_hour_return": -0.18934983140777473
    },
    {
        "pre_lunch_momentum": -0.2024374468176859,
        "morning_trend_strength": 1
    },
    {
        "pre_lunch_momentum": -0.2024374468176859,
        "close_strength": 0.10688013846819366
    },
    {
        "first_5min_range": 0.29882356110142155,
        "first_5min_high_test": 0.05336736274318468
    }
]

def parse_args():
    parser = argparse.ArgumentParser(description="Find days matching metric pairs and extract patterns.")
    parser.add_argument('--start-date', type=str, default='2025-06-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD), defaults to last date in minute data')
    parser.add_argument('--pattern-version', '-pv', type=int, help='Pattern database version number to use (e.g., 2 for v2). If not provided, uses the latest.')
    parser.add_argument('--minute-data', type=str, default='data/SPX_full_1min_CT.txt', help='Path to minute data file')
    parser.add_argument('--output', type=str, default='matched_patterns.json', help='Output JSON file for results')
    parser.add_argument('--generate-metrics-bucket', '-gmb', action='store_true', help='Generate all_metrics_bucket.json from minute data')
    parser.add_argument('--transform-matched-patterns', action='store_true', help='Transform matched_patterns.json to current_detected_patterns.json style')
    return parser.parse_args()


def load_metric_pairs(metrics_json_path):
    with open(metrics_json_path, 'r') as f:
        metric_pairs = json.load(f)
    # Expecting a list of dicts: [{"metric1": ..., "value1": ..., "metric2": ..., "value2": ...}, ...]
    if not isinstance(metric_pairs, list):
        raise ValueError("metrics-json must be a list of metric pair dicts")
    return metric_pairs

def load_minute_data(minute_data_path):
    df = pd.read_csv(minute_data_path, header=None, names=['datetime', 'open', 'high', 'low', 'close'], parse_dates=['datetime'])
    df['date'] = df['datetime'].dt.date
    return df


def filter_dates(df, start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()
    mask = (df['date'] >= start) & (df['date'] <= end)
    return df[mask]


def calculate_metrics_for_days(filtered_df):
    metrics_by_date = {}
    for date, group in filtered_df.groupby('date'):
        try:
            metrics = calculate_metrics(group.reset_index(drop=True))
            metrics_by_date[date] = metrics
        except Exception as e:
            print(f"Warning: Could not calculate metrics for {date}: {e}")
    return metrics_by_date


BUCKETS = [-float('inf'), -1.0, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1.0, float('inf')]

def get_bucket_label(value, buckets=BUCKETS):
    for i in range(len(buckets) - 1):
        if buckets[i] < value <= buckets[i + 1]:
            left = buckets[i]
            right = buckets[i + 1]
            return f"({left}, {right}]", left, right
    return None, None, None

def save_metrics_bucket(metrics_by_date, output_path='all_metrics_bucket.json'):
    bucketed = {}
    for date, metrics in metrics_by_date.items():
        bucketed_metrics = {}
        for metric, value in metrics.items():
            bucketed_metrics[metric] = value
            if isinstance(value, (int, float)):
                bucket_label, left, right = get_bucket_label(value)
                bucketed_metrics[f"{metric}_bucket"] = bucket_label
        bucketed[str(date)] = bucketed_metrics
    with open(output_path, 'w') as f:
        json.dump(bucketed, f, indent=2)
    print(f"Saved all metrics (with buckets) by date to {output_path}")


def load_all_metrics_bucket(path='all_metrics_bucket.json'):
    with open(path, 'r') as f:
        return json.load(f)

def load_pattern_db(pattern_db_path):
    with open(pattern_db_path, 'r') as f:
        return json.load(f)

def find_matching_dates_and_patterns(metric_pairs, all_metrics_bucket, pattern_db, output_path):
    results = []
    for pair in metric_pairs:
        if len(pair) != 2:
            continue
        metrics = list(pair.keys())
        values = list(pair.values())
        bucketed = {}
        bucket_bounds = {}
        for m, v in zip(metrics, values):
            bucket_label, left, right = get_bucket_label(v)
            bucketed[m] = bucket_label
            bucket_bounds[m] = (left, right)
        for date, metrics_dict in all_metrics_bucket.items():
            match = True
            for m in metrics:
                if metrics_dict.get(f"{m}_bucket") != bucketed[m]:
                    match = False
                    break
            if match:
                patterns = []
                for pat in pattern_db:
                    if (
                        pat.get("pattern1") == metrics[0]
                        and pat.get("pattern2") == metrics[1]
                        and pat.get("range1_min") == bucket_bounds[metrics[0]][0]
                        and pat.get("range1_max") == bucket_bounds[metrics[0]][1]
                        and pat.get("range2_min") == bucket_bounds[metrics[1]][0]
                        and pat.get("range2_max") == bucket_bounds[metrics[1]][1]
                    ):
                        patterns.append(pat)
                result = {
                    "pattern_date": date,
                    "matched_metric_values": {m: metrics_dict[m] for m in metrics},
                    "bucketed": {m: metrics_dict[f"{m}_bucket"] for m in metrics},
                    "patterns": patterns
                }
                results.append(result)
    # Sort results by pattern_date before writing
    results.sort(key=lambda x: x["pattern_date"])
    all_matched_dates = [r["pattern_date"] for r in results]
    output_obj = {
        "all_matched_dates": all_matched_dates,
        "results": results
    }
    with open(output_path, 'w') as f:
        json.dump(output_obj, f, indent=2)
    print(f"Wrote all matches and patterns to {output_path}")

def transform_matched_patterns_to_detected(matched_patterns_path, all_metrics_bucket_path, pattern_db_path, output_path):
    # Load matched patterns
    with open(matched_patterns_path, 'r') as f:
        matches = json.load(f)
    # Load all_metrics_bucket for close price lookup
    with open(all_metrics_bucket_path, 'r') as f:
        all_metrics_bucket = json.load(f)
    # Load pattern DB (needed for PatternDetector)
    detector = PatternDetector(pattern_db_path)

    # For each match (date), build detected pattern structure
    detected = []
    for match in matches:
        date = match["pattern_date"]
        metrics_dict = all_metrics_bucket.get(date, {})
        close_price = metrics_dict.get("close")
        if close_price is None:
            # Fallback: try to get from matched_metric_values
            close_price = match["matched_metric_values"].get("close")
        # Compose sessions
        session_patterns = {"morning": [], "mixed": [], "afternoon": []}
        for pattern in match["patterns"]:
            strategy = detector.generate_trading_strategy(pattern)
            period = strategy.get("period", "mixed")
            # Parse entry/exit times
            timeframe = strategy.get("timeframe", "")
            if '-' in timeframe:
                entry_time, exit_time = [t.strip() for t in timeframe.split('-')]
            else:
                entry_time = exit_time = timeframe.strip()
            if 'CT' not in entry_time:
                entry_time += ' CT'
            if 'CT' not in exit_time:
                exit_time += ' CT'
            # Calculate target/stop in points
            if close_price is not None:
                try:
                    if strategy['direction'] == 'bullish':
                        tp_pct = float(strategy['target'].split('+')[1].split('%')[0])
                        sl_pct = float(strategy['stop_loss'].split('-')[1].split('%')[0])
                        tp_points = round(close_price * (tp_pct / 100), 2)
                        sl_points = round(close_price * (sl_pct / 100), 2)
                    else:
                        tp_pct = float(strategy['target'].split('-')[1].split('%')[0])
                        sl_pct = float(strategy['stop_loss'].split('+')[1].split('%')[0])
                        tp_points = round(close_price * (tp_pct / 100), 2)
                        sl_points = round(close_price * (sl_pct / 100), 2)
                except Exception:
                    tp_points = sl_points = None
            else:
                tp_points = sl_points = None
            # Compose pattern dict
            pattern_data = {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "direction": "Buy" if strategy['direction'] == 'bullish' else "Sell",
                "target_points": tp_points,
                "stop_loss_points": sl_points,
                "success_rate": round(float(strategy['success_rate']), 2),
                "sample_size": strategy['sample_size'],
                "historical_dates": strategy.get('pattern_dates', []),
                "matched_metric_values": match["matched_metric_values"]
            }
            session_patterns[period].append(pattern_data)
        # Compose output structure
        output_data = {
            "pattern_day": datetime.strptime(date, "%Y-%m-%d").strftime('%A'),
            "pattern_date": date,
            "close_price": close_price,
            "filter_level": None,  # Not tracked in this workflow
            "patterns": {"sessions": session_patterns}
        }
        detected.append(output_data)
    # Save to output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(detected, f, indent=2)
    print(f"Transformed matched_patterns.json to {output_path}")

def main():
    args = parse_args()
    metric_pairs = HARDCODED_METRIC_PAIRS
    print(f"Loaded {len(metric_pairs)} metric pair(s) from hardcoded values")

    # Set up project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    sys.path.insert(0, project_root)

    if args.generate_metrics_bucket:
        # Load and filter minute data
        minute_df = load_minute_data(args.minute_data)
        # Set default end date if not provided
        if args.end_date is None:
            last_date = str(minute_df['date'].max())
            end_date = last_date
        else:
            end_date = args.end_date
        start_date = args.start_date
        filtered_df = filter_dates(minute_df, start_date, end_date)
        days = filtered_df['date'].unique()
        print(f"Loaded minute data for {len(days)} day(s) in range {start_date} to {end_date}")

        # Calculate metrics for each day
        metrics_by_date = calculate_metrics_for_days(filtered_df)
        print(f"Calculated metrics for {len(metrics_by_date)} day(s)")

        # Save all metrics (with buckets) by date
        save_metrics_bucket(metrics_by_date, output_path='all_metrics_bucket.json')
    else:
        print("Skipping metrics bucket generation. Using existing all_metrics_bucket.json.")

    # Get pattern database path based on version
    try:
        pattern_db_path, version = get_pattern_database(project_root, args.pattern_version)
        print(f"Using pattern database version v{version}")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Load all_metrics_bucket.json and pattern DB
    all_metrics_bucket = load_all_metrics_bucket('all_metrics_bucket.json')
    pattern_db = load_pattern_db(pattern_db_path)
    # Find and write matches
    find_matching_dates_and_patterns(metric_pairs, all_metrics_bucket, pattern_db, args.output)

    # If transform flag is set, run transformation
    if hasattr(args, 'transform_matched_patterns') and args.transform_matched_patterns:
        transform_matched_patterns_to_detected(
            matched_patterns_path=args.output,
            all_metrics_bucket_path='all_metrics_bucket.json',
            pattern_db_path=pattern_db_path,
            output_path=os.path.join(project_root, 'logs', 'current_detected_patterns.json')
        )

if __name__ == "__main__":
    # Add transform flag to argparse
    import argparse as _argparse
    parser = _argparse.ArgumentParser()
    parser.add_argument('--transform-matched-patterns', action='store_true', help='Transform matched_patterns.json to current_detected_patterns.json style')
    args, unknown = parser.parse_known_args()
    if args.transform_matched_patterns:
        # Patch parse_args to include this flag
        orig_parse_args = parse_args
        def patched_parse_args():
            orig = orig_parse_args()
            setattr(orig, 'transform_matched_patterns', True)
            return orig
        parse_args = patched_parse_args
    main() 