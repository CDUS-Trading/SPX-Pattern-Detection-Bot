import argparse
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime, timedelta
from src.patterns.utils.run_detector_class import RunDetector
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

DATA_FILE = 'data/SPX_full_1min_CT.txt'
OUTPUT_FILE = 'runs_database.json'
DEFAULT_START_DATE = '2022-10-01'

COL_NAMES = ['timestamp', 'open', 'high', 'low', 'close']


def parse_args():
    parser = argparse.ArgumentParser(description='Collect runs for each trading day in a date range.')
    parser.add_argument('--start-date', type=str, default=DEFAULT_START_DATE, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD), default is most recent in data file')
    return parser.parse_args()


def get_all_dates(data_file):
    dates = set()
    with open(data_file, 'r') as f:
        for line in f:
            date_str = line.split(',')[0][:10]
            dates.add(date_str)
    return sorted(dates)


def get_last_date(data_file):
    last_date = None
    with open(data_file, 'r') as f:
        for line in f:
            date_str = line.split(',')[0][:10]
            last_date = date_str
    return last_date


def load_day_data(data_file, target_date):
    # Efficiently load only lines for the target date
    rows = []
    with open(data_file, 'r') as f:
        for line in f:
            if line.startswith(target_date):
                parts = line.strip().split(',')
                if len(parts) == 5:
                    rows.append(parts)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=COL_NAMES)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.set_index('timestamp', inplace=True)
    return df


def main():
    args = parse_args()
    start_date = args.start_date
    end_date = args.end_date or get_last_date(DATA_FILE)

    all_dates = get_all_dates(DATA_FILE)
    # Filter dates within range
    all_dates = [d for d in all_dates if start_date <= d <= end_date]

    detector = RunDetector()
    results = []

    with tqdm(total=len(all_dates), ncols=100) as pbar:
        for date in all_dates:
            pbar.set_description(f"{date}")
            day_data = load_day_data(DATA_FILE, date)
            if day_data is None or day_data.empty:
                pbar.update(1)
                continue
            runs = detector.detect_runs(day_data)
            results.append({
                'date': date,
                'runs': runs
            })
            pbar.update(1)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} days of runs to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 