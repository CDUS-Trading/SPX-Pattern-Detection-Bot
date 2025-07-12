import argparse
import pandas as pd
import json
import os

# Mapping from sentiment to pattern direction
SENTIMENT_TO_DIRECTION = {
    'Bullish': 'Buy',
    'Bearish': 'Sell'
}

# Pattern file mapping (now using logs/backtest_patterns_<level>.json)
PATTERN_FILE_MAP = {
    'strict': 'logs/backtest_patterns_Strict.json',
    'moderate': 'logs/backtest_patterns_Moderate.json',
    'minimum': 'logs/backtest_patterns_Minimum.json',
    'recursive': 'logs/backtest_patterns_recursive.json',
}

SESSION_SHEET_MAP = {
    'morning': 'Morning_Session(8-12)',
    'afternoon': 'Afternoon_Session(12-3)'
}

PATTERN_SESSION_MAP = {
    'morning': 'morning',
    'afternoon': 'afternoon'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Filter patterns by sentiment prediction.')
    parser.add_argument('--pattern_level', choices=['strict', 'moderate', 'minimum', 'recursive'], default='recursive', help='Pattern strictness level')
    parser.add_argument('--output', default='filtered_sentiment_based_patterns.json', help='Output file for filtered patterns')
    return parser.parse_args()

def read_sentiment_data(xlsx_path):
    # Read both sheets into a dict
    xl = pd.ExcelFile(xlsx_path)
    data = {}
    for session, sheet in SESSION_SHEET_MAP.items():
        df = xl.parse(sheet)
        # Standardize column names
        df.columns = [c.strip().lower() for c in df.columns]
        data[session] = df
    return data

def load_pattern_data(pattern_file):
    with open(pattern_file, 'r') as f:
        data = json.load(f)
        # If the file has 'results', return only that list
        if isinstance(data, dict) and 'results' in data:
            return data['results']
        return data

def filter_patterns_by_sentiment(sentiment_data, pattern_data):
    filtered_days = []
    # Build lookup for sentiment by date and session
    sentiment_lookup = {}
    for session in ['morning', 'afternoon']:
        df = sentiment_data[session]
        for _, row in df.iterrows():
            date = str(row['date']).split()[0]
            sentiment = row['predicted_trend']
            direction = SENTIMENT_TO_DIRECTION.get(sentiment)
            if direction:
                sentiment_lookup[(date, session)] = direction
    # Count for diagnostics
    with_patterns = 0
    without_patterns = 0
    # Process each day in the pattern data
    for day in pattern_data:
        if 'patterns' not in day or 'sessions' not in day['patterns']:
            without_patterns += 1
            print(f"Warning: Skipping day without 'patterns' or 'sessions' key: {day.get('pattern_date', day)}")
            continue
        with_patterns += 1
        day_copy = dict(day)  # shallow copy
        sessions = day['patterns']['sessions']
        new_sessions = {}
        for session in ['morning', 'afternoon']:
            patterns = sessions.get(session, [])
            direction = sentiment_lookup.get((day['pattern_date'], session))
            if direction:
                filtered_patterns = [p for p in patterns if p['direction'].lower() == direction.lower()]
                if filtered_patterns:
                    new_sessions[session] = filtered_patterns
        if new_sessions:
            # Copy and update the sessions dict
            day_copy['patterns'] = dict(day['patterns'])
            day_copy['patterns']['sessions'] = new_sessions
            filtered_days.append(day_copy)
    print(f"Days with 'patterns': {with_patterns}")
    print(f"Days without 'patterns': {without_patterns}")
    return filtered_days

def main():
    args = parse_args()
    xlsx_path = os.path.join('data', 'padhu_sentiment_data.xlsx')
    pattern_file = PATTERN_FILE_MAP[args.pattern_level]
    sentiment_data = read_sentiment_data(xlsx_path)
    pattern_data = load_pattern_data(pattern_file)
    filtered_patterns = filter_patterns_by_sentiment(sentiment_data, pattern_data)
    # Output filtered patterns
    with open(args.output, 'w') as f:
        json.dump(filtered_patterns, f, indent=2)
    print(f"Filtered patterns saved to {args.output}")

if __name__ == '__main__':
    main()
