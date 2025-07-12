import pandas as pd
import json
from collections import Counter, defaultdict
from datetime import datetime

# File paths
SENTIMENT_FILE = 'data/padhu_sentiment_data.xlsx'
PATTERN_FILE = 'logs/backtest_patterns_recursive.json'

# Mapping
TREND_TO_DIRECTION = {'Bullish': 'Buy', 'Bearish': 'Sell'}

# Helper to get weekday name from date string (YYYY-MM-DD)
def get_weekday(date_str):
    return pd.to_datetime(date_str).strftime('%A')

# Load sentiment data
def load_sentiment(sheet):
    df = pd.read_excel(SENTIMENT_FILE, sheet_name=sheet)
    df = df[['date', 'predicted_trend']].copy()
    df['weekday'] = df['date'].apply(get_weekday)
    return df

def load_patterns_df():
    with open(PATTERN_FILE, 'r') as f:
        patterns_data = json.load(f)
    pattern_records = []
    for day in patterns_data:
        pattern_date = day.get('pattern_date')
        for session, patterns in day.get('patterns', {}).get('sessions', {}).items():
            for pattern in patterns:
                pattern_record = {
                    'date': pattern_date,
                    'session': session,
                    'entry_time': pattern['entry_time'],
                    'exit_time': pattern['exit_time'],
                    'direction': pattern['direction'],
                    'target_points': pattern['target_points'],
                    'stop_loss_points': pattern['stop_loss_points'],
                    'success_rate': pattern['success_rate']
                }
                pattern_records.append(pattern_record)
    return pd.DataFrame(pattern_records)

def analyze_session(sentiment_df, patterns_df, session, entry_time):
    stats = {
        'total': 0,
        'match': 0,
        'mismatch': 0,
        'no_pattern': 0,
        'by_weekday': defaultdict(lambda: Counter({'match':0, 'mismatch':0, 'no_pattern':0, 'total':0})),
        'direction_breakdown': Counter(),
        'success_rates': [],
        'mismatch_details': [],
        'debug_matches': []
    }
    for _, row in sentiment_df.iterrows():
        weekday = row['weekday']
        trend = row['predicted_trend']
        direction = TREND_TO_DIRECTION.get(trend)
        sentiment_date = pd.to_datetime(row['date']).date()
        stats['total'] += 1
        stats['by_weekday'][weekday]['total'] += 1
        # Filter DataFrame for this date, session, and entry_time
        matched = patterns_df[
            (pd.to_datetime(patterns_df['date']).dt.date == sentiment_date) &
            (patterns_df['session'] == session) &
            (patterns_df['entry_time'] == entry_time)
        ]
        if matched.empty:
            stats['no_pattern'] += 1
            stats['by_weekday'][weekday]['no_pattern'] += 1
            stats['debug_matches'].append({'sentiment_date': str(sentiment_date), 'matched_pattern_date': None, 'reason': 'no_pattern'})
            continue
        found_match = False
        for _, matched_pattern in matched.iterrows():
            stats['direction_breakdown'][matched_pattern['direction']] += 1
            stats['success_rates'].append(matched_pattern['success_rate'])
            stats['debug_matches'].append({
                'sentiment_date': str(sentiment_date),
                'matched_pattern_date': matched_pattern['date'],
                'pattern_direction': matched_pattern['direction'],
                'predicted_trend': trend
            })
            if matched_pattern['direction'] == direction:
                found_match = True
        if found_match:
            stats['match'] += 1
            stats['by_weekday'][weekday]['match'] += 1
        else:
            stats['mismatch'] += 1
            stats['by_weekday'][weekday]['mismatch'] += 1
            stats['mismatch_details'].append({
                'date': row['date'],
                'weekday': weekday,
                'predicted_trend': trend,
                'expected_direction': direction,
                'pattern_directions': list(matched['direction']),
                'pattern_dates': list(matched['date'])
            })
    return stats

def print_stats(session_name, entry_time, stats):
    print(f"\n{'='*40}\nSession: {session_name} ({entry_time})\n{'='*40}")
    print(f"Total sentiment days analyzed: {stats['total']}")
    print(f"Days with no pattern for this session/time: {stats['no_pattern']} ({stats['no_pattern']/stats['total']*100:.2f}%)")
    print(f"Days with matching trend and pattern direction: {stats['match']} ({stats['match']/stats['total']*100:.2f}%)")
    print(f"Days with mismatched trend and pattern direction: {stats['mismatch']} ({stats['mismatch']/stats['total']*100:.2f}%)")
    print("\nBreakdown by weekday:")
    for wd, cnt in stats['by_weekday'].items():
        print(f"  {wd}: total={cnt['total']}, match={cnt['match']}, mismatch={cnt['mismatch']}, no_pattern={cnt['no_pattern']}")
    print("\nPattern direction breakdown (in matched patterns):")
    for d, c in stats['direction_breakdown'].items():
        print(f"  {d}: {c}")
    if stats['success_rates']:
        print(f"\nAverage pattern success rate: {sum(stats['success_rates'])/len(stats['success_rates']):.2f}%")
        print(f"Median pattern success rate: {pd.Series(stats['success_rates']).median():.2f}%")
    print("\nMismatched cases (first 10 shown):")
    for m in stats['mismatch_details'][:10]:
        print(f"  Date: {m['date']} ({m['weekday']}), Predicted: {m['predicted_trend']} (expected {m['expected_direction']}), Patterns: {m['pattern_directions']} (pattern dates: {m['pattern_dates']})")
    print(f"... ({len(stats['mismatch_details'])} total mismatches)")
    print("\nDebug match log (first 10 shown):")
    for dbg in stats['debug_matches'][:10]:
        print(dbg)
    print(f"... ({len(stats['debug_matches'])} total matches)")

def main():
    print("Loading sentiment and pattern data...")
    morning_df = load_sentiment('Morning_Session(8-12)')
    afternoon_df = load_sentiment('Afternoon_Session(12-3)')
    patterns_df = load_patterns_df()
    # Analyze morning session (8:31AM CT)
    morning_stats = analyze_session(morning_df, patterns_df, 'morning', '8:31AM CT')
    # Analyze afternoon session (12:30PM CT)
    afternoon_stats = analyze_session(afternoon_df, patterns_df, 'afternoon', '12:30PM CT')
    # Print stats
    print_stats('Morning', '8:31AM CT', morning_stats)
    print_stats('Afternoon', '12:30PM CT', afternoon_stats)

if __name__ == '__main__':
    main() 