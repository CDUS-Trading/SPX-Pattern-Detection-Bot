import pandas as pd
from datetime import datetime
import csv

# Thresholds for reversal-continuation analysis
ADVERSE_MOVE_THRESHOLD = 18  # Points of adverse move to trigger analysis
STRONG_CONT_THRESHOLD = 27   # Points in SL direction to classify as strong continuation

def parse_ct_time(timestr):
    """Helper to parse time like '8:30AM CT' to datetime.time"""
    timestr = timestr.replace(' CT', '')
    return datetime.strptime(timestr, '%I:%M%p').time()

def analyze_reversal_continuation(results, minute_df):
    """
    Analyze trades for reversal-continuation patterns.
    
    Args:
        results: List of trade results from P&L simulation
        minute_df: DataFrame containing minute-by-minute price data
        
    Returns:
        List of results with reversal-continuation analysis added
    """
    rc_results = []
    for row in results:
        if row['exit_type'] == 'TP':
            row['adverse_move'] = ''
            row['retracement'] = ''
            row['continuation'] = ''
            row['full_retrace'] = ''
            row['partial_retrace'] = ''
            row['strong_continuation'] = ''
            row['moderate_continuation'] = ''
            rc_results.append(row)
            continue
            
        entry_time = parse_ct_time(row['entry_time'])
        exit_time = parse_ct_time(row['exit_time'])
        date = row['date']
        direction = row['direction']
        
        day_df = minute_df[minute_df['date'] == date]
        if day_df.empty:
            row['adverse_move'] = ''
            row['retracement'] = ''
            row['continuation'] = ''
            row['full_retrace'] = ''
            row['partial_retrace'] = ''
            row['strong_continuation'] = ''
            row['moderate_continuation'] = ''
            rc_results.append(row)
            continue
            
        entry_row = day_df[day_df['time'] >= entry_time].head(1)
        if entry_row.empty:
            row['adverse_move'] = ''
            row['retracement'] = ''
            row['continuation'] = ''
            row['full_retrace'] = ''
            row['partial_retrace'] = ''
            row['strong_continuation'] = ''
            row['moderate_continuation'] = ''
            rc_results.append(row)
            continue
            
        entry_idx = entry_row.index[0]
        entry_price = entry_row.iloc[0]['open']
        trade_df = day_df[(day_df.index >= entry_idx) & (day_df['time'] <= exit_time)]
        
        if trade_df.empty:
            row['adverse_move'] = ''
            row['retracement'] = ''
            row['continuation'] = ''
            row['full_retrace'] = ''
            row['partial_retrace'] = ''
            row['strong_continuation'] = ''
            row['moderate_continuation'] = ''
            rc_results.append(row)
            continue
            
        # Find if adverse move occurs
        adverse_move_found = False
        for idx, r in trade_df.iterrows():
            if direction == 'Buy':
                if r['low'] <= entry_price - ADVERSE_MOVE_THRESHOLD:
                    adverse_move_found = True
                    break
            else:  # Sell
                if r['high'] >= entry_price + ADVERSE_MOVE_THRESHOLD:
                    adverse_move_found = True
                    break
                    
        row['adverse_move'] = adverse_move_found
        row['retracement'] = ''
        row['continuation'] = ''
        row['full_retrace'] = ''
        row['partial_retrace'] = ''
        row['strong_continuation'] = ''
        row['moderate_continuation'] = ''
        
        if not adverse_move_found:
            rc_results.append(row)
            continue
            
        # At original exit time, check where price is relative to entry
        exit_row = trade_df.iloc[-1]
        exit_price = exit_row['close']
        
        if direction == 'Buy':
            diff = entry_price - exit_price
        else:  # Sell
            diff = exit_price - entry_price
            
        # Categorize
        if diff < ADVERSE_MOVE_THRESHOLD:  # Price moved back towards entry
            row['retracement'] = True
            row['continuation'] = False
            # Check if full or partial retrace
            if (direction == 'Buy' and exit_price >= entry_price) or (direction == 'Sell' and exit_price <= entry_price):
                row['full_retrace'] = True
                row['partial_retrace'] = False
            else:
                row['full_retrace'] = False
                row['partial_retrace'] = True
        else:  # Price continued past adverse move threshold in SL direction
            row['retracement'] = False
            row['continuation'] = True
            # Check if strong or moderate continuation
            if diff >= STRONG_CONT_THRESHOLD:
                row['strong_continuation'] = True
                row['moderate_continuation'] = False
            else:
                row['strong_continuation'] = False
                row['moderate_continuation'] = True
                
        rc_results.append(row)
    return rc_results

def print_analysis_summary(results):
    """Print a summary of the reversal-continuation analysis results."""
    rc_trades = [row for row in results if row.get('adverse_move')]
    total_rc = len(rc_trades)
    retrace = sum(1 for r in rc_trades if r.get('retracement'))
    cont = sum(1 for r in rc_trades if r.get('continuation'))
    full_retrace = sum(1 for r in rc_trades if r.get('full_retrace'))
    partial_retrace = sum(1 for r in rc_trades if r.get('partial_retrace'))
    strong_cont = sum(1 for r in rc_trades if r.get('strong_continuation'))
    moderate_cont = sum(1 for r in rc_trades if r.get('moderate_continuation'))
    
    print("\nReversal-Continuation Analysis:")
    print(f"Of {total_rc} trades with {ADVERSE_MOVE_THRESHOLD}pt adverse move:")
    print(f"  - {retrace} times, market retraced back towards entry at exit (hold = best)")
    print(f"    * {full_retrace} fully retraced to entry or better")
    print(f"    * {partial_retrace} partially retraced but not to entry")
    print(f"  - {cont} times, market continued past {ADVERSE_MOVE_THRESHOLD}pts in SL direction at exit (reverse = best)")
    print(f"    * {strong_cont} continued past {STRONG_CONT_THRESHOLD}pts in SL direction")
    print(f"    * {moderate_cont} continued {ADVERSE_MOVE_THRESHOLD}-{STRONG_CONT_THRESHOLD}pts in SL direction")
    
    if total_rc > 0:
        retrace_pct = retrace / total_rc * 100
        cont_pct = cont / total_rc * 100
        print("\nConclusion:")
        if retrace_pct > cont_pct:
            print(f"Most often, holding was best: {retrace}/{total_rc} ({retrace_pct:.1f}%) of trades retraced back towards entry at exit.")
        else:
            print(f"Most often, reversing was best: {cont}/{total_rc} ({cont_pct:.1f}%) of trades continued past {ADVERSE_MOVE_THRESHOLD}pts in SL direction at exit.")

def write_analysis_to_csv(results, output_file='output/reversal_continuation_analysis.csv'):
    """Write reversal-continuation analysis results to CSV."""
    rc_trades = [row for row in results if row.get('adverse_move')]
    rc_fieldnames = [
        'date', 'signal_num', 'entry_time', 'exit_time', 'direction',
        'success_rate', 'pnl', 'exit_type', 'filter_level',
        'adverse_move', 'retracement', 'continuation',
        'full_retrace', 'partial_retrace', 'strong_continuation', 'moderate_continuation'
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rc_fieldnames)
        writer.writeheader()
        for row in rc_trades:
            writer.writerow({k: row.get(k, '') for k in rc_fieldnames})
    print(f"Wrote {len(rc_trades)} rows to {output_file}") 