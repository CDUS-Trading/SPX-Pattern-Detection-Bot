import json
import os
import re
from datetime import datetime

TRANSLATORY_CONSTANTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'translatory_constants.json')

def load_timeframe_labels():
    with open(TRANSLATORY_CONSTANTS_PATH, 'r') as f:
        constants = json.load(f)
    return constants.get('timeframe_labels', {})

def format_direction(direction):
    if direction == 'bullish':
        return 'Buy \U0001F4C8'
    elif direction == 'bearish':
        return 'Sell \U0001F4C9'
    else:
        return direction

def format_action_plans_to_txt(patterns, output_path, analysis_day=None, based_on_day=None, based_on_date=None, close_price=None, top_n=3, show_all_slots=False, append=False):
    timeframe_labels = load_timeframe_labels()
    # Only keep top N patterns unless show_all_slots is True
    if not show_all_slots:
        patterns = patterns[:top_n]
    mode = 'a' if append else 'w'
    with open(output_path, mode) as f:
        # Write header if info is provided
        if analysis_day and based_on_day and based_on_date and close_price is not None:
            f.write(f"=== SPX Pattern Analysis for {analysis_day} ===\n")
            f.write(f"Based on data from {based_on_day}, {based_on_date}\n")
            f.write(f"Close from {based_on_day}: {close_price:.2f}\n\n")
        
        if show_all_slots:
            f.write(f"=== All Time Slots Action Plans ===\n\n")
        else:
            f.write(f"=== Top {top_n} Action Plans ===\n\n")

        # Sort patterns by entry time (parsed from label)
        def parse_entry_time(label):
            # Try to extract time from label (e.g., '8:30am CT - 9:30am CT')
            if '-' in label:
                entry = label.split('-')[0].strip()
            else:
                entry = label.strip()
            # Remove timezone
            entry = re.sub(r'\s*CT$', '', entry)
            # Try parsing as time
            try:
                # Accept both 8:30am and 08:30am
                t = datetime.strptime(entry, '%I:%M%p')
            except ValueError:
                try:
                    t = datetime.strptime(entry, '%I%p')
                except ValueError:
                    # If parsing fails, sort as last
                    t = datetime.strptime('11:59PM', '%I:%M%p')
            return t
        # Attach label to each pattern for sorting
        patterns_with_label = []
        for p in patterns:
            timeframe = p.get('timeframe', '')
            label = timeframe_labels.get(timeframe, timeframe)
            patterns_with_label.append((p, label))
        # Sort by parsed entry time
        patterns_with_label.sort(key=lambda x: parse_entry_time(x[1]))
        # Write sorted action plans
        for i, (p, label) in enumerate(patterns_with_label[:top_n], 1):
            if '-' in label:
                entry = label.split('-')[0].strip()
                exit_ = label.split('-')[1].strip()
            else:
                entry = label
                exit_ = label
            direction = format_direction(p.get('direction', ''))
            avg_move = p.get('favorable_avg_move', 0)
            median_move = p.get('favorable_median_move', 0)
            
            # Get counts for display
            fav_direction_count = p.get('fav_direction_count', 0)
            fav_direction_rate = p.get('fav_direction_rate', 0)
            tp_hit_count = p.get('tp_hit_count', 0)
            sl_hit_count = p.get('sl_hit_count', 0)
            sample_size = p.get('sample_size', 0)
            
            # Calculate percentages from counts
            fav_direction_rate_pct = (fav_direction_count / sample_size * 100) if sample_size > 0 else 0
            tp_hit_rate_pct = (tp_hit_count / sample_size * 100) if sample_size > 0 else 0
            sl_hit_rate_pct = (sl_hit_count / sample_size * 100) if sample_size > 0 else 0
            
            # New: Favorable/Adverse stats
            fav_mean = p.get('favorable_avg_move', 0)
            fav_median = p.get('favorable_median_move', 0)
            fav_std = p.get('favorable_std', 0)
            adv_mean = p.get('adverse_avg_move', 0)
            adv_median = p.get('adverse_median_move', 0)
            adv_std = p.get('adverse_std', 0)

            # All values are now in points
            avg_move_disp = avg_move
            median_move_disp = median_move
            move_unit = 'points'

            f.write(f"===== Action Plan {i} =====\n")
            f.write(f"Entry: {entry}\n")
            f.write(f"Exit: {exit_}\n")
            f.write(f"Direction: {direction}\n")
            f.write(f"Avg Move: {avg_move_disp:.2f} {move_unit}\n")
            f.write(f"Median Move: {median_move_disp:.2f} {move_unit}\n")
            f.write(f"Favorable Direction Rate: {fav_direction_count}/{sample_size} ({fav_direction_rate_pct:.0f}%)\n")
            f.write(f"TP Hit Rate: {tp_hit_count}/{sample_size} ({tp_hit_rate_pct:.0f}%)\n")
            f.write(f"SL Hit Rate: {sl_hit_count}/{sample_size} ({sl_hit_rate_pct:.0f}%)\n")
            f.write("------------------------------\n\n")

if __name__ == "__main__":
    input_file = 'current_detected_patterns.json'
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
    else:
        with open(input_file, 'r') as f:
            patterns = json.load(f)
        # Only keep top N patterns for both outputs
        top_n = 3
        top_patterns = patterns[:top_n]
        # Write JSON output
        with open('current_detected_patterns.json', 'w') as jf:
            json.dump(top_patterns, jf, indent=2)
        # Write TXT output
        format_action_plans_to_txt(top_patterns, 'current_detected_patterns.txt', top_n=top_n)
        print(f"Wrote top {top_n} formatted action plans to current_detected_patterns.txt and current_detected_patterns.json") 