import json
from datetime import datetime
import argparse

def format_patterns(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Format the header
    output = []
    output.append(f"=== SPX Pattern Analysis for {data['pattern_day']}, {data['pattern_date']} ===")
    output.append(f"Based on data from {data['based_on_day']}, {data['based_on_date']}")
    output.append(f"Close from {data['based_on_day']}: {data['close_price']:.2f}")
    output.append("")
    output.append(f"Filter Level: {data['filter_level']}")
    output.append("")

    # Format each session
    sessions = {
        "morning": "MORNING SESSION PATTERNS",
        "mixed": "MIXED SESSION PATTERNS",
        "afternoon": "AFTERNOON SESSION PATTERNS"
    }

    for session_name, session_title in sessions.items():
        if session_name in data['patterns']['sessions']:
            output.append(f"{session_title}:")
            output.append("=" * 50)
            output.append("")

            for pattern in data['patterns']['sessions'][session_name]:
                output.append("===== Action Plan =====")
                output.append(f"Entry: {pattern['entry_time']}")
                output.append(f"Exit: {pattern['exit_time']}")
                
                # Add emoji based on direction
                direction_emoji = "📈" if pattern['direction'] == "Buy" else "📉"
                output.append(f"Direction: {pattern['direction']} {direction_emoji}")
                
                output.append(f"TP: {pattern['target_points']:.2f} points")
                output.append(f"SL: {pattern['stop_loss_points']:.2f} points")
                output.append(f"Success Rate: {pattern['success_rate']:.2f}%")
                output.append("-" * 30)
                output.append("")

    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description='Format SPX pattern analysis JSON into readable text')
    parser.add_argument('--input', '-i', 
                      default='logs/current_detected_patterns.json',
                      help='Path to the JSON pattern file (default: logs/current_detected_patterns.json)')
    
    args = parser.parse_args()
    formatted_text = format_patterns(args.input)
    print(formatted_text)

if __name__ == "__main__":
    main() 