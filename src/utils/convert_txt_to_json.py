#!/usr/bin/env python3

import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def remove_emojis(text: str) -> str:
    """
    Remove emojis from text.
    
    Args:
        text (str): Text that may contain emojis
        
    Returns:
        str: Text with emojis removed
    """
    # Remove emojis and any extra whitespace
    return re.sub(r'[^\w\s]', '', text).strip()

def parse_date_string(date_str: str) -> Tuple[str, str]:
    """
    Parse a date string like "Tuesday, 2025-04-15" into day and date.
    
    Args:
        date_str (str): Date string in format "Day, YYYY-MM-DD"
        
    Returns:
        Tuple[str, str]: Tuple of (day, date)
    """
    try:
        # Split on comma and strip whitespace
        day, date = [part.strip() for part in date_str.split(',')]
        return day, date
    except Exception as e:
        logger.error(f"Error parsing date string '{date_str}': {str(e)}")
        return None, None

def parse_pattern_file(file_path: str) -> List[Dict]:
    """
    Parse a pattern text file and convert it to a list of structured dictionaries.
    
    Args:
        file_path (str): Path to the pattern text file
        
    Returns:
        List[Dict]: List of dictionaries containing the pattern data
    """
    logger.debug(f"Reading file: {file_path}")
    with open(file_path, 'r') as f:
        content = f.read()
    
    logger.debug(f"File content length: {len(content)} characters")
    
    # More robust section splitting: allow for extra whitespace and missing newlines
    sections = re.finditer(r'=== SPX Pattern Analysis for (.*?) ===\s*(.*?)(?=\n=== SPX Pattern Analysis for|\Z)', content, re.DOTALL)
    sections = list(sections)
    
    logger.debug(f"Found {len(sections)} day sections")
    print(f"[DEBUG] Found {len(sections)} pattern sections in TXT file.")
    
    result = []
    
    for i, section in enumerate(sections):
        print(f"\n[DEBUG] Processing section {i+1} header: {section.group(1).strip()}")
        
        # Extract the date from the header
        full_date = section.group(1).strip()
        pattern_day, pattern_date = parse_date_string(full_date)
        print(f"[DEBUG] Section {i+1} pattern_day: {pattern_day}, pattern_date: {pattern_date}")
        
        # Get the content for this section
        section_content = section.group(2)
        
        # Initialize the structure for this date
        day_data = {
            "pattern_day": pattern_day,
            "pattern_date": pattern_date,
            "based_on_day": None,
            "based_on_date": None,
            "close_price": None,
            "filter_level": None,
            "patterns": {
                "sessions": {
                    "morning": [],
                    "mixed": [],
                    "afternoon": []
                }
            }
        }
        
        # More robust extraction for based_on and close_price
        based_on_match = re.search(r'Based on data from\s*(.*?)\s*\n', section_content)
        close_price_match = re.search(r'Close from .*?:\s*([\d.]+)', section_content)
        if based_on_match:
            based_on_full = based_on_match.group(1)
            based_on_day, based_on_date = parse_date_string(based_on_full)
            day_data["based_on_day"] = based_on_day
            day_data["based_on_date"] = based_on_date
        else:
            print(f"[DEBUG] Section {i+1} missing based_on_date.")
            logger.warning(f"Could not find based_on_date in section {i+1}")
        if close_price_match:
            try:
                day_data["close_price"] = float(close_price_match.group(1))
            except Exception as e:
                print(f"[DEBUG] Section {i+1} could not parse close_price: {e}")
                logger.warning(f"Could not parse close_price in section {i+1}: {e}")
        else:
            print(f"[DEBUG] Section {i+1} missing close_price.")
            logger.warning(f"Could not find close_price in section {i+1}")
        
        # Extract filter level
        filter_match = re.search(r'Filter Level: (.*?)\s*\n', section_content)
        if filter_match:
            day_data["filter_level"] = filter_match.group(1)
            print(f"[DEBUG] Section {i+1} filter_level: {filter_match.group(1)}")
        else:
            print(f"[DEBUG] Section {i+1} missing filter_level.")
            logger.warning(f"Could not find filter_level in section {i+1}")
        
        # Split content into sessions
        session_splits = re.split(r'(MORNING|MIXED|AFTERNOON) SESSION PATTERNS:', section_content)
        
        # Process each session's patterns
        current_session = None
        for split in session_splits:
            if split.strip() in ['MORNING', 'MIXED', 'AFTERNOON']:
                current_session = split.strip().lower()
                continue
            elif current_session is None:
                continue
                
            # More robust action plan regex: allow for extra whitespace
            action_plan_pattern = r'===== Action Plan =====\s*' \
                                r'Entry: (.*?)\s*\n' \
                                r'Exit: (.*?)\s*\n' \
                                r'Direction: (.*?)\s*\n' \
                                r'TP: ([\d.]+) points\s*\n' \
                                r'SL: ([\d.]+) points\s*\n' \
                                r'Success Rate: ([\d.]+)%'
            
            action_plans = list(re.finditer(action_plan_pattern, split))
            print(f"[DEBUG] Section {i+1} found {len(action_plans)} action plans in {current_session} session.")
            
            for plan in action_plans:
                pattern = {
                    "entry_time": plan.group(1).strip(),
                    "exit_time": plan.group(2).strip(),
                    "direction": remove_emojis(plan.group(3)),  # Remove emojis from direction
                    "target_points": float(plan.group(4)),
                    "stop_loss_points": float(plan.group(5)),
                    "success_rate": float(plan.group(6))
                }
                
                day_data["patterns"]["sessions"][current_session].append(pattern)
                print(f"[DEBUG] Section {i+1} added pattern to {current_session} session.")
        
        print(f"[DEBUG] Section {i+1} added to result with pattern_date: {pattern_date}")
        result.append(day_data)
    
    logger.debug(f"Final result has {len(result)} dates")
    print(f"[DEBUG] Final result has {len(result)} pattern objects.")
    return result

def convert_to_json(input_file: str, output_file: Optional[str] = None) -> None:
    """
    Convert a pattern text file to JSON format.
    
    Args:
        input_file (str): Path to the input text file
        output_file (Optional[str]): Path to the output JSON file. If not provided,
                                   will use the same name as input file with .json extension
    """
    try:
        # Parse the pattern file
        pattern_data = parse_pattern_file(input_file)
        
        # Generate output filename if not provided
        if output_file is None:
            output_file = os.path.splitext(input_file)[0] + '.json'
        
        # Write to JSON file
        with open(output_file, 'w') as f:
            json.dump(pattern_data, f, indent=2)
        
        logger.info(f"Successfully converted {input_file} to {output_file}")
        
    except Exception as e:
        logger.error(f"Error converting file: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Convert pattern text files to JSON format')
    parser.add_argument('input_file', help='Path to the input text file')
    parser.add_argument('--output', '-o', help='Path to the output JSON file (optional)')
    
    args = parser.parse_args()
    
    convert_to_json(args.input_file, args.output)

def compare_txt_json_patterns(txt_file: str, json_file: str) -> None:
    """
    Compare the number of pattern sections in the txt file with the number of pattern objects in the json file.
    Print any mismatches in pattern dates.
    """
    # Read txt file and extract all pattern dates
    with open(txt_file, 'r') as f:
        txt_content = f.read()
    txt_pattern_headers = re.findall(r'=== SPX Pattern Analysis for (.*?) ===', txt_content)
    txt_pattern_dates = []
    for header in txt_pattern_headers:
        # header is like 'Tuesday, 2025-04-15'
        parts = header.split(',')
        if len(parts) == 2:
            txt_pattern_dates.append(parts[1].strip())
        else:
            txt_pattern_dates.append(header.strip())

    # Read json file and extract all pattern_date fields
    with open(json_file, 'r') as f:
        try:
            json_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return
    json_pattern_dates = [item.get('pattern_date') for item in json_data if isinstance(item, dict)]

    print(f"TXT pattern count: {len(txt_pattern_dates)}")
    print(f"JSON pattern count: {len(json_pattern_dates)}")

    # Find missing in JSON
    missing_in_json = [d for d in txt_pattern_dates if d not in json_pattern_dates]
    if missing_in_json:
        print("Pattern dates in TXT but missing in JSON:")
        for d in missing_in_json:
            print(f"  {d}")
    else:
        print("All TXT pattern dates are present in JSON.")

    # Find extra in JSON
    missing_in_txt = [d for d in json_pattern_dates if d not in txt_pattern_dates]
    if missing_in_txt:
        print("Pattern dates in JSON but missing in TXT:")
        for d in missing_in_txt:
            print(f"  {d}")
    else:
        print("All JSON pattern dates are present in TXT.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == '--compare':
        compare_txt_json_patterns(sys.argv[2], sys.argv[3])
    else:
        main() 