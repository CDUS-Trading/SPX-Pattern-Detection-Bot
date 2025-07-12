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
    
    # Find all sections starting with the pattern analysis header
    sections = re.finditer(r'=== SPX Pattern Analysis for (.*?) ===\n(.*?)(?=== SPX Pattern Analysis for|$)', content, re.DOTALL)
    sections = list(sections)
    
    logger.debug(f"Found {len(sections)} day sections")
    
    result = []
    
    for i, section in enumerate(sections):
        logger.debug(f"\nProcessing section {i+1}:")
        
        # Extract the date from the header
        full_date = section.group(1).strip()
        pattern_day, pattern_date = parse_date_string(full_date)
        logger.debug(f"Found pattern day: {pattern_day}, date: {pattern_date}")
        
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
        
        # Extract based on date and close price
        header_match = re.search(r'Based on data from (.*?)\nClose from .*?: ([\d.]+)', section_content)
        if header_match:
            based_on_full = header_match.group(1)
            based_on_day, based_on_date = parse_date_string(based_on_full)
            day_data["based_on_day"] = based_on_day
            day_data["based_on_date"] = based_on_date
            day_data["close_price"] = float(header_match.group(2))
            logger.debug(f"Found based_on_day: {based_on_day}, based_on_date: {based_on_date}")
            logger.debug(f"Found close_price: {header_match.group(2)}")
        else:
            logger.warning(f"Could not find based_on_date and close_price in section {i+1}")
        
        # Extract filter level
        filter_match = re.search(r'Filter Level: (.*?)\n', section_content)
        if filter_match:
            day_data["filter_level"] = filter_match.group(1)
            logger.debug(f"Found filter_level: {filter_match.group(1)}")
        else:
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
                
            # Regular expression to match action plans
            action_plan_pattern = r'===== Action Plan =====\n' \
                                r'Entry: (.*?)\n' \
                                r'Exit: (.*?)\n' \
                                r'Direction: (.*?)\n' \
                                r'TP: ([\d.]+) points\n' \
                                r'SL: ([\d.]+) points\n' \
                                r'Success Rate: ([\d.]+)%'
            
            # Find all action plans in this session
            action_plans = list(re.finditer(action_plan_pattern, split))
            logger.debug(f"Found {len(action_plans)} action plans in {current_session} session")
            
            for plan in action_plans:
                pattern = {
                    "entry_time": plan.group(1),
                    "exit_time": plan.group(2),
                    "direction": remove_emojis(plan.group(3)),  # Remove emojis from direction
                    "target_points": float(plan.group(4)),
                    "stop_loss_points": float(plan.group(5)),
                    "success_rate": float(plan.group(6))
                }
                
                day_data["patterns"]["sessions"][current_session].append(pattern)
                logger.debug(f"Added pattern to {current_session} session")
        
        result.append(day_data)
    
    logger.debug(f"Final result has {len(result)} dates")
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

if __name__ == "__main__":
    main() 