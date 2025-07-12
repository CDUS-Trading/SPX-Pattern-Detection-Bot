#!/usr/bin/env python3

import json
import os
import logging
from typing import Dict, Optional, List, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_json_to_txt(json_data: Union[Dict, List[Dict]], output_file: Optional[str] = None, show_dates_list: bool = False) -> str:
    """
    Convert pattern JSON data to a readable TXT format.
    Handles both single pattern analysis and backtest results.
    
    Args:
        json_data (Union[Dict, List[Dict]]): Pattern data in JSON format. Can be either:
            - A single pattern analysis dict
            - A list of pattern results from backtest
            - A backtest info dict with metadata and results
        output_file (Optional[str]): Path to save the TXT output. If not provided,
                                   will return the text content as a string.
        show_dates_list (bool): Whether to show historical dates in the output
    
    Returns:
        str: The text content if output_file is not provided, otherwise the path to the output file
    """
    try:
        text_content = ""
        
        # Handle backtest results
        if isinstance(json_data, dict) and 'metadata' in json_data and 'results' in json_data:
            # Write backtest metadata
            text_content += f"Backtest Results - {json_data['metadata']['timestamp']}\n"
            text_content += f"Date Range: {json_data['metadata']['start_date']} to {json_data['metadata']['end_date']}\n"
            text_content += f"Filter Level: {json_data['metadata']['filter_level']}\n"
            text_content += f"Total Trading Days: {json_data['metadata']['total_trading_days']}\n"
            text_content += "=" * 80 + "\n\n"
            
            # Process each result
            for result in json_data['results']:
                if 'status' in result:  # Error or no-data result
                    text_content += f"Date: {result['date']}\n"
                    text_content += f"Status: {result['status']}\n"
                    text_content += f"Message: {result['message']}\n"
                    if 'traceback' in result:
                        text_content += f"Traceback:\n{result['traceback']}\n"
                    text_content += "-" * 50 + "\n\n"
                else:
                    # Pattern result
                    text_content += write_pattern_section(result, show_dates_list)
        
        # Handle single pattern analysis
        elif isinstance(json_data, dict):
            text_content = write_pattern_section(json_data, show_dates_list)
        
        # Handle list of pattern results
        elif isinstance(json_data, list):
            for result in json_data:
                if isinstance(result, dict):
                    if 'status' in result:  # Error or no-data result
                        text_content += f"Date: {result['date']}\n"
                        text_content += f"Status: {result['status']}\n"
                        text_content += f"Message: {result['message']}\n"
                        text_content += "-" * 50 + "\n\n"
                    else:
                        text_content += write_pattern_section(result, show_dates_list)
        
        if output_file:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write(text_content)
            logger.info(f"Successfully wrote pattern analysis to {output_file}")
            return output_file
        else:
            return text_content
            
    except Exception as e:
        error_msg = f"Error converting JSON to TXT: {str(e)}"
        logger.error(error_msg)
        raise

def write_pattern_section(pattern_data: Dict, show_dates_list: bool = False) -> str:
    """
    Write a pattern section to the output text.
    
    Args:
        pattern_data (Dict): Pattern data to write
        show_dates_list (bool): Whether to show historical dates
        
    Returns:
        str: Formatted text content for the pattern
    """
    text_content = f"=== SPX Pattern Analysis for {pattern_data['pattern_day']}, {pattern_data['pattern_date']} ===\n"
    text_content += f"Based on data from {pattern_data['based_on_day']}, {pattern_data['based_on_date']}\n"
    text_content += f"Close from {pattern_data['based_on_day']}: {pattern_data['close_price']}\n\n"
    text_content += f"Filter Level: {pattern_data['filter_level']}\n\n"
    
    for session in ['morning', 'mixed', 'afternoon']:
        patterns = pattern_data['patterns']['sessions'][session]
        if patterns:
            text_content += f"{session.upper()} SESSION PATTERNS:\n"
            text_content += f"{'=' * 50}\n\n"
            
            for pattern in patterns:
                text_content += f"===== Action Plan =====\n"
                text_content += f"Entry: {pattern['entry_time']}\n"
                text_content += f"Exit: {pattern['exit_time']}\n"
                text_content += f"Direction: {pattern['direction']} {'📈' if pattern['direction'] == 'Buy' else '📉'}\n"
                text_content += f"TP: {pattern['target_points']} points\n"
                text_content += f"SL: {pattern['stop_loss_points']} points\n"
                text_content += f"Success Rate: {pattern['success_rate']}%\n"
                
                # Add historical dates if present and flag is set
                if show_dates_list and pattern.get('historical_dates'):
                    text_content += f"\nHistorical Dates: [ "
                    text_content += ", ".join(pattern['historical_dates'])
                    text_content += f" ]\n"
                
                text_content += f"{'-' * 30}\n\n"
    
    return text_content

def main():
    """
    Command line interface for converting JSON pattern files to TXT format.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert pattern JSON files to TXT format')
    parser.add_argument('input_file', help='Path to the input JSON file')
    parser.add_argument('--output', '-o', help='Path to the output TXT file (optional)')
    parser.add_argument('--show-dates-list', '-sdl', action='store_true',
                      help='Show historical dates for each pattern as a side-by-side list in the TXT output')
    
    args = parser.parse_args()
    
    try:
        # Read input JSON file
        with open(args.input_file, 'r') as f:
            json_data = json.load(f)
        
        # Generate output filename if not provided
        if args.output is None:
            output_file = os.path.splitext(args.input_file)[0] + '.txt'
        else:
            output_file = args.output
        
        # Convert and save
        convert_json_to_txt(json_data, output_file, args.show_dates_list)
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise

if __name__ == "__main__":
    main() 