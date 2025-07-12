#!/usr/bin/env python3

import pandas as pd
import pytz
from pathlib import Path
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_timezone(input_file: str, output_file: str) -> None:
    """
    Convert minute data from ET to CT timezone.
    
    Args:
        input_file (str): Path to input CSV file with ET data
        output_file (str): Path to save converted CT data
    """
    try:
        logger.info(f"Loading data from {input_file}")
        
        # First, read a few lines to determine the number of columns and if there are headers
        with open(input_file, 'r') as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
        
        num_columns = len(first_line.split(','))
        
        # Check if first line is a header by trying to parse it as datetime
        try:
            pd.to_datetime(first_line.split(',')[0])
            has_header = False
        except (ValueError, TypeError):
            has_header = True
        
        # Define column names based on the number of columns
        if num_columns == 5:
            column_names = ['datetime', 'open', 'high', 'low', 'close']
        elif num_columns == 6:
            column_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        else:
            raise ValueError(f"Unexpected number of columns: {num_columns}. Expected 5 or 6.")
        
        # Read the data with or without headers based on detection
        if has_header:
            df = pd.read_csv(input_file, header=0)
            # Ensure column names are correct
            df.columns = column_names
        else:
            df = pd.read_csv(input_file, header=None, names=column_names)
        
        # Convert datetime column to pandas datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Set timezone to ET
        et = pytz.timezone('US/Eastern')
        df['datetime'] = df['datetime'].dt.tz_localize(et)
        
        # Convert to CT with progress bar
        ct = pytz.timezone('US/Central')
        logger.info("Converting timezone from ET to CT...")
        
        # Create a new column for the converted timestamps
        converted_timestamps = []
        with tqdm(total=len(df), desc="Converting timestamps", unit="row") as pbar:
            for timestamp in df['datetime']:
                converted_timestamps.append(timestamp.astimezone(ct))
                pbar.update(1)
        
        # Update the datetime column with converted timestamps
        df['datetime'] = converted_timestamps
        
        # Remove timezone info for compatibility
        df['datetime'] = df['datetime'].dt.tz_localize(None)
        
        # Save to CSV with headers
        df.to_csv(output_file, header=True, index=False)
        logger.info(f"Successfully converted and saved data to {output_file}")
        
    except Exception as e:
        logger.error(f"Error converting timezone: {str(e)}")
        raise

def main():
    """
    Main execution function.
    """
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Set up file paths
    data_dir = project_root / 'data'
    
    # Files to convert
    files_to_convert = [
        ('SPX_full_1min.txt', 'SPX_full_1min_CT.txt'),
        ('SPX_week_1min.txt', 'SPX_week_1min_CT.txt')
    ]
    
    # Convert each file
    for input_file, output_file in files_to_convert:
        input_path = data_dir / input_file
        output_path = data_dir / output_file
        
        if input_path.exists():
            convert_timezone(input_path, output_path)
        else:
            logger.warning(f"Input file {input_path} does not exist, skipping")

if __name__ == "__main__":
    main()
