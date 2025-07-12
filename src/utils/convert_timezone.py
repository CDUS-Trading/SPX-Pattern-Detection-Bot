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
        
        # Read the data without headers and assign column names
        df = pd.read_csv(input_file, header=None, 
                        names=['datetime', 'open', 'high', 'low', 'close'])
        
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
        
        # Save to CSV
        df.to_csv(output_file, header=False, index=False)
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