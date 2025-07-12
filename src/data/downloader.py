#!/usr/bin/env python3

import argparse
import requests
import zipfile
import os
import logging
from pathlib import Path
from typing import Tuple, Optional
import sys
import pandas as pd
from datetime import datetime, time

# Add parent directory to Python path to import from src/utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.convert_timezone import convert_timezone

# Set up module-level logger
logger = logging.getLogger(__name__)

class DownloadError(Exception):
    """Custom exception for download-related errors"""
    pass

def setup_logging(level: int = logging.WARNING) -> None:
    """
    Configure logging for the module.
    
    Args:
        level (int): Logging level to use (default: logging.WARNING)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def download_file(url: str, output_path: str) -> bool:
    """
    Download a file from the given URL and save it to the specified path.
    
    Args:
        url (str): URL to download from
        output_path (str): Path to save the downloaded file
        
    Returns:
        bool: True if download was successful, False otherwise
        
    Raises:
        DownloadError: If there's an error during the download process
    """
    try:
        logger.info(f"Downloading file from {url}")
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, 'wb') as file:
                file.write(response.content)
            logger.info(f"Successfully downloaded file to {output_path}")
            return True
        else:
            error_msg = f"Failed to download file. Status code: {response.status_code}"
            logger.error(error_msg)
            raise DownloadError(error_msg)
    except Exception as e:
        error_msg = f"Error downloading file: {str(e)}"
        logger.error(error_msg)
        raise DownloadError(error_msg)

def extract_zip(zip_path: str, extract_path: str) -> Tuple[bool, str]:
    """
    Extract a ZIP file to the specified directory.
    
    Args:
        zip_path (str): Path to the ZIP file
        extract_path (str): Directory to extract to
        
    Returns:
        Tuple[bool, str]: (Success status, Actual filename from ZIP)
        
    Raises:
        DownloadError: If there's an error during the extraction process
    """
    try:
        logger.info(f"Extracting ZIP file from {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get the first file name from the ZIP
            file_list = zip_ref.namelist()
            if not file_list:
                raise DownloadError("ZIP file is empty")
            actual_filename = file_list[0]
            zip_ref.extractall(extract_path)
        logger.info(f"Successfully extracted files to {extract_path}")
        return True, actual_filename
    except Exception as e:
        error_msg = f"Error extracting ZIP file: {str(e)}"
        logger.error(error_msg)
        raise DownloadError(error_msg)

def get_data_urls(data_type: str, symbol: str = 'SPX') -> Tuple[str, str]:
    """
    Get the appropriate URL and filename based on the data type and symbol.
    
    Args:
        data_type (str): Type of data to download ('weekly' or 'full')
        symbol (str): Symbol to download ('SPX', 'MES', or 'VIX')
        
    Returns:
        Tuple[str, str]: URL and output filename
        
    Raises:
        ValueError: If data_type is not 'weekly' or 'full' or symbol is not 'SPX', 'MES', or 'VIX'
    """
    if symbol == 'SPX':
        if data_type == 'weekly':
            return (
                "https://firstratedata.com/api/data_file2/?userID=DHxFL3RKBEan2u1t6GztRA&fileUrlID=15791",
                "SPX_week_1min.txt"
            )
        elif data_type == 'full':
            return (
                "https://firstratedata.com/api/data_file2/?userID=DHxFL3RKBEan2u1t6GztRA&fileUrlID=15786",
                "SPX_full_1min.txt"
            )
    elif symbol == 'MES':
        if data_type == 'weekly':
            return (
                "https://firstratedata.com//api/data_file2/?userID=DHxFL3RKBEan2u1t6GztRA&fileUrlID=23390",
                "MES_week_1min.txt"
            )
        elif data_type == 'full':
            return (
                "https://firstratedata.com//api/data_file2/?userID=DHxFL3RKBEan2u1t6GztRA&fileUrlID=23389",
                "MES_full_1min_continuous_absolute_adjusted.txt"
            )
    elif symbol == 'VIX':
        # VIX data is daily, so data_type is ignored for VIX
        return (
            "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
            "VIX_daily.txt"
        )
    else:
        raise ValueError("symbol must be either 'SPX', 'MES', or 'VIX'")
    
    raise ValueError("data_type must be either 'weekly' or 'full'")

def split_mes_data_by_market_hours(input_file: str, output_dir: str) -> Tuple[str, str]:
    """
    Split MES data into market open (8:30 AM - 3:00 PM CT) and market close (3:00 PM - 12:00 AM CT) periods.
    
    Args:
        input_file (str): Path to the input MES data file
        output_dir (str): Directory to save the split files
        
    Returns:
        Tuple[str, str]: Paths to market open and market close files
        
    Raises:
        DownloadError: If there's an error during the splitting process
    """
    try:
        logger.info(f"Splitting MES data from {input_file}")
        
        # Read the MES data file with headers (6 columns: timestamp, open, high, low, close, volume)
        df = pd.read_csv(input_file, header=0, 
                        names=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        
        if df.empty:
            raise DownloadError("Input MES data file is empty")
        
        # Convert datetime column to datetime type
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Extract time component for filtering
        df['time'] = df['datetime'].dt.time
        
        # Define market hours
        market_open_start = time(8, 30)  # 8:30 AM
        market_open_end = time(15, 0)    # 3:00 PM
        market_close_start = time(15, 0) # 3:00 PM
        market_close_end = time(0, 0)    # 12:00 AM (midnight)
        
        # Split data into market open and market close periods
        market_open_mask = (df['time'] >= market_open_start) & (df['time'] <= market_open_end)
        market_close_mask = (df['time'] >= market_close_start) | (df['time'] < market_open_start)
        
        # Filter data
        market_open_df = df[market_open_mask].copy()
        market_close_df = df[market_close_mask].copy()
        
        # Remove the temporary time column
        market_open_df = market_open_df.drop('time', axis=1)
        market_close_df = market_close_df.drop('time', axis=1)
        
        # Generate output filenames
        base_name = Path(input_file).stem
        market_open_file = Path(output_dir) / f"{base_name}_market_open.txt"
        market_close_file = Path(output_dir) / f"{base_name}_market_close.txt"
        
        # Save the split files
        market_open_df.to_csv(market_open_file, header=True, index=False)
        market_close_df.to_csv(market_close_file, header=True, index=False)
        
        logger.info(f"Market open data saved to: {market_open_file}")
        logger.info(f"Market close data saved to: {market_close_file}")
        logger.info(f"Market open records: {len(market_open_df)}")
        logger.info(f"Market close records: {len(market_close_df)}")
        
        # Remove the original file
        os.remove(input_file)
        logger.info(f"Removed original file: {input_file}")
        
        return str(market_open_file), str(market_close_file)
        
    except Exception as e:
        error_msg = f"Error splitting MES data: {str(e)}"
        logger.error(error_msg)
        raise DownloadError(error_msg)

def download_and_extract_data(
    data_type: str,
    output_dir: str = 'data',
    cleanup: bool = True,
    symbol: str = 'SPX'
) -> Optional[str]:
    """
    Download and extract data from FirstRateData.
    
    Args:
        data_type (str): Type of data to download ('weekly' or 'full')
        output_dir (str): Directory to save downloaded files (default: 'data')
        cleanup (bool): Whether to clean up temporary files (default: True)
        symbol (str): Symbol to download ('SPX' or 'MES')
        
    Returns:
        Optional[str]: Path to the extracted data file if successful, None otherwise
        
    Raises:
        DownloadError: If there's an error during the download or extraction process
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get appropriate URL and filename
        url, expected_filename = get_data_urls(data_type, symbol)
        
        # Download the file
        temp_zip = output_path / "temp.zip"
        download_file(url, str(temp_zip))
        
        # Extract the ZIP file and get the actual filename
        success, actual_filename = extract_zip(str(temp_zip), str(output_path))
        
        # Clean up the temporary ZIP file if requested
        if cleanup:
            try:
                os.remove(temp_zip)
                logger.info("Cleaned up temporary ZIP file")
            except Exception as e:
                logger.warning(f"Could not remove temporary ZIP file: {e}")
        
        # Use the actual filename from the ZIP
        output_file = output_path / actual_filename
        logger.info(f"Successfully processed {actual_filename} to {output_path}")
        
        # Load and validate the data
        if symbol == 'MES':
            # MES files have 6 columns: timestamp, open, high, low, close, volume
            df = pd.read_csv(output_file, header=None, 
                           names=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        else:
            # SPX files have 5 columns: timestamp, open, high, low, close
            df = pd.read_csv(output_file, header=None, 
                           names=['datetime', 'open', 'high', 'low', 'close'])
        
        if df.empty:
            logger.error("Error: The data file is empty. Exiting.")
            return None
        
        # Add headers to the file for better usability
        df.to_csv(output_file, header=True, index=False)
        logger.info(f"Added headers to {actual_filename}")
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error in download_and_extract_data: {str(e)}")
        raise DownloadError(f"Failed to download and extract data: {str(e)}")

def download_vix_data(output_dir: str = 'data') -> Optional[str]:
    """
    Download VIX daily data from CBOE and convert to .txt format.
    
    Args:
        output_dir (str): Directory to save downloaded files (default: 'data')
        
    Returns:
        Optional[str]: Path to the processed VIX data file if successful, None otherwise
        
    Raises:
        DownloadError: If there's an error during the download or processing process
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get VIX URL and filename
        url, filename = get_data_urls('full', 'VIX')
        
        # Download the CSV file directly
        csv_file = output_path / "VIX_History.csv"
        download_file(url, str(csv_file))
        
        # Read the CSV file
        logger.info(f"Reading VIX data from {csv_file}")
        df = pd.read_csv(csv_file)
        
        if df.empty:
            raise DownloadError("VIX data file is empty")
        
        # Convert DATE column to datetime
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        # Sort by date to ensure chronological order
        df = df.sort_values('DATE')
        
        # Convert to the same format as other data files (datetime, open, high, low, close)
        # Format datetime as string in the same format as other files
        df['datetime'] = df['DATE'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Select and reorder columns to match the expected format
        output_df = df[['datetime', 'OPEN', 'HIGH', 'LOW', 'CLOSE']].copy()
        
        # Rename columns to match the expected format
        output_df.columns = ['datetime', 'open', 'high', 'low', 'close']
        
        # Save as .txt file with header
        txt_file = output_path / filename
        output_df.to_csv(txt_file, header=True, index=False)
        
        # Clean up the temporary CSV file
        try:
            os.remove(csv_file)
            logger.info("Cleaned up temporary CSV file")
        except Exception as e:
            logger.warning(f"Could not remove temporary CSV file: {e}")
        
        logger.info(f"Successfully processed VIX data to {txt_file}")
        logger.info(f"VIX data records: {len(output_df)}")
        logger.info(f"Date range: {output_df['datetime'].iloc[0]} to {output_df['datetime'].iloc[-1]}")
        
        return str(txt_file)
        
    except Exception as e:
        logger.error(f"Error in download_vix_data: {str(e)}")
        raise DownloadError(f"Failed to download and process VIX data: {str(e)}")

def main():
    """Main function for standalone execution"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Download and extract SPX, MES, and VIX data from various sources')
    parser.add_argument('--output-dir', type=str, default='data', help='Directory to save downloaded files (default: data/)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    try:
        # Download SPX data
        logger.info("Downloading SPX data...")
        for data_type in ['full']:  # Only full data
            logger.info(f"Downloading {data_type} SPX data...")
            spx_file = download_and_extract_data(data_type, args.output_dir, symbol='SPX')
            if spx_file:
                logger.info(f"{data_type.capitalize()} SPX data successfully processed to: {spx_file}")
                # Convert to CT
                spx_ct_file = spx_file.replace('.txt', '_CT.txt')
                convert_timezone(spx_file, spx_ct_file)
        
        # Download MES data
        logger.info("Downloading MES data...")
        for data_type in ['full']:  # Only full data
            logger.info(f"Downloading {data_type} MES data...")
            mes_file = download_and_extract_data(data_type, args.output_dir, symbol='MES')
            if mes_file:
                logger.info(f"{data_type.capitalize()} MES data successfully processed to: {mes_file}")
                # Convert to CT
                mes_ct_file = mes_file.replace('.txt', '_CT.txt')
                convert_timezone(mes_file, mes_ct_file)
                
                # Split MES data into market open and market close periods
                logger.info("Splitting MES data into market open and market close periods...")
                market_open_file, market_close_file = split_mes_data_by_market_hours(mes_ct_file, args.output_dir)
                logger.info(f"MES data split completed. Market open: {market_open_file}, Market close: {market_close_file}")
                # Remove the original MES files (both .txt and _CT.txt)
                try:
                    if os.path.exists(mes_file):
                        os.remove(mes_file)
                        logger.info(f"Removed intermediate MES file: {mes_file}")
                    if os.path.exists(mes_ct_file):
                        os.remove(mes_ct_file)
                        logger.info(f"Removed intermediate MES file: {mes_ct_file}")
                except Exception as e:
                    logger.warning(f"Could not remove intermediate MES files: {e}")
        
        # Download VIX data
        logger.info("Downloading VIX data...")
        vix_file = download_vix_data(args.output_dir)
        if vix_file:
            logger.info(f"VIX data successfully processed to: {vix_file}")
            
    except DownloadError as e:
        logger.error(f"Failed to process data: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 