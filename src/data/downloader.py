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

def extract_zip(zip_path: str, extract_path: str) -> bool:
    """
    Extract a ZIP file to the specified directory.
    
    Args:
        zip_path (str): Path to the ZIP file
        extract_path (str): Directory to extract to
        
    Returns:
        bool: True if extraction was successful, False otherwise
        
    Raises:
        DownloadError: If there's an error during the extraction process
    """
    try:
        logger.info(f"Extracting ZIP file from {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        logger.info(f"Successfully extracted files to {extract_path}")
        return True
    except Exception as e:
        error_msg = f"Error extracting ZIP file: {str(e)}"
        logger.error(error_msg)
        raise DownloadError(error_msg)

def get_data_urls(data_type: str) -> Tuple[str, str]:
    """
    Get the appropriate URL and filename based on the data type.
    
    Args:
        data_type (str): Type of data to download ('weekly' or 'full')
        
    Returns:
        Tuple[str, str]: URL and output filename
        
    Raises:
        ValueError: If data_type is not 'weekly' or 'full'
    """
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
    else:
        raise ValueError("data_type must be either 'weekly' or 'full'")

def download_and_extract_data(
    data_type: str,
    output_dir: str = 'data',
    cleanup: bool = True
) -> Optional[str]:
    """
    Download and extract SPX data from FirstRateData.
    
    Args:
        data_type (str): Type of data to download ('weekly' or 'full')
        output_dir (str): Directory to save downloaded files (default: 'data')
        cleanup (bool): Whether to clean up temporary files (default: True)
        
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
        url, output_filename = get_data_urls(data_type)
        
        # Download the file
        temp_zip = output_path / "temp.zip"
        download_file(url, str(temp_zip))
        
        # Extract the ZIP file
        extract_zip(str(temp_zip), str(output_path))
        
        # Clean up the temporary ZIP file if requested
        if cleanup:
            try:
                os.remove(temp_zip)
                logger.info("Cleaned up temporary ZIP file")
            except Exception as e:
                logger.warning(f"Could not remove temporary ZIP file: {e}")
        
        output_file = output_path / output_filename
        logger.info(f"Successfully processed {output_filename} to {output_path}")
        
        # After loading the weekly data
        df = pd.read_csv(output_file, header=None, names=['datetime', 'open', 'high', 'low', 'close'])
        if df.empty:
            logger.error("Error: The weekly data file is empty. Exiting.")
            return None
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error in download_and_extract_data: {str(e)}")
        raise DownloadError(f"Failed to download and extract data: {str(e)}")

def main():
    """Main function for standalone execution"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Download and extract SPX data from FirstRateData')
    parser.add_argument('--output-dir', type=str, default='data', help='Directory to save downloaded files (default: data/)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    try:
        # Download weekly data
        logger.info("Downloading weekly SPX data...")
        weekly_file = download_and_extract_data('weekly', args.output_dir)
        if weekly_file:
            logger.info(f"Weekly data successfully processed to: {weekly_file}")
            # Convert weekly data to CT
            weekly_ct_file = weekly_file.replace('.txt', '_CT.txt')
            convert_timezone(weekly_file, weekly_ct_file)
        
        # Download full data
        logger.info("Downloading full SPX data...")
        full_file = download_and_extract_data('full', args.output_dir)
        if full_file:
            logger.info(f"Full data successfully processed to: {full_file}")
            # Convert full data to CT
            full_ct_file = full_file.replace('.txt', '_CT.txt')
            convert_timezone(full_file, full_ct_file)
            
    except DownloadError as e:
        logger.error(f"Failed to process data: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 