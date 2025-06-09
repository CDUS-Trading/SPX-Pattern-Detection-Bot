"""
Pattern Detector
--------------
Main class for detecting market patterns in minute data.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import os
import glob

from ..utils import metrics, dates, io
from . import filter

logger = logging.getLogger(__name__)

class PatternDetector:
    """
    A class for detecting market patterns and generating trading strategies.
    
    Attributes:
        patterns (List[Dict]): List of pattern definitions loaded from the database
    """
    
    def __init__(self, pattern_database_path: Optional[str] = None) -> None:
        """
        Initialize the pattern detector with a database of known patterns.
        
        Args:
            pattern_database_path (Optional[str]): Path to the pattern database file
            
        Raises:
            PatternDatabaseError: If no valid pattern database is found
        """
        self.patterns: List[Dict] = []
        
        try:
            if pattern_database_path and os.path.exists(pattern_database_path):
                self.load_patterns(pattern_database_path)
            else:
                # Default pattern database path
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
                processed_dir = os.path.join(project_root, 'data', 'processed')
                default_path = os.path.join(processed_dir, 'master_pattern_database.json')
                # Find the latest versioned file if master_pattern_database.json does not exist
                if os.path.exists(default_path):
                    logger.debug(f"Looking for pattern database at: {default_path}")
                    self.load_patterns(default_path)
                else:
                    # Find all versioned files
                    versioned_files = glob.glob(os.path.join(processed_dir, 'master_pattern_database_v*_*.json'))
                    if versioned_files:
                        # Sort by version number and timestamp
                        def extract_version_and_time(f):
                            import re
                            base = os.path.basename(f)
                            m = re.match(r"master_pattern_database_v(\d+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.json", base)
                            if m:
                                v = int(m.group(1))
                                t = m.group(2)
                                return (v, t)
                            return (0, '')
                        versioned_files.sort(key=extract_version_and_time, reverse=True)
                        latest_file = versioned_files[0]
                        logger.info(f"No master_pattern_database.json found, loading latest versioned file: {latest_file}")
                        self.load_patterns(latest_file)
                    else:
                        error_msg = f"No pattern database found in {processed_dir}. Please provide a valid pattern database file."
                        logger.error(error_msg)
                        raise PatternDatabaseError(error_msg)
                
            logger.info(f"PatternDetector initialized with {len(self.patterns)} patterns")
            
        except Exception as e:
            error_msg = f"Error initializing PatternDetector: {str(e)}"
            logger.error(error_msg)
            raise PatternDatabaseError(error_msg)
    
    def load_patterns(self, file_path: str) -> None:
        """
        Load patterns from a JSON file.
        
        Args:
            file_path (str): Path to the pattern database JSON file
            
        Raises:
            PatternDatabaseError: If there's an error loading the pattern database
        """
        try:
            self.patterns = io.load_patterns(file_path)
        except ValueError as e:
            raise PatternDatabaseError(str(e))
    
    def save_patterns(self, file_path: str) -> None:
        """
        Save patterns to a JSON file.
        
        Args:
            file_path (str): Path to save the pattern database
            
        Raises:
            PatternDatabaseError: If there's an error saving the pattern database
        """
        try:
            io.save_patterns(self.patterns, file_path)
        except ValueError as e:
            raise PatternDatabaseError(str(e))
    
    def detect_patterns(self, metrics: Dict[str, float], min_tp: float, min_success_rate: float, 
                       min_occurrences: int, min_risk_reward: float) -> List[Dict]:
        """
        Detect patterns in the current market metrics based on specified filters.
        
        Args:
            metrics (Dict[str, float]): Dictionary of market metrics
            min_tp (float): Minimum target profit percentage
            min_success_rate (float): Minimum success rate (0-1)
            min_occurrences (int): Minimum number of pattern occurrences
            min_risk_reward (float): Minimum risk/reward ratio
            
        Returns:
            List[Dict]: List of matched patterns that meet the filter criteria
            
        Raises:
            PatternError: If there's an error during pattern detection
        """
        try:
            matched_patterns = []
            
            for pattern in self.patterns:
                pattern1 = pattern['pattern1']
                pattern2 = pattern['pattern2']
                
                # Skip if metrics don't exist
                if pattern1 not in metrics or pattern2 not in metrics:
                    logger.debug(f"Skipping pattern: missing metrics {pattern1} or {pattern2}")
                    continue
                
                # Get metric values
                metric1 = metrics[pattern1]
                metric2 = metrics[pattern2]
                
                # Handle large numbers as infinity
                range1_min = pattern['range1_min']
                range1_max = pattern['range1_max']
                range2_min = pattern['range2_min']
                range2_max = pattern['range2_max']
                
                # Convert large numbers back to infinity for comparison
                if range1_min <= -999999:
                    range1_min = float('-inf')
                if range1_max >= 999999:
                    range1_max = float('inf')
                if range2_min <= -999999:
                    range2_min = float('-inf')
                if range2_max >= 999999:
                    range2_max = float('inf')
                
                # Check if metrics match pattern ranges
                if (range1_min <= metric1 <= range1_max and 
                    range2_min <= metric2 <= range2_max):
                    matched_patterns.append(pattern)
            
            # Apply all filters to matched patterns
            filtered_patterns = filter.apply_all_filters(
                matched_patterns,
                min_tp=min_tp,
                min_success_rate=min_success_rate,
                min_occurrences=min_occurrences,
                min_risk_reward=min_risk_reward
            )
            
            logger.info(f"Found {len(filtered_patterns)} patterns matching criteria")
            return filtered_patterns
            
        except Exception as e:
            error_msg = f"Error detecting patterns: {str(e)}"
            logger.error(error_msg)
            raise PatternError(error_msg)
    
    def generate_trading_strategy(self, pattern: Dict) -> Dict:
        """
        Generate a trading strategy based on the detected pattern.
        
        Args:
            pattern (Dict): Pattern definition containing timeframe, direction, and metrics
            
        Returns:
            Dict: Trading strategy with entry, target, and stop-loss levels
            
        Raises:
            PatternError: If there's an error generating the strategy
        """
        try:
            timeframe = pattern['timeframe']
            direction = pattern['direction']
            success_rate = pattern['success_rate']
            avg_move = pattern['avg_move']
            risk_reward = pattern['risk_reward']
            
            # Map timeframes to CT time ranges and trading periods
            timeframe_mapping = {
                'first_hour': ('8:31AM-9:30AM CT', 'morning'),
                'hour2': ('9:30AM-10:30AM CT', 'morning'),
                'hour3': ('10:30AM-11:30AM CT', 'morning'),
                'hour4': ('11:30AM-12:30PM CT', 'morning'),
                
                # Afternoon session (12:30PM-3:00PM CT)
                'hour5': ('12:30PM-1:30PM CT', 'afternoon'),
                'hour6': ('1:30PM-2:30PM CT', 'afternoon'),
                'last_30_min': ('2:30PM-3:00PM CT', 'afternoon'),
                
                # Multi-hour periods
                'hours_2_3': ('9:30AM-11:30AM CT', 'morning'),
                'hours_3_4': ('10:30AM-12:30PM CT', 'morning'),
                'hours_4_5': ('11:30AM-1:30PM CT', 'mixed'),
                'hours_5_6': ('12:30PM-2:30PM CT', 'afternoon'),
                'hours_5_6_30': ('12:30PM-3:00PM CT', 'afternoon'),
                
                # 15-minute intervals (morning only)
                'hour1_q1': ('8:31AM-8:45AM CT', 'morning'),
                'hour1_q2': ('8:45AM-9:00AM CT', 'morning'),
                'hour1_q3': ('9:00AM-9:15AM CT', 'morning'),
                'hour1_q4': ('9:15AM-9:30AM CT', 'morning'),
                'hour2_q1': ('9:30AM-9:45AM CT', 'morning'),
                'hour2_q2': ('9:45AM-10:00AM CT', 'morning'),
                'hour2_q3': ('10:00AM-10:15AM CT', 'morning'),
                'hour2_q4': ('10:15AM-10:30AM CT', 'morning'),
                
                # 30-minute intervals (morning only)
                'hour1_h1': ('8:31AM-9:00AM CT', 'morning'),
                'hour1_h2': ('9:00AM-9:30AM CT', 'morning'),
                'hour2_h1': ('9:30AM-10:00AM CT', 'morning'),
                'hour2_h2': ('10:00AM-10:30AM CT', 'morning'),
                
                # Transitions (morning only)
                'momentum_hour1_q1_to_hour1_q2': ('8:31AM-9:00AM CT', 'morning'),
                'momentum_hour1_q2_to_hour1_q3': ('8:45AM-9:15AM CT', 'morning'),
                'momentum_hour1_q3_to_hour1_q4': ('9:00AM-9:30AM CT', 'morning'),
                'momentum_hour1_q4_to_hour2_q1': ('9:15AM-9:45AM CT', 'morning'),
                'momentum_hour2_q1_to_hour2_q2': ('9:30AM-10:00AM CT', 'morning'),
                'momentum_hour2_q2_to_hour2_q3': ('9:45AM-10:15AM CT', 'morning'),
                'momentum_hour2_q3_to_hour2_q4': ('10:00AM-10:30AM CT', 'morning'),
                'momentum_hour1_h1_to_hour1_h2': ('8:31AM-9:30AM CT', 'morning'),
                'momentum_hour1_h2_to_hour2_h1': ('9:00AM-10:00AM CT', 'morning'),
                'momentum_hour2_h1_to_hour2_h2': ('9:30AM-10:30AM CT', 'morning'),

                # New time-based patterns
                'first_5min': ('8:31AM-8:35AM CT', 'morning'),
                'power_hour': ('9:30AM-10:30AM CT', 'morning'),
                'lunch_hour': ('12:00PM-1:00PM CT', 'afternoon'),
                'pre_close': ('2:00PM-3:00PM CT', 'afternoon'),
                
                # Strongest move periods
                'strongest_hour': ('Variable', 'variable'),
                'strongest_30min': ('Variable', 'variable'),
                'strongest_15min': ('Variable', 'variable')
            }
            
            # Get the display format and period for the timeframe
            timeframe_info = timeframe_mapping.get(timeframe, (timeframe.replace('_', ' '), 'unknown'))
            timeframe_display, period = timeframe_info
            
            # Handle variable timeframes for strongest move periods
            if timeframe in ['strongest_hour', 'strongest_30min', 'strongest_15min']:
                timeframe_display, period = self._handle_variable_timeframe(pattern, timeframe)
            
            strategy = {
                'timeframe': timeframe_display,
                'period': period,
                'direction': direction,
                'success_rate': success_rate,
                'avg_move': avg_move,
                'risk_reward': risk_reward,
                'entry': None,
                'target': None,
                'stop_loss': None,
                'strategy': None
            }
            
            # Generate entry, target, and stop-loss levels
            if direction == 'bullish':
                strategy['strategy'] = f"Place a BUY trade for the {timeframe_display} period"
                strategy['target'] = f"Take profit at +{pattern['avg_win']:.2f}%"
                strategy['stop_loss'] = f"Set stop loss at -{pattern['avg_loss']:.2f}%"
            else:  # bearish
                strategy['strategy'] = f"Place a SELL/SHORT trade for the {timeframe_display} period"
                strategy['target'] = f"Take profit at -{pattern['avg_win']:.2f}%"
                strategy['stop_loss'] = f"Set stop loss at +{pattern['avg_loss']:.2f}%"
            
            logger.debug(f"Generated strategy for {timeframe_display}: {direction} with {success_rate}% success rate")
            return strategy
            
        except Exception as e:
            error_msg = f"Error generating trading strategy: {str(e)}"
            logger.error(error_msg)
            raise PatternError(error_msg)
    
    def _handle_variable_timeframe(self, pattern: Dict, timeframe: str) -> Tuple[str, str]:
        """
        Handle variable timeframes for strongest move periods.
        
        Args:
            pattern (Dict): Pattern definition
            timeframe (str): Timeframe identifier
            
        Returns:
            Tuple[str, str]: Display format and period for the timeframe
        """
        if timeframe == 'strongest_hour' and 'strongest_hour' in pattern:
            hour = pattern['strongest_hour']
            timeframe_display = f"{8+hour}:30AM-{9+hour}:30AM CT"
            period = 'morning' if hour <= 4 else 'afternoon'
        elif timeframe == 'strongest_30min' and 'strongest_30min_period' in pattern:
            period_num = pattern['strongest_30min_period']
            start_hour = 8 + (period_num - 1) // 2
            start_min = 30 if (period_num - 1) % 2 == 0 else 0
            end_hour = start_hour
            end_min = start_min + 30
            if end_min >= 60:
                end_hour += 1
                end_min -= 60
            timeframe_display = f"{start_hour}:{start_min:02d}AM-{end_hour}:{end_min:02d}AM CT"
            period = 'morning' if period_num <= 8 else 'afternoon'
        elif timeframe == 'strongest_15min' and 'strongest_15min_period' in pattern:
            period_num = pattern['strongest_15min_period']
            start_hour = 8 + (period_num - 1) // 4
            start_min = 30 + ((period_num - 1) % 4) * 15
            if start_min >= 60:
                start_hour += 1
                start_min -= 60
            end_min = start_min + 15
            end_hour = start_hour
            if end_min >= 60:
                end_hour += 1
                end_min -= 60
            timeframe_display = f"{start_hour}:{start_min:02d}AM-{end_hour}:{end_min:02d}AM CT"
            period = 'morning' if period_num <= 16 else 'afternoon'
        else:
            return timeframe.replace('_', ' '), 'variable'
    
    def get_pattern_by_date(self, date) -> Optional[Dict]:
        """
        Get pattern from database by date.
        
        Args:
            date (datetime.date): The date to find patterns for
            
        Returns:
            Optional[Dict]: Pattern data if found, None otherwise
        """
        try:
            # Convert date to string format
            date_str = date.strftime('%Y-%m-%d')
            
            # Look for patterns with this date
            for pattern in self.patterns:
                if pattern.get('date') == date_str:
                    return pattern
                    
            # If not found, look for patterns with date in 'dates' list if present
            for pattern in self.patterns:
                if 'dates' in pattern and date_str in pattern.get('dates', []):
                    return pattern
            
            logger.debug(f"No pattern found for date: {date_str}")
            return None
            
        except Exception as e:
            logger.warning(f"Error finding pattern by date {date}: {str(e)}")
            return None

class PatternError(Exception):
    """Custom exception for pattern detection errors"""
    pass

class PatternDatabaseError(Exception):
    """Custom exception for pattern database errors"""
    pass 