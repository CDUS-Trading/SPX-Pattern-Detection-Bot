"""
Pattern Filtering
---------------
Functions for filtering and selecting market patterns based on various criteria.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

def filter_by_success_rate(patterns: List[Dict], min_success_rate: float) -> List[Dict]:
    """
    Filter patterns by minimum success rate.
    
    Args:
        patterns (List[Dict]): List of patterns to filter
        min_success_rate (float): Minimum success rate (0-1)
        
    Returns:
        List[Dict]: Filtered patterns
    """
    return [p for p in patterns if p['success_rate']/100 >= min_success_rate]

def filter_by_occurrences(patterns: List[Dict], min_occurrences: int) -> List[Dict]:
    """
    Filter patterns by minimum number of occurrences.
    
    Args:
        patterns (List[Dict]): List of patterns to filter
        min_occurrences (int): Minimum number of occurrences
        
    Returns:
        List[Dict]: Filtered patterns
    """
    return [p for p in patterns if p['sample_size'] >= min_occurrences]

def filter_by_risk_reward(patterns: List[Dict], min_risk_reward: float) -> List[Dict]:
    """
    Filter patterns by minimum risk/reward ratio.
    
    Args:
        patterns (List[Dict]): List of patterns to filter
        min_risk_reward (float): Minimum risk/reward ratio
        
    Returns:
        List[Dict]: Filtered patterns
    """
    return [p for p in patterns if p['risk_reward'] >= min_risk_reward]

def filter_by_take_profit(patterns: List[Dict], min_tp: float) -> List[Dict]:
    """
    Filter patterns by minimum take profit.
    
    Args:
        patterns (List[Dict]): List of patterns to filter
        min_tp (float): Minimum take profit
        
    Returns:
        List[Dict]: Filtered patterns
    """
    return [p for p in patterns if p['avg_win'] >= min_tp]


def filter_by_highest_probability(patterns: List[Dict]) -> Dict[str, Dict]:
    """
    Filter patterns to keep only the highest probability pattern for each timeframe.
    
    Args:
        patterns (List[Dict]): List of patterns to filter
        
    Returns:
        Dict[str, Dict]: Dictionary mapping timeframes to their highest probability patterns
    """
    highest_prob_patterns = {}
    for pattern in patterns:
        timeframe = pattern['timeframe']
        if (timeframe not in highest_prob_patterns or 
            pattern['success_rate'] > highest_prob_patterns[timeframe]['success_rate']):
            highest_prob_patterns[timeframe] = pattern
    return highest_prob_patterns

def sort_patterns_by_entry_time(patterns: List[Dict]) -> List[Dict]:
    """
    Sort patterns by their entry time.
    
    Args:
        patterns (List[Dict]): List of patterns to sort
        
    Returns:
        List[Dict]: Sorted patterns
    """
    def get_entry_time(strategy: Dict) -> int:
        """Convert entry time to minutes since midnight for sorting"""
        timeframe = strategy['timeframe']
        if '-' in timeframe:
            entry_time = timeframe.split('-')[0].strip()
        else:
            entry_time = timeframe.strip()
            
        # Convert to minutes since midnight
        if 'AM' in entry_time:
            time_str = entry_time.replace('AM', '').strip()
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        elif 'PM' in entry_time:
            time_str = entry_time.replace('PM', '').strip()
            hours, minutes = map(int, time_str.split(':'))
            return (hours + 12) * 60 + minutes
        return 0
    
    return sorted(patterns, key=get_entry_time)

def is_interval_contained(interval1: str, interval2: str) -> bool:
    """
    Check if interval1 is completely contained within interval2.
    
    Args:
        interval1 (str): First time interval (e.g., "9:30AM-10:00AM CT")
        interval2 (str): Second time interval (e.g., "9:00AM-11:00AM CT")
        
    Returns:
        bool: True if interval1 is contained within interval2
        
    Raises:
        ValueError: If there's an error parsing the intervals
    """
    try:
        def parse_time(time_str: str) -> int:
            """Convert time string to minutes since midnight"""
            time_str = time_str.strip().replace(' CT', '')
            if 'AM' in time_str:
                time_str = time_str.replace('AM', '').strip()
                hours, minutes = map(int, time_str.split(':'))
                return hours * 60 + minutes
            elif 'PM' in time_str:
                time_str = time_str.replace('PM', '').strip()
                hours, minutes = map(int, time_str.split(':'))
                return (hours + 12) * 60 + minutes
            return 0

        def get_times(interval: str) -> Tuple[int, int]:
            """Get start and end times in minutes for an interval"""
            if '-' in interval:
                start, end = interval.split('-')
                return parse_time(start), parse_time(end)
            time = parse_time(interval)
            return time, time

        start1, end1 = get_times(interval1)
        start2, end2 = get_times(interval2)
        
        # Check if interval1 is completely contained within interval2
        return start2 <= start1 and end1 <= end2
        
    except Exception as e:
        error_msg = f"Error checking interval containment: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def filter_overlapping_patterns(patterns: List[Dict]) -> List[Dict]:
    """
    Filter out patterns that are contained within larger patterns.
    
    Args:
        patterns (List[Dict]): List of patterns to filter
        
    Returns:
        List[Dict]: Filtered patterns with no overlaps
    """
    filtered_patterns = []
    for i, pattern1 in enumerate(patterns):
        is_contained = False
        for j, pattern2 in enumerate(patterns):
            if i != j and is_interval_contained(pattern1['timeframe'], pattern2['timeframe']):
                is_contained = True
                break
        if not is_contained:
            filtered_patterns.append(pattern1)
    return filtered_patterns 

# Applying all filters
def apply_all_filters(patterns: List[Dict], min_tp: float, min_success_rate: float,
                     min_occurrences: int, min_risk_reward: float) -> List[Dict]:
    """
    Apply all filters to a list of patterns.
    
    Args:
        patterns (List[Dict]): List of patterns to filter
        min_tp (float): Minimum take profit
        min_success_rate (float): Minimum success rate (0-1)
        min_occurrences (int): Minimum number of occurrences
        min_risk_reward (float): Minimum risk/reward ratio
        
    Returns:
        List[Dict]: Filtered patterns that meet all criteria
    """
    filtered = filter_by_success_rate(patterns, min_success_rate)
    filtered = filter_by_occurrences(filtered, min_occurrences)
    filtered = filter_by_risk_reward(filtered, min_risk_reward)
    filtered = filter_by_take_profit(filtered, min_tp)
    return filtered

def detect_patterns_with_levels(detector, metrics: Dict[str, float], 
                              filter_levels: List[Dict], current_level: int = 0) -> Tuple[List[Dict], Optional[str], bool]:
    """
    Try to detect patterns using progressively relaxed filter levels.
    
    Args:
        detector: Pattern detector instance
        metrics (Dict[str, float]): Dictionary of market metrics
        filter_levels (List[Dict]): List of filter level configurations
        current_level (int): Current filter level index
        
    Returns:
        Tuple[List[Dict], Optional[str], bool]: Tuple containing:
            - List of matched patterns
            - Name of the filter level used (or None)
            - Flag indicating if no patterns were found
    """
    if current_level >= len(filter_levels):
        logger.warning("No patterns found with any filter level")
        return [], None, True
    
    level = filter_levels[current_level]
    logger.info(f"Trying filter level: {level['name']}")
    
    matched_patterns = detector.detect_patterns(
        metrics,
        min_tp=level['min_tp'],
        min_success_rate=level['min_success_rate'],
        min_occurrences=level['min_occurrences'],
        min_risk_reward=level['min_risk_reward']
    )
    
    if matched_patterns:
        logger.info(f"Found {len(matched_patterns)} patterns using {level['name']} filters")
        return matched_patterns, level['name'], False
    else:
        logger.info(f"No patterns found with {level['name']} filters. Trying next level...")
        return detect_patterns_with_levels(detector, metrics, filter_levels, current_level + 1)