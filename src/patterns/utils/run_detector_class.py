"""
Pattern Detector Runner
--------------------
Main script for running the pattern detection system.
"""

import logging
from typing import List, Dict
import pandas as pd


logger = logging.getLogger(__name__)

class RunDetector:
    """
    Detects robust, non-overlapping market runs (bull or bear) based on configurable thresholds:
    - min_run_points: Minimum move required to establish a run (e.g., 10 points)
    - pullback_tracking_threshold: Minimum adverse move before tracking a potential run break (e.g., 10 points)
    - max_pullback_points: Maximum allowed pullback before a run is broken (e.g., 20 points)
    - max_stagnation_minutes: Maximum time (bars) without a new extreme before ending the run (e.g., 15 minutes) #TODO: Make this configurable
    - Each run output includes an 'end_reason' key explaining why the run ended (e.g., 'min_move', 'pullback', 'stagnation').
    - Run start is always the open of the candidate bar; run end is always the close of the last extreme bar.
    """
    def __init__(
        self,
        min_run_points: float = 10.0,
        pullback_tracking_threshold: float = 10.0,
        max_pullback_points: float = 20.0,
        max_stagnation_minutes: int = 15,  # TODO: Make this configurable as needed
    ):
        """
        Initialize the RunDetector with robust run detection parameters.
        Args:
            min_run_points (float): Minimum move required to establish a run (in points).
            pullback_tracking_threshold (float): Minimum adverse move before tracking a potential run break (in points).
            max_pullback_points (float): Maximum allowed pullback before a run is broken (in points).
            max_stagnation_minutes (int): Maximum time (bars/minutes) without a new extreme before ending the run (default: 15). #TODO: Tune as needed
        """
        self.min_run_points = min_run_points
        self.pullback_tracking_threshold = pullback_tracking_threshold
        self.max_pullback_points = max_pullback_points
        self.max_stagnation_minutes = max_stagnation_minutes

    def detect_runs(self, minute_data: pd.DataFrame) -> List[Dict]:
        """
        Improved detection loop:
        - After a run ends, enter a 'watching' state.
        - Track the most recent local high and low (extremes) as candidate run starts.
        - For each new bar:
            - If a new high (since last run end) is made, update the candidate high.
            - If a new low is made, update the candidate low.
            - If the price moves at least min_run_points from the candidate high (down) or candidate low (up), start a new run from the extreme bar.
        - Only commit to a run when the threshold move is observed from the extreme, and the run's start bar is the extreme bar.
        - Track both bull and bear candidates in parallel.
        """
        runs = []
        opens = minute_data['open'].values
        closes = minute_data['close'].values
        timestamps = minute_data.index if minute_data.index.name else minute_data['timestamp']
        n = len(closes)
        i = 0
        while i < n:
            # Watching state: find most recent high and low as candidate run starts
            candidate_high_idx = i
            candidate_high = closes[i]
            candidate_low_idx = i
            candidate_low = closes[i]
            for j in range(i + 1, n):
                price = closes[j]
                # Update candidate high/low
                if price > candidate_high:
                    candidate_high = price
                    candidate_high_idx = j
                if price < candidate_low:
                    candidate_low = price
                    candidate_low_idx = j
                # Check for bull run (up from low)
                bull_move = price - candidate_low
                if bull_move >= self.min_run_points:
                    # Start bull run from candidate_low_idx
                    run_start_idx = candidate_low_idx
                    run_direction = 'bull'
                    start_open = opens[run_start_idx]
                    run_extreme_idx = run_start_idx
                    run_extreme_close = closes[run_extreme_idx]
                    pullback_tracking = False
                    pullback_max_adverse = 0.0
                    stagnation_counter = 0
                    last_extreme_idx = run_extreme_idx
                    for k in range(j + 1, n):
                        price_k = closes[k]
                        new_extreme = False
                        if price_k > run_extreme_close:
                            run_extreme_close = price_k
                            run_extreme_idx = k
                            pullback_tracking = False
                            pullback_max_adverse = 0.0
                            stagnation_counter = 0
                            new_extreme = True
                        adverse_move = run_extreme_close - price_k
                        if not pullback_tracking and adverse_move >= self.pullback_tracking_threshold:
                            pullback_tracking = True
                            pullback_max_adverse = adverse_move
                        if pullback_tracking:
                            if adverse_move > pullback_max_adverse:
                                pullback_max_adverse = adverse_move
                            if adverse_move >= self.max_pullback_points:
                                end_idx = run_extreme_idx
                                end_close = closes[end_idx]
                                run = {
                                    'start_idx': run_start_idx,
                                    'end_idx': end_idx,
                                    'start_time': str(timestamps[run_start_idx]),
                                    'end_time': str(timestamps[end_idx]),
                                    'direction': run_direction,
                                    'duration': end_idx - run_start_idx + 1,
                                    'total_move': end_close - start_open,
                                    'max_adverse': pullback_max_adverse,
                                    'end_reason': 'pullback'
                                }
                                runs.append(run)
                                i = end_idx + 1
                                break
                            if price_k > run_extreme_close:
                                pullback_tracking = False
                                pullback_max_adverse = 0.0
                        if not new_extreme:
                            stagnation_counter += 1
                        else:
                            stagnation_counter = 0
                        if stagnation_counter >= self.max_stagnation_minutes:
                            end_idx = run_extreme_idx
                            end_close = closes[end_idx]
                            run = {
                                'start_idx': run_start_idx,
                                'end_idx': end_idx,
                                'start_time': str(timestamps[run_start_idx]),
                                'end_time': str(timestamps[end_idx]),
                                'direction': run_direction,
                                'duration': end_idx - run_start_idx + 1,
                                'total_move': end_close - start_open,
                                'max_adverse': pullback_max_adverse,
                                'end_reason': 'stagnation'
                            }
                            runs.append(run)
                            i = end_idx + 1
                            break
                    else:
                        end_idx = run_extreme_idx
                        end_close = closes[end_idx]
                        run = {
                            'start_idx': run_start_idx,
                            'end_idx': end_idx,
                            'start_time': str(timestamps[run_start_idx]),
                            'end_time': str(timestamps[end_idx]),
                            'direction': run_direction,
                            'duration': end_idx - run_start_idx + 1,
                            'total_move': end_close - start_open,
                            'max_adverse': 0.0,
                            'end_reason': 'min_move'
                        }
                        runs.append(run)
                        i = end_idx + 1
                    break  # Only take the first valid run from this extreme
                # Check for bear run (down from high)
                bear_move = candidate_high - price
                if bear_move >= self.min_run_points:
                    run_start_idx = candidate_high_idx
                    run_direction = 'bear'
                    start_open = opens[run_start_idx]
                    run_extreme_idx = run_start_idx
                    run_extreme_close = closes[run_extreme_idx]
                    pullback_tracking = False
                    pullback_max_adverse = 0.0
                    stagnation_counter = 0
                    last_extreme_idx = run_extreme_idx
                    for k in range(j + 1, n):
                        price_k = closes[k]
                        new_extreme = False
                        if price_k < run_extreme_close:
                            run_extreme_close = price_k
                            run_extreme_idx = k
                            pullback_tracking = False
                            pullback_max_adverse = 0.0
                            stagnation_counter = 0
                            new_extreme = True
                        adverse_move = price_k - run_extreme_close
                        if not pullback_tracking and adverse_move >= self.pullback_tracking_threshold:
                            pullback_tracking = True
                            pullback_max_adverse = adverse_move
                        if pullback_tracking:
                            if adverse_move > pullback_max_adverse:
                                pullback_max_adverse = adverse_move
                            if adverse_move >= self.max_pullback_points:
                                end_idx = run_extreme_idx
                                end_close = closes[end_idx]
                                run = {
                                    'start_idx': run_start_idx,
                                    'end_idx': end_idx,
                                    'start_time': str(timestamps[run_start_idx]),
                                    'end_time': str(timestamps[end_idx]),
                                    'direction': run_direction,
                                    'duration': end_idx - run_start_idx + 1,
                                    'total_move': start_open - end_close,
                                    'max_adverse': pullback_max_adverse,
                                    'end_reason': 'pullback'
                                }
                                runs.append(run)
                                i = end_idx + 1
                                break
                            if price_k < run_extreme_close:
                                pullback_tracking = False
                                pullback_max_adverse = 0.0
                        if not new_extreme:
                            stagnation_counter += 1
                        else:
                            stagnation_counter = 0
                        if stagnation_counter >= self.max_stagnation_minutes:
                            end_idx = run_extreme_idx
                            end_close = closes[end_idx]
                            run = {
                                'start_idx': run_start_idx,
                                'end_idx': end_idx,
                                'start_time': str(timestamps[run_start_idx]),
                                'end_time': str(timestamps[end_idx]),
                                'direction': run_direction,
                                'duration': end_idx - run_start_idx + 1,
                                'total_move': start_open - end_close,
                                'max_adverse': pullback_max_adverse,
                                'end_reason': 'stagnation'
                            }
                            runs.append(run)
                            i = end_idx + 1
                            break
                    else:
                        end_idx = run_extreme_idx
                        end_close = closes[end_idx]
                        run = {
                            'start_idx': run_start_idx,
                            'end_idx': end_idx,
                            'start_time': str(timestamps[run_start_idx]),
                            'end_time': str(timestamps[end_idx]),
                            'direction': run_direction,
                            'duration': end_idx - run_start_idx + 1,
                            'total_move': start_open - end_close,
                            'max_adverse': 0.0,
                            'end_reason': 'min_move'
                        }
                        runs.append(run)
                        i = end_idx + 1
                    break  # Only take the first valid run from this extreme
            else:
                i += 1  # No run found, move to next bar
        return runs

    def validate_run(self, data_slice: pd.DataFrame, direction: str) -> bool:
        """
        Validate if a given data slice qualifies as a run based on criteria.
        Args:
            data_slice (pd.DataFrame): Slice of minute data representing a potential run.
            direction (str): 'bull' or 'bear'.
        Returns:
            bool: True if valid run, False otherwise.
        """
        pass  # To be implemented

    def calculate_run_metrics(self, run_data: pd.DataFrame, direction: str) -> Dict:
        """
        Calculate run metrics such as duration, total move, and max adverse excursion.
        Args:
            run_data (pd.DataFrame): Data for the detected run.
            direction (str): 'bull' or 'bear'.
        Returns:
            Dict: Dictionary of run metrics (all in points where applicable).
        """
        pass  # To be implemented

def main() -> None:
    """
    Main execution function for the pattern detector.
    """
    # TODO: Implement main execution logic
    pass

if __name__ == "__main__":
    main() 