#!/usr/bin/env python3

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.patterns.detector import PatternDetector
from src.patterns.utils.io import load_minute_data

class TradeEngine:
    def __init__(self, data_path: str, pattern_db_path: str):
        """
        Initialize the trade simulator.
        
        Args:
            data_path (str): Path to minute data file
            pattern_db_path (str): Path to pattern database
        """
        self.data_path = data_path
        self.pattern_db_path = pattern_db_path
        
        # Initialize pattern detector
        self.detector = PatternDetector(pattern_db_path)
        
        # Load minute data
        self.full_data = load_minute_data(data_path)
        
        # Initialize trade tracking
        self.trades: List[Dict] = []
        self.current_trade: Optional[Dict] = None  # Track currently open trade
        self.skipped_trades = 0  # Counter for skipped trades
        self.metrics: Dict = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit_loss': 0,
            'total_profit_loss_money': 0,  # In dollars
            'max_drawdown': 0,
            'max_drawdown_money': 0,  # In dollars
            'win_rate': 0,
            'average_win': 0,
            'average_win_money': 0,  # In dollars
            'average_loss': 0,
            'average_loss_money': 0,  # In dollars
            'profit_factor': 0
        }

    def parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime object."""
        time_str = time_str.replace(' CT', '')
        return datetime.strptime(time_str, '%I:%M%p')

    def can_open_trade(self, pattern: Dict) -> (bool, str):
        """
        Check if we can open a new trade.
        
        Args:
            pattern (Dict): Pattern to check
            
        Returns:
            (bool, str): (True, '') if we can open a trade, (False, reason) otherwise
        """
        # If we have an open trade, we can't open another one
        if self.current_trade is not None:
            return False, f"Skipping overlapping trade at {pattern['entry_time']} on {pattern['date']} - Already in trade from {self.current_trade['entry_time']}"

        # If we have a completed trade, check if we've missed the entry time
        if self.trades:
            last_trade = self.trades[-1]
            if last_trade['date'] == pattern['date']:
                last_exit_time = self.parse_time(last_trade['exit_time'])
                new_entry_time = self.parse_time(pattern['entry_time'])
                
                # If the new trade's entry time is before the last trade's exit time, we've missed it
                if new_entry_time < last_exit_time:
                    return False, f"Skipping trade at {pattern['entry_time']} on {pattern['date']} - Entry time has elapsed; not included in results"
            
        return True, ''

    def simulate_trade(self, date: datetime.date, entry_time: str, exit_time: str,
                      direction: str, tp_points: float, sl_points: float) -> Optional[Dict]:
        """
        Simulate a single trade.
        
        Args:
            date (datetime.date): Trading date
            entry_time (str): Entry time
            exit_time (str): Exit time
            direction (str): Trade direction ('bullish' or 'bearish')
            tp_points (float): Take profit points
            sl_points (float): Stop loss points
            
        Returns:
            Optional[Dict]: Trade results if successful, None otherwise
        """
        # Check if we can open a new trade
        can_open, reason = self.can_open_trade({'date': date, 'entry_time': entry_time})
        if not can_open:
            self.skipped_trades += 1  # Increment skipped trades counter
            return None

        # Get data for the trading day
        day_data = self.full_data[self.full_data['date'].dt.date == date].copy()
        if day_data.empty:
            logging.warning(f"No data available for date: {date}")
            return None

        # Parse entry and exit times
        entry_dt = self.parse_time(entry_time)
        exit_dt = self.parse_time(exit_time)
        
        # Convert to datetime objects with date
        entry_dt = datetime.combine(date, entry_dt.time())
        exit_dt = datetime.combine(date, exit_dt.time())
        
        # Get entry price
        entry_data = day_data[day_data['date'] >= entry_dt]
        if entry_data.empty:
            logging.warning(f"No data available at entry time: {entry_time} on {date}")
            return None
        
        entry_price = entry_data['open'].iloc[0]
        
        # Set current trade
        self.current_trade = {
            'date': date,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': direction,
            'entry_price': entry_price,
            'tp_points': tp_points,
            'sl_points': sl_points,
            'entry_dt': entry_dt,
            'exit_dt': exit_dt
        }
        
        # Track price movements
        trade_data = day_data[(day_data['date'] >= entry_dt) & (day_data['date'] <= exit_dt)]
        if trade_data.empty:
            logging.warning(f"No data available between {entry_time} and {exit_time} on {date}")
            self.current_trade = None
            return None
        
        # Initialize trade result
        trade_result = {
            'date': date,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': None,
            'exit_type': None,  # 'TP', 'SL', or 'TIME'
            'profit_loss': 0,
            'duration': None,
            'sl_points': sl_points
        }
        
        # Check for TP/SL hits
        for _, row in trade_data.iterrows():
            current_price = row['close']
            current_time = row['date']
            
            if direction == 'bullish':
                if current_price >= entry_price + tp_points:
                    trade_result['exit_price'] = entry_price + tp_points
                    trade_result['exit_type'] = 'TP'
                    trade_result['profit_loss'] = tp_points
                    trade_result['duration'] = (current_time - entry_dt).total_seconds() / 60
                    break
                elif current_price <= entry_price - sl_points:
                    trade_result['exit_price'] = entry_price - sl_points
                    trade_result['exit_type'] = 'SL'
                    trade_result['profit_loss'] = -sl_points
                    trade_result['duration'] = (current_time - entry_dt).total_seconds() / 60
                    break
            else:  # bearish
                if current_price <= entry_price - tp_points:
                    trade_result['exit_price'] = entry_price - tp_points
                    trade_result['exit_type'] = 'TP'
                    trade_result['profit_loss'] = tp_points
                    trade_result['duration'] = (current_time - entry_dt).total_seconds() / 60
                    break
                elif current_price >= entry_price + sl_points:
                    trade_result['exit_price'] = entry_price + sl_points
                    trade_result['exit_type'] = 'SL'
                    trade_result['duration'] = (current_time - entry_dt).total_seconds() / 60
                    break
        
        # If no TP/SL hit, exit at exit time
        if trade_result['exit_price'] is None:
            try:
                trade_result['exit_price'] = trade_data['close'].iloc[-1]
                trade_result['exit_type'] = 'TIME'
                if direction == 'bullish':
                    trade_result['profit_loss'] = trade_result['exit_price'] - entry_price
                else:
                    trade_result['profit_loss'] = entry_price - trade_result['exit_price']
                trade_result['duration'] = (trade_data['date'].iloc[-1] - entry_dt).total_seconds() / 60
            except IndexError:
                logging.warning(f"Could not get exit price for trade on {date} between {entry_time} and {exit_time}")
                self.current_trade = None
                return None
        
        # Clear current trade
        self.current_trade = None
        return trade_result

    def update_metrics(self, trade_result: Dict) -> None:
        """Update performance metrics based on trade result."""
        self.metrics['total_trades'] += 1
        self.metrics['total_profit_loss'] += trade_result['profit_loss']
        self.metrics['total_profit_loss_money'] += trade_result['profit_loss'] * 5  # $5 per point
        
        if trade_result['profit_loss'] > 0:
            self.metrics['winning_trades'] += 1
            self.metrics['average_win'] = (
                (self.metrics['average_win'] * (self.metrics['winning_trades'] - 1) + 
                 trade_result['profit_loss']) / self.metrics['winning_trades']
            )
            self.metrics['average_win_money'] = self.metrics['average_win'] * 5  # $5 per point
        else:
            self.metrics['losing_trades'] += 1
            self.metrics['average_loss'] = (
                (self.metrics['average_loss'] * (self.metrics['losing_trades'] - 1) + 
                 abs(trade_result['profit_loss'])) / self.metrics['losing_trades']
            )
            self.metrics['average_loss_money'] = self.metrics['average_loss'] * 5  # $5 per point
        
        # Calculate win rate
        self.metrics['win_rate'] = (
            self.metrics['winning_trades'] / self.metrics['total_trades'] * 100
        )
        
        # Calculate profit factor
        if self.metrics['average_loss'] > 0:
            self.metrics['profit_factor'] = (
                self.metrics['average_win'] / self.metrics['average_loss']
            )

    def run_simulation(self, patterns_by_date: Dict[datetime.date, List[Dict]], 
                      start_date: datetime.date, end_date: datetime.date,
                      show_progress: bool = True) -> List[Dict]:
        """
        Run the trade simulation using provided patterns.
        
        Args:
            patterns_by_date (Dict[datetime.date, List[Dict]]): Dictionary mapping dates to their patterns
            start_date (datetime.date): Start date for simulation
            end_date (datetime.date): End date for simulation
            show_progress (bool): Whether to show progress bar
            
        Returns:
            List[Dict]: List of trade results
        """
        if not patterns_by_date:
            logging.error("No patterns provided for simulation")
            return []
        
        # Calculate total days for progress bar
        total_days = (end_date - start_date).days + 1
        
        # Initialize progress bar if requested
        pbar = tqdm(total=total_days, desc="Simulating trades", unit="day") if show_progress else None
        
        current_date = start_date
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                if pbar:
                    pbar.update(1)
                continue
            
            # Get patterns for current date
            if current_date in patterns_by_date:
                patterns = patterns_by_date[current_date]
                
                # Update progress bar description
                if pbar:
                    pbar.set_description(f"Simulating {current_date.strftime('%Y-%m-%d')} ({len(patterns)} patterns)")
                
                # Simulate trades for each pattern
                for pattern in patterns:
                    trade_result = self.simulate_trade(
                        current_date,
                        pattern['entry_time'],
                        pattern['exit_time'],
                        pattern['direction'],
                        pattern['tp_points'],
                        pattern['sl_points']
                    )
                    
                    if trade_result:
                        self.trades.append(trade_result)
                        self.update_metrics(trade_result)
            
            current_date += timedelta(days=1)
            if pbar:
                pbar.update(1)
                # Update progress bar postfix with current metrics
                pbar.set_postfix({
                    'trades': self.metrics['total_trades'],
                    'win_rate': f"{self.metrics['win_rate']:.1f}%",
                    'P/L': f"{self.metrics['total_profit_loss']:.1f} points (${self.metrics['total_profit_loss_money']:.2f})"
                })
        
        if pbar:
            pbar.close()
        
        print(f"Skipped {self.skipped_trades} trades out of {self.metrics['total_trades'] + self.skipped_trades} total trades due to overlap or elapsed entry time.")
        return self.trades

    def get_metrics(self) -> Dict:
        """Get current simulation metrics."""
        return self.metrics.copy()

    def get_trades(self) -> List[Dict]:
        """Get list of all trades."""
        return self.trades.copy()

    def reset(self) -> None:
        """Reset the simulator state."""
        self.trades = []
        self.current_trade = None
        self.skipped_trades = 0
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit_loss': 0,
            'total_profit_loss_money': 0,
            'max_drawdown': 0,
            'max_drawdown_money': 0,
            'win_rate': 0,
            'average_win': 0,
            'average_win_money': 0,
            'average_loss': 0,
            'average_loss_money': 0,
            'profit_factor': 0
        } 