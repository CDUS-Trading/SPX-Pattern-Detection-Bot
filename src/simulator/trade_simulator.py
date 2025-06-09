#!/usr/bin/env python3

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.patterns.detector import (
    PatternDetector,
)
from src.patterns.utils.io import load_minute_data

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the trade simulator.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Simulate trades based on backtest patterns')
    
    parser.add_argument('--filter',
                       type=str,
                       choices=['Strict', 'Moderate', 'Minimum', 'Poor'],
                       help='Filter type to use for simulation. If not provided, will use recursive filtering starting from Strict')
    
    return parser.parse_args()

class TradeSimulator:
    def __init__(self, backtest_dates_path: str, data_path: str, pattern_db_path: str, filter_type: Optional[str] = None):
        """
        Initialize the trade simulator.
        
        Args:
            backtest_dates_path (str): Path to backtest_dates.json
            data_path (str): Path to minute data file
            pattern_db_path (str): Path to pattern database
            filter_type (Optional[str]): Filter type to use for simulation. If None, will use recursive filtering
        """
        self.backtest_dates_path = backtest_dates_path
        self.data_path = data_path
        self.pattern_db_path = pattern_db_path
        self.filter_type = filter_type
        
        # Load backtest dates
        with open(backtest_dates_path, 'r') as f:
            self.backtest_info = json.load(f)
        
        # Initialize pattern detector
        self.detector = PatternDetector(pattern_db_path)
        
        # Load minute data
        self.full_data = load_minute_data(data_path)
        
        # Initialize trade tracking
        self.trades: List[Dict] = []
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
        """
        Parse time string to datetime object.
        
        Args:
            time_str (str): Time string in format "HH:MMAM/PM CT"
            
        Returns:
            datetime: Parsed datetime object
        """
        time_str = time_str.replace(' CT', '')
        return datetime.strptime(time_str, '%I:%M%p')

    def calculate_dynamic_sl(self, entry_price: float, tp_points: float, pattern_success_rate: float, 
                           day_data: pd.DataFrame, entry_dt: datetime) -> float:
        """
        Calculate dynamic stop loss based on market conditions and pattern characteristics.
        
        Args:
            entry_price (float): Entry price
            tp_points (float): Take profit points
            pattern_success_rate (float): Pattern's historical success rate
            day_data (pd.DataFrame): Day's price data
            entry_dt (datetime): Entry datetime
            
        Returns:
            float: Calculated stop loss points
        """
        # Get recent price data before entry (last 30 minutes)
        recent_data = day_data[day_data['date'] < entry_dt].tail(30)
        if recent_data.empty:
            return tp_points * 0.5  # Default to 0.5x TP if no recent data
        
        # Calculate recent volatility (standard deviation of price changes)
        price_changes = recent_data['close'].pct_change().dropna()
        volatility = price_changes.std() * 100  # Convert to percentage
        
        # Calculate ATR (Average True Range) for the last 14 periods
        high_low = recent_data['high'] - recent_data['low']
        high_close = abs(recent_data['high'] - recent_data['close'].shift())
        low_close = abs(recent_data['low'] - recent_data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        # Base SL on ATR (1.5x ATR)
        base_sl = atr * 1.5
        
        # Adjust SL based on pattern success rate
        # Higher success rate = tighter stop loss
        success_rate_factor = 1.0 - (pattern_success_rate / 100) * 0.5  # 0.5 to 1.0
        adjusted_sl = base_sl * success_rate_factor
        
        # Ensure SL is not too tight (minimum 0.3x TP) or too wide (maximum 1.0x TP)
        min_sl = tp_points * 0.3
        max_sl = tp_points * 1.0
        final_sl = max(min_sl, min(adjusted_sl, max_sl))
        
        return final_sl

    def simulate_trade(self, date: datetime.date, entry_time: str, exit_time: str,
                      direction: str, tp_points: float, sl_points: float) -> Dict:
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
            Dict: Trade results
        """
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
        
        # Calculate dynamic stop loss if sl_points is 0
        if sl_points == 0:
            # Get pattern success rate from the pattern database
            pattern_success_rate = 50.0  # Default to 50% if not available
            try:
                pattern = self.detector.get_pattern_by_date(date)
                if pattern and 'success_rate' in pattern:
                    pattern_success_rate = pattern['success_rate']
            except Exception as e:
                logging.warning(f"Could not get pattern success rate: {e}")
            
            sl_points = self.calculate_dynamic_sl(entry_price, tp_points, pattern_success_rate, day_data, entry_dt)
            logging.info(f"Using dynamic stop loss of {sl_points:.2f} points for trade on {date}")
        
        # Track price movements
        trade_data = day_data[(day_data['date'] >= entry_dt) & (day_data['date'] <= exit_dt)]
        if trade_data.empty:
            logging.warning(f"No data available between {entry_time} and {exit_time} on {date}")
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
            'sl_points': sl_points  # Add SL points to trade result
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
                    trade_result['profit_loss'] = -sl_points
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
                return None
        
        return trade_result

    def update_metrics(self, trade_result: Dict) -> None:
        """
        Update performance metrics based on trade result.
        
        Args:
            trade_result (Dict): Trade result dictionary
        """
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

    def parse_patterns_from_file(self, file_path: str) -> Dict[datetime.date, List[Dict]]:
        """
        Parse trading patterns from backtest_patterns.json file.
        
        Args:
            file_path (str): Path to backtest_patterns.json
            
        Returns:
            Dict[datetime.date, List[Dict]]: Dictionary mapping dates to their patterns
        """
        patterns_by_date = {}
        
        try:
            with open(file_path, 'r') as f:
                backtest_data = json.load(f)
            
            # Process each result in the JSON file
            for result in backtest_data.get('results', []):
                # Skip error/no-data results
                if 'status' in result and result['status'] == 'no_data':
                    continue
                
                # Get the pattern date
                pattern_date = datetime.strptime(result['pattern_date'], '%Y-%m-%d').date()
                
                # Initialize patterns list for this date if not exists
                if pattern_date not in patterns_by_date:
                    patterns_by_date[pattern_date] = []
                
                # Process patterns from each session
                for session_type in ['morning', 'mixed', 'afternoon']:
                    session_patterns = result['patterns']['sessions'].get(session_type, [])
                    for pattern in session_patterns:
                        # Convert pattern to our format
                        formatted_pattern = {
                            'entry_time': pattern['entry_time'],
                            'exit_time': pattern['exit_time'],
                            'direction': 'bullish' if pattern['direction'] == 'Buy' else 'bearish',
                            'tp_points': pattern['target_points'],
                            'sl_points': pattern['stop_loss_points'],
                            'success_rate': pattern['success_rate']
                        }
                        patterns_by_date[pattern_date].append(formatted_pattern)
            
            return patterns_by_date
            
        except Exception as e:
            logging.error(f"Error parsing patterns from file: {str(e)}")
            return {}

    def run_simulation(self) -> None:
        """
        Run the trade simulation using patterns from the specified backtest file.
        """
        # Get the backtest file for the specified filter
        logs_dir = os.path.dirname(self.backtest_dates_path)
        
        # Use filter type (including 'recursive' if no specific filter provided)
        filter_type = self.filter_type if self.filter_type else 'recursive'
        backtest_file = f'backtest_patterns_{filter_type}.json'
        patterns_file = os.path.join(logs_dir, backtest_file)
        
        if not os.path.exists(patterns_file):
            logging.error(f"Backtest file not found: {patterns_file}")
            return
        
        logging.info(f"Processing {backtest_file} with filter type: {filter_type}")
        patterns_by_date = self.parse_patterns_from_file(patterns_file)
        
        if not patterns_by_date:
            logging.error("No patterns found in backtest file")
            return
        
        start_date = datetime.strptime(self.backtest_info['start_date'], '%Y-%m-%d').date()
        end_date = datetime.strptime(self.backtest_info['end_date'], '%Y-%m-%d').date()
        
        # Calculate total days for progress bar
        total_days = (end_date - start_date).days + 1
        
        # Initialize progress bar
        with tqdm(total=total_days, desc=f"Simulating trades ({filter_type})", unit="day") as pbar:
            current_date = start_date
            while current_date <= end_date:
                # Skip weekends
                if current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                    pbar.update(1)
                    continue
                
                # Get patterns for current date
                if current_date in patterns_by_date:
                    patterns = patterns_by_date[current_date]
                    
                    # Update progress bar description
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
                pbar.update(1)
                
                # Update progress bar postfix with current metrics
                pbar.set_postfix({
                    'trades': self.metrics['total_trades'],
                    'win_rate': f"{self.metrics['win_rate']:.1f}%",
                    'P/L': f"{self.metrics['total_profit_loss']:.1f} points (${self.metrics['total_profit_loss_money']:.2f})"
                })
        
        # Print summary
        self.print_summary()

    def print_summary(self) -> None:
        """
        Print simulation summary and performance metrics to a file.
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(self.backtest_dates_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output file path with filter type
        output_file = os.path.join(output_dir, f'simulation_summary_{self.filter_type if self.filter_type else "recursive"}.txt')
        
        with open(output_file, 'w') as f:
            f.write("\n=== Trade Simulation Summary ===\n")
            f.write(f"Period: {self.backtest_info['start_date']} to {self.backtest_info['end_date']}\n")
            f.write(f"Total Trading Days: {self.backtest_info['total_trading_days']}\n")
            f.write(f"Filter Type: {self.filter_type if self.filter_type else 'recursive'}\n")
            f.write(f"Total Trades: {self.metrics['total_trades']}\n")
            f.write(f"Winning Trades: {self.metrics['winning_trades']}\n")
            f.write(f"Losing Trades: {self.metrics['losing_trades']}\n")
            f.write(f"Win Rate: {self.metrics['win_rate']:.2f}%\n")
            f.write(f"Total Profit/Loss: {self.metrics['total_profit_loss']:.2f} points (${self.metrics['total_profit_loss_money']:.2f})\n")
            f.write(f"Average Win: {self.metrics['average_win']:.2f} points (${self.metrics['average_win_money']:.2f})\n")
            f.write(f"Average Loss: {self.metrics['average_loss']:.2f} points (${self.metrics['average_loss_money']:.2f})\n")
            f.write(f"Profit Factor: {self.metrics['profit_factor']:.2f}\n")
            
            # Calculate additional statistics
            if self.trades:
                # Calculate max drawdown
                cumulative_pl = 0
                max_drawdown = 0
                max_drawdown_money = 0
                peak = 0
                for trade in self.trades:
                    cumulative_pl += trade['profit_loss']
                    if cumulative_pl > peak:
                        peak = cumulative_pl
                    drawdown = peak - cumulative_pl
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                        max_drawdown_money = drawdown * 5  # $5 per point
                
                # Calculate win/loss streaks
                current_streak = 0
                max_win_streak = 0
                max_loss_streak = 0
                for trade in self.trades:
                    if trade['profit_loss'] > 0:
                        if current_streak > 0:
                            current_streak += 1
                        else:
                            current_streak = 1
                    else:
                        if current_streak < 0:
                            current_streak -= 1
                        else:
                            current_streak = -1
                    
                    if current_streak > max_win_streak:
                        max_win_streak = current_streak
                    elif current_streak < -max_loss_streak:
                        max_loss_streak = abs(current_streak)
                
                # Calculate average trade duration
                avg_duration = sum(trade['duration'] for trade in self.trades) / len(self.trades)
                
                # Calculate success rate by direction
                bullish_trades = [t for t in self.trades if t['direction'] == 'bullish']
                bearish_trades = [t for t in self.trades if t['direction'] == 'bearish']
                
                bullish_wins = sum(1 for t in bullish_trades if t['profit_loss'] > 0)
                bearish_wins = sum(1 for t in bearish_trades if t['profit_loss'] > 0)
                
                bullish_win_rate = (bullish_wins / len(bullish_trades) * 100) if bullish_trades else 0
                bearish_win_rate = (bearish_wins / len(bearish_trades) * 100) if bearish_trades else 0
                
                # Calculate exit type statistics
                tp_trades = [t for t in self.trades if t['exit_type'] == 'TP']
                sl_trades = [t for t in self.trades if t['exit_type'] == 'SL']
                time_trades = [t for t in self.trades if t['exit_type'] == 'TIME']
                
                tp_count = len(tp_trades)
                sl_count = len(sl_trades)
                time_count = len(time_trades)
                
                tp_win_rate = (sum(1 for t in tp_trades if t['profit_loss'] > 0) / tp_count * 100) if tp_count > 0 else 0
                sl_win_rate = (sum(1 for t in sl_trades if t['profit_loss'] > 0) / sl_count * 100) if sl_count > 0 else 0
                time_win_rate = (sum(1 for t in time_trades if t['profit_loss'] > 0) / time_count * 100) if time_count > 0 else 0
                
                # Write additional statistics
                f.write("\n=== Additional Statistics ===\n")
                f.write(f"Maximum Drawdown: {max_drawdown:.2f} points (${max_drawdown_money:.2f})\n")
                f.write(f"Longest Winning Streak: {max_win_streak}\n")
                f.write(f"Longest Losing Streak: {max_loss_streak}\n")
                f.write(f"Average Trade Duration: {avg_duration:.1f} minutes\n")
                f.write(f"Bullish Trades: {len(bullish_trades)} (Win Rate: {bullish_win_rate:.2f}%)\n")
                f.write(f"Bearish Trades: {len(bearish_trades)} (Win Rate: {bearish_win_rate:.2f}%)\n")
                
                # Write exit type statistics
                f.write("\n=== Exit Type Statistics ===\n")
                f.write(f"Take Profit (TP) Exits: {tp_count} ({tp_count/len(self.trades)*100:.1f}% of trades)\n")
                f.write(f"  - Win Rate: {tp_win_rate:.1f}%\n")
                f.write(f"  - Average P/L: {sum(t['profit_loss'] for t in tp_trades)/tp_count:.2f} points (${sum(t['profit_loss'] for t in tp_trades)*5/tp_count:.2f})\n")
                
                f.write(f"Stop Loss (SL) Exits: {sl_count} ({sl_count/len(self.trades)*100:.1f}% of trades)\n")
                f.write(f"  - Win Rate: {sl_win_rate:.1f}%\n")
                f.write(f"  - Average P/L: {sum(t['profit_loss'] for t in sl_trades)/sl_count:.2f} points (${sum(t['profit_loss'] for t in sl_trades)*5/sl_count:.2f})\n")
                
                f.write(f"Time-based Exits: {time_count} ({time_count/len(self.trades)*100:.1f}% of trades)\n")
                f.write(f"  - Win Rate: {time_win_rate:.1f}%\n")
                f.write(f"  - Average P/L: {sum(t['profit_loss'] for t in time_trades)/time_count:.2f} points (${sum(t['profit_loss'] for t in time_trades)*5/time_count:.2f})\n")
            
            # Write trade details
            f.write("\n=== Trade Details ===\n")
            for trade in self.trades:
                f.write(f"\nDate: {trade['date']}\n")
                f.write(f"Entry: {trade['entry_time']} @ {trade['entry_price']:.2f}\n")
                f.write(f"Exit: {trade['exit_time']} @ {trade['exit_price']:.2f}\n")
                f.write(f"Direction: {trade['direction']}\n")
                f.write(f"Exit Type: {trade['exit_type']}\n")
                f.write(f"Profit/Loss: {trade['profit_loss']:.2f} points (${trade['profit_loss'] * 5:.2f})\n")
                f.write(f"Duration: {trade['duration']:.1f} minutes\n")

def main():
    """
    Main execution function for the trade simulator.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Set up paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    backtest_dates_path = os.path.join(project_root, 'logs', 'backtest_dates.json')
    data_path = os.path.join(project_root, 'data', 'SPX_full_1min_CT.txt')
    pattern_db_path = os.path.join(project_root, 'data', 'processed', 'master_pattern_database.json')
    
    # Print input information
    print(f"Backtest Dates: {backtest_dates_path}")
    print(f"Price Data: {data_path}")
    print(f"Pattern Database: {pattern_db_path}")
    print(f"Filter Type: {args.filter if args.filter else 'recursive'}")
    
    # Initialize and run simulator
    simulator = TradeSimulator(backtest_dates_path, data_path, pattern_db_path, args.filter)
    simulator.run_simulation()
    
    # Print output information
    output_file = os.path.join(os.path.dirname(backtest_dates_path), 
                              f'simulation_summary_{args.filter if args.filter else "recursive"}.txt')
    print(f"Simulation Summary saved to {output_file}")

if __name__ == "__main__":
    main() 