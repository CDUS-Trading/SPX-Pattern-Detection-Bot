import os
import json
import time
import datetime
import threading
import pandas as pd
import re
import pytz
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Load environment variables from .env file
# First try to load from root directory, then from config directory
if os.path.exists(project_root / '.env'):
    load_dotenv(project_root / '.env')
elif os.path.exists(project_root / 'config' / '.env'):
    load_dotenv(project_root / 'config' / '.env')
else:
    print("WARNING: No .env file found. Please create one based on the template in config/env.template")

# Get Alpaca API credentials from environment variables
APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
APCA_API_BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')

# Check if credentials are loaded
if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
    raise ValueError("Alpaca API credentials not found. Please check your .env file.")

# Initialize Alpaca API connection
api = tradeapi.REST(
    APCA_API_KEY_ID,
    APCA_API_SECRET_KEY,
    APCA_API_BASE_URL,
    api_version='v2'
)

# Constants for trading
SPX_TO_SPY_RATIO = 10.0  # SPX is approximately 10x the value of SPY
SYMBOL = "SPY"  # S&P 500 ETF
TRADE_QTY = 20  # Number of contracts to trade
CENTRAL_TIMEZONE = pytz.timezone('US/Central')
EASTERN_TIMEZONE = pytz.timezone('US/Eastern')
CHECK_INTERVAL = 5  # Time in seconds between price checks

# Ensure logs directory exists
logs_dir = project_root / 'logs'
os.makedirs(logs_dir, exist_ok=True)
TRADE_LOG_FILE = logs_dir / 'paper_trades' / 'trade_log.csv'

# Position management - ensure only one position at a time
position_lock = threading.Lock()
active_position = False  # Global variable for position tracking

def is_market_open() -> bool:
    """
    Check if the market is currently open.
    
    Returns:
        bool: True if market is open, False otherwise
    """
    try:
        clock = api.get_clock()
        return clock.is_open
    except Exception as e:
        print(f"Error checking market status: {e}")
        return False

def validate_pattern_date(pattern_date: str) -> bool:
    """
    Validate that the pattern date is not in the future.
    
    Args:
        pattern_date (str): Pattern date in YYYY-MM-DD format
        
    Returns:
        bool: True if date is valid, False otherwise
    """
    try:
        pattern_dt = datetime.datetime.strptime(pattern_date, '%Y-%m-%d').date()
        current_date = datetime.datetime.now(CENTRAL_TIMEZONE).date()
        return pattern_dt <= current_date
    except Exception as e:
        print(f"Error validating pattern date: {e}")
        return False

def parse_pattern_file(file_path):
    """
    Parse the pattern file to extract trading signals and scale SPX values to SPY
    
    Args:
        file_path (str or Path): Path to the pattern file
        
    Returns:
        list: List of dictionaries containing signal data with SPY-scaled values
    """
    try:
        with open(file_path, 'r') as f:
            pattern_data = json.load(f)
        
        signals = []
        # Handle both list and dictionary formats
        if isinstance(pattern_data, dict):
            pattern_data = [pattern_data]  # Convert single day to list
            
        for day_data in pattern_data:
            # Validate pattern date
            pattern_date = day_data.get('pattern_date')
            if not pattern_date or not validate_pattern_date(pattern_date):
                print(f"Skipping patterns for future date: {pattern_date}")
                continue
                
            # Process patterns from each session
            for session in ['morning', 'mixed', 'afternoon']:
                session_patterns = day_data.get('patterns', {}).get('sessions', {}).get(session, [])
                for pattern in session_patterns:
                    # Scale down SPX values to SPY values
                    spy_tp = float(pattern['target_points']) / SPX_TO_SPY_RATIO
                    spy_sl = float(pattern['stop_loss_points']) / SPX_TO_SPY_RATIO
                    
                    # Remove any existing CT suffix before adding our own
                    entry_time = pattern['entry_time'].replace(' CT', '')
                    exit_time = pattern['exit_time'].replace(' CT', '')
                    
                    signals.append({
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "direction": pattern['direction'].upper(),
                        "tp": spy_tp,
                        "sl": spy_sl,
                        "success_rate": float(pattern['success_rate']),
                        "pattern_date": pattern_date
                    })
        
        print(f"Parsed {len(signals)} trading signals from {file_path}")
        print("\nScaled values for SPY trading:")
        for signal in signals:
            print(f"  Direction: {signal['direction']}")
            print(f"  TP: {signal['tp']:.2f} points (SPY)")
            print(f"  SL: {signal['sl']:.2f} points (SPY)")
            print("  ---")
        
        return signals
    
    except Exception as e:
        print(f"Error parsing pattern file: {e}")
        print(f"File path: {file_path}")
        print("File contents:")
        try:
            with open(file_path, 'r') as f:
                print(f.read())
        except Exception as read_error:
            print(f"Could not read file: {read_error}")
        return []

def convert_to_datetime(time_str, current_date=None):
    """
    Convert time string to datetime object
    
    Args:
        time_str (str): Time string in format "H:MM AM/PM"
        current_date (datetime.date): Date to use (defaults to today)
        
    Returns:
        datetime.datetime: Datetime object in Central Time
    """
    if current_date is None:
        current_date = datetime.datetime.now(CENTRAL_TIMEZONE).date()
    
    # Convert from "H:MM AM/PM" to datetime
    time_format = "%I:%M%p"
    time_str = time_str.replace(' ', '')  # Remove any spaces
    time_obj = datetime.datetime.strptime(time_str, time_format).time()
    
    # Combine with current date and set timezone
    dt = datetime.datetime.combine(current_date, time_obj)
    return CENTRAL_TIMEZONE.localize(dt)

def get_current_price():
    """
    Get the current price of SPX
    
    Returns:
        float: Current price of SPX
    """
    try:
        # Get last trade
        last_trade = api.get_latest_trade(SYMBOL)
        return float(last_trade.price)
    except Exception as e:
        print(f"Error getting current price: {e}")
        return None

def has_open_positions():
    """
    Check if there are any open positions for the symbol
    
    Returns:
        bool: True if there are open positions, False otherwise
    """
    try:
        positions = api.list_positions()
        for position in positions:
            if position.symbol == SYMBOL:
                return True
        return False
    except Exception as e:
        print(f"Error checking positions: {e}")
        return False

def place_market_order(signal):
    """
    Place a market order according to the signal
    
    Args:
        signal (dict): Trading signal
        
    Returns:
        dict: Order information
    """
    global active_position
    
    try:
        # Check if we already have an open position
        with position_lock:
            if active_position:
                print(f"Already have an active position for {SYMBOL}. Skipping this signal.")
                return None
            
            # Also double check with the API
            if has_open_positions():
                print(f"Open position detected for {SYMBOL}. Skipping this signal.")
                return None
            
            # Mark that we're starting a position
            active_position = True
        
        side = 'buy' if signal['direction'] == 'BUY' else 'sell'
        
        # Place market order
        order = api.submit_order(
            symbol=SYMBOL,
            qty=TRADE_QTY,
            side=side,
            type='market',
            time_in_force='day'
        )
        
        print(f"Market order placed: {side.upper()} {TRADE_QTY} {SYMBOL}")
        
        # Get entry price after order is filled
        filled_order = wait_for_order_fill(order.id)
        entry_price = float(filled_order.filled_avg_price) if filled_order else None
        
        return {
            'order_id': order.id,
            'entry_time': datetime.datetime.now(CENTRAL_TIMEZONE),
            'entry_price': entry_price,
            'side': side
        }
    
    except Exception as e:
        print(f"Error placing market order: {e}")
        # Reset position flag if order failed
        with position_lock:
            active_position = False
        return None

def wait_for_order_fill(order_id, timeout=60):
    """
    Wait for an order to be filled
    
    Args:
        order_id (str): Order ID
        timeout (int): Timeout in seconds
        
    Returns:
        Order: Filled order or None if timeout
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        order = api.get_order(order_id)
        if order.status == 'filled':
            return order
        time.sleep(1)
    
    print(f"Order {order_id} not filled within {timeout} seconds")
    return None

def close_position(symbol):
    """
    Close the position for the given symbol
    
    Args:
        symbol (str): Symbol to close position for
        
    Returns:
        dict: Order information
    """
    global active_position
    
    try:
        # Get current position
        position = api.get_position(symbol)
        
        # Determine side for closing position
        side = 'sell' if position.side == 'long' else 'buy'
        
        # Place market order to close position
        order = api.submit_order(
            symbol=symbol,
            qty=abs(float(position.qty)),
            side=side,
            type='market',
            time_in_force='day'
        )
        
        print(f"Position closed: {side.upper()} {abs(float(position.qty))} {symbol}")
        
        # Get exit price after order is filled
        filled_order = wait_for_order_fill(order.id)
        exit_price = float(filled_order.filled_avg_price) if filled_order else None
        
        # Mark position as closed
        with position_lock:
            active_position = False
        
        return {
            'order_id': order.id,
            'exit_time': datetime.datetime.now(CENTRAL_TIMEZONE),
            'exit_price': exit_price
        }
    
    except Exception as e:
        print(f"Error closing position: {e}")
        # Reset position flag to ensure we don't get stuck
        with position_lock:
            active_position = False
        return None

def monitor_position(signal, trade_info):
    """
    Monitor the position and exit based on conditions
    
    Args:
        signal (dict): Trading signal
        trade_info (dict): Trade information
        
    Returns:
        dict: Updated trade information
    """
    global active_position
    
    entry_price = trade_info['entry_price']
    if not entry_price:
        print("Entry price not available, cannot monitor position")
        # Reset position flag
        with position_lock:
            active_position = False
        return trade_info
    
    # Calculate target and stop prices
    if signal['direction'] == 'BUY':
        target_price = entry_price + signal['tp']
        stop_price = entry_price - signal['sl']
    else:  # SELL
        target_price = entry_price - signal['tp']
        stop_price = entry_price + signal['sl']
    
    # Convert exit time to datetime
    exit_time = convert_to_datetime(signal['exit_time'])
    # Set exit time 1 minute earlier to account for order execution delays
    exit_time = exit_time - datetime.timedelta(minutes=1)
    
    print(f"Monitoring position...")
    print(f"Entry Price: {entry_price}")
    print(f"Target Price: {target_price}")
    print(f"Stop Price: {stop_price}")
    print(f"Exit Time: {exit_time.strftime('%Y-%m-%d %H:%M:%S')} CT (1 min early to allow for order execution)")
    
    exit_reason = None
    
    while True:
        try:
            current_time = datetime.datetime.now(CENTRAL_TIMEZONE)
            
            # Check if it's time to exit
            if current_time >= exit_time:
                print("Exit time reached (1 min early), closing position")
                exit_reason = "TIME"
                break
            
            # Check if we're approaching market close (2 minutes before)
            market_close = current_time.replace(hour=15, minute=0, second=0, microsecond=0)
            if current_time >= market_close - datetime.timedelta(minutes=2):
                print("Approaching market close, closing position")
                exit_reason = "MARKET_CLOSE"
                break
            
            # Get current price
            current_price = get_current_price()
            if not current_price:
                print("Could not get current price, retrying...")
                time.sleep(CHECK_INTERVAL)
                continue
            
            print(f"Current price: {current_price}")
            
            # Check if target or stop price is hit
            if signal['direction'] == 'BUY':
                if current_price >= target_price:
                    print("Take profit target reached, closing position")
                    exit_reason = "TP"
                    break
                elif current_price <= stop_price:
                    print("Stop loss triggered, closing position")
                    exit_reason = "SL"
                    break
            else:  # SELL
                if current_price <= target_price:
                    print("Take profit target reached, closing position")
                    exit_reason = "TP"
                    break
                elif current_price >= stop_price:
                    print("Stop loss triggered, closing position")
                    exit_reason = "SL"
                    break
            
            time.sleep(CHECK_INTERVAL)
        
        except Exception as e:
            print(f"Error in position monitoring: {e}")
            time.sleep(CHECK_INTERVAL)
    
    # Close position
    exit_info = close_position(SYMBOL)
    
    if exit_info:
        trade_info.update(exit_info)
        trade_info['exit_reason'] = exit_reason
        
        # Calculate PnL
        if signal['direction'] == 'BUY':
            trade_info['pnl'] = trade_info['exit_price'] - trade_info['entry_price']
        else:  # SELL
            trade_info['pnl'] = trade_info['entry_price'] - trade_info['exit_price']
    
    return trade_info

def log_trade(trade_info, signal):
    """
    Log trade information to CSV file
    
    Args:
        trade_info (dict): Trade information
        signal (dict): Signal information
    """
    # Create log file with headers if it doesn't exist
    if not os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, 'w') as f:
            f.write("date,entry_time,exit_time,direction,entry_price,exit_price,tp,sl,exit_reason,pnl\n")
    
    # Format data for CSV
    date = trade_info['entry_time'].strftime('%Y-%m-%d')
    entry_time = trade_info['entry_time'].strftime('%H:%M:%S')
    exit_time = trade_info['exit_time'].strftime('%H:%M:%S') if 'exit_time' in trade_info else ''
    direction = signal['direction']
    entry_price = trade_info['entry_price']
    exit_price = trade_info['exit_price'] if 'exit_price' in trade_info else ''
    tp = signal['tp']
    sl = signal['sl']
    exit_reason = trade_info.get('exit_reason', '')
    pnl = trade_info.get('pnl', '')
    
    # Write to CSV
    with open(TRADE_LOG_FILE, 'a') as f:
        f.write(f"{date},{entry_time},{exit_time},{direction},{entry_price},{exit_price},{tp},{sl},{exit_reason},{pnl}\n")
    
    print(f"Trade logged to {TRADE_LOG_FILE}")

def execute_trading_plan(signal):
    """
    Execute a trading plan for a signal
    
    Args:
        signal (dict): Trading signal
    """
    # Check if market is open
    if not is_market_open():
        print("Market is currently closed. Skipping signal.")
        return
        
    # Convert times to datetime objects
    entry_time = convert_to_datetime(signal['entry_time'])
    
    # Wait until entry time
    now = datetime.datetime.now(CENTRAL_TIMEZONE)
    while entry_time > now:
        wait_seconds = (entry_time - now).total_seconds()
        print(f"Waiting {wait_seconds:.2f} seconds until entry time: {entry_time.strftime('%Y-%m-%d %H:%M:%S')} CT")
        time.sleep(min(wait_seconds, 60))  # Sleep for 60 seconds or the remaining time, whichever is smaller
        now = datetime.datetime.now(CENTRAL_TIMEZONE)
        
        # Check if market is still open
        if not is_market_open():
            print("Market closed while waiting for entry time. Skipping signal.")
            return
    
    # Place order at entry time
    trade_info = place_market_order(signal)
    
    if not trade_info:
        print("Failed to place order or position already exists, skipping signal")
        return
    
    # Monitor position
    trade_info = monitor_position(signal, trade_info)
    
    # Log trade
    log_trade(trade_info, signal)

def safety_net_close_all_positions():
    """
    Safety net: Close all open positions 2 minutes before market close (3:00 PM CT).
    Runs in a background thread.
    """
    while True:
        now = datetime.datetime.now(CENTRAL_TIMEZONE)
        # Calculate today's market close (3:00 PM CT)
        market_close = now.replace(hour=15, minute=0, second=0, microsecond=0)
        safety_net_time = market_close - datetime.timedelta(minutes=2)
        sleep_seconds = (safety_net_time - now).total_seconds()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        try:
            print("[Safety Net] Closing all open positions 2 minutes before market close...")
            positions = api.list_positions()
            for position in positions:
                try:
                    print(f"[Safety Net] Closing position for {position.symbol}")
                    close_position(position.symbol)
                except Exception as e:
                    print(f"[Safety Net] Error closing position for {position.symbol}: {e}")
            print("[Safety Net] All positions closed.")
        except Exception as e:
            print(f"[Safety Net] Error listing positions: {e}")
        # Sleep until next day
        time.sleep(60 * 60)  # 1 hour, then re-check

def main():
    print("SPX Pattern Trading Bot initialized.")
    
    # Check if market is open
    if not is_market_open():
        print("Market is currently closed. Please run this script during market hours.")
        return
    
    file_path = logs_dir / "current_detected_patterns.json"
    print(f"Using pattern file from logs directory: {file_path}")
    
    # Parse pattern file
    signals = parse_pattern_file(file_path)
    
    if not signals:
        print("No valid trading signals found in the pattern file.")
        return
    
    # Get current date
    current_date = datetime.datetime.now(CENTRAL_TIMEZONE).date()
    
    # Display the signals
    for i, signal in enumerate(signals):
        print(f"\nSignal {i+1}:")
        print(f"  Pattern Date: {signal['pattern_date']}")
        print(f"  Entry Time: {signal['entry_time']} CT")
        print(f"  Exit Time: {signal['exit_time']} CT")
        print(f"  Direction: {signal['direction']}")
        print(f"  Take Profit: {signal['tp']:.2f} points (SPY)")
        print(f"  Stop Loss: {signal['sl']:.2f} points (SPY)")
        print(f"  Success Rate: {signal['success_rate']}%")
    
    # Make sure we don't have any active positions from a previous run
    try:
        positions = api.list_positions()
        for position in positions:
            if position.symbol == SYMBOL:
                print(f"Closing existing position for {SYMBOL} before starting new signals")
                close_position(SYMBOL)
                break
    except Exception as e:
        print(f"Error checking existing positions: {e}")
    
    # Sort signals by entry time
    signals.sort(key=lambda x: convert_to_datetime(x['entry_time']))
    
    # Execute trading plans for each signal in sequence
    for i, signal in enumerate(signals):
        print(f"\nExecuting trading plan for Signal {i+1}...")
        execute_trading_plan(signal)

if __name__ == "__main__":
    # Start safety net thread
    safety_thread = threading.Thread(target=safety_net_close_all_positions, daemon=True)
    safety_thread.start()
    try:
        account = api.get_account()
        print(f"Successfully connected to Alpaca API!")
        print(f"Account status: {account.status}")
        print(f"Cash balance: ${float(account.cash)}")
        print(f"Portfolio value: ${float(account.portfolio_value)}")
        main()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected! Attempting to close any open positions...")
        try:
            positions = api.list_positions()
            for position in positions:
                if position.symbol == SYMBOL:
                    print(f"Closing open position for {SYMBOL} due to interruption.")
                    close_position(SYMBOL)
        except Exception as e:
            print(f"Error while closing position on interrupt: {e}")
        print("Exiting safely.")
    except Exception as e:
        print(f"Error connecting to Alpaca API: {e}")
