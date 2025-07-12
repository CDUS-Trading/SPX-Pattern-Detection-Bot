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
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from queue import Queue
from typing import Optional, Dict, Any
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'paper_trades' / 'trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Ensure logs directory exists
logs_dir = project_root / 'logs'
os.makedirs(logs_dir, exist_ok=True)
TRADE_LOG_FILE = logs_dir / 'paper_trades' / 'trade_log.csv'

# Position management - ensure only one position at a time
position_lock = threading.Lock()
active_position = False  # Global variable for position tracking

# Add signal queue for consecutive orders
signal_queue = Queue()
current_signal: Optional[Dict[str, Any]] = None
scheduler = BackgroundScheduler()

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_api_call(func, *args, **kwargs):
    """
    Safely execute an API call with retries
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"API call failed: {e}")
        raise

def is_market_open() -> bool:
    """
    Check if the market is currently open.
    
    Returns:
        bool: True if market is open, False otherwise
    """
    try:
        clock = safe_api_call(api.get_clock)
        return clock.is_open
    except Exception as e:
        logger.error(f"Error checking market status: {e}")
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

def process_next_signal():
    """
    Process the next signal in the queue if available
    """
    global current_signal
    
    if not signal_queue.empty():
        next_signal = signal_queue.get()
        current_signal = next_signal
        print(f"\nProcessing next signal in queue:")
        print(f"  Entry Time: {next_signal['entry_time']} CT")
        print(f"  Exit Time: {next_signal['exit_time']} CT")
        print(f"  Direction: {next_signal['direction']}")
        execute_trading_plan(next_signal)
    else:
        print("\nNo more signals in queue")
        current_signal = None

def place_market_order(signal):
    """
    Place a market order with OCO (One-Cancels-Other) for TP/SL
    
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
                logger.warning(f"Already have an active position for {SYMBOL}. Skipping this signal.")
                return None
            
            # Also double check with the API
            if has_open_positions():
                logger.warning(f"Open position detected for {SYMBOL}. Skipping this signal.")
                return None
            
            # Mark that we're starting a position
            active_position = True
        
        side = 'buy' if signal['direction'] == 'BUY' else 'sell'
        
        # Place market order
        order = safe_api_call(
            api.submit_order,
            symbol=SYMBOL,
            qty=TRADE_QTY,
            side=side,
            type='market',
            time_in_force='day'
        )
        
        logger.info(f"Market order placed: {side.upper()} {TRADE_QTY} {SYMBOL}")
        
        # Get entry price after order is filled
        filled_order = wait_for_order_fill(order.id)
        if not filled_order:
            logger.error("Market order not filled within timeout")
            with position_lock:
                active_position = False
            return None
            
        entry_price = float(filled_order.filled_avg_price)
        
        # Verify position was opened correctly
        position = safe_api_call(api.get_position, SYMBOL)
        if not validate_position(position):
            logger.error("Position validation failed after order fill")
            with position_lock:
                active_position = False
            return None
        
        # Calculate TP and SL prices
        if side == 'buy':
            tp_price = entry_price + signal['tp']
            sl_price = entry_price - signal['sl']
        else:
            tp_price = entry_price - signal['tp']
            sl_price = entry_price + signal['sl']
        
        # Place OCO orders
        try:
            oco_order = safe_api_call(
                api.submit_order,
                symbol=SYMBOL,
                qty=TRADE_QTY,
                side='sell' if side == 'buy' else 'buy',
                type='limit',
                time_in_force='day',
                limit_price=tp_price,
                stop_price=sl_price,
                stop_limit_price=sl_price,
                client_order_id=f"oco_{order.id}"
            )
            
            logger.info(f"OCO orders placed - TP: {tp_price}, SL: {sl_price}")
            
            # Set up callback for OCO order fill
            def check_oco_fill():
                while True:
                    try:
                        oco_status = safe_api_call(api.get_order, oco_order.id)
                        if oco_status.status == 'filled':
                            logger.info(f"OCO order filled: {oco_status.side} at {oco_status.filled_avg_price}")
                            # Process next signal after a short delay
                            time.sleep(2)  # Wait for orders to settle
                            process_next_signal()
                            break
                        time.sleep(1)
                    except Exception as e:
                        logger.error(f"Error checking OCO order status: {e}")
                        break
            
            # Start OCO monitoring in a separate thread
            oco_thread = threading.Thread(target=check_oco_fill, daemon=True)
            oco_thread.start()
            
        except Exception as e:
            logger.error(f"Error placing OCO orders: {e}")
            # Try to close the position if OCO orders fail
            try:
                close_position(SYMBOL)
            except Exception as close_error:
                logger.error(f"Error closing position after OCO failure: {close_error}")
            return None
        
        return {
            'order_id': order.id,
            'entry_time': datetime.datetime.now(CENTRAL_TIMEZONE),
            'entry_price': entry_price,
            'side': side
        }
    
    except Exception as e:
        logger.error(f"Error placing market order: {e}")
        # Reset position flag if order failed
        with position_lock:
            active_position = False
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

def validate_position(position):
    """
    Validate that a position is properly filled
    
    Args:
        position: Position object from Alpaca API
        
    Returns:
        bool: True if position is valid, False otherwise
    """
    try:
        if not position or not position.qty:
            return False
        if float(position.qty) != TRADE_QTY:
            logger.warning(f"Position quantity mismatch: {position.qty} != {TRADE_QTY}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating position: {e}")
        return False

def schedule_exit_time(signal, trade_info):
    """
    Schedule a position exit at the specified time and queue the next signal
    
    Args:
        signal (dict): Trading signal
        trade_info (dict): Trade information
    """
    exit_time = convert_to_datetime(signal['exit_time'])
    # Set exit time 1 minute earlier to account for order execution delays
    exit_time = exit_time - datetime.timedelta(minutes=1)
    
    def exit_position():
        try:
            if has_open_positions():
                print(f"Scheduled exit time reached for {SYMBOL}, closing position")
                exit_info = close_position(SYMBOL)
                if exit_info:
                    trade_info.update(exit_info)
                    trade_info['exit_reason'] = "TIME"
                    # Calculate PnL
                    if signal['direction'] == 'BUY':
                        trade_info['pnl'] = trade_info['exit_price'] - trade_info['entry_price']
                    else:  # SELL
                        trade_info['pnl'] = trade_info['entry_price'] - trade_info['exit_price']
                    # Log the trade
                    log_trade(trade_info, signal)
                    
                    # Process next signal after a short delay
                    time.sleep(2)  # Wait for orders to settle
                    process_next_signal()
        except Exception as e:
            print(f"Error in scheduled exit: {e}")
    
    # Schedule the exit
    scheduler.add_job(exit_position, DateTrigger(run_date=exit_time))

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
    
    # Schedule the time-based exit
    schedule_exit_time(signal, trade_info)

def safety_net_close_all_positions():
    """
    Safety net: Close all open positions 2 minutes before market close (3:00 PM CT).
    Runs in a background thread.
    """
    while True:
        try:
            now = datetime.datetime.now(CENTRAL_TIMEZONE)
            # Calculate today's market close (3:00 PM CT)
            market_close = now.replace(hour=15, minute=0, second=0, microsecond=0)
            safety_net_time = market_close - datetime.timedelta(minutes=2)
            sleep_seconds = (safety_net_time - now).total_seconds()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            
            logger.info("[Safety Net] Closing all open positions 2 minutes before market close...")
            positions = safe_api_call(api.list_positions)
            for position in positions:
                try:
                    logger.info(f"[Safety Net] Closing position for {position.symbol}")
                    close_position(position.symbol)
                except Exception as e:
                    logger.error(f"[Safety Net] Error closing position for {position.symbol}: {e}")
            logger.info("[Safety Net] All positions closed.")
            
            # Shutdown scheduler at market close
            try:
                scheduler.shutdown()
            except Exception as e:
                logger.error(f"[Safety Net] Error shutting down scheduler: {e}")
                
        except Exception as e:
            logger.error(f"[Safety Net] Error in safety net: {e}")
        
        # Sleep until next day
        time.sleep(60 * 60)  # 1 hour, then re-check

def main():
    logger.info("SPX Pattern Trading Bot initialized.")
    
    # Check if market is open
    if not is_market_open():
        logger.warning("Market is currently closed. Please run this script during market hours.")
        return
    
    file_path = logs_dir / "current_detected_patterns.json"
    logger.info(f"Using pattern file from logs directory: {file_path}")
    
    # Parse pattern file
    signals = parse_pattern_file(file_path)
    
    if not signals:
        logger.warning("No valid trading signals found in the pattern file.")
        return
    
    # Get current date
    current_date = datetime.datetime.now(CENTRAL_TIMEZONE).date()
    
    # Display the signals
    for i, signal in enumerate(signals):
        logger.info(f"\nSignal {i+1}:")
        logger.info(f"  Pattern Date: {signal['pattern_date']}")
        logger.info(f"  Entry Time: {signal['entry_time']} CT")
        logger.info(f"  Exit Time: {signal['exit_time']} CT")
        logger.info(f"  Direction: {signal['direction']}")
        logger.info(f"  Take Profit: {signal['tp']:.2f} points (SPY)")
        logger.info(f"  Stop Loss: {signal['sl']:.2f} points (SPY)")
        logger.info(f"  Success Rate: {signal['success_rate']}%")
    
    # Make sure we don't have any active positions from a previous run
    try:
        positions = safe_api_call(api.list_positions)
        for position in positions:
            if position.symbol == SYMBOL:
                logger.info(f"Closing existing position for {SYMBOL} before starting new signals")
                close_position(SYMBOL)
                break
    except Exception as e:
        logger.error(f"Error checking existing positions: {e}")
    
    # Sort signals by entry time
    signals.sort(key=lambda x: convert_to_datetime(x['entry_time']))
    
    # Queue all signals
    for signal in signals:
        signal_queue.put(signal)
    
    # Start the scheduler
    try:
        scheduler.start()
    except Exception as e:
        logger.error(f"Error starting scheduler: {e}")
        return
    
    # Process the first signal
    process_next_signal()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        try:
            scheduler.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}")

if __name__ == "__main__":
    # Start safety net thread
    safety_thread = threading.Thread(target=safety_net_close_all_positions, daemon=True)
    safety_thread.start()
    try:
        account = safe_api_call(api.get_account)
        logger.info(f"Successfully connected to Alpaca API!")
        logger.info(f"Account status: {account.status}")
        logger.info(f"Cash balance: ${float(account.cash)}")
        logger.info(f"Portfolio value: ${float(account.portfolio_value)}")
        main()
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt detected! Attempting to close any open positions...")
        try:
            positions = safe_api_call(api.list_positions)
            for position in positions:
                if position.symbol == SYMBOL:
                    logger.info(f"Closing open position for {SYMBOL} due to interruption.")
                    close_position(SYMBOL)
        except Exception as e:
            logger.error(f"Error while closing position on interrupt: {e}")
        logger.info("Exiting safely.")
    except Exception as e:
        logger.error(f"Error connecting to Alpaca API: {e}")
