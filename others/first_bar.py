import pandas as pd
from datetime import datetime, timedelta

# Load the data
data = pd.read_csv('SPX_full_1min.txt', 
                   header=None, 
                   names=['timestamp', 'open', 'high', 'low', 'close'], 
                   parse_dates=['timestamp'])

# Filter data for the past three years
three_years_ago = datetime.now() - timedelta(days=3*365)
data = data[data['timestamp'] >= three_years_ago]

# Function to determine if a bar is green or red
def is_green_bar(open_price, close_price):
    return close_price > open_price

# Initialize statistics
def analyze_trades(trades):
    total_trades = len(trades)
    wins = sum(1 for trade in trades if trade > 0)
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0
    average_win = sum(trade for trade in trades if trade > 0) / wins if wins > 0 else 0
    average_loss = sum(trade for trade in trades if trade < 0) / losses if losses > 0 else 0
    risk_reward_ratio = average_win / abs(average_loss) if average_loss != 0 else float('inf')
    win_loss_sequence = ''.join('W' if trade > 0 else 'L' for trade in trades)
    
    return total_trades, win_rate, risk_reward_ratio, win_loss_sequence

# Analyze the data
for bar_type in ['first_minute', 'first_five_minute']:
    for hold_hours in [1, 2, 3]:
        trades = []
        for date, group in data.groupby(data['timestamp'].dt.date):
            if group.empty:
                continue  # Skip days with no data

            # Determine the bar to analyze
            if bar_type == 'first_minute':
                bar = group.iloc[0]
                open_price = bar['open']
                close_price = bar['close']
            elif bar_type == 'first_five_minute' and len(group) >= 5:
                bar = group.iloc[:5]
                open_price = bar['open'].iloc[0]
                close_price = bar['close'].iloc[-1]
            else:
                continue

            # Determine trade direction
            if is_green_bar(open_price, close_price):
                # Simulate a buy trade
                entry_price = close_price
            else:
                # Simulate a short trade
                entry_price = open_price

            # Calculate exit price after holding for specified hours
            exit_time = group['timestamp'].iloc[0] + timedelta(hours=hold_hours)
            exit_group = group[group['timestamp'] <= exit_time]
            if not exit_group.empty:
                exit_price = exit_group['close'].iloc[-1]
                trade_result = exit_price - entry_price if is_green_bar(open_price, close_price) else entry_price - exit_price
                trades.append(trade_result)

        # Calculate and print statistics
        total_trades, win_rate, risk_reward_ratio, win_loss_sequence = analyze_trades(trades)
        print(f"Bar Type: {bar_type}, Hold Hours: {hold_hours}")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Risk-Reward Ratio: {risk_reward_ratio:.2f}")
        print(f"Win/Loss Sequence: {win_loss_sequence}\n")

