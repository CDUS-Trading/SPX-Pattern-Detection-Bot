import pandas as pd


def analyze_trade_logs(file_path, output_path):
    # Read the trade logs
    trades = pd.read_csv(file_path)
    
    # Calculate percentage returns
    trades['Percentage Return'] = ((trades['exit_price'] - trades['entry_price']) / trades['entry_price']) * 100
    
    # Calculate percentage increases
    trades['Percentage Increase'] = ((trades['exit_price'] - trades['entry_price']) / trades['entry_price']) * 100
    
    # Calculate other statistics
    total_trades = len(trades)
    total_profit = trades['pnl'].sum()
    average_return = trades['Percentage Return'].mean()
    
    # Calculate non-market-related insights
    profitable_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] < 0]
    
    percentage_profitable = (len(profitable_trades) / total_trades) * 100
    total_profit = profitable_trades['pnl'].sum()
    total_loss = losing_trades['pnl'].sum()
    profit_loss_ratio = total_profit / abs(total_loss) if total_loss != 0 else float('inf')
    
    average_profit = profitable_trades['pnl'].mean()
    average_loss = losing_trades['pnl'].mean()
    risk_reward_ratio = average_profit / abs(average_loss) if average_loss != 0 else float('inf')
    
    # Calculate additional insights
    max_drawdown = (trades['pnl'].cumsum().min() - trades['pnl'].cumsum().max()) / trades['pnl'].cumsum().max()
    win_rate = (len(profitable_trades) / total_trades) * 100
    loss_rate = (len(losing_trades) / total_trades) * 100
    
    # Assuming 'entry_time' and 'exit_time' are in 'HH:MM:SS' format
    trades['entry_time'] = pd.to_datetime(trades['entry_time'], format='%H:%M:%S')
    trades['exit_time'] = pd.to_datetime(trades['exit_time'], format='%H:%M:%S')
    trades['holding_time'] = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 3600  # in hours
    average_holding_time = trades['holding_time'].mean()
    
    volatility = trades['Percentage Return'].std()
    sharpe_ratio = (average_return / volatility) if volatility != 0 else float('inf')
    
    # Prepare the output
    with open(output_path, 'w') as f:
        f.write("Comprehensive Trade Analysis:\n")
        f.write(f"Total Trades: {total_trades}\n")
        f.write(f"Total Profit/Loss: {total_profit}\n")
        f.write(f"Average Percentage Return: {average_return:.2f}%\n")
        f.write(f"Percentage of Profitable Trades: {percentage_profitable:.2f}%\n")
        f.write(f"Profit/Loss Ratio: {profit_loss_ratio:.2f}\n")
        f.write(f"Risk/Reward Ratio: {risk_reward_ratio:.2f}\n")
        f.write(f"Maximum Drawdown: {max_drawdown:.2f}\n")
        f.write(f"Win Rate: {win_rate:.2f}%\n")
        f.write(f"Loss Rate: {loss_rate:.2f}%\n")
        f.write(f"Average Holding Time: {average_holding_time:.2f} hours\n")
        f.write(f"Volatility: {volatility:.2f}\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")


# Example usage
analyze_trade_logs('logs/paper_trades/paper_trade_log.csv', 'logs/paper_trades/paper_trade_analysis.txt') 