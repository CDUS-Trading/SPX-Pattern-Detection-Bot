# SPX Pattern Automated Trading Bot

This bot automatically executes trades for SPX based on pattern signals using Alpaca's paper trading API.

## Features

- Automatically executes trades based on pattern signals from pattern files
- Maintains only one position at a time to prevent conflicting trades
- Monitors positions for take profit, stop loss, and time-based exits
- Logs trade details to a CSV file for record keeping
- Processes signals sequentially based on entry time

## Prerequisites

- Python 3.x
- Alpaca Paper Trading account

## Setup

1. Clone this repository
2. Install required packages:
   ```
   pip install python-dotenv alpaca-trade-api pandas pytz
   ```
3. Create a `.env` file in the root directory with your Alpaca API credentials:
   ```
   APCA_API_KEY_ID=your_api_key_here
   APCA_API_SECRET_KEY=your_api_secret_here
   APCA_API_BASE_URL=https://paper-api.alpaca.markets
   ```

## Usage

1. Ensure you have a pattern file in the format `moderate_patterns_YYYYMMDD.txt` in the root directory.
2. Run the bot:
   ```
   python src/simulator/main.py
   ```
3. The bot will:
   - Parse the most recent pattern file
   - Sort signals by entry time
   - Execute trades sequentially (one position at a time)
   - Monitor positions and exit based on TP/SL or exit time
   - Log all trades to `logs/trade_log.csv`

## Pattern File Format

The bot expects pattern files in the following format:

```
===== Action Plan =====
Entry: 10:30AM CT
Exit: 12:30PM CT
Direction: Sell 📉
TP: 26.54 points
SL: 16.59 points
Success Rate: 78.95%
------------------------------
```

## Project Structure

```
minute_data_patterns/
├── config/         # Configuration files
├── data/           # Market data files
├── logs/           # Trade logs
├── src/
│   ├── simulator/  # Trading simulator
│   │   └── main.py # Main trading bot script
│   ├── patterns/   # Pattern generation modules
│   └── utils/      # Utility functions
└── README.md
```

## Configuration

You can modify the following constants in `src/simulator/main.py`:

- `SYMBOL`: The symbol to trade (default: "SPY")
- `TRADE_QTY`: Number of contracts to trade (default: 1)
- `CHECK_INTERVAL`: Time in seconds between price checks (default: 5)

## Position Management

The bot implements position management to ensure only one position is active at a time:

- Before placing an order, the bot checks if there's already an active position
- Trades are processed sequentially in order of entry time
- If an existing position is found at startup, it will be closed before starting new trades
- Thread-safe locks prevent race conditions when checking/updating position status

## Logs

All trades are logged to `logs/trade_log.csv` with the following information:

- Date
- Entry time
- Exit time
- Direction (BUY/SELL)
- Entry price
- Exit price
- Take profit points
- Stop loss points
- Exit reason (TP/SL/TIME)
- PnL

## Disclaimer

This bot is for educational purposes only. Trading involves risk, and you should never trade with money you cannot afford to lose. Always test thoroughly using paper trading before using real money. 