import argparse
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import os
import json

def is_market_open(date):
    # For historical data in our file, we'll just check weekends
    if date.weekday() in [5, 6]:
        return False
    return True

def plot_stock_chart(date_str, show_consecutive=False, highlight_start=None, highlight_end=None, tp=10, sl=20, direction=None):
    try:
        # Convert date string to datetime (now accepting YYYY-MM-DD format)
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            # Fallback to mm/dd/yyyy format for backward compatibility
            date = datetime.strptime(date_str, '%m/%d/%Y')
        
        if not is_market_open(date):
            print(f"Stock market was closed on {date_str}")
            return
        
        # Read the data file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, '../../data/SPX_full_1min_CT.txt')
        df = pd.read_csv(data_path, header=None, names=['datetime', 'Open', 'High', 'Low', 'Close'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        # Assume the data is already in CT. If not, convert here.
        df['datetime'] = df['datetime'].dt.tz_localize('America/Chicago')
        
        # If show_consecutive is True, find the previous market day
        if show_consecutive:
            prev_date = date
            while True:
                prev_date = prev_date - pd.Timedelta(days=1)
                if is_market_open(prev_date):
                    break
            
            # Plot both days (previous day first, then current day)
            plot_single_day(df, prev_date, highlight_start, highlight_end, tp, sl, direction)
            plot_single_day(df, date, highlight_start, highlight_end, tp, sl, direction)
        else:
            # Plot just the requested day
            plot_single_day(df, date, highlight_start, highlight_end, tp, sl, direction)
            
    except ValueError:
        print("Please enter date in format YYYY-MM-DD")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def plot_single_day(df, date, highlight_start=None, highlight_end=None, tp=10, sl=20, direction=None):
    """Helper function to plot a single day's chart, with optional TP/SL visualization"""
    # Filter for the requested date and previous day
    current_day_mask = df['datetime'].dt.date == date.date()
    prev_day_mask = df['datetime'].dt.date < date.date()
    
    spx = df[current_day_mask].copy()
    prev_day_data = df[prev_day_mask]
    
    if len(spx) == 0:
        print(f"No data available for {date.strftime('%m/%d/%Y')}")
        return
            
    # Get previous day's close if available
    if len(prev_day_data) > 0:
        prev_close = prev_day_data['Close'].iloc[-1]
        overnight_change = spx['Open'].iloc[0] - prev_close
        overnight_pct = (overnight_change / prev_close) * 100
    else:
        overnight_change = None
        overnight_pct = None
    
    spx.set_index('datetime', inplace=True)
    
    # Calculate key points
    day_high = spx['High'].max()
    day_low = spx['Low'].min()
    day_open = spx['Open'].iloc[0]
    day_close = spx['Close'].iloc[-1]
    price_change = day_close - day_open
    pct_change = (price_change / day_open) * 100
    
    # Short title
    title_text = f"SPX {date.strftime('%m/%d/%Y')} | Day Change: {price_change:+.2f} ({pct_change:+.2f}%)"
    
    # Set colors
    is_up_day = day_close >= day_open
    bg_color = 'rgba(230, 243, 230, 0.3)' if is_up_day else 'rgba(247, 230, 230, 0.3)'
    price_line_color = '#4B0082'  # Deep indigo
    high_color = '#2ecc71'    # Emerald green
    low_color = '#e74c3c'     # Soft red
    open_color = '#3498db'    # Bright blue
    close_color = '#9b59b6'   # Purple
    
    # Create figure
    fig = go.Figure()
    
    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=spx.index,
        open=spx['Open'],
        high=spx['High'],
        low=spx['Low'],
        close=spx['Close'],
        name='OHLC',
        increasing_line_color='#2ecc71',  # green
        decreasing_line_color='#e74c3c',  # red
        showlegend=False
    ))
    
    # Add horizontal lines for OHLC
    for price, color, name in [
        (day_high, high_color, f'High: {day_high:.2f}'),
        (day_low, low_color, f'Low: {day_low:.2f}'),
        (day_open, open_color, f'Open: {day_open:.2f}'),
        (day_close, close_color, f'Close: {day_close:.2f}')
    ]:
        fig.add_trace(go.Scatter(
            x=[spx.index[0], spx.index[-1]],
            y=[price, price],
            name=name,
            line=dict(color=color, dash='dot'),
            hoverinfo='name',
            showlegend=False  # Remove OHLC from legend
        ))
    
    # Add markers for OHLC points
    fig.add_trace(go.Scatter(
        x=[spx['High'].idxmax()],
        y=[day_high],
        mode='markers',
        marker=dict(color=high_color, size=10),
        name='High Point',
        showlegend=False,
        hovertemplate='High: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[spx['Low'].idxmin()],
        y=[day_low],
        mode='markers',
        marker=dict(color=low_color, size=10),
        name='Low Point',
        showlegend=False,
        hovertemplate='Low: %{y:.2f}<extra></extra>'
    ))
    
    # Calculate additional insights
    day_range = day_high - day_low
    range_pct = (day_range / day_open) * 100
    time_to_high = spx['High'].idxmax().strftime('%H:%M')
    time_to_low = spx['Low'].idxmin().strftime('%H:%M')

    # Build multi-line stats for subtitle with <br> for wrapping
    stats_line = (
        f"Date: {date.strftime('%Y-%m-%d')} | "
        f"O: {day_open:.2f} | H: {day_high:.2f} | L: {day_low:.2f} | C: {day_close:.2f} <br>"
        f"Range: {day_range:.2f} pts, {range_pct:.2f}% | "
        f"Overnight: "
    )
    if overnight_change is not None:
        stats_line += f"{overnight_change:+.2f} pts, {overnight_pct:+.2f}% <br>"
    else:
        stats_line += "N/A <br>"
    stats_line += f"Day Change: {price_change:+.2f} pts, {pct_change:+.2f}%"

    # Add custom OHLC legend in the top right
    ohlc_legend = (
        f'<span style="color:{open_color};font-size:18px;">●</span> O: {day_open:.2f}<br>'
        f'<span style="color:{high_color};font-size:18px;">●</span> H: {day_high:.2f}<br>'
        f'<span style="color:{low_color};font-size:18px;">●</span> L: {day_low:.2f}<br>'
        f'<span style="color:{close_color};font-size:18px;">●</span> C: {day_close:.2f}'
    )
    fig.add_annotation(
        text=ohlc_legend,
        xref="paper", yref="paper",
        x=0.99, y=0.99,
        showarrow=False,
        align="right",
        bordercolor="#cccccc",
        borderwidth=1,
        borderpad=8,
        bgcolor="rgba(255,255,255,0.92)",
        opacity=0.98,
        font=dict(size=14, color="#222")
    )
    
    # --- TP/SL logic ---
    if highlight_start and highlight_end and direction:
        try:
            start_dt = pd.to_datetime(f"{date.strftime('%Y-%m-%d')} {highlight_start}")
            end_dt = pd.to_datetime(f"{date.strftime('%Y-%m-%d')} {highlight_end}")
            # Find closest index for start and end
            start_idx = spx.index.get_indexer([start_dt], method='nearest')[0]
            end_idx = spx.index.get_indexer([end_dt], method='nearest')[0]
            window = spx.iloc[start_idx:end_idx+1]
            entry_time = window.index[0]
            entry_price = window.iloc[0]['Close']
            if direction == 'long':
                tp_level = entry_price + tp
                sl_level = entry_price - sl
                tp_hit = window[window['High'] >= tp_level]
                sl_hit = window[window['Low'] <= sl_level]
            elif direction == 'short':
                tp_level = entry_price - tp
                sl_level = entry_price + sl
                tp_hit = window[window['Low'] <= tp_level]
                sl_hit = window[window['High'] >= sl_level]
            else:
                raise ValueError('direction must be "long" or "short"')
            # Find which comes first
            tp_time = tp_hit.index[0] if not tp_hit.empty else None
            sl_time = sl_hit.index[0] if not sl_hit.empty else None
            outcome = None
            hit_time = None
            hit_price = None
            if tp_time and (not sl_time or tp_time <= sl_time):
                outcome = 'TP'
                hit_time = tp_time
                hit_price = tp_level
            elif sl_time:
                outcome = 'SL'
                hit_time = sl_time
                hit_price = sl_level
            # Draw entry to hit line and marker
            if outcome == 'TP':
                fig.add_trace(go.Scatter(
                    x=[entry_time, hit_time],
                    y=[entry_price, hit_price],
                    mode='lines',
                    line=dict(color='green', width=3),
                    name='Entry→TP',
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=[hit_time],
                    y=[hit_price],
                    mode='markers',
                    marker=dict(color='green', size=14, symbol='circle'),
                    name='TP Hit',
                    showlegend=True,
                    hovertemplate="TP hit at %{x|%H:%M CT}<br>Price: %{y:.2f}<extra></extra>"
                ))
                # Add annotation for points move
                mid_x = entry_time + (hit_time - entry_time) / 2
                mid_y = (entry_price + hit_price) / 2
                y_offset = (spx['High'].max() - spx['Low'].min()) * 0.01
                point_diff = hit_price - entry_price
                fig.add_annotation(
                    x=mid_x,
                    y=mid_y + y_offset,
                    text=f"{point_diff:+.2f} pts",
                    showarrow=False,
                    font=dict(size=14, color='green'),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor='green',
                    borderwidth=1,
                    borderpad=4
                )
            elif outcome == 'SL':
                fig.add_trace(go.Scatter(
                    x=[entry_time, hit_time],
                    y=[entry_price, hit_price],
                    mode='lines',
                    line=dict(color='red', width=3),
                    name='Entry→SL',
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=[hit_time],
                    y=[hit_price],
                    mode='markers',
                    marker=dict(color='red', size=14, symbol='x'),
                    name='SL Hit',
                    showlegend=True,
                    hovertemplate="SL hit at %{x|%H:%M CT}<br>Price: %{y:.2f}<extra></extra>"
                ))
                # Add annotation for points move
                mid_x = entry_time + (hit_time - entry_time) / 2
                mid_y = (entry_price + hit_price) / 2
                y_offset = (spx['High'].max() - spx['Low'].min()) * 0.01
                point_diff = hit_price - entry_price
                fig.add_annotation(
                    x=mid_x,
                    y=mid_y + y_offset,
                    text=f"{point_diff:+.2f} pts",
                    showarrow=False,
                    font=dict(size=14, color='red'),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor='red',
                    borderwidth=1,
                    borderpad=4
                )
            else:
                # Neither hit: draw gray line to end and marker
                end_time = window.index[-1]
                end_price = window.iloc[-1]['Close']
                fig.add_trace(go.Scatter(
                    x=[entry_time, end_time],
                    y=[entry_price, end_price],
                    mode='lines',
                    line=dict(color='gray', width=2, dash='dot'),
                    name='Entry→End',
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=[end_time],
                    y=[end_price],
                    mode='markers',
                    marker=dict(color='gray', size=12, symbol='diamond'),
                    name='No TP/SL',
                    showlegend=True,
                    hovertemplate="No TP/SL hit<br>Price: %{y:.2f}<extra></extra>"
                ))
                # Add annotation for points move
                mid_x = entry_time + (end_time - entry_time) / 2
                mid_y = (entry_price + end_price) / 2
                y_offset = (spx['High'].max() - spx['Low'].min()) * 0.01
                point_diff = end_price - entry_price
                fig.add_annotation(
                    x=mid_x,
                    y=mid_y + y_offset,
                    text=f"{point_diff:+.2f} pts",
                    showarrow=False,
                    font=dict(size=14, color='gray'),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor='gray',
                    borderwidth=1,
                    borderpad=4
                )
        except Exception as e:
            print(f"Could not process TP/SL window: {e}")
    # Highlight time window if specified (keep original highlight for context)
    if highlight_start and highlight_end:
        try:
            start_dt = pd.to_datetime(f"{date.strftime('%Y-%m-%d')} {highlight_start}")
            end_dt = pd.to_datetime(f"{date.strftime('%Y-%m-%d')} {highlight_end}")
            fig.add_vrect(
                x0=start_dt, x1=end_dt,
                fillcolor="rgba(255, 215, 0, 0.18)",
                line_width=0,
                layer="below",
            )
        except Exception as e:
            print(f"Could not highlight window: {e}")
    # Update layout with stats line as subtitle above the plot
    fig.update_layout(
        title=dict(
            text=title_text + '<br><span style="font-size:15px; color:#444;">' + stats_line + ' (All times CT)</span>',
            x=0.5,
            y=0.97,
            font=dict(size=22)
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor='white',
        hovermode='x unified',
        xaxis=dict(
            title='Time (CT)',
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True,
            tickformat='%H:%M',
        ),
        yaxis=dict(
            title='Price',
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cccccc",
            borderwidth=1
        ),
        margin=dict(r=40, t=100, b=60, l=60)  # Increase top margin for subtitle
    )
    
    # Show the plot
    fig.show()

def plot_multiple_dates(dates_list, highlight_start=None, highlight_end=None, tp=10, sl=20, direction=None):
    for date_str in dates_list:
        # Remove double quotes and extra whitespace
        clean_date = date_str.replace('"', '').replace("'", '').strip()
        print(f"\nPlotting for {clean_date}...")
        plot_stock_chart(clean_date, show_consecutive=False, highlight_start=highlight_start, highlight_end=highlight_end, tp=tp, sl=sl, direction=direction)

def main():
    parser = argparse.ArgumentParser(description='Plot SPX stock chart for a specific date or list of dates')
    parser.add_argument('--date', help='Date in YYYY-MM-DD format')
    parser.add_argument('--dates', nargs='?', const='test.txt', default=None,
                      help='Path to a file containing a list of dates (default: test.txt) or comma-separated list of dates (YYYY-MM-DD)')
    parser.add_argument('--consecutive', action='store_true', 
                      help='If set, shows charts for the previous market day and the specified date (single-date mode only)')
    parser.add_argument('--highlight_start', '-hs', type=str, default=None, help='Highlight window start time (HH:MM, e.g. 09:30)')
    parser.add_argument('--highlight_end', '-he', type=str, default=None, help='Highlight window end time (HH:MM, e.g. 10:45)')
    parser.add_argument('--tp', type=float, default=10, help='Take-profit in points (default: 10)')
    parser.add_argument('--sl', type=float, default=20, help='Stop-loss in points (default: 20)')
    parser.add_argument('--direction', type=str, choices=['long', 'short'], default=None, help='Trade direction: long or short (required if using highlight window)')
    args = parser.parse_args()

    # If highlight window is used, direction is required
    if (args.highlight_start or args.highlight_end) and not args.direction:
        print("Error: --direction must be specified if using --highlight_start and --highlight_end.")
        return

    if args.dates:
        if os.path.isfile(args.dates):
            with open(args.dates, 'r') as f:
                content = f.read().strip()
                # Try to parse as JSON first
                try:
                    # Handle trailing commas by wrapping in square brackets
                    if not content.startswith('['):
                        content = '[' + content
                    if not content.endswith(']'):
                        content = content.rstrip(',') + ']'
                    dates_list = json.loads(content)
                except Exception:
                    # Fallback: parse as line-by-line dates with potential indentation and trailing commas
                    dates_list = [line.strip().strip('"').strip("'").rstrip(',').strip() for line in content.split('\n') if line.strip()]
        else:
            # Assume comma-separated string
            dates_list = [d.strip() for d in args.dates.split(',') if d.strip()]
        plot_multiple_dates(dates_list, args.highlight_start, args.highlight_end, args.tp, args.sl, args.direction)
    elif args.date:
        plot_stock_chart(args.date, args.consecutive, args.highlight_start, args.highlight_end, args.tp, args.sl, args.direction)
    else:
        print("Please provide either --date or --dates.")

if __name__ == "__main__":
    main()
