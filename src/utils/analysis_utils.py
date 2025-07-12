def analyze_consecutive_patterns(results):
    """Analyze patterns of consecutive wins and losses"""
    # Convert results to a list of 1s (wins) and 0s (losses), filtering out None values
    outcomes = []
    for r in results:
        if r['original_tp_sl_pnl'] is not None:  # Only include trades with valid P&L
            outcomes.append(1 if r['original_tp_sl_pnl'] > 0 else 0)
    
    if not outcomes:
        print("\nNo valid trades found for consecutive pattern analysis")
        return
        
    # Find consecutive sequences
    current_sequence = []
    sequences = []
    current_value = outcomes[0]
    current_count = 1
    
    for value in outcomes[1:]:
        if value == current_value:
            current_count += 1
        else:
            sequences.append((current_value, current_count))
            current_value = value
            current_count = 1
    sequences.append((current_value, current_count))
    
    # Analyze sequences
    win_sequences = [count for value, count in sequences if value == 1]
    loss_sequences = [count for value, count in sequences if value == 0]
    
    # Calculate statistics
    total_trades = len(outcomes)
    total_wins = sum(win_sequences)
    total_losses = sum(loss_sequences)
    
    print("\n=== Consecutive Wins/Losses Analysis ===")
    print(f"Total Valid Trades: {total_trades}")
    print(f"Total Wins: {total_wins} ({total_wins/total_trades*100:.1f}%)")
    print(f"Total Losses: {total_losses} ({total_losses/total_trades*100:.1f}%)")
    
    print("\nWin Streaks (W = Win, L = Loss):")
    print(f"Longest Win Streak: {max(win_sequences) if win_sequences else 0} (e.g., {'W' * max(win_sequences) if win_sequences else 'N/A'})")
    print(f"Average Win Streak: {sum(win_sequences)/len(win_sequences):.1f}" if win_sequences else "N/A")
    print(f"Most Common Win Streak: {max(set(win_sequences), key=win_sequences.count) if win_sequences else 0}")
    
    print("\nLoss Streaks (W = Win, L = Loss):")
    print(f"Longest Loss Streak: {max(loss_sequences) if loss_sequences else 0} (e.g., {'L' * max(loss_sequences) if loss_sequences else 'N/A'})")
    print(f"Average Loss Streak: {sum(loss_sequences)/len(loss_sequences):.1f}" if loss_sequences else "N/A")
    print(f"Most Common Loss Streak: {max(set(loss_sequences), key=loss_sequences.count) if loss_sequences else 0}")
    
    # Analyze distribution of streak lengths
    print("\nWin Streak Distribution (W = Win, L = Loss):")
    for length in sorted(set(win_sequences)):
        count = win_sequences.count(length)
        print(f"{length} consecutive wins: {count} times ({count/len(win_sequences)*100:.1f}% of win streaks) (e.g., {'W' * length})")
    
    print("\nLoss Streak Distribution (W = Win, L = Loss):")
    for length in sorted(set(loss_sequences)):
        count = loss_sequences.count(length)
        print(f"{length} consecutive losses: {count} times ({count/len(loss_sequences)*100:.1f}% of loss streaks) (e.g., {'L' * length})")
    
    # Add sequence examples
    print("\nExample Sequences from the Data:")
    example_length = min(20, len(outcomes))  # Show first 20 trades or less
    sequence_str = ''.join(['W' if x == 1 else 'L' for x in outcomes[:example_length]])
    print(f"First {example_length} trades: {sequence_str}") 