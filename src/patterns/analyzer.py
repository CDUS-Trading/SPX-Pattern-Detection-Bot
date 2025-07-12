import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# Set style for better visualizations
plt.style.use('seaborn-v0_8')  # Using a built-in style that's similar to seaborn
sns.set_palette("husl")

def load_pattern_database():
    with open('master_pattern_database.json', 'r') as f:
        return json.load(f)

def analyze_pattern_frequency(patterns):
    # Count pattern1 and pattern2 frequencies
    pattern1_counts = Counter(p['pattern1'] for p in patterns)
    pattern2_counts = Counter(p['pattern2'] for p in patterns)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot pattern1 frequencies
    pattern1_df = pd.DataFrame.from_dict(pattern1_counts, orient='index', columns=['count'])
    pattern1_df.sort_values('count', ascending=True).plot(kind='barh', ax=ax1)
    ax1.set_title('Pattern1 Frequency')
    ax1.set_xlabel('Count')
    
    # Plot pattern2 frequencies
    pattern2_df = pd.DataFrame.from_dict(pattern2_counts, orient='index', columns=['count'])
    pattern2_df.sort_values('count', ascending=True).plot(kind='barh', ax=ax2)
    ax2.set_title('Pattern2 Frequency')
    ax2.set_xlabel('Count')
    
    plt.tight_layout()
    plt.show()

def analyze_timeframe_performance(patterns):
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(patterns)
    
    # Calculate average success rate and risk-reward ratio by timeframe
    timeframe_stats = df.groupby('timeframe').agg({
        'success_rate': 'mean',
        'risk_reward': 'mean',
        'sample_size': 'sum'
    }).round(2)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot success rates
    timeframe_stats['success_rate'].sort_values(ascending=True).plot(kind='barh', ax=ax1)
    ax1.set_title('Average Success Rate by Timeframe')
    ax1.set_xlabel('Success Rate (%)')
    
    # Plot risk-reward ratios
    timeframe_stats['risk_reward'].sort_values(ascending=True).plot(kind='barh', ax=ax2)
    ax2.set_title('Average Risk-Reward Ratio by Timeframe')
    ax2.set_xlabel('Risk-Reward Ratio')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("\nTimeframe Performance Statistics:")
    print(timeframe_stats)

def analyze_direction_bias(patterns):
    df = pd.DataFrame(patterns)
    
    # Count directions by timeframe
    direction_counts = pd.crosstab(df['timeframe'], df['direction'])
    
    # Create stacked bar chart
    plt.figure(figsize=(12, 6))
    direction_counts.plot(kind='bar', stacked=True)
    plt.title('Direction Bias by Timeframe')
    plt.xlabel('Timeframe')
    plt.ylabel('Count')
    plt.legend(title='Direction')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_win_loss_ratios(patterns):
    df = pd.DataFrame(patterns)
    
    # Calculate win-loss ratios
    df['win_loss_ratio'] = df['avg_win'] / df['avg_loss']
    
    # Create box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='direction', y='win_loss_ratio', data=df)
    plt.title('Win-Loss Ratio Distribution by Direction')
    plt.ylabel('Win-Loss Ratio')
    plt.tight_layout()
    plt.show()

def analyze_sample_size_distribution(patterns):
    df = pd.DataFrame(patterns)
    
    # Create histogram of sample sizes
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='sample_size', bins=30)
    plt.title('Distribution of Sample Sizes')
    plt.xlabel('Sample Size')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def analyze_timeframe_patterns(patterns):
    df = pd.DataFrame(patterns)
    
    # Create a cross-tabulation of timeframes and pattern combinations
    pattern_combinations = df['pattern1'] + ' + ' + df['pattern2']
    timeframe_patterns = pd.crosstab(df['timeframe'], pattern_combinations)
    
    # Create heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(timeframe_patterns, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Pattern Combinations by Timeframe')
    plt.xlabel('Pattern Combinations')
    plt.ylabel('Timeframe')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Print most common pattern combinations for each timeframe
    print("\nMost Common Pattern Combinations by Timeframe:")
    for timeframe in timeframe_patterns.index:
        top_patterns = timeframe_patterns.loc[timeframe].nlargest(3)
        print(f"\n{timeframe}:")
        for pattern, count in top_patterns.items():
            print(f"  {pattern}: {count} occurrences")

def main():
    # Load patterns
    patterns = load_pattern_database()
    
    # Run all analyses
    analyze_pattern_frequency(patterns)
    analyze_timeframe_performance(patterns)
    analyze_direction_bias(patterns)
    analyze_win_loss_ratios(patterns)
    analyze_sample_size_distribution(patterns)
    analyze_timeframe_patterns(patterns)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 