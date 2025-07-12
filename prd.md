# Product Requirements Document (PRD)

## Project: Minute-Data Pattern Trading System

---

## 1. Overview

The system is designed to identify highly specific market patterns by analyzing minute-by-minute price data for the SPX (S&P 500 index). It operates in two main phases: historical pattern discovery and real-time inference.

In the historical phase, the system processes a large dataset of 1-minute bars spanning the last three years. For each pair of consecutive trading days, it computes a rich set of market metrics. These metrics capture various aspects of market behavior, such as closing strength, volatility, and returns over different time intervals (e.g., the last hour, first 5 minutes, power hour, lunch hour, etc.). By analyzing these metrics, the system identifies patterns where specific conditions on one day (e.g., high volatility in the last hour) correlate with predictable outcomes on the next day (e.g., a strong move in a particular intra-day period). These patterns are stored in a structured format, detailing the conditions, the expected outcomes, and statistical measures like success rate, average move, and risk-reward ratios.

During the real-time inference phase, the system takes the latest available minute data for the current and previous trading days. It calculates the same set of market metrics and compares them against the stored patterns. If the current metrics fall within the ranges defined by a pattern, the system generates a trading strategy. This strategy includes specific entry and exit times, trade direction (buy or sell), take-profit and stop-loss levels, and the historical success rate of the pattern.

The filtering system plays a crucial role in refining these strategies. It applies a series of filters to ensure that only high-quality patterns are considered. These filters include minimum thresholds for take-profit, success rate, number of occurrences, and risk-reward ratio. By progressively relaxing these filters, the system can adapt to varying market conditions, ensuring that it provides actionable strategies even in less predictable environments. The strategies are then organized into sessions (morning, mixed, afternoon) and further filtered to remove overlapping or low-probability setups. The final output is a set of actionable trading plans, which are logged for review and potential execution.

This approach allows the system to learn from historical data and apply those insights to current market conditions, providing traders with data-driven strategies based on observed market patterns.

---

## 2. Workflow

### A. Historical Pattern Discovery
- For each day T:
  - Extract a comprehensive set of metrics (see below).
  - For each pair (T, T+1), record the outcome metrics for T+1.
  - Group and analyze these pairs to find statistically significant patterns.

### B. Real-Time Inference
- For the most recent day (T):
  - Extract the same metrics as above.
  - Find historical days with similar metrics.
  - Use the outcomes from those historical T+1 days to generate a trading plan for the next session.

### C. Filtering and Output
- Apply filters on take-profit, success rate, occurrences, and risk-reward.
- Organize strategies by session (morning, mixed, afternoon).
- Remove overlapping or low-probability setups.
- Output actionable trading plans for review and execution.

---

## 3. Terminology

- **Setup Day (Day T):** The day whose metrics are used to find historical analogs (also called "feature vector", "condition metrics", or "pattern features").
- **Outcome Day (Day T+1):** The day after a setup, whose realized moves are used as targets ("outcome metrics", "result metrics", or "targets").

---

## 4. Metrics

### A. Day T (Setup Day) Metrics

- last_hour
- last_30min
- last_15min
- close_vol
- last_hour_vol
- close_strength
- day_range

#### Early Day Volatility
- first_15min_vol
- first_30min_vol
- first_60min_vol

#### Momentum & Trend
- pre_lunch_momentum
- morning_trend_strength

#### First 5 Minutes
- first_5min_return
- first_5min_range
- first_5min_high_test
- first_5min_low_test
- first_5min_vol

#### Power Hour (9:30-10:30)
- power_hour_return
- power_hour_range
- power_hour_trend_changes
- power_hour_vol
- power_hour_momentum

#### Lunch Hour (12:00-1:00)
- lunch_hour_range
- lunch_hour_return
- lunch_hour_direction_changes
- lunch_hour_vol
- lunch_hour_range_contraction

#### Strongest Periods
- strongest_hour
- strongest_hour_range
- strongest_30min_period
- strongest_30min_range
- strongest_15min_period
- strongest_15min_range

#### Hourly and Multi-Hour Returns
- hour1
- hour2
- hour3
- hour4
- hour5
- hour6

#### Quarter-hour returns (first two hours)
- hour1_q1
- hour1_q2
- hour1_q3
- hour1_q4
- hour2_q1
- hour2_q2
- hour2_q3
- hour2_q4

#### Momentum between quarters
- momentum_hour1_q1_to_hour1_q2
- momentum_hour1_q2_to_hour1_q3
- momentum_hour1_q3_to_hour1_q4
- momentum_hour1_q4_to_hour2_q1
- momentum_hour2_q1_to_hour2_q2
- momentum_hour2_q2_to_hour2_q3
- momentum_hour2_q3_to_hour2_q4

#### Half-hour and hour splits
- hour1_h1
- hour1_h2
- hour2_h1
- hour2_h2
- momentum_hour1_h1_to_hour1_h2
- momentum_hour1_h2_to_hour2_h1
- momentum_hour2_h1_to_hour2_h2

#### End of day and multi-hour periods
- last_30_min
- hours_1_2
- hours_2_3
- hours_3_4
- hours_4_5
- hours_5_6
- hours_5_6_30

#### Gaps (if calculated for T)
- official_gap
- effective_gap

---

### B. Day T+1 (Outcome Day) Metrics

- today_official_gap
- today_effective_gap
- today_first_15min
- today_first_30min
- today_first_hour
- today_hour1
- today_hour2
- today_hour3
- today_hour4
- today_hour5
- today_hour6
- today_hour1_q1
- today_hour1_q2
- today_hour1_q3
- today_hour1_q4
- today_hour2_q1
- today_hour2_q2
- today_hour2_q3
- today_hour2_q4
- today_momentum_hour1_q1_to_hour1_q2
- today_momentum_hour1_q2_to_hour1_q3
- today_momentum_hour1_q3_to_hour1_q4
- today_momentum_hour1_q4_to_hour2_q1
- today_momentum_hour2_q1_to_hour2_q2
- today_momentum_hour2_q2_to_hour2_q3
- today_momentum_hour2_q3_to_hour2_q4
- today_hour1_h1
- today_hour1_h2
- today_hour2_h1
- today_hour2_h2
- today_momentum_hour1_h1_to_hour1_h2
- today_momentum_hour1_h2_to_hour2_h1
- today_momentum_hour2_h1_to_hour2_h2
- today_last_30_min
- today_today_high_15min
- today_today_low_15min
- today_today_open_15min
- today_today_high_30min
- today_today_low_30min
- today_today_open_30min
- today_today_high_60min
- today_today_low_60min
- today_today_open_60min
- today_hours_1_2
- today_hours_2_3
- today_hours_3_4
- today_hours_4_5
- today_hours_5_6
- today_hours_5_6_30

---

## 5. Filtering and Strategy Generation

- Patterns are filtered by minimum thresholds for take-profit, success rate, number of occurrences, and risk-reward ratio.
- Filters can be progressively relaxed to adapt to market conditions.
- Strategies are organized by session and further filtered to remove overlaps and low-probability setups.
- Final output is a set of actionable trading plans, logged for review and execution.

---

## 6. Glossary

- **Feature Vector / Setup Metrics:** The set of metrics extracted from day T.
- **Outcome Metrics / Target Variables:** The set of realized moves on day T+1.
- **Pattern:** A combination of setup metrics and associated outcome statistics.
- **Session:** A market period (e.g., morning, lunch, afternoon) used for organizing strategies.

---

## 7. Future Enhancements

- Add new metrics or refine existing ones as new market behaviors are discovered.
- Integrate with execution systems for automated trading.
- Expand to other instruments or timeframes.

--- 