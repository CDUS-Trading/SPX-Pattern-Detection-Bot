# PRD: Setup → Run Pattern Trading System (feat-flex-time)

## Executive Summary

This document outlines the architectural transformation of the SPX Pattern Trading System from fixed time windows to dynamic run-based pattern recognition. The goal is to align pattern detection with actual market behavior by detecting significant directional movements ("runs") and predicting their characteristics based on setup conditions.

## Current System Architecture & Limitations

### Current Implementation
- **Pattern Collection (`collector.py`)**: Extracts setup metrics from Day T, calculates fixed time window returns from Day T+1
- **Pattern Detection (`detector.py`)**: Matches current metrics against historical patterns for fixed time periods
- **Pattern Database**: ~9.2MB JSON containing setup conditions mapped to fixed timeframe outcomes

### Current Pattern Structure
```json
{
  "pattern1": "last_15min",           // Setup metric 1
  "range1_min": -0.1, "range1_max": 0.1,
  "pattern2": "close_vol",            // Setup metric 2  
  "range2_min": -0.1, "range2_max": 0.1,
  "timeframe": "hour2",               // Fixed 60-minute window
  "direction": "bullish",
  "success_rate": 75.0,
  "avg_move": 0.11821873834466352,
  "sample_size": 4
}
```

### Critical Limitations
1. **Artificial Time Constraints**: Real profitable runs don't respect hourly boundaries
2. **Missed Opportunities**: Runs spanning multiple fixed buckets get fragmented
3. **Poor Risk/Reward**: Fixed windows may capture only part of actual moves
4. **Inflexible**: Market dynamics change but time windows remain static

### Current Fixed Timeframes
```json
{
  "hour1": [0, 60],    // 8:30am-9:30am CT
  "hour2": [60, 120],  // 9:30am-10:30am CT
  "hour3": [120, 180], // 10:30am-11:30am CT
  "hour4": [180, 240], // 11:30am-12:30pm CT
  "hour5": [240, 300], // 12:30pm-1:30pm CT
  "hour6": [300, 360]  // 1:30pm-2:30pm CT
}
```

## Proposed "Setup → Run" System

### Core Concept
**Transform**: Setup Conditions → Fixed Time Returns  
**Into**: Setup Conditions → Dynamic Run Characteristics

### Run Definition
A **run** is a significant directional market movement with:
- **Minimum Duration**: 10+ minutes
- **Minimum Magnitude**: 0.3%+ move
- **Maximum Pullback**: ≤20 against direction
- **Momentum Consistency**: Sustained directional pressure

### New Pattern Structure
```json
{
  "setup_conditions": {
    "last_15min": 0.17,
    "close_vol": 0.52,
    "close_strength": 0.73
  },
  "run_characteristics": {
    "start_time": "9:47AM",
    "end_time": "11:23AM", 
    "duration": 96,
    "total_move": 0.73,
    "direction": "bull",
    "max_adverse": -0.21
  },
  "historical_count": 23,
  "success_rate": 68.0
}
```

### Target Output Format
```
Expected Run: 9:47AM - 11:23AM CT (96 minutes)
Direction: Buy 📈
Avg Move: +0.73%
Success Rate: 68%
Max Adverse: -0.21%
Based on 23 similar setups
```

## Technical Implementation

### Phase 1: Run Detection Algorithm

#### Core Components

**1. Run Detection Engine (`src/patterns/run_detector.py`)**
```python
class RunDetector:
    def __init__(self, min_move=0.003, min_duration=10, max_pullback=0.0015):
        self.min_move = min_move        # 0.3% minimum
        self.min_duration = min_duration # 10 minutes minimum
        self.max_pullback = max_pullback # 0.15% maximum pullback
    
    def detect_runs(self, minute_data: pd.DataFrame) -> List[Dict]:
        """Detect all significant runs in minute data"""
        
    def validate_run(self, data_slice: pd.DataFrame) -> bool:
        """Validate if data slice qualifies as a run"""
        
    def calculate_run_metrics(self, run_data: pd.DataFrame) -> Dict:
        """Calculate duration, move, adverse excursion"""
```

**2. Run Detection Algorithm**
```python
def detect_runs(self, minute_data):
    runs = []
    i = 0
    
    while i < len(minute_data) - self.min_duration:
        # Look for momentum start
        if self._detect_momentum_start(minute_data[i:]):
            run_end = self._find_run_end(minute_data, i)
            if run_end - i >= self.min_duration:
                run_data = minute_data[i:run_end+1]
                if self._validate_run(run_data):
                    runs.append(self._create_run_object(run_data, i, run_end))
                    i = run_end + 1
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    
    return runs
```

**3. Run Validation Criteria**
- **Total Move**: Net percentage change ≥ threshold
- **Direction Consistency**: >70% of minutes in same direction
- **Momentum**: Average move per minute ≥ 0.01%
- **Pullback Tolerance**: Max adverse ≤ threshold

#### Run Categories
- **Short Runs**: 10-30 minutes, 0.3-0.8% moves
- **Medium Runs**: 30-90 minutes, 0.8-1.5% moves  
- **Extended Runs**: 90+ minutes, 1.5%+ moves

### Phase 2: Pattern Collection Transformation

**Modified `collector.py`**
```python
def analyze_market_correlation_with_runs(minute_file, start_date, end_date):
    """Enhanced analysis using run detection instead of fixed windows"""
    
    for day_pair in trading_days:
        # Extract setup conditions (unchanged)
        yesterday_patterns = extract_setup_metrics(yesterday_data)
        
        # NEW: Detect runs instead of calculating fixed windows
        detected_runs = run_detector.detect_runs(today_data)
        
        # Create setup → run mappings
        for run in detected_runs:
            pattern_entry = {
                'setup_conditions': yesterday_patterns,
                'run': run,
                'date': today_date
            }
            patterns.append(pattern_entry)
    
    return patterns
```

**New Pattern Database Structure**
```json
[
  {
    "setup_pattern1": "last_15min",
    "setup_range1": [-0.1, 0.1],
    "setup_pattern2": "close_vol", 
    "setup_range2": [0.4, 0.6],
    "run_start_time": "9:47AM",
    "run_duration": 96,
    "run_move": 0.0073,
    "run_direction": "bull",
    "run_max_adverse": -0.0021,
    "historical_dates": ["2024-01-15", "2024-02-03", "..."],
    "sample_size": 23
  }
]
```

### Phase 3: Detection System Transformation

**Modified `detector.py`**
```python
class RunPatternDetector:
    def detect_run_patterns(self, current_metrics):
        """Find expected runs based on current setup conditions"""
        
        matched_runs = []
        for pattern in self.run_patterns:
            if self._setup_matches(current_metrics, pattern):
                matched_runs.append(pattern)
        
        # Aggregate similar runs
        aggregated_runs = self._aggregate_run_predictions(matched_runs)
        return aggregated_runs
    
    def _aggregate_run_predictions(self, runs):
        """Combine similar runs into statistical expectations"""
        
        # Group by start time windows (15-min buckets)
        time_groups = self._group_by_start_time(runs)
        
        for group in time_groups:
            yield {
                'expected_start': self._calc_avg_start_time(group),
                'expected_duration': self._calc_avg_duration(group),
                'expected_move': self._calc_avg_move(group),
                'success_rate': len(group) / total_matches,
                'sample_size': len(group),
                'confidence': self._calc_confidence(group)
            }
```

### Phase 4: Integration & Testing

**1. Backward Compatibility**
- Maintain current system during transition
- A/B testing framework for comparison
- Gradual migration of detection logic

**2. Performance Optimization**
```python
# Efficient run detection with vectorized operations
def vectorized_run_detection(minute_data):
    # Use numpy for fast momentum calculations
    returns = np.diff(minute_data['close']) / minute_data['close'][:-1]
    cumulative_returns = np.cumsum(returns)
    
    # Vectorized threshold detection
    momentum_starts = np.where(np.abs(returns) > momentum_threshold)[0]
    
    # Fast run validation using rolling windows
    for start_idx in momentum_starts:
        # ... optimized validation logic
```

## File Structure Changes

### New Files
```
src/patterns/
├── run_detector.py              # Core run detection algorithms
├── run_pattern_collector.py     # Modified collector for runs
├── run_pattern_detector.py      # Modified detector for runs
└── utils/
    ├── run_analytics.py         # Run statistics and validation
    └── run_visualization.py     # Run plotting and analysis

data/processed/
├── run_patterns_v1.json        # Run-based pattern database
└── run_validation_results.json # Validation metrics

config/
└── run_detection_params.json   # Run detection parameters
```

### Modified Files
```
src/patterns/
├── collector.py                 # Add run detection option
├── detector.py                  # Add run-based detection
└── core/
    └── pattern_detector_class.py # Support both systems
```

## Configuration

**`config/run_detection_params.json`**
```json
{
  "run_detection": {
    "min_move_threshold": 0.003,      
    "min_duration_minutes": 10,
    "max_pullback_threshold": 0.0015, 
    "momentum_threshold": 0.001       
  },
  "run_categories": {
    "short": {"min_duration": 10, "max_duration": 30},
    "medium": {"min_duration": 30, "max_duration": 90}, 
    "extended": {"min_duration": 90, "max_duration": 999}
  },
  "aggregation": {
    "start_time_bucket_minutes": 15,
    "min_sample_size": 5,
    "confidence_threshold": 0.6
  }
}
```

## Success Metrics

### Validation Framework
1. **Historical Accuracy**: % of predicted runs that actually occurred
2. **Timing Precision**: Average deviation between predicted and actual start times
3. **Move Accuracy**: Correlation between predicted and actual move sizes
4. **Risk Management**: Improvement in predicted vs. actual adverse excursions

### Performance Benchmarks
```python
# Target improvements vs current system
metrics = {
    'pattern_accuracy': 0.75,        # 75% of predictions accurate
    'timing_precision': 15,          # ±15 minutes average deviation
    'move_correlation': 0.65,        # 65% correlation with actual moves
    'risk_improvement': 0.25         # 25% better risk prediction
}
```

### Backtesting Framework
```python
def backtest_run_system(historical_data, start_date, end_date):
    """Compare run-based vs fixed-window performance"""
    
    run_results = []
    fixed_results = []
    
    for test_date in date_range:
        # Test both systems
        run_predictions = run_detector.predict(test_date)
        fixed_predictions = current_detector.predict(test_date)
        
        # Measure actual outcomes
        actual_moves = get_actual_moves(test_date)
        
        # Calculate performance metrics
        run_performance = calculate_performance(run_predictions, actual_moves)
        fixed_performance = calculate_performance(fixed_predictions, actual_moves)
        
        run_results.append(run_performance)
        fixed_results.append(fixed_performance)
    
    return compare_systems(run_results, fixed_results)
```
## Executive Summary

This document outlines the architectural transformation of the SPX Pattern Trading System from fixed time windows to dynamic run-based pattern recognition. The goal is to align pattern detection with actual market behavior by detecting significant directional movements ("runs") and predicting their characteristics based on setup conditions.


This document serves as the comprehensive specification for transforming from fixed time windows to dynamic run-based pattern recognition. Each implementation phase should reference this document for context and architectural guidance.

Key Success Indicator: When the system can accurately predict that after specific setup conditions, the market will likely run from 9:47AM to 11:23AM with an average move of +0.73%, based on 23 similar historical instances, rather than just saying "buy from 12:30PM to 3:00PM."