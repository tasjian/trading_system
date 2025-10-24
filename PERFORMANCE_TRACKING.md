# FinRL Performance Tracking System

## Overview

The **Performance Tracking System** provides comprehensive evaluation, comparison, and visualization of FinRL agent performance across training cycles. All metrics are automatically computed and stored in a SQLite database for historical tracking and analysis.

### Key Features

✅ **30+ Comprehensive Metrics** - Sharpe, Sortino, Calmar ratios, Win rate, Max drawdown, VaR, CVaR, and more
✅ **SQLite Persistence** - All metrics saved to database for historical tracking
✅ **Automatic Evaluation** - Performance computed after every training/fine-tuning run
✅ **Model Comparison** - Side-by-side comparison of different models and agents
✅ **Visualization Tools** - matplotlib charts for performance analysis
✅ **Best Model Ranking** - Find top performers by any metric
✅ **Production Ready** - Integrated with both full training and incremental learning

---

## Architecture

### Components

```
utils/performance_tracker.py       # Core tracking system
├── PerformanceMetrics             # Dataclass for all metrics
├── PerformanceTracker             # Main tracker class
│   ├── evaluate_agent()           # Evaluate model on test env
│   ├── save_metrics()             # Save to database
│   ├── get_metrics()              # Query database
│   ├── compare_models()           # Compare two models
│   ├── get_best_model()           # Find top performers
│   ├── plot_performance_comparison()
│   ├── plot_historical_performance()
│   └── generate_performance_report()
└── Convenience functions

data/performance_metrics.db         # SQLite database
examples/performance_analysis_example.py  # Usage examples
```

### Integration Points

The performance tracker is automatically integrated with:

1. **Full Training** (`train_finrl_models.py`)
   - Evaluates all 5 agents on 2024 test data after training
   - Saves metrics to database automatically

2. **Incremental Training** (`train_finrl_incremental.py`)
   - Evaluates fine-tuned models on validation split
   - Compares performance before/after fine-tuning

3. **Agent Wrapper** (`agents/finrl_agent_wrapper.py`)
   - Can be called programmatically for custom evaluations

---

## Metrics Tracked

### Core Performance Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Sharpe Ratio** | Risk-adjusted return | (Annual Return - Risk Free) / Volatility |
| **Sortino Ratio** | Downside risk-adjusted return | Annual Return / Downside Deviation |
| **Calmar Ratio** | Return per unit of drawdown | Annual Return / Max Drawdown |
| **Win Rate** | Percentage of profitable trades | Winning Trades / Total Trades |
| **Loss Rate** | Percentage of losing trades | Losing Trades / Total Trades |
| **Profit Factor** | Ratio of profits to losses | Gross Profit / Gross Loss |

### Risk Metrics

| Metric | Description |
|--------|-------------|
| **Max Drawdown** | Maximum peak-to-trough decline |
| **Max Drawdown Duration** | Longest drawdown period (days) |
| **Volatility** | Annualized standard deviation of returns |
| **Downside Deviation** | Volatility of negative returns only |
| **VaR (95%)** | Value at Risk at 95% confidence |
| **CVaR (95%)** | Conditional VaR (expected shortfall) |

### Return Metrics

| Metric | Description |
|--------|-------------|
| **Total Return** | Overall portfolio gain/loss |
| **Annualized Return** | Return adjusted to annual basis |
| **Cumulative Return** | Total return over evaluation period |
| **Final Portfolio Value** | Ending portfolio value |
| **Initial Portfolio Value** | Starting portfolio value |

### Trade Statistics

- Total Trades
- Winning Trades
- Losing Trades
- Average Win
- Average Loss
- Largest Win
- Largest Loss
- Average Trade Duration

---

## Usage

### 1. Automatic Evaluation (Recommended)

Performance metrics are **automatically computed and saved** when you run training:

```bash
# Full training - automatically evaluates after completion
python train_finrl_models.py

# Incremental training - automatically evaluates after fine-tuning
python train_finrl_incremental.py --days 60
```

After training completes, metrics are saved to `data/performance_metrics.db`.

### 2. Programmatic Evaluation

```python
from utils.performance_tracker import PerformanceTracker
from stable_baselines3 import PPO

# Initialize tracker
tracker = PerformanceTracker(db_path="data/performance_metrics.db")

# Load your model
model = PPO.load("data/finrl_models/agent_ppo.zip")

# Load your test environment (FinRL StockTradingEnv)
# ... create test_env ...

# Evaluate
metrics = tracker.evaluate_agent(
    model=model,
    test_env=test_env,
    model_name="ppo_base",
    agent_type="ppo",
    notes="Baseline model"
)

# Save to database
tracker.save_metrics(metrics)

print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
print(f"Win Rate: {metrics.win_rate:.2%}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
```

### 3. Query Metrics

```python
# Get all metrics
all_metrics = tracker.get_metrics()

# Get metrics for specific agent
ppo_metrics = tracker.get_metrics(agent_type='ppo')

# Get metrics for specific model
model_metrics = tracker.get_metrics(model_name='ppo_20251003_1430')

# Limit results
recent_metrics = tracker.get_metrics(limit=10)
```

### 4. Compare Models

```python
# Compare two models side-by-side
comparison = tracker.compare_models(
    model1_name='ppo_base',
    model2_name='ppo_20251003_finetuned'
)

print(comparison)
```

### 5. Find Best Models

```python
# Best model by Sharpe ratio
best_sharpe = tracker.get_best_model(metric='sharpe_ratio', top_n=1)

# Top 3 PPO models by Sortino ratio
best_ppo = tracker.get_best_model(
    agent_type='ppo',
    metric='sortino_ratio',
    top_n=3
)

# Best model with lowest max drawdown
safest = tracker.get_best_model(metric='max_drawdown', top_n=1)
```

### 6. Generate Reports

```python
# Report for all PPO models
report = tracker.generate_performance_report(agent_type='ppo')
print(report)

# Save report to file
report = tracker.generate_performance_report(
    agent_type='ppo',
    save_path='reports/ppo_performance.txt'
)
```

### 7. Create Visualizations

```python
# Compare all models of an agent type
tracker.plot_performance_comparison(
    agent_type='ppo',
    metrics_to_plot=['sharpe_ratio', 'sortino_ratio', 'win_rate', 'max_drawdown'],
    save_path='plots/ppo_comparison.png'
)

# Compare specific models
tracker.plot_performance_comparison(
    models=['ppo_base', 'ppo_finetuned_20251003'],
    save_path='plots/ppo_before_after.png'
)

# Historical performance trend
tracker.plot_historical_performance(
    agent_type='ppo',
    metric='sharpe_ratio',
    save_path='plots/ppo_sharpe_history.png'
)
```

### 8. Convenience Functions

```python
from utils.performance_tracker import evaluate_and_save, compare_and_plot

# Evaluate and save in one call
metrics = evaluate_and_save(
    model=model,
    test_env=test_env,
    model_name='ppo_test',
    agent_type='ppo',
    notes='Testing new hyperparameters'
)

# Compare models and create plot
compare_and_plot(
    model_names=['ppo_base', 'ppo_v2', 'ppo_v3'],
    save_path='plots/ppo_versions.png'
)
```

---

## Example Workflow

### After Full Training

```bash
# 1. Train models (automatically evaluates at the end)
python train_finrl_models.py

# 2. Run analysis example
python examples/performance_analysis_example.py

# 3. View results
cat data/performance_comparison_ppo.png
```

### After Incremental Learning

```bash
# 1. Fine-tune on recent 60 days (automatically evaluates)
python train_finrl_incremental.py --days 60

# 2. Compare base vs fine-tuned
python -c "
from utils.performance_tracker import PerformanceTracker
tracker = PerformanceTracker()
comparison = tracker.compare_models('ppo_base', 'ppo_20251003_1430')
print(comparison)
"
```

### Weekly Performance Monitoring

```python
# monitor_weekly.py
from utils.performance_tracker import PerformanceTracker
import pandas as pd

tracker = PerformanceTracker()

# Get all PPO metrics
ppo_metrics = tracker.get_metrics(agent_type='ppo')

# Filter last 7 days
ppo_metrics['timestamp'] = pd.to_datetime(ppo_metrics['timestamp'])
last_week = ppo_metrics[
    ppo_metrics['timestamp'] >= pd.Timestamp.now() - pd.Timedelta(days=7)
]

# Check if performance is degrading
if len(last_week) > 0:
    latest_sharpe = last_week.iloc[0]['sharpe_ratio']

    if latest_sharpe < 1.0:
        print("⚠️ Warning: Sharpe ratio below 1.0. Consider retraining.")
    else:
        print("✅ Performance looks good!")
```

---

## Database Schema

The SQLite database stores metrics in the following schema:

```sql
CREATE TABLE performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,

    -- Core Performance
    sharpe_ratio REAL,
    sortino_ratio REAL,
    calmar_ratio REAL,
    win_rate REAL,
    loss_rate REAL,
    profit_factor REAL,

    -- Risk Metrics
    max_drawdown REAL,
    max_drawdown_duration INTEGER,
    volatility REAL,
    downside_deviation REAL,
    var_95 REAL,
    cvar_95 REAL,

    -- Return Metrics
    total_return REAL,
    annualized_return REAL,
    cumulative_return REAL,
    final_portfolio_value REAL,
    initial_portfolio_value REAL,

    -- Trade Statistics
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    avg_win REAL,
    avg_loss REAL,
    largest_win REAL,
    largest_loss REAL,
    avg_trade_duration REAL,

    -- Metadata
    evaluation_days INTEGER,
    test_data_start TEXT,
    test_data_end TEXT,
    notes TEXT,

    UNIQUE(model_name, timestamp)
);

CREATE INDEX idx_agent_type ON performance_metrics(agent_type);
CREATE INDEX idx_timestamp ON performance_metrics(timestamp);
```

You can query the database directly using:

```bash
sqlite3 data/performance_metrics.db "SELECT model_name, sharpe_ratio, win_rate FROM performance_metrics ORDER BY sharpe_ratio DESC LIMIT 5;"
```

---

## Best Practices

### ✅ DO

1. **Evaluate regularly** - Track performance after every training/fine-tuning
2. **Compare before deployment** - Always compare new models to baseline
3. **Monitor trends** - Use historical plots to spot performance degradation
4. **Use multiple metrics** - Don't rely on Sharpe ratio alone
5. **Save visualizations** - Keep plots for documentation and reports
6. **Document changes** - Use the `notes` parameter to track what changed

### ❌ DON'T

1. **Don't ignore drawdown** - High Sharpe with high drawdown is risky
2. **Don't overfit to one metric** - Balance return, risk, and win rate
3. **Don't delete the database** - It contains valuable historical data
4. **Don't skip evaluation** - Always evaluate before deploying to production
5. **Don't ignore validation data** - Evaluate on unseen test data

---

## Interpretation Guide

### Sharpe Ratio

- **> 2.0** - Excellent risk-adjusted returns
- **1.0 - 2.0** - Good performance
- **0.5 - 1.0** - Acceptable
- **< 0.5** - Poor, consider retraining

### Win Rate

- **> 60%** - Very strong
- **50% - 60%** - Good
- **40% - 50%** - Acceptable (if profit factor > 1.5)
- **< 40%** - Needs improvement

### Max Drawdown

- **< 10%** - Excellent risk management
- **10% - 20%** - Acceptable
- **20% - 30%** - High risk
- **> 30%** - Very high risk, review strategy

### Sortino Ratio

- Similar to Sharpe, but focuses on downside risk
- Higher is better
- Should be > Sharpe ratio (indicates good downside protection)

---

## Troubleshooting

### Issue: No metrics in database

**Cause:** Training hasn't completed yet or evaluation failed

**Solution:**
```bash
# Check if training is complete
ps aux | grep train_finrl

# Check logs
tail -100 logs/finrl_training.log

# Manually evaluate
python examples/performance_analysis_example.py
```

### Issue: Evaluation fails during training

**Cause:** Insufficient test data or environment creation error

**Solution:**
- Check that test data has at least 100 records
- Verify FinRL environment is properly configured
- Check logs for specific error messages

### Issue: Database locked error

**Cause:** Multiple processes accessing database simultaneously

**Solution:**
```python
# Add timeout when initializing
tracker = PerformanceTracker(db_path="data/performance_metrics.db")
# SQLite handles locking automatically with timeout
```

### Issue: Plots not generating

**Cause:** matplotlib not installed

**Solution:**
```bash
pip install matplotlib
```

---

## Advanced Usage

### Custom Metrics

You can extend the `PerformanceMetrics` dataclass to track additional metrics:

```python
from dataclasses import dataclass
from utils.performance_tracker import PerformanceMetrics

@dataclass
class ExtendedMetrics(PerformanceMetrics):
    information_ratio: float
    omega_ratio: float
    tail_ratio: float
```

### Custom Evaluation Logic

Override `_extract_trade_statistics` for your specific action space:

```python
class CustomTracker(PerformanceTracker):
    def _extract_trade_statistics(self, actions_log, daily_returns):
        # Your custom logic based on your action space
        # ...
        return stats_dict
```

---

## Examples

See `examples/performance_analysis_example.py` for a comprehensive demonstration of all features:

```bash
python examples/performance_analysis_example.py
```

This will:
- Retrieve all metrics from database
- Show PPO performance history
- Find best models by Sharpe ratio
- Compare two models
- Generate performance report
- Create visualizations
- Show best model per agent type

---

## Summary

The Performance Tracking System provides:

✅ **Automated evaluation** after every training run
✅ **Comprehensive metrics** covering risk, return, and trades
✅ **Historical tracking** in SQLite database
✅ **Easy comparison** between models and agents
✅ **Visual analysis** with matplotlib plots
✅ **Production-ready** integration with training workflows

**Database:** `data/performance_metrics.db`
**Visualizations:** `data/performance_*.png`
**Reports:** Generate on-demand with `generate_performance_report()`

For questions or issues, check the logs or refer to this documentation.

**Happy Trading! 📈**
