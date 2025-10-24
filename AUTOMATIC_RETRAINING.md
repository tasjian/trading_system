# FinRL Automatic Retraining System

## Overview

The **Automatic Retraining System** provides production-ready scheduled retraining with adaptive lookback periods, performance monitoring, and automatic rollback capabilities. It's designed to keep your FinRL models up-to-date with current market conditions without manual intervention.

### Key Features

✅ **Adaptive Lookback** - Automatically adjusts training window based on market volatility
✅ **Market Regime Detection** - Classifies volatility (low/normal/high) and recommends optimal lookback
✅ **Performance Monitoring** - Validates retrained models against baseline metrics
✅ **Automatic Rollback** - Reverts to previous models if performance degrades
✅ **Model Versioning** - Automatic backup before retraining with timestamped versions
✅ **Flexible Scheduling** - Integration with cron, Airflow, Prefect
✅ **Production Ready** - Error handling, logging, notifications

---

## Architecture

### Components

```
scheduled_retraining.py                # Main retraining orchestrator
├── AutomaticRetrainingScheduler       # Core scheduler class
│   ├── run_scheduled_retraining()     # Main workflow
│   ├── _detect_regime()               # Market volatility detection
│   ├── _run_incremental_training()    # Execute training
│   ├── _evaluate_performance()        # Validate new models
│   └── _validate_and_rollback()       # Rollback if needed

utils/market_regime_detector.py        # Volatility detection
├── MarketRegimeDetector               # Regime classification
│   ├── detect_current_regime()        # Current market state
│   ├── _calculate_realized_volatility()
│   └── should_retrain_urgently()      # Emergency retraining check

agents/finrl_agent_wrapper.py          # Integration points
├── automatic_retraining()             # Full auto retraining
├── adaptive_incremental_training()    # Simple adaptive training
└── incremental_train_models()         # Manual training

workflows/                              # Orchestration integrations
├── airflow_dag.py                     # Apache Airflow DAG
├── prefect_flow.py                    # Prefect 2.0 flow
└── setup_cron_retraining.sh           # Cron setup script
```

---

## Quick Start

### Option 1: Manual Execution

```bash
# Run retraining once
python scheduled_retraining.py
```

### Option 2: Cron Schedule (Recommended for Production)

```bash
# Interactive setup
./setup_cron_retraining.sh

# Select schedule:
#   1. Weekly (Sunday 2:00 AM) - Recommended
#   2. Bi-weekly (1st and 15th)
#   3. Custom cron expression
```

### Option 3: Programmatic (from Agent Wrapper)

```python
from agents.finrl_agent_wrapper import FinRLAgentWrapper

# Initialize agent
agent = FinRLAgentWrapper(symbols=['AAPL', 'MSFT', 'GOOGL'])
await agent.initialize_system()

# Full automatic retraining with validation and rollback
results = await agent.automatic_retraining(
    min_sharpe_threshold=0.8,
    max_performance_degradation=0.10,
    enable_rollback=True
)

# Simpler: Adaptive lookback only (no validation/rollback)
metrics = await agent.adaptive_incremental_training(
    use_market_regime=True
)

# Manual: Fixed lookback
metrics = await agent.incremental_train_models(lookback_days=60)
```

### Option 4: Airflow (Enterprise)

```bash
# Copy DAG to Airflow folder
cp workflows/airflow_dag.py ~/airflow/dags/

# Restart Airflow scheduler
airflow scheduler
```

### Option 5: Prefect (Modern Workflow)

```bash
# Install Prefect
pip install prefect

# Deploy flow
python workflows/prefect_flow.py
```

---

## Adaptive Lookback Strategy

The system automatically adjusts the lookback period based on market volatility:

### Volatility Regimes

| Regime | Volatility Range | Lookback Days | Rationale |
|--------|------------------|---------------|-----------|
| **Low Volatility** | < 10% annualized | 90 days | Stable markets - longer window for robust learning |
| **Normal Volatility** | 10% - 25% annualized | 60 days | Standard conditions - balanced approach |
| **High Volatility** | > 25% annualized | 30 days | Rapid changes - shorter window for quick adaptation |

### Volatility Detection

The system uses **realized volatility** calculated from recent market data:

```python
from utils.market_regime_detector import MarketRegimeDetector

detector = MarketRegimeDetector(symbols=['SPY', 'QQQ'])  # Market proxies
regime = detector.detect_current_regime()

print(f"Regime: {regime.regime}")                    # 'low', 'normal', 'high'
print(f"Volatility: {regime.volatility:.2%}")        # e.g., 18.5%
print(f"Lookback: {regime.recommended_lookback}")    # e.g., 60 days
```

### Example Scenarios

**Scenario 1: Normal Market (March 2024)**
- Realized volatility: 15%
- Regime: `normal`
- Lookback: **60 days**
- Agents learn from balanced recent data

**Scenario 2: Market Crash (COVID-19 crash)**
- Realized volatility: 45%
- Regime: `high`
- Lookback: **30 days**
- Agents quickly adapt to new volatile conditions

**Scenario 3: Calm Bull Market**
- Realized volatility: 8%
- Regime: `low`
- Lookback: **90 days**
- Agents learn from extended stable period

---

## Performance Monitoring & Rollback

### Validation Metrics

After each retraining, the system validates performance using:

1. **Absolute Threshold**: Sharpe ratio must be >= `min_sharpe_threshold` (default: 0.8)
2. **Relative Performance**: New Sharpe must not degrade by more than `max_performance_degradation` (default: 10%)

### Rollback Logic

```python
# Example: PPO agent retraining

# Baseline (before retraining)
baseline_sharpe = 1.5

# After retraining
new_sharpe = 1.3

# Performance degradation
degradation = (1.5 - 1.3) / 1.5 = 13.3%

# Result: ROLLBACK (degradation > 10% threshold)
# System automatically restores previous PPO model from backup
```

### Rollback Flow

```
1. Backup models before retraining
   ↓
2. Run incremental training
   ↓
3. Evaluate new models on test data
   ↓
4. Compare to baseline:
   - Sharpe >= threshold? ✓
   - Degradation <= max? ✓
   ↓
   YES → Keep new models
   NO  → Restore from backup
   ↓
5. Log results and send notifications
```

### Example: Successful Retraining

```
📊 PPO Evaluation:
   Baseline Sharpe: 1.42
   New Sharpe: 1.48 (+0.06, +4.2%)
   ✅ PASS - Performance improved

📊 SAC Evaluation:
   Baseline Sharpe: 1.38
   New Sharpe: 1.35 (-0.03, -2.2%)
   ✅ PASS - Degradation within threshold (2.2% < 10%)

✅ All agents passed validation, keeping new models
```

### Example: Rollback Required

```
📊 DDPG Evaluation:
   Baseline Sharpe: 1.55
   New Sharpe: 1.28 (-0.27, -17.4%)
   ❌ FAIL - Degradation exceeds threshold (17.4% > 10%)

⚠️ DDPG: ROLLING BACK - Performance degraded by 17.4%
   Restored: agent_ddpg_20251001_1430.zip

🔄 Rolled back 1 agent(s): ddpg
```

---

## Configuration

### Default Configuration

```python
config = {
    # Performance thresholds
    'min_sharpe_threshold': 0.8,          # Minimum acceptable Sharpe
    'max_performance_degradation': 0.10,  # Max 10% performance drop

    # Lookback defaults (overridden by regime detection)
    'default_lookback_days': 60,
    'low_volatility_lookback': 90,
    'high_volatility_lookback': 30,

    # Agents to retrain
    'agents_to_train': ['a2c', 'ppo', 'ddpg', 'sac', 'td3'],

    # Safety features
    'enable_rollback': True,
    'backup_before_retraining': True,

    # Urgent retraining (extreme volatility)
    'enable_urgent_retraining': True,
    'urgent_volatility_threshold': 0.30,  # 30%

    # Notifications (optional)
    'send_notifications': False,
    'notification_email': None
}
```

### Custom Configuration

```python
from scheduled_retraining import AutomaticRetrainingScheduler

# Conservative: Higher thresholds, bi-weekly
config = {
    'min_sharpe_threshold': 1.0,           # Stricter threshold
    'max_performance_degradation': 0.05,   # Only 5% max drop
    'enable_rollback': True
}

scheduler = AutomaticRetrainingScheduler(config=config)
results = scheduler.run_scheduled_retraining()
```

---

## Scheduling Options

### 1. Cron (Simple & Reliable)

**Weekly retraining (Sundays at 2:00 AM)**
```bash
0 2 * * 0 cd /path/to/trading_system && python scheduled_retraining.py >> logs/cron_retraining.log 2>&1
```

**Bi-weekly (1st and 15th at 2:00 AM)**
```bash
0 2 1,15 * * cd /path/to/trading_system && python scheduled_retraining.py >> logs/cron_retraining.log 2>&1
```

**Monthly (first Sunday of month)**
```bash
0 2 1-7 * 0 cd /path/to/trading_system && python scheduled_retraining.py >> logs/cron_retraining.log 2>&1
```

### 2. Airflow (Enterprise Orchestration)

Features:
- Task dependencies
- Retry logic
- Web UI for monitoring
- Email notifications
- Distributed execution

```python
# See workflows/airflow_dag.py

# Schedule: Every Sunday at 2:00 AM
schedule_interval='0 2 * * 0'

# Tasks:
# 1. Health check
# 2. Run retraining
# 3. Validate results
# 4. Update system
```

### 3. Prefect (Modern Orchestration)

Features:
- Real-time monitoring
- Task caching
- Automatic retries
- Cloud deployment
- Python-native

```python
# See workflows/prefect_flow.py

# Deploy:
prefect deployment build workflows/prefect_flow.py:finrl_retraining_flow -n weekly
prefect deployment apply finrl_retraining_flow-deployment.yaml
prefect agent start
```

### 4. Systemd Timer (Linux)

Create `/etc/systemd/system/finrl-retraining.service`:
```ini
[Unit]
Description=FinRL Automatic Retraining
After=network.target

[Service]
Type=oneshot
User=your-user
WorkingDirectory=/path/to/trading_system
ExecStart=/usr/bin/python3 scheduled_retraining.py
```

Create `/etc/systemd/system/finrl-retraining.timer`:
```ini
[Unit]
Description=Weekly FinRL Retraining Timer

[Timer]
OnCalendar=Sun *-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable:
```bash
sudo systemctl enable finrl-retraining.timer
sudo systemctl start finrl-retraining.timer
```

---

## Usage Examples

### Example 1: Weekly Scheduled Retraining

```bash
# Set up weekly cron job
./setup_cron_retraining.sh

# Select option 1 (Weekly - Sunday 2:00 AM)

# Monitor logs
tail -f logs/cron_retraining.log
```

### Example 2: Programmatic with Custom Thresholds

```python
import asyncio
from agents.finrl_agent_wrapper import FinRLAgentWrapper

async def main():
    agent = FinRLAgentWrapper()
    await agent.initialize_system()

    # Conservative: Only accept improvements
    results = await agent.automatic_retraining(
        min_sharpe_threshold=1.2,
        max_performance_degradation=0.0,  # No degradation allowed
        enable_rollback=True
    )

    print(f"Success: {results['success']}")
    print(f"Rollbacks: {results['rollbacks']}")

asyncio.run(main())
```

### Example 3: Manual with Regime Detection

```python
from utils.market_regime_detector import MarketRegimeDetector
from train_finrl_incremental import IncrementalTrainer

# Detect regime
detector = MarketRegimeDetector()
regime = detector.detect_current_regime()

print(f"Market: {regime.regime}, Lookback: {regime.recommended_lookback} days")

# Train with adaptive lookback
trainer = IncrementalTrainer(lookback_days=regime.recommended_lookback)
metrics = trainer.fine_tune_all_agents()
```

### Example 4: Emergency Retraining (High Volatility)

```python
from utils.market_regime_detector import MarketRegimeDetector

detector = MarketRegimeDetector()

# Check if urgent retraining needed
if detector.should_retrain_urgently(threshold_volatility=0.30):
    print("🚨 URGENT RETRAINING NEEDED!")

    # Use shortest lookback for quick adaptation
    from train_finrl_incremental import IncrementalTrainer
    trainer = IncrementalTrainer(lookback_days=30)
    metrics = trainer.fine_tune_all_agents()
```

---

## Monitoring & Logs

### Log Files

```
logs/
├── scheduled_retraining.log      # Main retraining log
├── cron_retraining.log           # Cron execution log
└── finrl_training.log            # Training details

data/retraining_summaries/
└── retraining_summary_YYYYMMDD_HHMMSS.json  # Detailed results
```

### Monitoring Commands

```bash
# Watch retraining log
tail -f logs/scheduled_retraining.log

# Check last retraining result
ls -lt data/retraining_summaries/ | head -5

# View summary
cat data/retraining_summaries/retraining_summary_20251003_020015.json | jq .

# Check rollbacks
grep "ROLLING BACK" logs/scheduled_retraining.log

# Monitor cron execution
tail -f logs/cron_retraining.log
```

### Summary JSON Structure

```json
{
  "timestamp": "2025-10-03T02:00:15",
  "success": true,
  "regime": "normal",
  "lookback_days": 60,
  "agents_retrained": ["a2c", "ppo", "ddpg", "sac", "td3"],
  "rollbacks": [],
  "errors": [],
  "training_results": {
    "ppo": {
      "timesteps": 30000,
      "training_time_seconds": 95.3,
      "performance_metrics": {
        "sharpe_ratio": 1.48,
        "win_rate": 0.56,
        "max_drawdown": -0.082
      }
    }
  },
  "evaluation_results": {
    "ppo": {
      "new_sharpe": 1.48,
      "baseline_sharpe": 1.42,
      "sharpe_change": 0.06,
      "sharpe_change_pct": 0.042,
      "passes_threshold": true,
      "performance_acceptable": true
    }
  }
}
```

---

## Best Practices

### ✅ DO

1. **Start with weekly schedule** - Bi-weekly for conservative approach
2. **Monitor first few runs** - Watch logs and verify rollbacks work
3. **Use market regime detection** - Let volatility guide lookback
4. **Enable rollback** - Safety net for production systems
5. **Backup before retraining** - Automatic but verify backups exist
6. **Test during market hours** - Ensure no conflicts with trading
7. **Use conservative thresholds** - Start strict, relax if needed
8. **Monitor performance metrics** - Track Sharpe, win rate, drawdown
9. **Schedule during off-hours** - 2-4 AM to avoid market hours
10. **Keep logs** - Retain for debugging and analysis

### ❌ DON'T

1. **Don't retrain daily** - Too frequent, models need time to prove themselves
2. **Don't disable rollback in production** - Critical safety feature
3. **Don't ignore regime changes** - High volatility needs different lookback
4. **Don't use very low thresholds** - Sharpe < 0.5 is too risky
5. **Don't retrain during market hours** - May interfere with trading
6. **Don't delete backups** - Need for rollback capability
7. **Don't skip validation** - Always evaluate before deploying
8. **Don't ignore failed retraining** - Investigate and fix issues
9. **Don't use untested thresholds** - Validate on historical data first
10. **Don't over-optimize** - Balance performance with stability

---

## Recommended Schedules

### Conservative (Recommended for Production)

- **Frequency**: Bi-weekly (1st and 15th)
- **Lookback**: Adaptive (30-90 days based on volatility)
- **Thresholds**:
  - Min Sharpe: 0.8
  - Max degradation: 10%
- **Rollback**: Enabled
- **Best for**: Live trading, risk-averse strategies

### Balanced (Standard)

- **Frequency**: Weekly (every Sunday)
- **Lookback**: Adaptive (30-90 days)
- **Thresholds**:
  - Min Sharpe: 0.7
  - Max degradation: 15%
- **Rollback**: Enabled
- **Best for**: Most production systems

### Aggressive (High Adaptation)

- **Frequency**: Weekly (Sundays + urgent retraining)
- **Lookback**: Adaptive with shorter windows
- **Thresholds**:
  - Min Sharpe: 0.6
  - Max degradation: 20%
- **Rollback**: Enabled
- **Best for**: Volatile markets, experimental strategies

---

## Troubleshooting

### Issue: Retraining fails silently

**Solution:**
```bash
# Check cron logs
tail -100 logs/cron_retraining.log

# Test manually
python scheduled_retraining.py

# Check cron job is active
crontab -l | grep scheduled_retraining
```

### Issue: All models get rolled back

**Cause:** Thresholds too strict or market regime change

**Solution:**
```python
# Relax thresholds
config = {
    'min_sharpe_threshold': 0.6,  # Lower threshold
    'max_performance_degradation': 0.15  # Allow 15% degradation
}
```

### Issue: Training takes too long

**Cause:** Too many timesteps or large lookback window

**Solution:**
```python
# Reduce timesteps
timesteps_override = {
    'ppo': 20000,  # Reduce from 30000
    'sac': 15000   # Reduce from 24000
}

# Or reduce lookback
lookback_days = 45  # Instead of 60
```

### Issue: Models don't improve

**Cause:** Lookback too short or not enough training

**Solution:**
```python
# Increase lookback
lookback_days = 90  # More data

# Increase timesteps
timesteps_override = {
    'ppo': 50000,  # More training
    'sac': 40000
}
```

---

## Summary

**The Automatic Retraining System provides:**

✅ **Adaptive lookback** based on market volatility (30-90 days)
✅ **Performance validation** with automatic rollback on degradation
✅ **Model versioning** with timestamped backups
✅ **Flexible scheduling** via cron, Airflow, Prefect
✅ **Production-ready** error handling and logging
✅ **Integration** with FinRL agent wrapper

**Recommended Setup:**

1. **Schedule:** Weekly (every Sunday at 2:00 AM)
2. **Lookback:** Adaptive (use regime detection)
3. **Thresholds:** min_sharpe=0.8, max_degradation=10%
4. **Rollback:** Enabled
5. **Monitoring:** Check logs weekly

**Quick Start:**
```bash
# Set up cron schedule
./setup_cron_retraining.sh

# Monitor first run
tail -f logs/cron_retraining.log
```

**Programmatic:**
```python
results = await agent.automatic_retraining()
```

For questions or issues, check the logs or refer to this documentation.

**Happy Trading! 📈**
