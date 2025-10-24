"""
Prefect Flow for FinRL Automatic Retraining

Modern workflow orchestration with Prefect 2.0.

Installation:
    pip install prefect

Setup:
    1. Start Prefect server: prefect server start
    2. Configure deployment: python workflows/prefect_flow.py
    3. Create deployment: prefect deployment build -n finrl-retraining
    4. Apply deployment: prefect deployment apply

Features:
    - Automatic retries with exponential backoff
    - Task caching for efficiency
    - Real-time monitoring via Prefect UI
    - Email/Slack notifications
    - Distributed execution support
"""

from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import sys
from typing import Dict, Any

# Add trading system to path
TRADING_SYSTEM_PATH = '/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system'
sys.path.insert(0, TRADING_SYSTEM_PATH)

from scheduled_retraining import AutomaticRetrainingScheduler
from utils.market_regime_detector import MarketRegimeDetector
from utils.performance_tracker import PerformanceTracker


@task(
    name="Detect Market Regime",
    description="Detect current market volatility regime",
    retries=2,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)
def detect_market_regime() -> Dict[str, Any]:
    """Detect current market regime."""
    detector = MarketRegimeDetector(symbols=['SPY', 'QQQ'])
    regime = detector.detect_current_regime()

    return {
        'regime': regime.regime,
        'volatility': regime.volatility,
        'recommended_lookback': regime.recommended_lookback,
        'confidence': regime.confidence
    }


@task(
    name="Get Baseline Metrics",
    description="Retrieve baseline performance metrics",
    retries=2,
    retry_delay_seconds=30
)
def get_baseline_metrics(agents: list) -> Dict[str, Any]:
    """Get current performance metrics as baseline."""
    tracker = PerformanceTracker()
    baseline = {}

    for agent_type in agents:
        metrics = tracker.get_metrics(agent_type=agent_type, limit=1)
        if not metrics.empty:
            baseline[agent_type] = {
                'sharpe_ratio': metrics.iloc[0]['sharpe_ratio'],
                'win_rate': metrics.iloc[0]['win_rate'],
                'model_name': metrics.iloc[0]['model_name']
            }

    return baseline


@task(
    name="Run Incremental Training",
    description="Execute incremental model training",
    retries=1,
    retry_delay_seconds=300,
    timeout_seconds=7200  # 2 hours
)
def run_incremental_training(lookback_days: int, agents: list) -> Dict[str, Any]:
    """Run incremental training."""
    from train_finrl_incremental import IncrementalTrainer

    trainer = IncrementalTrainer(lookback_days=lookback_days)
    results = trainer.fine_tune_all_agents(agents_to_train=agents)

    return results


@task(
    name="Validate Performance",
    description="Validate retrained model performance",
    retries=1,
    retry_delay_seconds=30
)
def validate_performance(
    training_results: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    min_sharpe_threshold: float = 0.8,
    max_degradation: float = 0.10
) -> Dict[str, Any]:
    """Validate performance and determine if rollback needed."""
    validation_results = {}

    for agent_type, training_info in training_results.items():
        if 'error' in training_info:
            validation_results[agent_type] = {
                'valid': False,
                'reason': training_info['error']
            }
            continue

        perf_metrics = training_info.get('performance_metrics')
        if not perf_metrics:
            validation_results[agent_type] = {
                'valid': False,
                'reason': 'No performance metrics'
            }
            continue

        baseline = baseline_metrics.get(agent_type)
        new_sharpe = perf_metrics['sharpe_ratio']

        # Check absolute threshold
        passes_threshold = new_sharpe >= min_sharpe_threshold

        # Check degradation vs baseline
        performance_acceptable = True
        if baseline:
            baseline_sharpe = baseline['sharpe_ratio']
            degradation = (baseline_sharpe - new_sharpe) / baseline_sharpe if baseline_sharpe > 0 else 0
            performance_acceptable = degradation <= max_degradation

        validation_results[agent_type] = {
            'valid': passes_threshold and performance_acceptable,
            'new_sharpe': new_sharpe,
            'baseline_sharpe': baseline.get('sharpe_ratio') if baseline else None,
            'passes_threshold': passes_threshold,
            'performance_acceptable': performance_acceptable
        }

    return validation_results


@task(
    name="Rollback Models",
    description="Rollback models that failed validation",
    retries=1,
    retry_delay_seconds=30
)
def rollback_failed_models(validation_results: Dict[str, Any]) -> list:
    """Rollback models that failed validation."""
    from pathlib import Path
    import shutil

    rollback_list = []
    backup_dir = Path("data/finrl_models_backup")
    incremental_dir = Path("data/finrl_models_incremental")

    for agent_type, validation_info in validation_results.items():
        if not validation_info.get('valid', True):
            # Find most recent backup
            backup_dirs = sorted(
                backup_dir.glob("backup_*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            if backup_dirs:
                latest_backup = backup_dirs[0]
                for backup_file in latest_backup.glob(f"agent_{agent_type}_*.zip"):
                    target = incremental_dir / backup_file.name
                    shutil.copy2(backup_file, target)
                    rollback_list.append(agent_type)

    return rollback_list


@flow(
    name="FinRL Automatic Retraining",
    description="End-to-end automatic retraining with adaptive lookback and rollback",
    retries=1,
    retry_delay_seconds=600
)
def finrl_retraining_flow(
    agents: list = None,
    min_sharpe_threshold: float = 0.8,
    max_degradation: float = 0.10
) -> Dict[str, Any]:
    """
    Main retraining flow.

    Args:
        agents: List of agents to retrain (default: all)
        min_sharpe_threshold: Minimum acceptable Sharpe ratio
        max_degradation: Maximum acceptable performance degradation

    Returns:
        Flow execution results
    """
    agents = agents or ['a2c', 'ppo', 'ddpg', 'sac', 'td3']

    # Step 1: Detect market regime
    regime_info = detect_market_regime()
    lookback_days = regime_info['recommended_lookback']

    print(f"📊 Market Regime: {regime_info['regime'].upper()}")
    print(f"   Lookback: {lookback_days} days")

    # Step 2: Get baseline metrics
    baseline_metrics = get_baseline_metrics(agents)

    # Step 3: Run incremental training
    training_results = run_incremental_training(lookback_days, agents)

    # Step 4: Validate performance
    validation_results = validate_performance(
        training_results,
        baseline_metrics,
        min_sharpe_threshold,
        max_degradation
    )

    # Step 5: Rollback if needed
    rollback_list = rollback_failed_models(validation_results)

    # Compile results
    results = {
        'regime': regime_info,
        'agents_retrained': list(training_results.keys()),
        'validation': validation_results,
        'rollbacks': rollback_list,
        'success': all(v.get('valid', False) for v in validation_results.values())
    }

    print(f"\n✅ Retraining flow completed")
    print(f"   Success: {results['success']}")
    if rollback_list:
        print(f"   ⚠️ Rolled back: {', '.join(rollback_list)}")

    return results


@flow(
    name="FinRL Simple Retraining",
    description="Simplified retraining without validation (use for testing)"
)
def simple_retraining_flow(lookback_days: int = 60):
    """Simplified retraining flow for testing."""
    scheduler = AutomaticRetrainingScheduler()
    results = scheduler.run_scheduled_retraining()
    return results


# Deployment configuration
if __name__ == "__main__":
    # Run locally for testing
    print("Running Prefect retraining flow locally...")
    results = finrl_retraining_flow()
    print(f"\n📊 Results: {results}")

    # To create a deployment:
    # 1. prefect deployment build workflows/prefect_flow.py:finrl_retraining_flow -n weekly-retraining
    # 2. prefect deployment apply finrl_retraining_flow-deployment.yaml
    # 3. prefect agent start -q default
