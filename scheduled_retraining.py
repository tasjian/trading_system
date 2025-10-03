#!/usr/bin/env python3
"""
Automatic FinRL Retraining Scheduler

Production-ready scheduled retraining system with:
- Adaptive lookback based on market volatility
- Automatic performance monitoring
- Rollback on performance degradation
- Model versioning and backup
- Integration with cron, Airflow, Prefect

Usage:
    # Run manually
    python scheduled_retraining.py

    # Schedule with cron (add to crontab -e)
    0 2 * * 0 cd /path/to/trading_system && python scheduled_retraining.py

    # Schedule with Airflow (see workflows/airflow_dag.py)
    # Schedule with Prefect (see workflows/prefect_flow.py)
"""

import sys
import os
import logging
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil

# Add trading system to path
sys.path.insert(0, '/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system')

from train_finrl_incremental import IncrementalTrainer
from utils.performance_tracker import PerformanceTracker
from utils.market_regime_detector import MarketRegimeDetector
from agents.finrl_agent_wrapper import FinRLAgentWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduled_retraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AutomaticRetrainingScheduler:
    """
    Production-ready automatic retraining system.

    Features:
    - Adaptive lookback based on volatility
    - Performance monitoring with automatic rollback
    - Model versioning and backup
    - Configurable thresholds
    - Integration with orchestration tools
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize retraining scheduler.

        Args:
            config: Optional configuration overrides
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        self.symbols = self.config['symbols']
        self.performance_tracker = PerformanceTracker(db_path="data/performance_metrics.db")
        self.regime_detector = MarketRegimeDetector(symbols=['SPY', 'QQQ'])

        # Directories
        self.model_dir = Path("data/finrl_models")
        self.incremental_dir = Path("data/finrl_models_incremental")
        self.backup_dir = Path("data/finrl_models_backup")

        # Create directories
        for dir_path in [self.model_dir, self.incremental_dir, self.backup_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info("🔄 Initialized Automatic Retraining Scheduler")
        logger.info(f"   Symbols: {len(self.symbols)}")
        logger.info(f"   Min Sharpe Threshold: {self.config['min_sharpe_threshold']}")
        logger.info(f"   Performance Degradation: {self.config['max_performance_degradation']:.1%}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V'],

            # Performance thresholds
            'min_sharpe_threshold': 0.8,  # Minimum acceptable Sharpe ratio
            'max_performance_degradation': 0.10,  # Max 10% performance drop vs baseline

            # Lookback defaults (overridden by volatility detection)
            'default_lookback_days': 60,
            'low_volatility_lookback': 90,
            'high_volatility_lookback': 30,

            # Retraining configuration
            'agents_to_train': ['a2c', 'ppo', 'ddpg', 'sac', 'td3'],
            'enable_rollback': True,
            'backup_before_retraining': True,

            # Monitoring
            'send_notifications': False,  # Set to True to enable email notifications
            'notification_email': None,

            # Aggressive retraining (during extreme volatility)
            'enable_urgent_retraining': True,
            'urgent_volatility_threshold': 0.30,  # 30% annualized volatility
        }

    def run_scheduled_retraining(self) -> Dict[str, Any]:
        """
        Run scheduled retraining workflow.

        Returns:
            Dictionary with retraining results
        """
        logger.info("\n" + "=" * 80)
        logger.info("🚀 SCHEDULED RETRAINING WORKFLOW")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        results = {
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'agents_retrained': [],
            'rollbacks': [],
            'errors': [],
            'regime': None,
            'lookback_days': None
        }

        try:
            # Step 1: Detect market regime
            logger.info("\n📊 Step 1: Market Regime Detection")
            regime = self.regime_detector.detect_current_regime()
            results['regime'] = regime.regime
            results['lookback_days'] = regime.recommended_lookback

            logger.info(f"   Market Regime: {regime.regime.upper()}")
            logger.info(f"   Volatility: {regime.volatility:.2%}")
            logger.info(f"   Recommended Lookback: {regime.recommended_lookback} days")

            # Check for urgent retraining
            if self.config['enable_urgent_retraining']:
                if self.regime_detector.should_retrain_urgently(
                    self.config['urgent_volatility_threshold']
                ):
                    logger.warning("🚨 URGENT RETRAINING TRIGGERED!")
                    regime.recommended_lookback = self.config['high_volatility_lookback']

            # Step 2: Backup existing models
            if self.config['backup_before_retraining']:
                logger.info("\n💾 Step 2: Backing up existing models")
                self._backup_models()

            # Step 3: Get baseline performance
            logger.info("\n📊 Step 3: Recording baseline performance")
            baseline_metrics = self._get_baseline_metrics()

            # Step 4: Run incremental training
            logger.info("\n🏋️ Step 4: Running incremental training")
            training_results = self._run_incremental_training(
                lookback_days=regime.recommended_lookback
            )

            results['agents_retrained'] = list(training_results.keys())

            # Step 5: Evaluate new models
            logger.info("\n📊 Step 5: Evaluating retrained models")
            evaluation_results = self._evaluate_retrained_models(
                training_results,
                baseline_metrics
            )

            # Step 6: Rollback if needed
            logger.info("\n🔄 Step 6: Performance validation and rollback")
            rollback_results = self._validate_and_rollback(
                evaluation_results,
                baseline_metrics
            )

            results['rollbacks'] = rollback_results

            # Step 7: Save summary
            logger.info("\n💾 Step 7: Saving retraining summary")
            self._save_summary(results, training_results, evaluation_results)

            results['success'] = True

            logger.info("\n" + "=" * 80)
            logger.info("✅ SCHEDULED RETRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"\n❌ Retraining failed: {e}")
            results['errors'].append(str(e))
            results['success'] = False

            # Send notification if enabled
            if self.config['send_notifications']:
                self._send_notification(f"Retraining failed: {e}")

        return results

    def _backup_models(self):
        """Backup current models before retraining."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / f"backup_{timestamp}"
        backup_subdir.mkdir(exist_ok=True)

        # Backup incremental models
        if self.incremental_dir.exists():
            for model_file in self.incremental_dir.glob("agent_*.zip"):
                shutil.copy2(model_file, backup_subdir / model_file.name)
                logger.info(f"   Backed up: {model_file.name}")

        logger.info(f"✅ Models backed up to: {backup_subdir}")

    def _get_baseline_metrics(self) -> Dict[str, Any]:
        """Get baseline performance metrics before retraining."""
        baseline = {}

        for agent_type in self.config['agents_to_train']:
            # Get most recent metrics for this agent
            metrics = self.performance_tracker.get_metrics(
                agent_type=agent_type,
                limit=1
            )

            if not metrics.empty:
                baseline[agent_type] = {
                    'sharpe_ratio': metrics.iloc[0]['sharpe_ratio'],
                    'win_rate': metrics.iloc[0]['win_rate'],
                    'max_drawdown': metrics.iloc[0]['max_drawdown'],
                    'model_name': metrics.iloc[0]['model_name']
                }
                logger.info(f"   {agent_type.upper()}: Sharpe={baseline[agent_type]['sharpe_ratio']:.3f}")
            else:
                logger.warning(f"   ⚠️ No baseline metrics for {agent_type}")
                baseline[agent_type] = None

        return baseline

    def _run_incremental_training(self, lookback_days: int) -> Dict[str, Dict]:
        """
        Run incremental training on all agents.

        Args:
            lookback_days: Days of recent data for training

        Returns:
            Dictionary of training results by agent
        """
        trainer = IncrementalTrainer(
            symbols=self.symbols,
            lookback_days=lookback_days,
            checkpoint_dir=str(self.model_dir),
            incremental_dir=str(self.incremental_dir)
        )

        training_results = trainer.fine_tune_all_agents(
            agents_to_train=self.config['agents_to_train']
        )

        return training_results

    def _evaluate_retrained_models(
        self,
        training_results: Dict[str, Dict],
        baseline_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate retrained models and compare to baseline.

        Args:
            training_results: Results from training
            baseline_metrics: Baseline performance before retraining

        Returns:
            Evaluation results
        """
        evaluation_results = {}

        for agent_type, training_info in training_results.items():
            if 'error' in training_info:
                evaluation_results[agent_type] = {'error': training_info['error']}
                continue

            # Get new performance metrics
            perf_metrics = training_info.get('performance_metrics')

            if not perf_metrics:
                logger.warning(f"⚠️ No performance metrics for {agent_type}")
                evaluation_results[agent_type] = {'error': 'No performance metrics'}
                continue

            baseline = baseline_metrics.get(agent_type)

            if baseline:
                # Calculate performance change
                sharpe_change = perf_metrics['sharpe_ratio'] - baseline['sharpe_ratio']
                sharpe_change_pct = (sharpe_change / baseline['sharpe_ratio']) if baseline['sharpe_ratio'] != 0 else 0

                evaluation_results[agent_type] = {
                    'new_sharpe': perf_metrics['sharpe_ratio'],
                    'baseline_sharpe': baseline['sharpe_ratio'],
                    'sharpe_change': sharpe_change,
                    'sharpe_change_pct': sharpe_change_pct,
                    'new_win_rate': perf_metrics['win_rate'],
                    'baseline_win_rate': baseline['win_rate'],
                    'new_max_drawdown': perf_metrics['max_drawdown'],
                    'baseline_max_drawdown': baseline['max_drawdown'],
                    'passes_threshold': perf_metrics['sharpe_ratio'] >= self.config['min_sharpe_threshold'],
                    'performance_acceptable': sharpe_change_pct >= -self.config['max_performance_degradation']
                }

                logger.info(f"   {agent_type.upper()}:")
                logger.info(f"      Sharpe: {baseline['sharpe_ratio']:.3f} → {perf_metrics['sharpe_ratio']:.3f} "
                           f"({sharpe_change:+.3f}, {sharpe_change_pct:+.1%})")
                logger.info(f"      Status: {'✅ PASS' if evaluation_results[agent_type]['performance_acceptable'] else '❌ FAIL'}")
            else:
                # No baseline - just check absolute threshold
                evaluation_results[agent_type] = {
                    'new_sharpe': perf_metrics['sharpe_ratio'],
                    'baseline_sharpe': None,
                    'passes_threshold': perf_metrics['sharpe_ratio'] >= self.config['min_sharpe_threshold'],
                    'performance_acceptable': True  # No baseline to compare
                }

        return evaluation_results

    def _validate_and_rollback(
        self,
        evaluation_results: Dict[str, Any],
        baseline_metrics: Dict[str, Any]
    ) -> List[str]:
        """
        Validate retrained models and rollback if performance degrades.

        Args:
            evaluation_results: Evaluation results
            baseline_metrics: Baseline metrics

        Returns:
            List of agents that were rolled back
        """
        if not self.config['enable_rollback']:
            logger.info("   Rollback disabled in configuration")
            return []

        rollback_list = []

        for agent_type, eval_info in evaluation_results.items():
            if 'error' in eval_info:
                continue

            # Check if rollback needed
            needs_rollback = False
            reason = ""

            if not eval_info.get('passes_threshold', True):
                needs_rollback = True
                reason = f"Sharpe below threshold ({eval_info['new_sharpe']:.3f} < {self.config['min_sharpe_threshold']})"

            elif not eval_info.get('performance_acceptable', True):
                needs_rollback = True
                sharpe_change_pct = eval_info.get('sharpe_change_pct', 0)
                reason = f"Performance degraded by {abs(sharpe_change_pct):.1%}"

            if needs_rollback:
                logger.warning(f"   ⚠️ {agent_type.upper()}: ROLLING BACK - {reason}")

                # Find most recent backup
                backup_dirs = sorted(self.backup_dir.glob("backup_*"), key=lambda p: p.stat().st_mtime, reverse=True)

                if backup_dirs:
                    latest_backup = backup_dirs[0]

                    # Restore from backup
                    for backup_file in latest_backup.glob(f"agent_{agent_type}_*.zip"):
                        target = self.incremental_dir / backup_file.name
                        shutil.copy2(backup_file, target)
                        logger.info(f"      Restored: {backup_file.name}")

                    rollback_list.append(agent_type)
                else:
                    logger.error(f"      ❌ No backup found for {agent_type}")

            else:
                logger.info(f"   ✅ {agent_type.upper()}: Performance acceptable, keeping new model")

        if rollback_list:
            logger.warning(f"\n🔄 Rolled back {len(rollback_list)} agent(s): {', '.join(rollback_list)}")
        else:
            logger.info("\n✅ All agents passed validation, no rollbacks needed")

        return rollback_list

    def _save_summary(
        self,
        results: Dict[str, Any],
        training_results: Dict[str, Dict],
        evaluation_results: Dict[str, Any]
    ):
        """Save retraining summary to file."""
        summary_dir = Path("data/retraining_summaries")
        summary_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = summary_dir / f"retraining_summary_{timestamp}.json"

        summary = {
            'results': results,
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"   Summary saved: {summary_file}")

    def _send_notification(self, message: str):
        """Send notification about retraining status."""
        # Placeholder for notification logic
        # Could integrate with email, Slack, PagerDuty, etc.
        logger.info(f"📧 Notification: {message}")


async def run_async_retraining(config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Async wrapper for retraining (for integration with async workflows).

    Args:
        config: Optional configuration

    Returns:
        Retraining results
    """
    scheduler = AutomaticRetrainingScheduler(config=config)
    return scheduler.run_scheduled_retraining()


def main():
    """Main entry point for scheduled execution."""
    logger.info("=" * 80)
    logger.info("🚀 Starting Scheduled Retraining")
    logger.info("=" * 80)

    # Create logs directory
    os.makedirs('logs', exist_ok=True)

    # Run retraining
    scheduler = AutomaticRetrainingScheduler()
    results = scheduler.run_scheduled_retraining()

    # Exit with appropriate code
    if results['success']:
        logger.info("\n✅ Retraining completed successfully")
        exit(0)
    else:
        logger.error("\n❌ Retraining failed")
        exit(1)


if __name__ == "__main__":
    main()
