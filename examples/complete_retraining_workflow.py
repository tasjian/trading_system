#!/usr/bin/env python3
"""
Complete Automatic Retraining Workflow Example

This example demonstrates the full end-to-end automatic retraining workflow:
1. Detect market regime
2. Run adaptive incremental training
3. Evaluate performance
4. Compare to baseline
5. Generate visualizations and reports

Usage:
    python examples/complete_retraining_workflow.py
"""

import sys
import os
import asyncio
from datetime import datetime

# Add trading system to path
sys.path.insert(0, '/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system')

from utils.market_regime_detector import MarketRegimeDetector
from utils.performance_tracker import PerformanceTracker
from agents.finrl_agent_wrapper import FinRLAgentWrapper
from train_finrl_incremental import IncrementalTrainer


async def main():
    print("=" * 80)
    print("FinRL Complete Automatic Retraining Workflow")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ========================================================================
    # Step 1: Detect Market Regime
    # ========================================================================
    print("📊 Step 1: Market Regime Detection")
    print("-" * 80)

    detector = MarketRegimeDetector(symbols=['SPY', 'QQQ', 'IWM'])
    regime = detector.detect_current_regime()

    print(f"Market Regime: {regime.regime.upper()}")
    print(f"Volatility: {regime.volatility:.2%}")
    print(f"Recommended Lookback: {regime.recommended_lookback} days")
    print(f"Confidence: {regime.confidence:.1%}")
    print()

    # ========================================================================
    # Step 2: Get Baseline Performance
    # ========================================================================
    print("📊 Step 2: Baseline Performance Metrics")
    print("-" * 80)

    tracker = PerformanceTracker()
    baseline_metrics = {}

    agents = ['a2c', 'ppo', 'ddpg', 'sac', 'td3']
    for agent_type in agents:
        metrics = tracker.get_metrics(agent_type=agent_type, limit=1)
        if not metrics.empty:
            baseline_metrics[agent_type] = {
                'sharpe': metrics.iloc[0]['sharpe_ratio'],
                'win_rate': metrics.iloc[0]['win_rate'],
                'model_name': metrics.iloc[0]['model_name']
            }
            print(f"{agent_type.upper():6} - Sharpe: {baseline_metrics[agent_type]['sharpe']:.3f}, "
                  f"Win Rate: {baseline_metrics[agent_type]['win_rate']:.2%}")
        else:
            print(f"{agent_type.upper():6} - No baseline metrics found")

    print()

    # ========================================================================
    # Step 3: Run Incremental Training (Option A: Direct)
    # ========================================================================
    print("🏋️  Step 3: Incremental Training with Adaptive Lookback")
    print("-" * 80)

    print(f"Training with {regime.recommended_lookback}-day lookback...")
    print()

    trainer = IncrementalTrainer(
        lookback_days=regime.recommended_lookback,
        checkpoint_dir="data/finrl_models",
        incremental_dir="data/finrl_models_incremental"
    )

    # Train all agents
    training_results = trainer.fine_tune_all_agents()

    print()
    print("Training Summary:")
    for agent_type, result in training_results.items():
        if 'error' in result:
            print(f"  {agent_type.upper()}: ❌ FAILED - {result['error']}")
        else:
            print(f"  {agent_type.upper()}: ✅ SUCCESS - {result['timesteps']:,} timesteps in {result['training_time_seconds']:.1f}s")

    print()

    # ========================================================================
    # Step 4: Evaluate New Performance
    # ========================================================================
    print("📊 Step 4: Performance Evaluation")
    print("-" * 80)

    performance_changes = {}

    for agent_type, result in training_results.items():
        if 'error' in result or 'performance_metrics' not in result:
            continue

        perf = result['performance_metrics']
        baseline = baseline_metrics.get(agent_type)

        if baseline:
            sharpe_change = perf['sharpe_ratio'] - baseline['sharpe']
            sharpe_change_pct = (sharpe_change / baseline['sharpe']) * 100 if baseline['sharpe'] > 0 else 0

            performance_changes[agent_type] = {
                'new_sharpe': perf['sharpe_ratio'],
                'baseline_sharpe': baseline['sharpe'],
                'change': sharpe_change,
                'change_pct': sharpe_change_pct,
                'improved': sharpe_change > 0
            }

            status = "📈" if sharpe_change > 0 else "📉"
            print(f"{agent_type.upper():6} {status} Sharpe: {baseline['sharpe']:.3f} → {perf['sharpe_ratio']:.3f} "
                  f"({sharpe_change:+.3f}, {sharpe_change_pct:+.1f}%)")
        else:
            print(f"{agent_type.upper():6} - New Sharpe: {perf['sharpe_ratio']:.3f} (no baseline)")

    print()

    # ========================================================================
    # Step 5: Decision Summary
    # ========================================================================
    print("🎯 Step 5: Performance Validation")
    print("-" * 80)

    MIN_SHARPE_THRESHOLD = 0.8
    MAX_DEGRADATION = 0.10

    decisions = {}

    for agent_type, change_info in performance_changes.items():
        passes_threshold = change_info['new_sharpe'] >= MIN_SHARPE_THRESHOLD
        degradation_ok = change_info['change_pct'] >= -(MAX_DEGRADATION * 100)

        decision = "✅ ACCEPT" if (passes_threshold and degradation_ok) else "❌ REJECT"
        reason = []

        if not passes_threshold:
            reason.append(f"Sharpe below {MIN_SHARPE_THRESHOLD}")
        if not degradation_ok:
            reason.append(f"Degradation > {MAX_DEGRADATION:.0%}")

        decisions[agent_type] = {
            'accept': passes_threshold and degradation_ok,
            'reason': ', '.join(reason) if reason else 'Performance acceptable'
        }

        print(f"{agent_type.upper():6} {decision} - {decisions[agent_type]['reason']}")

    print()

    # ========================================================================
    # Step 6: Generate Report
    # ========================================================================
    print("📄 Step 6: Performance Report")
    print("-" * 80)

    # Generate report for all agents
    report = tracker.generate_performance_report()
    print(report[:1000])  # Print first 1000 characters
    print("... (see full report in data/performance_report.txt)")

    # Save full report
    with open("data/performance_report.txt", "w") as f:
        f.write(report)

    print()

    # ========================================================================
    # Step 7: Generate Visualizations
    # ========================================================================
    print("📊 Step 7: Visualizations")
    print("-" * 80)

    try:
        # Compare all PPO models
        tracker.plot_performance_comparison(
            agent_type='ppo',
            save_path='data/ppo_performance_comparison.png'
        )
        print("✅ Saved: data/ppo_performance_comparison.png")

        # Historical performance trend
        tracker.plot_historical_performance(
            agent_type='ppo',
            metric='sharpe_ratio',
            save_path='data/ppo_sharpe_history.png'
        )
        print("✅ Saved: data/ppo_sharpe_history.png")

    except ImportError:
        print("⚠️  matplotlib not installed - skipping visualizations")

    print()

    # ========================================================================
    # Step 8: Summary Statistics
    # ========================================================================
    print("📈 Step 8: Summary Statistics")
    print("-" * 80)

    total_agents = len(training_results)
    successful_training = sum(1 for r in training_results.values() if 'error' not in r)
    accepted_models = sum(1 for d in decisions.values() if d['accept'])
    improved_models = sum(1 for c in performance_changes.values() if c['improved'])

    print(f"Total Agents: {total_agents}")
    print(f"Successfully Trained: {successful_training}")
    print(f"Accepted Models: {accepted_models}")
    print(f"Improved Performance: {improved_models}")
    print(f"Market Regime: {regime.regime.upper()}")
    print(f"Lookback Used: {regime.recommended_lookback} days")

    print()
    print("=" * 80)
    print("✅ Workflow Complete!")
    print("=" * 80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Next Steps:")
    print("  1. Review performance report: data/performance_report.txt")
    print("  2. Check visualizations: data/*.png")
    print("  3. If satisfied, models are ready for production")
    print("  4. Monitor performance in live trading")
    print()


# ============================================================================
# Alternative: Use Agent Wrapper (Simpler)
# ============================================================================
async def simplified_workflow():
    """
    Simplified workflow using FinRL Agent Wrapper.
    This is the recommended approach for production.
    """
    print("=" * 80)
    print("Simplified Automatic Retraining (via Agent Wrapper)")
    print("=" * 80)
    print()

    # Initialize agent
    agent = FinRLAgentWrapper(
        symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    )
    await agent.initialize_system()

    # Option 1: Full automatic retraining with validation and rollback
    print("Running full automatic retraining...")
    results = await agent.automatic_retraining(
        min_sharpe_threshold=0.8,
        max_performance_degradation=0.10,
        enable_rollback=True
    )

    print(f"\nSuccess: {results['success']}")
    print(f"Agents Retrained: {', '.join(results['agents_retrained'])}")
    if results['rollbacks']:
        print(f"Rollbacks: {', '.join(results['rollbacks'])}")

    print()

    # Option 2: Simpler adaptive training (no validation/rollback)
    print("Or use simplified adaptive training...")
    metrics = await agent.adaptive_incremental_training()

    print(f"Trained {len(metrics)} agents")

    print()
    print("✅ Complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Complete retraining workflow example")
    parser.add_argument('--simplified', action='store_true',
                       help='Use simplified workflow via agent wrapper')
    args = parser.parse_args()

    if args.simplified:
        asyncio.run(simplified_workflow())
    else:
        asyncio.run(main())
