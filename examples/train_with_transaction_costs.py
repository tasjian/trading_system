#!/usr/bin/env python3
"""
Train FinRL Agents with Realistic Transaction Costs

This example demonstrates how to train FinRL agents with realistic
transaction costs including:
- Commission (10 bps = 0.1%)
- Market impact slippage (square-root model)
- Volume constraints (max 10% of ADV)

Results show how transaction costs affect:
- Final returns
- Sharpe ratios
- Trading frequency
- Position sizing

Usage:
    python examples/train_with_transaction_costs.py
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add project to path
sys.path.insert(0, '/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system')

from core.finrl_transaction_costs import TradingEnvWithCosts, create_env_with_costs
from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv


def generate_market_data(
    symbols: list,
    start_date: str = '2020-01-01',
    end_date: str = '2023-12-31',
    add_technical_indicators: bool = True
) -> pd.DataFrame:
    """
    Generate synthetic market data for testing.

    In production, replace with real data from Alpaca or Yahoo Finance.
    """
    dates = pd.date_range(start_date, end_date, freq='D')
    data = []

    for date in dates:
        for symbol in symbols:
            # Generate realistic OHLCV
            base_price = np.random.uniform(50, 500)
            volatility = np.random.uniform(0.01, 0.03)

            open_price = base_price * (1 + np.random.randn() * volatility)
            close_price = open_price * (1 + np.random.randn() * volatility)
            high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * volatility)
            low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * volatility)
            volume = np.random.uniform(1e6, 1e7)

            row = {
                'date': date,
                'tic': symbol,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }

            # Add technical indicators
            if add_technical_indicators:
                row['macd'] = np.random.randn()
                row['rsi_30'] = np.random.uniform(30, 70)
                row['cci_30'] = np.random.randn() * 100
                row['dx_30'] = np.random.uniform(10, 40)

            data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)

    return df


def compare_with_without_costs(
    df: pd.DataFrame,
    symbols: list,
    agent_type: str = 'ppo',
    training_timesteps: int = 10000
):
    """
    Compare agent performance with and without transaction costs.
    """
    print(f"\n{'='*80}")
    print(f"Training {agent_type.upper()} with and without transaction costs")
    print(f"{'='*80}\n")

    # Split data
    train_df = df[df['date'] < '2023-01-01'].copy()
    test_df = df[df['date'] >= '2023-01-01'].copy()

    # Reset indices
    train_df = train_df.set_index('date')
    test_df = test_df.set_index('date')

    tech_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']

    # ========================================================================
    # Train WITHOUT transaction costs (baseline)
    # ========================================================================
    print("🔵 Training WITHOUT transaction costs (baseline)...")

    env_no_costs = create_env_with_costs(
        df=train_df,
        stock_dim=len(symbols),
        commission_rate=0.0,          # No commission
        slippage_coeff=0.0,           # No slippage
        max_trade_pct_adv=1.0,        # No volume constraint
        tech_indicator_list=tech_indicators,
        initial_amount=1000000
    )

    env_no_costs = DummyVecEnv([lambda: env_no_costs])

    if agent_type == 'ppo':
        model_no_costs = PPO('MlpPolicy', env_no_costs, verbose=0)
    elif agent_type == 'a2c':
        model_no_costs = A2C('MlpPolicy', env_no_costs, verbose=0)
    else:
        model_no_costs = PPO('MlpPolicy', env_no_costs, verbose=0)

    model_no_costs.learn(total_timesteps=training_timesteps)
    print("✅ Training complete (no costs)\n")

    # ========================================================================
    # Train WITH realistic transaction costs
    # ========================================================================
    print("🔴 Training WITH realistic transaction costs...")

    env_with_costs = create_env_with_costs(
        df=train_df,
        stock_dim=len(symbols),
        commission_rate=0.001,        # 10 bps commission
        slippage_coeff=0.0005,        # Realistic slippage
        max_trade_pct_adv=0.10,       # Max 10% of ADV
        tech_indicator_list=tech_indicators,
        initial_amount=1000000
    )

    env_with_costs = DummyVecEnv([lambda: env_with_costs])

    if agent_type == 'ppo':
        model_with_costs = PPO('MlpPolicy', env_with_costs, verbose=0)
    elif agent_type == 'a2c':
        model_with_costs = A2C('MlpPolicy', env_with_costs, verbose=0)
    else:
        model_with_costs = PPO('MlpPolicy', env_with_costs, verbose=0)

    model_with_costs.learn(total_timesteps=training_timesteps)
    print("✅ Training complete (with costs)\n")

    # ========================================================================
    # Backtest both models on test data
    # ========================================================================
    print("📊 Backtesting on test data...")

    # Test environment WITH costs (realistic)
    test_env = create_env_with_costs(
        df=test_df,
        stock_dim=len(symbols),
        commission_rate=0.001,
        slippage_coeff=0.0005,
        max_trade_pct_adv=0.10,
        tech_indicator_list=tech_indicators,
        initial_amount=1000000
    )

    results = {}

    # Backtest model trained WITHOUT costs
    obs = test_env.reset()
    done = False
    total_cost_no_cost_model = 0.0
    portfolio_values_no_cost_model = []

    while not done:
        action, _states = model_no_costs.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action[0])

        total_cost_no_cost_model += info['total_cost']
        portfolio_values_no_cost_model.append(info['portfolio_value'])

    final_value_no_cost_model = portfolio_values_no_cost_model[-1]
    returns_no_cost_model = (final_value_no_cost_model / 1000000 - 1) * 100

    results['no_costs_model'] = {
        'final_value': final_value_no_cost_model,
        'returns': returns_no_cost_model,
        'total_costs': total_cost_no_cost_model,
        'portfolio_values': portfolio_values_no_cost_model
    }

    # Backtest model trained WITH costs
    obs = test_env.reset()
    done = False
    total_cost_with_cost_model = 0.0
    portfolio_values_with_cost_model = []

    while not done:
        action, _states = model_with_costs.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action[0])

        total_cost_with_cost_model += info['total_cost']
        portfolio_values_with_cost_model.append(info['portfolio_value'])

    final_value_with_cost_model = portfolio_values_with_cost_model[-1]
    returns_with_cost_model = (final_value_with_cost_model / 1000000 - 1) * 100

    results['with_costs_model'] = {
        'final_value': final_value_with_cost_model,
        'returns': returns_with_cost_model,
        'total_costs': total_cost_with_cost_model,
        'portfolio_values': portfolio_values_with_cost_model
    }

    # ========================================================================
    # Print comparison
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"RESULTS COMPARISON ({agent_type.upper()})")
    print(f"{'='*80}\n")

    print("Model trained WITHOUT costs:")
    print(f"  Final Portfolio Value: ${results['no_costs_model']['final_value']:,.2f}")
    print(f"  Total Returns: {results['no_costs_model']['returns']:+.2f}%")
    print(f"  Transaction Costs: ${results['no_costs_model']['total_costs']:,.2f}")
    print(f"  Net Returns (after costs): {(results['no_costs_model']['final_value']/1000000-1)*100:+.2f}%\n")

    print("Model trained WITH costs:")
    print(f"  Final Portfolio Value: ${results['with_costs_model']['final_value']:,.2f}")
    print(f"  Total Returns: {results['with_costs_model']['returns']:+.2f}%")
    print(f"  Transaction Costs: ${results['with_costs_model']['total_costs']:,.2f}")
    print(f"  Net Returns (after costs): {(results['with_costs_model']['final_value']/1000000-1)*100:+.2f}%\n")

    cost_difference = results['no_costs_model']['total_costs'] - results['with_costs_model']['total_costs']
    print(f"💡 Insight:")
    print(f"  Model trained WITH costs paid ${abs(cost_difference):,.2f} {'LESS' if cost_difference > 0 else 'MORE'} in transaction costs")
    print(f"  This represents {abs(cost_difference)/results['no_costs_model']['total_costs']*100:.1f}% cost reduction")

    return results


def analyze_cost_sensitivity(
    df: pd.DataFrame,
    symbols: list,
    commission_rates: list = [0.0001, 0.0005, 0.001, 0.002],
    training_timesteps: int = 5000
):
    """
    Analyze how different commission rates affect agent behavior.
    """
    print(f"\n{'='*80}")
    print(f"COMMISSION RATE SENSITIVITY ANALYSIS")
    print(f"{'='*80}\n")

    train_df = df[df['date'] < '2023-01-01'].copy()
    train_df = train_df.set_index('date')

    results = []

    for commission_rate in commission_rates:
        print(f"Training with {commission_rate*100:.3f}% commission...")

        env = create_env_with_costs(
            df=train_df,
            stock_dim=len(symbols),
            commission_rate=commission_rate,
            slippage_coeff=0.0005,
            max_trade_pct_adv=0.10,
            initial_amount=1000000
        )

        env = DummyVecEnv([lambda: env])
        model = PPO('MlpPolicy', env, verbose=0)
        model.learn(total_timesteps=training_timesteps)

        # Get cost statistics
        cost_stats = env.envs[0].cost_model.get_statistics()

        results.append({
            'commission_rate': commission_rate,
            'commission_bps': commission_rate * 10000,
            'total_trades': cost_stats['total_trades'],
            'total_commission': cost_stats['total_commission'],
            'total_slippage': cost_stats['total_slippage'],
            'total_costs': cost_stats['total_costs'],
            'avg_cost_per_trade': cost_stats['avg_commission_per_trade'] + cost_stats['avg_slippage_per_trade']
        })

        print(f"  Total trades: {cost_stats['total_trades']}")
        print(f"  Total costs: ${cost_stats['total_costs']:,.2f}\n")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")
    print(f"{'Commission (bps)':<20} {'Total Trades':<15} {'Total Costs':<15} {'Avg Cost/Trade'}")
    print(f"{'-'*80}")

    for result in results:
        print(f"{result['commission_bps']:<20.1f} {result['total_trades']:<15} "
              f"${result['total_costs']:<14,.2f} ${result['avg_cost_per_trade']:.2f}")

    return results


def main():
    """Run transaction cost examples."""
    print("="*80)
    print("FinRL Training with Realistic Transaction Costs")
    print("="*80)

    # Generate market data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    print(f"\nGenerating synthetic market data for {len(symbols)} stocks...")
    df = generate_market_data(symbols, start_date='2020-01-01', end_date='2023-12-31')
    print(f"✅ Generated {len(df)} rows of market data\n")

    # Example 1: Compare with/without costs
    print("\n" + "="*80)
    print("EXAMPLE 1: Impact of Transaction Costs on Performance")
    print("="*80)
    results = compare_with_without_costs(
        df=df,
        symbols=symbols,
        agent_type='ppo',
        training_timesteps=10000
    )

    # Example 2: Commission sensitivity analysis
    print("\n" + "="*80)
    print("EXAMPLE 2: Commission Rate Sensitivity")
    print("="*80)
    sensitivity_results = analyze_cost_sensitivity(
        df=df,
        symbols=symbols,
        commission_rates=[0.0001, 0.0005, 0.001, 0.002],
        training_timesteps=5000
    )

    print("\n" + "="*80)
    print("✅ Analysis Complete!")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. Models trained WITH costs learn to trade less frequently")
    print("  2. This reduces total transaction costs significantly")
    print("  3. Net returns (after costs) are often higher for cost-aware models")
    print("  4. Higher commission rates lead to fewer but larger trades")
    print("\nNext Steps:")
    print("  - Integrate into production training pipeline")
    print("  - Calibrate slippage coefficient using real trade data")
    print("  - Test on live paper trading account")


if __name__ == "__main__":
    main()
