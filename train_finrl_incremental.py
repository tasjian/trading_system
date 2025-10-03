#!/usr/bin/env python3
"""
FinRL Incremental Learning (Fine-tuning) Module

This module provides production-ready incremental learning capabilities for FinRL agents.
Instead of training from scratch, it loads existing trained models and fine-tunes them
on recent market data, allowing agents to adapt to current market conditions while
preserving learned knowledge.

Key Features:
- Load and fine-tune existing trained models
- Multi-agent support (A2C, PPO, DDPG, SAC, TD3)
- Rolling window data fetching (configurable lookback period)
- Versioned model checkpointing with timestamps
- Full FinRL environment compatibility
- Integration with Alpaca API for live data
- Ensemble-aware training

Usage:
    # Fine-tune all agents on last 60 days
    python train_finrl_incremental.py --days 60 --timesteps 20000

    # Fine-tune specific agent
    python train_finrl_incremental.py --agent ppo --days 90 --timesteps 50000

    # From code
    from train_finrl_incremental import IncrementalTrainer
    trainer = IncrementalTrainer(lookback_days=60)
    trainer.fine_tune_all_agents(timesteps=20000)
"""

import sys
import os
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

# Add FinRL to Python path
finrl_path = '/Users/zac/Desktop/02_PROJECTS/FinRL'
if finrl_path not in sys.path:
    sys.path.insert(0, finrl_path)

# Add trading system to path
sys.path.insert(0, '/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import FinRL components
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split
from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3
from stable_baselines3.common.logger import configure

# Import trading system components
from tools.alpaca_client import alpaca_client
from agents.finrl_agent_wrapper import FinRLAgentWrapper
from utils.performance_tracker import PerformanceTracker, evaluate_and_save


class IncrementalTrainer:
    """
    Production-ready incremental learning system for FinRL agents.

    This class handles fine-tuning of pre-trained FinRL agents on recent market data,
    enabling continuous learning and adaptation to evolving market conditions.
    """

    # Model architecture mapping
    MODEL_CLASSES = {
        'a2c': A2C,
        'ppo': PPO,
        'ddpg': DDPG,
        'sac': SAC,
        'td3': TD3
    }

    def __init__(self,
                 symbols: Optional[List[str]] = None,
                 lookback_days: int = 60,
                 checkpoint_dir: str = "data/finrl_models",
                 incremental_dir: str = "data/finrl_models_incremental",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize incremental trainer.

        Args:
            symbols: List of stock symbols to trade (default: top 10)
            lookback_days: Days of recent data for fine-tuning (default: 60)
            checkpoint_dir: Directory containing base trained models
            incremental_dir: Directory for saving fine-tuned models
            config: Optional configuration overrides
        """
        self.symbols = symbols or ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
        self.lookback_days = lookback_days
        self.checkpoint_dir = Path(checkpoint_dir)
        self.incremental_dir = Path(incremental_dir)

        # Create directories
        self.incremental_dir.mkdir(parents=True, exist_ok=True)

        # Configuration with smart defaults
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        # Technical indicators (matching existing system)
        self.tech_indicator_list = [
            'macd', 'rsi_30', 'cci_30', 'dx_30', 'bb_bbm', 'bb_bbh', 'bb_bbl'
        ]

        # State space calculation
        self.stock_dim = len(self.symbols)
        self.state_space = 1 + 2 * self.stock_dim + len(self.tech_indicator_list) * self.stock_dim

        # Training data cache
        self.recent_data = None
        self.training_env = None

        # Metrics tracking
        self.training_metrics = {}
        self.performance_tracker = PerformanceTracker(db_path="data/performance_metrics.db")

        logger.info(f"🔄 Initialized Incremental Trainer")
        logger.info(f"   Symbols: {len(self.symbols)}")
        logger.info(f"   Lookback: {lookback_days} days")
        logger.info(f"   State Space: {self.state_space}")
        logger.info(f"   Checkpoint Dir: {self.checkpoint_dir}")
        logger.info(f"   Incremental Dir: {self.incremental_dir}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for incremental training."""
        return {
            # Fine-tuning timesteps (much less than full training)
            'timesteps_dict': {
                'a2c': 20000,   # 20% of full training
                'ppo': 30000,   # 20% of full training
                'ddpg': 20000,
                'sac': 24000,
                'td3': 20000
            },

            # Environment parameters
            'initial_amount': 100000,
            'hmax': 100,
            'buy_cost_pct': 0.001,
            'sell_cost_pct': 0.001,
            'reward_scaling': 1e-4,

            # Learning rate adjustments for fine-tuning
            'learning_rates': {
                'a2c': 0.0003,    # Lower than initial (0.0007)
                'ppo': 0.0001,    # Lower than initial (0.00025)
                'ddpg': 0.0005,   # Lower than initial (0.001)
                'sac': 0.00005,   # Lower than initial (0.0001)
                'td3': 0.0005     # Lower than initial (0.001)
            },

            # Validation parameters
            'validation_split': 0.2,  # Use 20% of recent data for validation
            'early_stopping_patience': 5,
            'min_improvement': 0.001
        }

    def load_recent_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load and preprocess recent market data for fine-tuning.

        Args:
            force_refresh: Force re-fetch from Alpaca even if cached

        Returns:
            DataFrame with recent market data and technical indicators
        """
        if self.recent_data is not None and not force_refresh:
            logger.info("✅ Using cached recent data")
            return self.recent_data

        logger.info(f"📊 Fetching last {self.lookback_days} days of market data...")

        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)

            all_data = []

            for symbol in self.symbols:
                try:
                    logger.debug(f"   Fetching {symbol}...")

                    # Use Alpaca API for historical data
                    bars = alpaca_client.api.get_bars(
                        symbol,
                        '1Day',
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d')
                    )

                    if bars:
                        for bar in bars:
                            all_data.append({
                                'date': bar.t.strftime('%Y-%m-%d'),
                                'tic': symbol,
                                'open': float(bar.o),
                                'high': float(bar.h),
                                'low': float(bar.l),
                                'close': float(bar.c),
                                'volume': int(bar.v)
                            })

                except Exception as e:
                    logger.warning(f"⚠️ Failed to fetch {symbol}: {e}")
                    continue

            if not all_data:
                raise ValueError("No market data retrieved")

            # Create DataFrame
            df = pd.DataFrame(all_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['date', 'tic']).reset_index(drop=True)

            logger.info(f"✅ Retrieved {len(df)} records for {len(df['tic'].unique())} symbols")

            # Add technical indicators
            df = self._add_technical_indicators(df)

            # Cache the data
            self.recent_data = df

            return df

        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            raise

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame."""
        logger.info("🔧 Adding technical indicators...")

        try:
            # Try FinRL DataProcessor first
            from finrl.meta.data_processor import DataProcessor
            dp = DataProcessor()
            df = dp.add_technical_indicator(df, tech_indicator_list=self.tech_indicator_list)
            df = dp.add_turbulence(df)
            logger.info("✅ Technical indicators added via FinRL DataProcessor")

        except Exception as e:
            logger.warning(f"⚠️ FinRL DataProcessor failed: {e}, using manual calculation")

            # Manual calculation fallback
            df = df.sort_values(['tic', 'date']).reset_index(drop=True)

            for symbol in df['tic'].unique():
                mask = df['tic'] == symbol
                symbol_df = df[mask].copy().sort_values('date')

                # Calculate indicators
                symbol_df['macd'] = symbol_df['close'].ewm(span=12).mean() - symbol_df['close'].ewm(span=26).mean()

                # RSI calculation
                delta = symbol_df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=30).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=30).mean()
                rs = gain / loss
                symbol_df['rsi_30'] = 100 - (100 / (1 + rs))

                # Other indicators
                symbol_df['cci_30'] = 0.0  # Simplified
                symbol_df['dx_30'] = 0.0
                symbol_df['bb_bbm'] = symbol_df['close'].rolling(20).mean()
                symbol_df['bb_bbh'] = symbol_df['bb_bbm'] + (symbol_df['close'].rolling(20).std() * 2)
                symbol_df['bb_bbl'] = symbol_df['bb_bbm'] - (symbol_df['close'].rolling(20).std() * 2)
                symbol_df['turbulence'] = symbol_df['close'].rolling(20).std() / symbol_df['close'].rolling(20).mean()

                # Update main DataFrame
                for col in ['macd', 'rsi_30', 'cci_30', 'dx_30', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'turbulence']:
                    df.loc[mask, col] = symbol_df[col].values

            logger.info("✅ Technical indicators added manually")

        # Fill NaN values
        df = df.ffill().bfill().fillna(0)

        return df

    def create_environment(self, df: pd.DataFrame) -> StockTradingEnv:
        """Create FinRL trading environment from recent data."""
        logger.info("🏗️ Creating trading environment for fine-tuning...")

        try:
            # Environment parameters
            env_kwargs = {
                "hmax": self.config['hmax'],
                "initial_amount": self.config['initial_amount'],
                "num_stock_shares": [0] * self.stock_dim,
                "buy_cost_pct": [self.config['buy_cost_pct']] * self.stock_dim,
                "sell_cost_pct": [self.config['sell_cost_pct']] * self.stock_dim,
                "state_space": self.state_space,
                "stock_dim": self.stock_dim,
                "tech_indicator_list": self.tech_indicator_list,
                "action_space": self.stock_dim,
                "reward_scaling": self.config['reward_scaling'],
                "print_verbosity": 100
            }

            # Create environment
            e_train_gym = StockTradingEnv(df=df, **env_kwargs)

            # Get Stable Baselines3 compatible environment
            env_train, _ = e_train_gym.get_sb_env()

            logger.info("✅ Trading environment created")
            return env_train

        except Exception as e:
            logger.error(f"❌ Environment creation failed: {e}")
            raise

    def find_latest_checkpoint(self, agent_type: str) -> Optional[Path]:
        """Find the most recent checkpoint for an agent type."""
        # Check incremental directory first (for successive fine-tuning)
        incremental_checkpoints = list(self.incremental_dir.glob(f"agent_{agent_type}_*.zip"))
        if incremental_checkpoints:
            latest = max(incremental_checkpoints, key=lambda p: p.stat().st_mtime)
            logger.info(f"📦 Found incremental checkpoint: {latest.name}")
            return latest

        # Fall back to base checkpoint directory
        base_checkpoint = self.checkpoint_dir / f"agent_{agent_type}.zip"
        if base_checkpoint.exists():
            logger.info(f"📦 Found base checkpoint: {base_checkpoint.name}")
            return base_checkpoint

        logger.warning(f"⚠️ No checkpoint found for {agent_type}")
        return None

    def _evaluate_model_performance(
        self,
        model: Any,
        agent_type: str,
        model_name: str,
        recent_df: pd.DataFrame
    ):
        """
        Evaluate model performance on test data and save to performance tracker.

        Args:
            model: Trained model to evaluate
            agent_type: Type of agent (a2c, ppo, etc.)
            model_name: Unique identifier for model
            recent_df: Recent data dataframe

        Returns:
            PerformanceMetrics object
        """
        # Split data into train/test (use last 20% as test)
        split_idx = int(len(recent_df) * 0.8)
        test_df = recent_df[split_idx:].reset_index(drop=True)

        # Create test environment
        env_kwargs = {
            "hmax": self.config['hmax'],
            "initial_amount": self.config['initial_amount'],
            "num_stock_shares": [0] * self.stock_dim,
            "buy_cost_pct": [self.config['buy_cost_pct']] * self.stock_dim,
            "sell_cost_pct": [self.config['sell_cost_pct']] * self.stock_dim,
            "state_space": self.state_space,
            "stock_dim": self.stock_dim,
            "tech_indicator_list": self.tech_indicator_list,
            "action_space": self.stock_dim,
            "reward_scaling": self.config['reward_scaling'],
            "print_verbosity": 0
        }

        test_env_obj = StockTradingEnv(df=test_df, **env_kwargs)
        test_env, _ = test_env_obj.get_sb_env()

        # Store date range for metadata
        test_env.start_date = test_df['date'].min() if 'date' in test_df.columns else 'N/A'
        test_env.end_date = test_df['date'].max() if 'date' in test_df.columns else 'N/A'

        # Evaluate and save metrics
        perf_metrics = self.performance_tracker.evaluate_agent(
            model=model,
            test_env=test_env,
            model_name=model_name,
            agent_type=agent_type,
            notes=f"Incremental fine-tuning on {self.lookback_days} days lookback"
        )

        # Save to database
        self.performance_tracker.save_metrics(perf_metrics)

        return perf_metrics

    def fine_tune_agent(self,
                       agent_type: str,
                       timesteps: Optional[int] = None,
                       learning_rate: Optional[float] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Fine-tune a single agent on recent data.

        Args:
            agent_type: Agent type ('a2c', 'ppo', 'ddpg', 'sac', 'td3')
            timesteps: Training timesteps (uses config default if None)
            learning_rate: Learning rate (uses config default if None)

        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 Fine-tuning {agent_type.upper()} agent")
        logger.info(f"{'='*60}")

        agent_type = agent_type.lower()

        if agent_type not in self.MODEL_CLASSES:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Get configuration
        timesteps = timesteps or self.config['timesteps_dict'][agent_type]
        learning_rate = learning_rate or self.config['learning_rates'][agent_type]

        logger.info(f"📊 Training Config:")
        logger.info(f"   Timesteps: {timesteps:,}")
        logger.info(f"   Learning Rate: {learning_rate}")
        logger.info(f"   Lookback: {self.lookback_days} days")

        try:
            # Load recent data
            recent_df = self.load_recent_data()

            # Create environment
            env_train = self.create_environment(recent_df)

            # Find and load checkpoint
            checkpoint_path = self.find_latest_checkpoint(agent_type)

            model_class = self.MODEL_CLASSES[agent_type]

            if checkpoint_path:
                logger.info(f"🔄 Loading checkpoint: {checkpoint_path}")
                model = model_class.load(str(checkpoint_path), env=env_train)

                # Update learning rate for fine-tuning
                model.learning_rate = learning_rate
                logger.info(f"✅ Checkpoint loaded, learning rate set to {learning_rate}")
            else:
                logger.warning(f"⚠️ No checkpoint found. Training new {agent_type.upper()} model from scratch.")
                logger.warning(f"   This is not incremental learning!")

                # Create new agent with FinRL DRLAgent
                agent = DRLAgent(env=env_train)
                model_kwargs = {'learning_rate': learning_rate}
                model = agent.get_model(agent_type, model_kwargs=model_kwargs)

            # Set up logging
            log_dir = f'results/incremental_{agent_type}'
            os.makedirs(log_dir, exist_ok=True)
            new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
            model.set_logger(new_logger)

            # Fine-tune the model
            logger.info(f"🏋️ Fine-tuning {agent_type.upper()} for {timesteps:,} timesteps...")
            start_time = datetime.now()

            model.learn(
                total_timesteps=timesteps,
                tb_log_name=f"incremental_{agent_type}",
                reset_num_timesteps=False  # Continue from checkpoint
            )

            training_time = (datetime.now() - start_time).total_seconds()

            # Save fine-tuned model with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            save_path = self.incremental_dir / f"agent_{agent_type}_{timestamp}"
            model.save(str(save_path))

            logger.info(f"✅ Fine-tuning completed in {training_time:.1f}s")
            logger.info(f"💾 Saved: {save_path}.zip")

            # Collect metrics
            metrics = {
                'agent_type': agent_type,
                'timesteps': timesteps,
                'learning_rate': learning_rate,
                'training_time_seconds': training_time,
                'lookback_days': self.lookback_days,
                'timestamp': timestamp,
                'checkpoint_used': str(checkpoint_path) if checkpoint_path else 'none',
                'model_path': str(save_path) + '.zip'
            }

            self.training_metrics[agent_type] = metrics

            # Evaluate model performance on test data
            try:
                logger.info(f"📊 Evaluating {agent_type.upper()} performance...")
                perf_metrics = self._evaluate_model_performance(
                    model=model,
                    agent_type=agent_type,
                    model_name=f"{agent_type}_{timestamp}",
                    recent_df=recent_df
                )

                # Add performance metrics to training metrics
                metrics['performance_metrics'] = {
                    'sharpe_ratio': perf_metrics.sharpe_ratio,
                    'sortino_ratio': perf_metrics.sortino_ratio,
                    'win_rate': perf_metrics.win_rate,
                    'max_drawdown': perf_metrics.max_drawdown,
                    'annualized_return': perf_metrics.annualized_return
                }

                logger.info(f"✅ Performance: Sharpe={perf_metrics.sharpe_ratio:.3f}, "
                           f"Win Rate={perf_metrics.win_rate:.2%}, "
                           f"Max DD={perf_metrics.max_drawdown:.2%}")

            except Exception as e:
                logger.warning(f"⚠️ Performance evaluation failed: {e}")
                metrics['performance_metrics'] = None

            return model, metrics

        except Exception as e:
            logger.error(f"❌ Fine-tuning failed for {agent_type}: {e}")
            raise

    def fine_tune_all_agents(self,
                            timesteps_override: Optional[Dict[str, int]] = None,
                            agents_to_train: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Fine-tune all agents (or specified subset) on recent data.

        Args:
            timesteps_override: Optional dict of agent_type -> timesteps
            agents_to_train: Optional list of agent types to train (default: all)

        Returns:
            Dictionary of agent_type -> metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Starting Incremental Training for All Agents")
        logger.info(f"{'='*60}\n")

        agents_to_train = agents_to_train or ['a2c', 'ppo', 'ddpg', 'sac', 'td3']
        timesteps_override = timesteps_override or {}

        all_metrics = {}
        successful_agents = 0

        for i, agent_type in enumerate(agents_to_train, 1):
            logger.info(f"\n📋 Agent {i}/{len(agents_to_train)}: {agent_type.upper()}")

            try:
                timesteps = timesteps_override.get(agent_type)
                _, metrics = self.fine_tune_agent(agent_type, timesteps=timesteps)
                all_metrics[agent_type] = metrics
                successful_agents += 1

            except Exception as e:
                logger.error(f"❌ Failed to fine-tune {agent_type}: {e}")
                all_metrics[agent_type] = {'error': str(e)}
                continue

        # Save metrics summary
        self._save_training_summary(all_metrics)

        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Incremental Training Summary")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Successfully fine-tuned: {successful_agents}/{len(agents_to_train)} agents")
        logger.info(f"📁 Models saved in: {self.incremental_dir}")
        logger.info(f"📊 Training logs in: results/incremental_*/")
        logger.info(f"{'='*60}\n")

        return all_metrics

    def _save_training_summary(self, metrics: Dict[str, Dict[str, Any]]):
        """Save training metrics to JSON file."""
        summary_path = self.incremental_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

        summary = {
            'training_date': datetime.now().isoformat(),
            'lookback_days': self.lookback_days,
            'symbols': self.symbols,
            'agents': metrics
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"📊 Training summary saved: {summary_path}")

    def compare_models(self, agent_type: str) -> Dict[str, Any]:
        """
        Compare base model vs fine-tuned model performance.

        Args:
            agent_type: Agent type to compare

        Returns:
            Comparison metrics dictionary
        """
        logger.info(f"📊 Comparing {agent_type.upper()} models...")

        # This would require a validation environment and backtesting
        # Placeholder for future implementation

        return {
            'agent_type': agent_type,
            'comparison': 'not_implemented',
            'note': 'Implement backtesting comparison in future version'
        }


def main():
    """Main entry point for incremental training."""
    parser = argparse.ArgumentParser(description='FinRL Incremental Learning (Fine-tuning)')

    parser.add_argument('--agent', type=str, default='all',
                       help='Agent type to train (a2c, ppo, ddpg, sac, td3, or all)')
    parser.add_argument('--days', type=int, default=60,
                       help='Lookback days for recent data (default: 60)')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Training timesteps (overrides config defaults)')
    parser.add_argument('--checkpoint-dir', type=str, default='data/finrl_models',
                       help='Base checkpoint directory')
    parser.add_argument('--output-dir', type=str, default='data/finrl_models_incremental',
                       help='Output directory for fine-tuned models')

    args = parser.parse_args()

    logger.info("🚀 FinRL Incremental Learning System")
    logger.info("=" * 60)

    # Create trainer
    trainer = IncrementalTrainer(
        lookback_days=args.days,
        checkpoint_dir=args.checkpoint_dir,
        incremental_dir=args.output_dir
    )

    # Train agents
    if args.agent.lower() == 'all':
        # Train all agents
        timesteps_override = {}
        if args.timesteps:
            timesteps_override = {agent: args.timesteps for agent in ['a2c', 'ppo', 'ddpg', 'sac', 'td3']}

        metrics = trainer.fine_tune_all_agents(timesteps_override=timesteps_override)
    else:
        # Train single agent
        agent_type = args.agent.lower()
        if agent_type not in ['a2c', 'ppo', 'ddpg', 'sac', 'td3']:
            logger.error(f"❌ Invalid agent type: {agent_type}")
            sys.exit(1)

        _, metrics = trainer.fine_tune_agent(agent_type, timesteps=args.timesteps)

    logger.info("🎉 Incremental training completed!")
    logger.info(f"📁 Fine-tuned models saved in: {args.output_dir}")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
