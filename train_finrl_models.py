#!/usr/bin/env python3
"""
Train FinRL DRL Models

This script trains FinRL DRL agents using historical market data, following the
official FinRL training examples from https://github.com/AI4Finance-Foundation/FinRL

Usage:
python train_finrl_models.py
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add FinRL to Python path
finrl_path = '/Users/zac/Desktop/02_PROJECTS/FinRL'
if finrl_path not in sys.path:
    sys.path.insert(0, finrl_path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import FinRL components
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.data_processor import DataProcessor
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories

from stable_baselines3.common.logger import configure

# Import our Alpaca client for real data
sys.path.insert(0, '/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system')
from tools.alpaca_client import alpaca_client
from utils.performance_tracker import PerformanceTracker

class FinRLModelTrainer:
    """FinRL Model Trainer using real Alpaca market data."""
    
    def __init__(self, symbols=None, start_date='2008-01-01', end_date='2024-12-31'):
        self.symbols = symbols or ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
        self.start_date = start_date
        self.end_date = end_date
        
        # FinRL configuration
        self.tech_indicator_list = [
            'macd', 'rsi_30', 'cci_30', 'dx_30', 'bb_bbm', 'bb_bbh', 'bb_bbl'
        ]
        
        self.training_data = None
        self.stock_dimension = len(self.symbols)
        self.state_space = 1 + 2 * self.stock_dimension + len(self.tech_indicator_list) * self.stock_dimension
        
        # Create directories
        os.makedirs('data/finrl_models', exist_ok=True)
        os.makedirs('results', exist_ok=True)

        # Performance tracking
        self.performance_tracker = PerformanceTracker(db_path="data/performance_metrics.db")

        logger.info(f"🎯 Initialized FinRL Trainer for {len(self.symbols)} symbols")
        logger.info(f"📊 State Space: {self.state_space}, Stock Dimension: {self.stock_dimension}")
        logger.info(f"📅 Training Period: {start_date} to {end_date} (includes 2008-2010 financial crisis)")
        logger.info(f"💪 Crisis-Resilient Training: Agents will learn from extreme market conditions")
    
    def prepare_data(self):
        """Prepare training data from Alpaca."""
        logger.info("📊 Preparing training data from Alpaca...")
        
        try:
            # Get historical data from Alpaca for all symbols
            all_data = []
            
            for symbol in self.symbols:
                logger.info(f"📈 Fetching data for {symbol}...")
                
                try:
                    # Use the raw Alpaca API for historical data
                    bars = alpaca_client.api.get_bars(
                        symbol, 
                        '1Day', 
                        start=self.start_date, 
                        end=self.end_date
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
                    logger.warning(f"⚠️ Failed to get data for {symbol}: {e}")
            
            if not all_data:
                raise ValueError("No market data retrieved")
            
            # Create DataFrame
            df = pd.DataFrame(all_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['date', 'tic']).reset_index(drop=True)
            
            logger.info(f"✅ Retrieved {len(df)} records for {len(df['tic'].unique())} symbols")
            
            # Add technical indicators using FinRL
            logger.info("🔧 Adding technical indicators...")
            try:
                # Try with standard FinRL approach first
                dp = DataProcessor()
                
                # Add technical indicators
                df = dp.add_technical_indicator(df, tech_indicator_list=self.tech_indicator_list)
                
                # Add turbulence (market volatility indicator)
                df = dp.add_turbulence(df)
                
                logger.info(f"✅ Added technical indicators using FinRL DataProcessor")
                
            except Exception as e:
                logger.warning(f"⚠️ FinRL DataProcessor failed: {e}")
                logger.info("🔧 Adding technical indicators manually...")
                
                # Manual technical indicators calculation
                df = df.sort_values(['tic', 'date']).reset_index(drop=True)
                
                for symbol in df['tic'].unique():
                    symbol_mask = df['tic'] == symbol
                    symbol_df = df[symbol_mask].copy().sort_values('date')
                    
                    # Simple technical indicators
                    symbol_df['macd'] = symbol_df['close'].ewm(span=12).mean() - symbol_df['close'].ewm(span=26).mean()
                    symbol_df['rsi_30'] = 50.0  # Simplified RSI
                    symbol_df['cci_30'] = 0.0
                    symbol_df['dx_30'] = 0.0
                    symbol_df['bb_bbm'] = symbol_df['close'].rolling(20).mean()
                    symbol_df['bb_bbh'] = symbol_df['bb_bbm'] + (symbol_df['close'].rolling(20).std() * 2)
                    symbol_df['bb_bbl'] = symbol_df['bb_bbm'] - (symbol_df['close'].rolling(20).std() * 2)
                    symbol_df['turbulence'] = symbol_df['close'].rolling(20).std() / symbol_df['close'].rolling(20).mean()
                    
                    # Update main dataframe
                    for col in ['macd', 'rsi_30', 'cci_30', 'dx_30', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'turbulence']:
                        df.loc[symbol_mask, col] = symbol_df[col].values
                
                logger.info(f"✅ Added technical indicators manually")
            
            # Fill any NaN values
            df = df.ffill().bfill().fillna(0)
            
            logger.info(f"✅ Added technical indicators. Final dataset: {len(df)} records")
            
            self.training_data = df
            return True
            
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}")
            return False
    
    def create_environment(self, df):
        """Create FinRL trading environment."""
        logger.info("🏗️ Creating FinRL trading environment...")
        
        try:
            # Environment parameters
            env_kwargs = {
                "hmax": 100,  # Max shares per trade
                "initial_amount": 100000,  # Starting capital
                "num_stock_shares": [0] * self.stock_dimension,
                "buy_cost_pct": [0.001] * self.stock_dimension,  # 0.1% transaction cost
                "sell_cost_pct": [0.001] * self.stock_dimension,
                "state_space": self.state_space,
                "stock_dim": self.stock_dimension,
                "tech_indicator_list": self.tech_indicator_list,
                "action_space": self.stock_dimension,
                "reward_scaling": 1e-4,
                "print_verbosity": 100
            }
            
            # Create environment
            e_train_gym = StockTradingEnv(df=df, **env_kwargs)
            
            # Get Stable Baselines3 compatible environment
            env_train, _ = e_train_gym.get_sb_env()
            
            logger.info("✅ FinRL trading environment created successfully")
            return env_train
            
        except Exception as e:
            logger.error(f"❌ Environment creation failed: {e}")
            return None
    
    def train_agent(self, env_train, agent_type, timesteps, model_params=None):
        """Train a specific DRL agent."""
        logger.info(f"🚀 Training {agent_type.upper()} agent for {timesteps} timesteps...")
        
        try:
            # Create DRL agent
            agent = DRLAgent(env=env_train)
            model = agent.get_model(agent_type, model_kwargs=model_params)
            
            # Set up logging
            tmp_path = f'results/{agent_type}'
            os.makedirs(tmp_path, exist_ok=True)
            new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            model.set_logger(new_logger)
            
            # Train the model
            trained_model = agent.train_model(
                model=model,
                tb_log_name=agent_type,
                total_timesteps=timesteps
            )
            
            # Save the model
            model_path = f"data/finrl_models/agent_{agent_type}"
            trained_model.save(model_path)
            
            logger.info(f"✅ {agent_type.upper()} training completed. Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ {agent_type.upper()} training failed: {e}")
            return False
    
    def train_all_models(self):
        """Train all DRL models."""
        logger.info("🚀 Starting FinRL DRL Model Training...")
        
        # Prepare data
        if not self.prepare_data():
            logger.error("❌ Data preparation failed. Cannot proceed with training.")
            return False
        
        # Split data for training - include financial crisis period for robust learning
        train_data = data_split(
            self.training_data,
            start=self.start_date,  # Now includes 2008-2010 crisis period
            end='2023-12-31'  # Use data up to 2023 for training (16 years including crisis)
        )
        
        if len(train_data) < 1000:
            logger.error(f"❌ Insufficient training data: {len(train_data)} records")
            return False
        
        logger.info(f"📚 Training data: {len(train_data)} records")
        
        # Create environment
        env_train = self.create_environment(train_data)
        if env_train is None:
            logger.error("❌ Environment creation failed. Cannot proceed with training.")
            return False
        
        # Enhanced training configurations for crisis-resilient learning (16 years of data including 2008-2010)
        training_configs = {
            'a2c': {
                'timesteps': 100000,  # Increased for crisis period learning
                'params': {
                    'n_steps': 5, 
                    'ent_coef': 0.01, 
                    'learning_rate': 0.0007,
                    'gamma': 0.99  # Slightly higher discount for long-term crisis recovery
                }
            },
            'ppo': {
                'timesteps': 150000,  # More training for policy gradient robustness
                'params': {
                    'n_steps': 2048,
                    'ent_coef': 0.01,
                    'learning_rate': 0.00025,
                    'batch_size': 128,
                    'gamma': 0.99,
                    'clip_range': 0.2  # Standard clipping for stability
                }
            },
            'ddpg': {
                'timesteps': 100000,  # Extended for off-policy learning from crisis
                'params': {
                    'batch_size': 100, 
                    'buffer_size': 200000,  # Larger buffer for crisis experience
                    'learning_rate': 0.001,
                    'gamma': 0.99,
                    'tau': 0.005  # Soft target updates for stability
                }
            },
            'sac': {
                'timesteps': 120000,  # SAC often performs well in volatile markets
                'params': {
                    'batch_size': 128,
                    'buffer_size': 200000,  # Larger buffer for diverse experiences
                    'learning_rate': 0.0001,
                    'learning_starts': 1000,  # More random exploration initially
                    'ent_coef': 'auto_0.1',
                    'gamma': 0.99
                }
            },
            'td3': {
                'timesteps': 100000,  # Extended for robust Q-learning
                'params': {
                    'batch_size': 100, 
                    'buffer_size': 200000,  # Larger experience replay
                    'learning_rate': 0.001,
                    'gamma': 0.99,
                    'tau': 0.005,  # Soft updates
                    'policy_delay': 2  # TD3 specific parameter
                }
            }
        }
        
        # Train each agent
        successful_agents = 0
        total_agents = len(training_configs)
        
        for agent_type, config in training_configs.items():
            logger.info(f"\n📋 Training Agent {successful_agents + 1}/{total_agents}: {agent_type.upper()}")
            
            if self.train_agent(env_train, agent_type, config['timesteps'], config['params']):
                successful_agents += 1
            else:
                logger.warning(f"⚠️ {agent_type.upper()} training failed, continuing with other agents...")
        
        # Training summary
        logger.info(f"\n🎯 Training Summary: {successful_agents}/{total_agents} agents trained successfully")

        if successful_agents > 0:
            logger.info("✅ FinRL DRL model training completed successfully!")
            logger.info(f"📁 Models saved in: data/finrl_models/")
            logger.info(f"📊 Training logs in: results/")

            # Evaluate trained models
            logger.info("\n📊 Evaluating trained models on test data...")
            self.evaluate_all_models(train_data, training_configs)

            return True
        else:
            logger.error("❌ No agents were successfully trained")
            return False

    def evaluate_all_models(self, train_data, training_configs):
        """
        Evaluate all trained models on test data and save metrics.

        Args:
            train_data: Training data for reference
            training_configs: Training configurations used
        """
        try:
            from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3

            # Split data for testing (use 2024 data as test)
            test_data = data_split(
                self.training_data,
                start='2024-01-01',
                end=self.end_date
            )

            if len(test_data) < 100:
                logger.warning(f"⚠️ Insufficient test data: {len(test_data)} records. Skipping evaluation.")
                return

            logger.info(f"📊 Test data: {len(test_data)} records (2024 data)")

            # Create test environment
            test_env = self.create_environment(test_data)
            if test_env is None:
                logger.warning("⚠️ Failed to create test environment. Skipping evaluation.")
                return

            # Store metadata for test environment
            test_env.start_date = '2024-01-01'
            test_env.end_date = self.end_date

            # Model class mapping
            model_classes = {
                'a2c': A2C,
                'ppo': PPO,
                'ddpg': DDPG,
                'sac': SAC,
                'td3': TD3
            }

            # Evaluate each trained model
            for agent_type in training_configs.keys():
                model_path = f"data/finrl_models/agent_{agent_type}.zip"

                if not os.path.exists(model_path):
                    logger.warning(f"⚠️ Model not found: {model_path}")
                    continue

                try:
                    logger.info(f"📊 Evaluating {agent_type.upper()}...")

                    # Load model
                    model_class = model_classes[agent_type]
                    model = model_class.load(model_path)

                    # Evaluate and save metrics
                    perf_metrics = self.performance_tracker.evaluate_agent(
                        model=model,
                        test_env=test_env,
                        model_name=f"{agent_type}_base",
                        agent_type=agent_type,
                        notes="Full training from 2008-2024, tested on 2024 data"
                    )

                    # Save to database
                    self.performance_tracker.save_metrics(perf_metrics)

                    logger.info(f"✅ {agent_type.upper()}: Sharpe={perf_metrics.sharpe_ratio:.3f}, "
                               f"Win Rate={perf_metrics.win_rate:.2%}, "
                               f"Max DD={perf_metrics.max_drawdown:.2%}")

                except Exception as e:
                    logger.error(f"❌ Failed to evaluate {agent_type}: {e}")
                    continue

            logger.info("\n✅ Model evaluation complete!")
            logger.info("📊 Metrics saved to data/performance_metrics.db")

        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            # Don't raise - evaluation failure shouldn't stop the training script

def main():
    """Main training function."""
    logger.info("🚀 Starting FinRL DRL Model Training")
    logger.info("=" * 60)
    
    # Create trainer with extended historical data including 2008-2010 financial crisis
    trainer = FinRLModelTrainer(
        symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V'],
        start_date='2008-01-01',  # Include 2008-2010 financial crisis for robust training
        end_date='2024-09-01'
    )
    
    # Train all models
    success = trainer.train_all_models()
    
    if success:
        logger.info("🎉 FinRL DRL models are now ready for trading!")
        logger.info("🔧 Integration with trading system will automatically use these trained models")
        return True
    else:
        logger.error("💥 Training failed. Please check the logs and try again.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)