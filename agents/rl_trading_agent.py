"""
Reinforcement Learning Trading Agent
PPO-based agent for portfolio allocation and trade execution with LLM integration
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import pickle
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from agents.rl_trading_env import LLMTradingEnvironment, MarketRegime, TradingState

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for RL agent training."""
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Environment settings
    n_envs: int = 4
    lookback_window: int = 20
    transaction_cost: float = 0.001
    sentiment_weight: float = 0.3
    
    # Evaluation settings
    eval_freq: int = 10000
    n_eval_episodes: int = 5
    
    # Model saving
    save_path: str = "models/rl_trading_agent"
    log_path: str = "logs/rl_training"


@dataclass
class TradingPerformance:
    """Trading performance metrics."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades_count: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float


class LLMTradingFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that processes LLM sentiment features
    alongside traditional market features.
    """
    
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        # Input dimension from observation space
        input_dim = observation_space.shape[0]
        
        # Market features network (first 7 features)
        self.market_net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # LLM sentiment features network (next 6 features)
        self.sentiment_net = nn.Sequential(
            nn.Linear(6, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU()
        )
        
        # Portfolio features network (next 4 features)
        self.portfolio_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        # Risk and regime features network (next 5 features)
        self.risk_regime_net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        # Time features network (last 3 features)
        self.time_net = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU()
        )
        
        # Combined features network
        combined_dim = 32 + 24 + 16 + 16 + 8  # 96
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split observations into feature groups
        market_features = observations[:, :7]
        sentiment_features = observations[:, 7:13]
        portfolio_features = observations[:, 13:17]
        risk_regime_features = observations[:, 17:22]
        time_features = observations[:, 22:25]
        
        # Process each feature group
        market_out = self.market_net(market_features)
        sentiment_out = self.sentiment_net(sentiment_features)
        portfolio_out = self.portfolio_net(portfolio_features)
        risk_regime_out = self.risk_regime_net(risk_regime_features)
        time_out = self.time_net(time_features)
        
        # Combine all features
        combined = torch.cat([
            market_out, sentiment_out, portfolio_out, 
            risk_regime_out, time_out
        ], dim=1)
        
        return self.combined_net(combined)


class TradingCallback(BaseCallback):
    """Custom callback for monitoring training progress."""
    
    def __init__(self, eval_env, eval_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.performance_history = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the model
            mean_reward, performance = self._evaluate_model()
            self.performance_history.append(performance)
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward:.4f}")
                
                # Save best model
                self.model.save(f"{self.model.logger.dir}/best_model")
            
            # Log performance metrics
            if self.verbose > 0:
                print(f"Evaluation at step {self.n_calls}:")
                print(f"  Mean reward: {mean_reward:.4f}")
                print(f"  Total return: {performance.total_return:.2%}")
                print(f"  Sharpe ratio: {performance.sharpe_ratio:.3f}")
                print(f"  Max drawdown: {performance.max_drawdown:.2%}")
        
        return True
    
    def _evaluate_model(self, n_eval_episodes: int = 5):
        """Evaluate the model and return performance metrics."""
        episode_rewards = []
        episode_returns = []
        episode_trades = []
        
        for _ in range(n_eval_episodes):
            obs = self.eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward
            
            metrics = self.eval_env.get_performance_metrics()
            episode_rewards.append(episode_reward)
            episode_returns.append(metrics.get('total_return', 0))
            episode_trades.append(metrics.get('trades_count', 0))
        
        mean_reward = np.mean(episode_rewards)
        mean_return = np.mean(episode_returns)
        
        # Calculate performance metrics
        performance = TradingPerformance(
            total_return=mean_return,
            sharpe_ratio=mean_return / (np.std(episode_returns) + 1e-8) * np.sqrt(252),
            max_drawdown=0.0,  # Would need more detailed tracking
            win_rate=0.0,  # Would need trade-level analysis
            profit_factor=0.0,  # Would need trade-level analysis
            trades_count=int(np.mean(episode_trades)),
            avg_trade_return=0.0,  # Would need trade-level analysis
            volatility=np.std(episode_returns) * np.sqrt(252),
            calmar_ratio=0.0,  # Would need drawdown calculation
            sortino_ratio=0.0   # Would need downside deviation
        )
        
        return mean_reward, performance


class RLTradingAgent:
    """
    Main RL Trading Agent using PPO with LLM integration.
    """
    
    def __init__(self, 
                 symbol: str = "SPY",
                 config: TrainingConfig = None,
                 device: str = "auto"):
        
        self.symbol = symbol
        self.config = config or TrainingConfig()
        self.device = device
        
        # Initialize components
        self.model = None
        self.training_env = None
        self.eval_env = None
        self.vec_env = None
        self.callbacks = []
        
        # Performance tracking
        self.training_history = []
        self.evaluation_history = []
        
        # Model management
        self.model_path = self.config.save_path
        self.is_trained = False
        
        logger.info(f"✅ Initialized RL Trading Agent for {symbol}")
    
    def _create_environment(self, is_training: bool = True) -> LLMTradingEnvironment:
        """Create trading environment."""
        env = LLMTradingEnvironment(
            symbol=self.symbol,
            lookback_window=self.config.lookback_window,
            transaction_cost=self.config.transaction_cost,
            sentiment_weight=self.config.sentiment_weight,
            use_continuous_actions=True
        )
        
        # Wrap with Monitor for logging
        if is_training:
            os.makedirs(self.config.log_path, exist_ok=True)
            env = Monitor(env, self.config.log_path)
        
        return env
    
    def _setup_vectorized_env(self, market_data: pd.DataFrame):
        """Set up vectorized environment for training."""
        def make_env():
            env = self._create_environment(is_training=True)
            env.set_data(market_data)
            return env
        
        # Create multiple environments
        self.vec_env = DummyVecEnv([make_env for _ in range(self.config.n_envs)])
        
        # Add normalization
        self.vec_env = VecNormalize(
            self.vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=self.config.gamma
        )
    
    def _create_model(self):
        """Create PPO model with custom architecture."""
        
        # Custom policy with LLM feature extractor
        policy_kwargs = dict(
            features_extractor_class=LLMTradingFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=[256, 128, 64]  # Actor-Critic network architecture
        )
        
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.config.learning_rate,
            n_steps=2048 // self.config.n_envs,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=self.config.log_path,
            device=self.device
        )
        
        logger.info("✅ Created PPO model with custom LLM feature extractor")
    
    def _setup_callbacks(self, eval_data: pd.DataFrame):
        """Set up training callbacks."""
        # Create evaluation environment
        self.eval_env = self._create_environment(is_training=False)
        self.eval_env.set_data(eval_data)
        
        # Create callbacks
        eval_callback = TradingCallback(
            eval_env=self.eval_env,
            eval_freq=self.config.eval_freq,
            verbose=1
        )
        
        self.callbacks = [eval_callback]
    
    async def train(self, 
              train_data: pd.DataFrame,
              eval_data: pd.DataFrame = None,
              save_model: bool = True):
        """Train the RL agent."""
        
        logger.info(f"🚀 Starting RL agent training for {self.symbol}")
        logger.info(f"📊 Training data: {len(train_data)} steps")
        
        try:
            # Setup environment
            self._setup_vectorized_env(train_data)
            
            # Create model
            self._create_model()
            
            # Setup callbacks
            if eval_data is not None:
                self._setup_callbacks(eval_data)
                logger.info(f"📊 Evaluation data: {len(eval_data)} steps")
            
            # Start training
            logger.info(f"🏋️ Training for {self.config.total_timesteps} timesteps...")
            
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=self.callbacks,
                tb_log_name=f"rl_trading_{self.symbol}",
                reset_num_timesteps=False
            )
            
            self.is_trained = True
            
            # Save model
            if save_model:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.model.save(self.model_path)
                
                # Save vectorization stats
                if self.vec_env:
                    self.vec_env.save(f"{self.model_path}_vec_normalize.pkl")
                
                logger.info(f"💾 Model saved to {self.model_path}")
            
            logger.info("✅ Training completed successfully!")
            
            return self._get_training_summary()
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def load_model(self, model_path: str = None):
        """Load trained model."""
        model_path = model_path or self.model_path
        
        try:
            self.model = PPO.load(model_path, device=self.device)
            
            # Load vectorization stats if available
            vec_path = f"{model_path}_vec_normalize.pkl"
            if os.path.exists(vec_path):
                self.vec_env = VecNormalize.load(vec_path, DummyVecEnv([lambda: None]))
            
            self.is_trained = True
            logger.info(f"✅ Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    async def predict(self, 
                state: Union[np.ndarray, TradingState],
                deterministic: bool = True) -> Tuple[float, float]:
        """Predict action given current state."""
        
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained or loaded")
        
        try:
            # Convert TradingState to array if needed
            if isinstance(state, TradingState):
                from agents.rl_trading_env import LLMTradingEnvironment
                env = LLMTradingEnvironment()
                obs = env._state_to_array(state)
            else:
                obs = state
            
            # Normalize observation if vec_env is available
            if self.vec_env is not None:
                obs = self.vec_env.normalize_obs(obs.reshape(1, -1))
                obs = obs.flatten()
            
            # Predict action
            action, _ = self.model.predict(obs, deterministic=deterministic)
            
            # Extract action value and confidence
            action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            confidence = 0.8 if deterministic else 0.6  # Placeholder confidence
            
            return action_value, confidence
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 0.0, 0.0  # Default to hold position
    
    async def backtest(self, 
                 test_data: pd.DataFrame,
                 initial_balance: float = 100000.0) -> Dict[str, Any]:
        """Backtest the trained agent."""
        
        if not self.is_trained:
            raise ValueError("Agent must be trained before backtesting")
        
        logger.info(f"📈 Starting backtest for {self.symbol}")
        logger.info(f"📊 Test data: {len(test_data)} steps")
        
        # Create test environment
        test_env = self._create_environment(is_training=False)
        test_env.initial_balance = initial_balance
        test_env.set_data(test_data)
        
        # Run backtest
        obs = await test_env.reset()
        done = False
        step = 0
        
        actions_taken = []
        rewards_received = []
        portfolio_values = []
        
        while not done:
            # Get action from model
            action, confidence = await self.predict(obs, deterministic=True)
            actions_taken.append(action)
            
            # Execute action
            obs, reward, done, info = await test_env.step([action])
            rewards_received.append(reward)
            portfolio_values.append(info['portfolio_value'])
            
            step += 1
            
            if step % 100 == 0:
                logger.info(f"Backtest step {step}: Portfolio value ${info['portfolio_value']:.2f}")
        
        # Get final performance metrics
        performance_metrics = test_env.get_performance_metrics()
        
        # Calculate additional metrics
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        additional_metrics = self._calculate_performance_metrics(
            portfolio_values, returns, initial_balance
        )
        
        backtest_results = {
            'performance_metrics': performance_metrics,
            'additional_metrics': additional_metrics,
            'portfolio_values': portfolio_values.tolist(),
            'actions_taken': actions_taken,
            'rewards_received': rewards_received,
            'total_steps': step,
            'trade_history': test_env.trade_history
        }
        
        await test_env.close()
        
        logger.info("✅ Backtest completed")
        logger.info(f"💰 Final return: {additional_metrics['total_return']:.2%}")
        logger.info(f"📊 Sharpe ratio: {additional_metrics['sharpe_ratio']:.3f}")
        
        return backtest_results
    
    def _calculate_performance_metrics(self, 
                                     portfolio_values: np.ndarray,
                                     returns: np.ndarray,
                                     initial_balance: float) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        
        total_return = (portfolio_values[-1] / initial_balance) - 1
        
        # Sharpe ratio (assuming 2% risk-free rate)
        excess_returns = returns - (0.02 / 252)
        sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)
        
        # Maximum drawdown
        cumulative_returns = portfolio_values / initial_balance
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(252)
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'symbol': self.symbol,
            'total_timesteps': self.config.total_timesteps,
            'training_completed': True,
            'model_path': self.model_path,
            'config': self.config.__dict__
        }
    
    async def close(self):
        """Clean up resources."""
        if self.vec_env:
            self.vec_env.close()
        if self.eval_env:
            await self.eval_env.close()


# Utility functions
def create_training_config(
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    sentiment_weight: float = 0.3
) -> TrainingConfig:
    """Create training configuration with common settings."""
    
    return TrainingConfig(
        total_timesteps=total_timesteps,
        learning_rate=learning_rate,
        sentiment_weight=sentiment_weight,
        batch_size=64,
        n_epochs=10,
        n_envs=4,
        eval_freq=10000
    )


async def train_rl_agent(
    symbol: str,
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame = None,
    config: TrainingConfig = None,
    device: str = "auto"
) -> RLTradingAgent:
    """Train an RL agent with the given data."""
    
    agent = RLTradingAgent(symbol=symbol, config=config, device=device)
    
    training_summary = await agent.train(
        train_data=train_data,
        eval_data=eval_data,
        save_model=True
    )
    
    logger.info(f"✅ Training completed: {training_summary}")
    return agent


if __name__ == "__main__":
    # Test the RL agent
    async def test_rl_agent():
        import yfinance as yf
        
        # Get sample data
        data = yf.download("AAPL", period="3mo", interval="1h")
        
        # Split data
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        # Create and train agent
        config = create_training_config(total_timesteps=50000)
        agent = RLTradingAgent("AAPL", config=config)
        
        # Train agent
        await agent.train(train_data, test_data)
        
        # Backtest
        results = await agent.backtest(test_data)
        
        print(f"Backtest results: {results['additional_metrics']}")
        
        await agent.close()
    
    asyncio.run(test_rl_agent())