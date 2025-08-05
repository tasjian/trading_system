"""
Reinforcement Learning Trading Environment
Integrates with LLM sentiment analysis for enhanced state representation
"""

import gym
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from enum import Enum

from gym import spaces
from stable_baselines3.common.env_checker import check_env

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Trading action types."""
    HOLD = 0
    BUY = 1
    SELL = 2


class MarketRegime(Enum):
    """Market regime types."""
    BULL = 0
    BEAR = 1
    SIDEWAYS = 2
    VOLATILE = 3


@dataclass
class TradingState:
    """Complete trading state information."""
    # Market data
    price: float
    volume: float
    volatility: float
    returns: np.ndarray  # Historical returns
    
    # Technical indicators
    rsi: float
    macd: float
    bollinger_upper: float
    bollinger_lower: float
    
    # LLM-derived features
    sentiment_score: float
    sentiment_confidence: float
    news_sentiment: float
    social_sentiment: float
    earnings_sentiment: float
    sec_filings_sentiment: float
    
    # Portfolio state
    position: float  # Current position size (-1 to 1)
    cash: float
    portfolio_value: float
    unrealized_pnl: float
    
    # Risk metrics
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk
    
    # Market regime
    regime: MarketRegime
    regime_confidence: float
    
    # Time features
    hour_of_day: int
    day_of_week: int
    is_market_open: bool


@dataclass
class RewardComponents:
    """Components of the reward function."""
    pnl_reward: float
    risk_penalty: float
    transaction_cost: float
    sentiment_alignment: float
    regime_bonus: float
    total_reward: float


class LLMTradingEnvironment(gym.Env):
    """
    Custom trading environment that integrates LLM sentiment analysis
    with traditional market data for RL training.
    """
    
    def __init__(self, 
                 symbol: str = "SPY",
                 initial_balance: float = 100000.0,
                 lookback_window: int = 20,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 1.0,
                 sentiment_weight: float = 0.3,
                 use_continuous_actions: bool = True):
        
        super().__init__()
        
        # Environment configuration
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.sentiment_weight = sentiment_weight
        self.use_continuous_actions = use_continuous_actions
        
        # State tracking
        self.current_step = 0
        self.max_steps = 1000  # Will be set based on data length
        self.done = False
        
        # Portfolio tracking
        self.cash = initial_balance
        self.position = 0.0
        self.portfolio_value = initial_balance
        self.trade_history = []
        self.reward_history = []
        
        # Performance tracking
        self.max_portfolio_value = initial_balance
        self.total_return = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        
        # Data storage
        self.market_data = None
        self.sentiment_data = None
        self.states_history = []
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Initialize LLM integration
        self.sentiment_agent = None
        self._initialize_sentiment_integration()
    
    def _setup_spaces(self):
        """Set up action and observation spaces."""
        
        # Action space
        if self.use_continuous_actions:
            # Continuous actions: position size (-1 to 1)
            self.action_space = spaces.Box(
                low=-self.max_position_size, 
                high=self.max_position_size, 
                shape=(1,), 
                dtype=np.float32
            )
        else:
            # Discrete actions: Hold, Buy, Sell
            self.action_space = spaces.Discrete(3)
        
        # Observation space
        # Market features (7) + LLM features (6) + Portfolio features (4) + Risk features (3) + Regime features (2) + Time features (3)
        obs_dim = 25
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
    
    def _initialize_sentiment_integration(self):
        """Initialize LLM sentiment analysis integration."""
        try:
            from agents.sentiment_agent import SentimentAgent
            self.sentiment_agent = SentimentAgent()
            logger.info("✅ LLM sentiment integration initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize sentiment integration: {e}")
            self.sentiment_agent = None
    
    async def _get_llm_features(self, timestamp: datetime) -> Dict[str, float]:
        """Get LLM-derived features for the current timestamp."""
        if not self.sentiment_agent:
            return {
                'sentiment_score': 0.0,
                'sentiment_confidence': 0.5,
                'news_sentiment': 0.0,
                'social_sentiment': 0.0,
                'earnings_sentiment': 0.0,
                'sec_filings_sentiment': 0.0
            }
        
        try:
            # Get comprehensive sentiment analysis
            sentiment_result = await self.sentiment_agent.analyze_comprehensive_sentiment(self.symbol)
            
            return {
                'sentiment_score': sentiment_result.overall_score,
                'sentiment_confidence': sentiment_result.confidence,
                'news_sentiment': sentiment_result.news_sentiment.score if sentiment_result.news_sentiment else 0.0,
                'social_sentiment': np.mean([s.score for s in sentiment_result.social_sentiment.values()]) if sentiment_result.social_sentiment else 0.0,
                'earnings_sentiment': np.mean([s.score for s in sentiment_result.earnings_sentiment.values()]) if sentiment_result.earnings_sentiment else 0.0,
                'sec_filings_sentiment': np.mean([s.score for s in sentiment_result.sec_filings_sentiment.values()]) if sentiment_result.sec_filings_sentiment else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting LLM features: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_confidence': 0.5,
                'news_sentiment': 0.0,
                'social_sentiment': 0.0,
                'earnings_sentiment': 0.0,
                'sec_filings_sentiment': 0.0
            }
    
    def _calculate_technical_indicators(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """Calculate technical indicators from price and volume data."""
        try:
            # RSI calculation
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            if len(gains) >= 14:
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                rs = avg_gain / (avg_loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50.0
            
            # MACD calculation
            if len(prices) >= 26:
                ema_12 = self._ema(prices, 12)
                ema_26 = self._ema(prices, 26)
                macd = ema_12 - ema_26
            else:
                macd = 0.0
            
            # Bollinger Bands
            if len(prices) >= 20:
                sma_20 = np.mean(prices[-20:])
                std_20 = np.std(prices[-20:])
                bollinger_upper = sma_20 + (2 * std_20)
                bollinger_lower = sma_20 - (2 * std_20)
            else:
                current_price = prices[-1]
                bollinger_upper = current_price * 1.02
                bollinger_lower = current_price * 0.98
            
            # Volatility
            if len(prices) >= 2:
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
            else:
                volatility = 0.2  # Default volatility
            
            return {
                'rsi': rsi,
                'macd': macd,
                'bollinger_upper': bollinger_upper,
                'bollinger_lower': bollinger_lower,
                'volatility': volatility
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'bollinger_upper': prices[-1] * 1.02,
                'bollinger_lower': prices[-1] * 0.98,
                'volatility': 0.2
            }
    
    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _detect_market_regime(self, prices: np.ndarray, volumes: np.ndarray) -> Tuple[MarketRegime, float]:
        """Detect current market regime using price and volume patterns."""
        try:
            if len(prices) < 20:
                return MarketRegime.SIDEWAYS, 0.5
            
            # Calculate recent returns and volatility
            returns = np.diff(prices[-20:]) / prices[-20:-1]
            avg_return = np.mean(returns)
            volatility = np.std(returns)
            
            # Volume trend
            volume_trend = np.mean(volumes[-5:]) / np.mean(volumes[-20:-5]) if len(volumes) >= 20 else 1.0
            
            confidence = 0.7  # Base confidence
            
            # Regime detection logic
            if avg_return > 0.002 and volatility < 0.02:
                regime = MarketRegime.BULL
                confidence = min(0.9, 0.7 + abs(avg_return) * 10)
            elif avg_return < -0.002 and volatility < 0.02:
                regime = MarketRegime.BEAR
                confidence = min(0.9, 0.7 + abs(avg_return) * 10)
            elif volatility > 0.03:
                regime = MarketRegime.VOLATILE
                confidence = min(0.9, 0.7 + (volatility - 0.03) * 10)
            else:
                regime = MarketRegime.SIDEWAYS
                confidence = 0.6
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.SIDEWAYS, 0.5
    
    def _calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk metrics from historical returns."""
        try:
            if len(returns) < 2:
                return {
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'var_95': 0.0
                }
            
            # Sharpe ratio (assuming risk-free rate of 2%)
            excess_returns = returns - 0.02/252  # Daily risk-free rate
            sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(returns, 5)
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'var_95': 0.0
            }
    
    def _get_time_features(self, timestamp: datetime) -> Dict[str, float]:
        """Get time-based features."""
        return {
            'hour_of_day': timestamp.hour / 24.0,
            'day_of_week': timestamp.weekday() / 6.0,
            'is_market_open': 1.0 if 9 <= timestamp.hour <= 16 else 0.0
        }
    
    async def _get_current_state(self) -> TradingState:
        """Get current trading state with all features."""
        if self.market_data is None or self.current_step >= len(self.market_data):
            raise ValueError("No market data available or step out of bounds")
        
        # Get current market data
        current_data = self.market_data.iloc[self.current_step]
        timestamp = current_data.name if hasattr(current_data, 'name') else datetime.now()
        
        # Get historical data for calculations
        start_idx = max(0, self.current_step - self.lookback_window)
        historical_prices = self.market_data['close'].iloc[start_idx:self.current_step+1].values
        historical_volumes = self.market_data['volume'].iloc[start_idx:self.current_step+1].values
        
        # Calculate technical indicators
        tech_indicators = self._calculate_technical_indicators(historical_prices, historical_volumes)
        
        # Get LLM features
        llm_features = await self._get_llm_features(timestamp)
        
        # Detect market regime
        regime, regime_confidence = self._detect_market_regime(historical_prices, historical_volumes)
        
        # Calculate risk metrics
        if len(historical_prices) >= 2:
            returns = np.diff(historical_prices) / historical_prices[:-1]
            risk_metrics = self._calculate_risk_metrics(returns)
        else:
            risk_metrics = {'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'var_95': 0.0}
        
        # Get time features
        time_features = self._get_time_features(timestamp)
        
        # Calculate portfolio metrics
        current_price = float(current_data['close'])
        portfolio_value = self.cash + (self.position * current_price)
        unrealized_pnl = (self.position * current_price) - (self.position * self.entry_price if hasattr(self, 'entry_price') and self.position != 0 else 0)
        
        return TradingState(
            # Market data
            price=current_price,
            volume=float(current_data['volume']),
            volatility=tech_indicators['volatility'],
            returns=returns if len(historical_prices) >= 2 else np.array([0.0]),
            
            # Technical indicators
            rsi=tech_indicators['rsi'],
            macd=tech_indicators['macd'],
            bollinger_upper=tech_indicators['bollinger_upper'],
            bollinger_lower=tech_indicators['bollinger_lower'],
            
            # LLM features
            sentiment_score=llm_features['sentiment_score'],
            sentiment_confidence=llm_features['sentiment_confidence'],
            news_sentiment=llm_features['news_sentiment'],
            social_sentiment=llm_features['social_sentiment'],
            earnings_sentiment=llm_features['earnings_sentiment'],
            sec_filings_sentiment=llm_features['sec_filings_sentiment'],
            
            # Portfolio state
            position=self.position,
            cash=self.cash,
            portfolio_value=portfolio_value,
            unrealized_pnl=unrealized_pnl,
            
            # Risk metrics
            sharpe_ratio=risk_metrics['sharpe_ratio'],
            max_drawdown=risk_metrics['max_drawdown'],
            var_95=risk_metrics['var_95'],
            
            # Market regime
            regime=regime,
            regime_confidence=regime_confidence,
            
            # Time features
            hour_of_day=int(time_features['hour_of_day'] * 24),
            day_of_week=int(time_features['day_of_week'] * 6),
            is_market_open=bool(time_features['is_market_open'])
        )
    
    def _state_to_array(self, state: TradingState) -> np.ndarray:
        """Convert TradingState to numpy array for RL agent."""
        return np.array([
            # Market features (7)
            state.price / 1000.0,  # Normalize price
            state.volume / 1e6,    # Normalize volume
            state.volatility,
            np.mean(state.returns[-5:]) if len(state.returns) >= 5 else 0.0,
            state.rsi / 100.0,
            state.macd,
            (state.price - state.bollinger_lower) / (state.bollinger_upper - state.bollinger_lower + 1e-8),
            
            # LLM features (6)
            state.sentiment_score,
            state.sentiment_confidence,
            state.news_sentiment,
            state.social_sentiment,
            state.earnings_sentiment,
            state.sec_filings_sentiment,
            
            # Portfolio features (4)
            state.position,
            state.cash / self.initial_balance,
            state.portfolio_value / self.initial_balance,
            state.unrealized_pnl / self.initial_balance,
            
            # Risk features (3)
            np.clip(state.sharpe_ratio / 3.0, -1, 1),  # Normalize Sharpe
            state.max_drawdown,
            state.var_95,
            
            # Regime features (2)
            state.regime.value / 3.0,  # Normalize regime
            state.regime_confidence,
            
            # Time features (3)
            state.hour_of_day / 24.0,
            state.day_of_week / 6.0,
            float(state.is_market_open)
        ], dtype=np.float32)
    
    def _calculate_reward(self, action: float, prev_state: TradingState, current_state: TradingState) -> RewardComponents:
        """Calculate comprehensive reward with multiple components."""
        
        # PnL-based reward
        portfolio_change = current_state.portfolio_value - prev_state.portfolio_value
        pnl_reward = portfolio_change / self.initial_balance
        
        # Risk penalty
        risk_penalty = 0.0
        if current_state.max_drawdown < -0.1:  # Penalize large drawdowns
            risk_penalty = current_state.max_drawdown * 2.0
        
        # Transaction cost
        position_change = abs(current_state.position - prev_state.position)
        transaction_cost = -position_change * self.transaction_cost
        
        # Sentiment alignment reward
        sentiment_alignment = 0.0
        if abs(current_state.sentiment_score) > 0.1:  # Only if sentiment is significant
            if (current_state.position > 0 and current_state.sentiment_score > 0) or \
               (current_state.position < 0 and current_state.sentiment_score < 0):
                sentiment_alignment = abs(current_state.sentiment_score) * current_state.sentiment_confidence * 0.1
        
        # Market regime bonus
        regime_bonus = 0.0
        if current_state.regime == MarketRegime.BULL and current_state.position > 0:
            regime_bonus = 0.05 * current_state.regime_confidence
        elif current_state.regime == MarketRegime.BEAR and current_state.position < 0:
            regime_bonus = 0.05 * current_state.regime_confidence
        
        # Total reward
        total_reward = pnl_reward + risk_penalty + transaction_cost + \
                      (sentiment_alignment * self.sentiment_weight) + regime_bonus
        
        return RewardComponents(
            pnl_reward=pnl_reward,
            risk_penalty=risk_penalty,
            transaction_cost=transaction_cost,
            sentiment_alignment=sentiment_alignment,
            regime_bonus=regime_bonus,
            total_reward=total_reward
        )
    
    def set_data(self, market_data: pd.DataFrame):
        """Set market data for the environment."""
        self.market_data = market_data.copy()
        self.max_steps = len(market_data) - 1
        logger.info(f"✅ Set market data: {len(market_data)} steps for {self.symbol}")
    
    async def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = self.lookback_window  # Start after lookback window
        self.cash = self.initial_balance
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        self.done = False
        
        self.trade_history = []
        self.reward_history = []
        self.states_history = []
        
        self.max_portfolio_value = self.initial_balance
        
        # Get initial state
        state = await self._get_current_state()
        self.states_history.append(state)
        
        return self._state_to_array(state)
    
    async def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        if self.done:
            raise ValueError("Environment is done, call reset()")
        
        # Get current state
        prev_state = self.states_history[-1] if self.states_history else await self._get_current_state()
        
        # Execute action
        if self.use_continuous_actions:
            new_position = np.clip(float(action[0]), -self.max_position_size, self.max_position_size)
        else:
            if action == ActionType.BUY.value:
                new_position = min(self.max_position_size, self.position + 0.1)
            elif action == ActionType.SELL.value:
                new_position = max(-self.max_position_size, self.position - 0.1)
            else:  # HOLD
                new_position = self.position
        
        # Update position and cash
        current_price = float(self.market_data.iloc[self.current_step]['close'])
        position_change = new_position - self.position
        trade_value = position_change * current_price
        
        # Check if we have enough cash
        if trade_value > self.cash:
            trade_value = self.cash
            new_position = self.position + (trade_value / current_price)
        
        self.cash -= trade_value
        self.position = new_position
        
        if abs(position_change) > 1e-6:  # Record trade
            self.trade_history.append({
                'step': self.current_step,
                'action': action,
                'position_change': position_change,
                'price': current_price,
                'value': trade_value
            })
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        self.done = (self.current_step >= self.max_steps) or (self.cash + self.position * current_price <= 0)
        
        # Get new state
        current_state = await self._get_current_state()
        self.states_history.append(current_state)
        
        # Calculate reward
        reward_components = self._calculate_reward(action, prev_state, current_state)
        self.reward_history.append(reward_components)
        
        # Update performance tracking
        self.portfolio_value = current_state.portfolio_value
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        
        # Prepare info dict
        info = {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'cash': self.cash,
            'reward_components': reward_components,
            'sentiment_score': current_state.sentiment_score,
            'regime': current_state.regime.name,
            'trades_count': len(self.trade_history)
        }
        
        return self._state_to_array(current_state), reward_components.total_reward, self.done, info
    
    def render(self, mode='human'):
        """Render the environment state."""
        if not self.states_history:
            return
        
        current_state = self.states_history[-1]
        
        print(f"\n=== Trading Environment State ===")
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Price: ${current_state.price:.2f}")
        print(f"Position: {current_state.position:.4f}")
        print(f"Cash: ${current_state.cash:.2f}")
        print(f"Portfolio Value: ${current_state.portfolio_value:.2f}")
        print(f"Total Return: {((current_state.portfolio_value / self.initial_balance) - 1) * 100:.2f}%")
        print(f"Sentiment: {current_state.sentiment_score:.3f} (confidence: {current_state.sentiment_confidence:.3f})")
        print(f"Market Regime: {current_state.regime.name} (confidence: {current_state.regime_confidence:.3f})")
        print(f"RSI: {current_state.rsi:.1f}")
        print(f"Trades: {len(self.trade_history)}")
        
        if self.reward_history:
            last_reward = self.reward_history[-1]
            print(f"Last Reward: {last_reward.total_reward:.6f}")
            print(f"  PnL: {last_reward.pnl_reward:.6f}")
            print(f"  Risk Penalty: {last_reward.risk_penalty:.6f}")
            print(f"  Transaction Cost: {last_reward.transaction_cost:.6f}")
            print(f"  Sentiment Alignment: {last_reward.sentiment_alignment:.6f}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics."""
        if not self.states_history:
            return {}
        
        current_state = self.states_history[-1]
        
        # Calculate returns
        total_return = (current_state.portfolio_value / self.initial_balance) - 1
        
        # Calculate metrics from reward history
        total_rewards = [r.total_reward for r in self.reward_history]
        avg_reward = np.mean(total_rewards) if total_rewards else 0.0
        
        return {
            'total_return': total_return,
            'portfolio_value': current_state.portfolio_value,
            'max_drawdown': current_state.max_drawdown,
            'sharpe_ratio': current_state.sharpe_ratio,
            'trades_count': len(self.trade_history),
            'avg_reward': avg_reward,
            'final_position': current_state.position,
            'steps_completed': self.current_step
        }
    
    async def close(self):
        """Clean up resources."""
        if self.sentiment_agent:
            await self.sentiment_agent.close()


# Environment validation
def validate_environment():
    """Validate the trading environment setup."""
    env = LLMTradingEnvironment(symbol="AAPL")
    
    try:
        check_env(env)
        logger.info("✅ Environment validation passed")
        return True
    except Exception as e:
        logger.error(f"❌ Environment validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test the environment
    async def test_environment():
        env = LLMTradingEnvironment(symbol="AAPL")
        
        # Create sample data
        import yfinance as yf
        data = yf.download("AAPL", period="1mo", interval="1h")
        env.set_data(data)
        
        # Test reset and steps
        obs = await env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = await env.step(action)
            env.render()
            if done:
                break
        
        metrics = env.get_performance_metrics()
        print(f"\nPerformance metrics: {metrics}")
        
        await env.close()
    
    asyncio.run(test_environment())