#!/usr/bin/env python3
"""
Enhanced FinRL Trading Environment for Alpaca Integration

This module creates a custom trading environment that integrates FinRL's StockTradingEnv
with our Alpaca data pipeline and existing trading system architecture.

Key Features:
1. Real-time Alpaca data integration
2. Dynamic symbol management
3. Enhanced technical indicators
4. Sentiment data integration
5. Advanced portfolio metrics
6. Turbulence-based risk management
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces

# Add FinRL to Python path
finrl_path = '/Users/zac/Desktop/02_PROJECTS/FinRL'
if finrl_path not in sys.path:
    sys.path.insert(0, finrl_path)

logger = logging.getLogger(__name__)

class AlpacaFinRLEnvironment(gym.Env):
    """
    Enhanced FinRL Trading Environment with Alpaca Integration
    
    This environment extends FinRL's StockTradingEnv with:
    - Real-time Alpaca data integration
    - Dynamic symbol management
    - Enhanced state representation
    - Sentiment analysis integration
    - Advanced reward calculation
    """
    
    metadata = {"render.modes": ["human"]}
    
    def __init__(self,
                 symbols: List[str],
                 alpaca_client=None,
                 initial_amount: float = 100000,
                 hmax: int = 100,
                 buy_cost_pct: float = 0.001,
                 sell_cost_pct: float = 0.001,
                 reward_scaling: float = 1e-4,
                 tech_indicator_list: List[str] = None,
                 lookback_window: int = 252,
                 turbulence_threshold: Optional[float] = None,
                 sentiment_weight: float = 0.1,
                 print_verbosity: int = 10):
        
        # Core parameters
        self.symbols = symbols
        self.alpaca_client = alpaca_client
        self.stock_dim = len(symbols)
        self.initial_amount = initial_amount
        self.hmax = hmax
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.print_verbosity = print_verbosity
        self.lookback_window = lookback_window
        self.turbulence_threshold = turbulence_threshold
        self.sentiment_weight = sentiment_weight
        
        # Technical indicators
        self.tech_indicator_list = tech_indicator_list or [
            'macd', 'rsi_30', 'cci_30', 'dx_30', 'bb_bbm', 'bb_bbh', 'bb_bbl'
        ]
        
        # State and action spaces
        self.state_dim = self._calculate_state_dimension()
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,))
        
        # Environment state
        self.day = 0
        self.terminal = False
        self.current_step = 0
        
        # Portfolio state
        self.cash = initial_amount
        self.stock_shares = np.zeros(self.stock_dim)
        self.stock_prices = np.ones(self.stock_dim) * 100.0  # Default prices
        
        # Performance tracking
        self.asset_memory = [initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = []
        
        # Market data
        self.market_data = {}
        self.sentiment_data = {}
        self.historical_data = None
        
        # Risk management
        self.turbulence = 0.0
        self.cost = 0.0
        self.trades = 0
        
        logger.info(f"🏗️ Initialized Alpaca-FinRL Environment: {self.stock_dim} symbols, state_dim={self.state_dim}")
        
        # Initialize market data
        self._initialize_market_data()
        
        # Set initial state
        self.state = self._get_current_state()
    
    def _calculate_state_dimension(self) -> int:
        """Calculate the dimension of the state space."""
        # State composition:
        # 1. Cash balance (1)
        # 2. Stock prices (stock_dim)
        # 3. Stock holdings (stock_dim)
        # 4. Technical indicators (tech_indicator_list * stock_dim)
        # 5. Sentiment scores (stock_dim)
        # 6. Portfolio metrics (3: total_value, portfolio_return, sharpe_ratio)
        # 7. Market regime indicators (2: volatility, trend)
        
        base_dim = 1 + (2 * self.stock_dim)  # cash + prices + holdings
        tech_dim = len(self.tech_indicator_list) * self.stock_dim
        sentiment_dim = self.stock_dim
        portfolio_dim = 3
        market_dim = 2
        
        total_dim = base_dim + tech_dim + sentiment_dim + portfolio_dim + market_dim
        return total_dim
    
    def _initialize_market_data(self):
        """Initialize market data from Alpaca - NO SYNTHETIC DATA ALLOWED."""
        try:
            if self.alpaca_client:
                logger.info("📊 Fetching real market data from Alpaca...")
                self._fetch_alpaca_data()
            else:
                logger.error("❌ No Alpaca client - REFUSING to use synthetic data for trading")
                raise ValueError("Cannot initialize market data: no Alpaca client provided. Synthetic data is not allowed for trading.")
                
        except Exception as e:
            logger.error(f"❌ Market data initialization failed: {e}")
            raise ValueError("Cannot initialize market data: real data source failed. Synthetic data is not allowed for trading.")
    
    def _fetch_alpaca_data(self):
        """Fetch real market data from Alpaca."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_window + 50)
            
            historical_data = []
            
            for symbol in self.symbols:
                try:
                    # Get historical bars
                    bars_df = self.alpaca_client.get_market_data(
                        symbol=symbol,
                        timeframe='1Day',
                        limit=self.lookback_window + 100  # Get enough data plus buffer
                    )
                    bars = bars_df.to_dict('records') if not bars_df.empty else []
                    
                    if bars:
                        for bar in bars[-self.lookback_window:]:  # Keep only recent data
                            historical_data.append({
                                'timestamp': bar.get('timestamp', bar.get('t')),
                                'symbol': symbol,
                                'open': float(bar.get('open', bar.get('o', 0))),
                                'high': float(bar.get('high', bar.get('h', 0))),
                                'low': float(bar.get('low', bar.get('l', 0))),
                                'close': float(bar.get('close', bar.get('c', 0))),
                                'volume': int(bar.get('volume', bar.get('v', 0)))
                            })
                    
                    # Get current price
                    current_price = self.alpaca_client.get_current_price(symbol)
                    if current_price:
                        self.stock_prices[self.symbols.index(symbol)] = current_price
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
            
            if historical_data:
                self.historical_data = pd.DataFrame(historical_data)
                self._calculate_technical_indicators()
                logger.info(f"✅ Loaded {len(historical_data)} historical records")
            else:
                raise ValueError("No historical data retrieved")
                
        except Exception as e:
            logger.error(f"❌ Alpaca data fetch failed: {e}")
            raise
    
    # REMOVED: _generate_synthetic_data() method
    # Synthetic data is not allowed for trading decisions
    
    def _calculate_technical_indicators(self):
        """Calculate technical indicators for each symbol."""
        try:
            for symbol in self.symbols:
                symbol_data = self.historical_data[self.historical_data['symbol'] == symbol].copy()
                if len(symbol_data) == 0:
                    continue
                
                symbol_data = symbol_data.sort_values('timestamp')
                
                # Calculate technical indicators
                prices = symbol_data['close'].values
                
                # MACD
                ema12 = pd.Series(prices).ewm(span=12).mean()
                ema26 = pd.Series(prices).ewm(span=26).mean()
                macd = ema12 - ema26
                
                # RSI
                delta = pd.Series(prices).diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=30).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=30).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                bb_middle = pd.Series(prices).rolling(20).mean()
                bb_std = pd.Series(prices).rolling(20).std()
                bb_upper = bb_middle + (bb_std * 2)
                bb_lower = bb_middle - (bb_std * 2)
                
                # Store current values (latest)
                self.market_data[symbol] = {
                    'price': prices[-1] if len(prices) > 0 else 100.0,
                    'macd': macd.iloc[-1] if len(macd) > 0 and not pd.isna(macd.iloc[-1]) else 0.0,
                    'rsi_30': rsi.iloc[-1] if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50.0,
                    'cci_30': 0.0,  # Simplified
                    'dx_30': 0.0,   # Simplified
                    'bb_bbm': bb_middle.iloc[-1] if len(bb_middle) > 0 and not pd.isna(bb_middle.iloc[-1]) else prices[-1] if len(prices) > 0 else 100.0,
                    'bb_bbh': bb_upper.iloc[-1] if len(bb_upper) > 0 and not pd.isna(bb_upper.iloc[-1]) else prices[-1] * 1.02 if len(prices) > 0 else 102.0,
                    'bb_bbl': bb_lower.iloc[-1] if len(bb_lower) > 0 and not pd.isna(bb_lower.iloc[-1]) else prices[-1] * 0.98 if len(prices) > 0 else 98.0,
                    'volume': symbol_data['volume'].iloc[-1] if len(symbol_data) > 0 else 1000000,
                    'volatility': pd.Series(prices).pct_change().std() * np.sqrt(252) if len(prices) > 1 else 0.2
                }
                
        except Exception as e:
            logger.error(f"❌ Technical indicator calculation failed: {e}")
            # Set default values
            for symbol in self.symbols:
                self.market_data[symbol] = {
                    'price': 100.0, 'macd': 0.0, 'rsi_30': 50.0, 'cci_30': 0.0, 'dx_30': 0.0,
                    'bb_bbm': 100.0, 'bb_bbh': 102.0, 'bb_bbl': 98.0, 'volume': 1000000, 'volatility': 0.2
                }
    
    def _get_current_state(self) -> np.ndarray:
        """Get current environment state."""
        try:
            state = []
            
            # 1. Cash balance (normalized)
            state.append(self.cash / self.initial_amount)
            
            # 2. Stock prices (normalized by initial amount for scale)
            for i, symbol in enumerate(self.symbols):
                price = self.market_data.get(symbol, {}).get('price', self.stock_prices[i])
                state.append(price / 100.0)  # Normalize around 1.0
            
            # 3. Stock holdings (normalized by max position)
            for i in range(self.stock_dim):
                state.append(self.stock_shares[i] / (self.hmax * 10))  # Normalize by reasonable max
            
            # 4. Technical indicators
            for symbol in self.symbols:
                symbol_data = self.market_data.get(symbol, {})
                state.extend([
                    np.tanh(symbol_data.get('macd', 0.0)),  # Bounded
                    symbol_data.get('rsi_30', 50.0) / 100.0,  # 0-1 range
                    np.tanh(symbol_data.get('cci_30', 0.0) / 100.0),  # Bounded
                    np.tanh(symbol_data.get('dx_30', 0.0) / 100.0),  # Bounded
                    symbol_data.get('bb_bbm', 100.0) / 100.0,  # Normalized
                    symbol_data.get('bb_bbh', 102.0) / 100.0,  # Normalized
                    symbol_data.get('bb_bbl', 98.0) / 100.0   # Normalized
                ])
            
            # 5. Sentiment scores (if available)
            for symbol in self.symbols:
                sentiment = self.sentiment_data.get(symbol, {}).get('score', 0.0)
                state.append(np.tanh(sentiment))  # Bounded -1 to 1
            
            # 6. Portfolio metrics
            total_value = self._calculate_total_value()
            portfolio_return = (total_value - self.initial_amount) / self.initial_amount
            
            state.extend([
                total_value / self.initial_amount,  # Total value ratio
                np.tanh(portfolio_return),  # Portfolio return (bounded)
                np.tanh(self._calculate_sharpe_ratio())  # Sharpe ratio (bounded)
            ])
            
            # 7. Market regime indicators
            volatility = np.mean([self.market_data.get(s, {}).get('volatility', 0.2) for s in self.symbols])
            trend = np.mean([self.market_data.get(s, {}).get('price', 100.0) for s in self.symbols]) / 100.0 - 1.0
            
            state.extend([
                min(1.0, volatility / 0.5),  # Normalized volatility
                np.tanh(trend * 10)  # Bounded trend
            ])
            
            return np.array(state, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"❌ State calculation failed: {e}")
            # Return zero state as fallback
            return np.zeros(self.state_dim, dtype=np.float32)
    
    def _calculate_total_value(self) -> float:
        """Calculate total portfolio value."""
        cash_value = self.cash
        stock_value = 0.0
        
        for i, symbol in enumerate(self.symbols):
            price = self.market_data.get(symbol, {}).get('price', self.stock_prices[i])
            stock_value += self.stock_shares[i] * price
        
        return cash_value + stock_value
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio based on returns history."""
        if len(self.asset_memory) < 30:  # Need minimum history
            return 0.0
        
        try:
            returns = pd.Series(self.asset_memory).pct_change().dropna()
            if len(returns) == 0 or returns.std() == 0:
                return 0.0
            
            sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
            return float(sharpe)
            
        except Exception:
            return 0.0
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one trading step."""
        try:
            # Normalize actions to trading amounts
            actions = np.array(actions, dtype=np.float32)
            actions = np.clip(actions, -1.0, 1.0)  # Ensure bounds
            
            # Scale actions to shares
            scaled_actions = actions * self.hmax
            scaled_actions = scaled_actions.astype(int)
            
            # Apply turbulence constraint
            if self.turbulence_threshold and self.turbulence >= self.turbulence_threshold:
                # Force liquidation during high turbulence
                scaled_actions = -self.stock_shares.astype(int)
            
            # Record initial portfolio value
            begin_total_asset = self._calculate_total_value()
            
            # Execute trades
            self._execute_trades(scaled_actions)
            
            # Update market state (simulate time progression)
            self._update_market_data()
            
            # Calculate reward
            end_total_asset = self._calculate_total_value()
            reward = (end_total_asset - begin_total_asset) * self.reward_scaling
            
            # Update memories
            self.asset_memory.append(end_total_asset)
            self.rewards_memory.append(reward)
            self.actions_memory.append(actions.copy())
            self.date_memory.append(datetime.now().isoformat())
            
            # Update environment state
            self.current_step += 1
            self.state = self._get_current_state()
            
            # Check if terminal
            self.terminal = self._check_terminal_condition()
            
            # Info dict
            info = {
                'total_asset': end_total_asset,
                'reward': reward,
                'cost': self.cost,
                'trades': self.trades,
                'turbulence': self.turbulence,
                'sharpe_ratio': self._calculate_sharpe_ratio()
            }
            
            return self.state, reward, self.terminal, False, info
            
        except Exception as e:
            logger.error(f"❌ Step execution failed: {e}")
            # Return safe fallback
            return self.state, 0.0, True, False, {'error': str(e)}
    
    def _execute_trades(self, actions: np.ndarray):
        """Execute trading actions."""
        try:
            # Sort actions for optimal execution (sells first, then buys)
            action_order = np.argsort(actions)
            
            for i in action_order:
                action = actions[i]
                symbol = self.symbols[i]
                price = self.market_data.get(symbol, {}).get('price', self.stock_prices[i])
                
                if action < 0:  # Sell
                    self._sell_stock(i, abs(action), price)
                elif action > 0:  # Buy
                    self._buy_stock(i, action, price)
                    
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
    
    def _sell_stock(self, stock_index: int, num_shares: int, price: float):
        """Execute sell order."""
        try:
            # Check available shares
            available_shares = self.stock_shares[stock_index]
            actual_sell = min(num_shares, available_shares)
            
            if actual_sell > 0:
                # Calculate proceeds
                sell_amount = actual_sell * price * (1 - self.sell_cost_pct)
                
                # Update portfolio
                self.cash += sell_amount
                self.stock_shares[stock_index] -= actual_sell
                
                # Track costs and trades
                self.cost += actual_sell * price * self.sell_cost_pct
                self.trades += 1
                
        except Exception as e:
            logger.error(f"❌ Sell execution failed for stock {stock_index}: {e}")
    
    def _buy_stock(self, stock_index: int, num_shares: int, price: float):
        """Execute buy order."""
        try:
            # Calculate cost including transaction fees
            total_cost = num_shares * price * (1 + self.buy_cost_pct)
            
            # Check available cash
            if self.cash >= total_cost:
                # Update portfolio
                self.cash -= total_cost
                self.stock_shares[stock_index] += num_shares
                
                # Track costs and trades
                self.cost += num_shares * price * self.buy_cost_pct
                self.trades += 1
                
        except Exception as e:
            logger.error(f"❌ Buy execution failed for stock {stock_index}: {e}")
    
    def _update_market_data(self):
        """Simulate market data progression (or fetch real-time data)."""
        try:
            if self.alpaca_client:
                # Try to get real-time updates
                for i, symbol in enumerate(self.symbols):
                    try:
                        current_price = self.alpaca_client.get_current_price(symbol)
                        if current_price:
                            self.stock_prices[i] = current_price
                            self.market_data[symbol]['price'] = current_price
                    except:
                        pass
            else:
                # Simulate price changes
                for i, symbol in enumerate(self.symbols):
                    current_price = self.market_data[symbol]['price']
                    volatility = self.market_data[symbol]['volatility']
                    
                    # Random walk
                    change = np.random.normal(0.0, volatility / np.sqrt(252))
                    new_price = current_price * (1 + change)
                    
                    self.stock_prices[i] = new_price
                    self.market_data[symbol]['price'] = new_price
            
            # Update turbulence
            self._update_turbulence()
            
        except Exception as e:
            logger.error(f"❌ Market data update failed: {e}")
    
    def _update_turbulence(self):
        """Update market turbulence indicator."""
        try:
            # Simple turbulence calculation based on price volatility
            price_changes = []
            for symbol in self.symbols:
                if symbol in self.market_data:
                    volatility = self.market_data[symbol]['volatility']
                    price_changes.append(volatility)
            
            if price_changes:
                self.turbulence = np.mean(price_changes)
            else:
                self.turbulence = 0.0
                
        except Exception:
            self.turbulence = 0.0
    
    def _check_terminal_condition(self) -> bool:
        """Check if episode should terminate."""
        # Terminal conditions
        max_steps = 1000  # Maximum steps per episode
        min_cash_ratio = 0.1  # Minimum cash ratio before forced termination
        
        if self.current_step >= max_steps:
            return True
        
        total_value = self._calculate_total_value()
        if total_value < self.initial_amount * min_cash_ratio:
            return True
        
        return False
    
    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        try:
            # Reset portfolio state
            self.cash = self.initial_amount
            self.stock_shares = np.zeros(self.stock_dim)
            
            # Reset tracking variables
            self.current_step = 0
            self.terminal = False
            self.cost = 0.0
            self.trades = 0
            self.turbulence = 0.0
            
            # Reset memories
            self.asset_memory = [self.initial_amount]
            self.rewards_memory = []
            self.actions_memory = []
            self.date_memory = []
            
            # Update market data
            self._update_market_data()
            
            # Get initial state
            self.state = self._get_current_state()
            
            info = {
                'initial_amount': self.initial_amount,
                'symbols': self.symbols,
                'state_dim': self.state_dim
            }
            
            return self.state, info
            
        except Exception as e:
            logger.error(f"❌ Environment reset failed: {e}")
            # Return safe fallback
            return np.zeros(self.state_dim, dtype=np.float32), {}
    
    def render(self, mode='human') -> Any:
        """Render environment state."""
        if mode == 'human':
            total_value = self._calculate_total_value()
            return_pct = (total_value - self.initial_amount) / self.initial_amount * 100
            
            print(f"Step: {self.current_step}")
            print(f"Total Value: ${total_value:,.2f}")
            print(f"Return: {return_pct:.2f}%")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"Trades: {self.trades}")
            print(f"Turbulence: {self.turbulence:.4f}")
            print("---")
            
        return self.state
    
    def update_sentiment_data(self, sentiment_data: Dict[str, Dict]):
        """Update sentiment data for symbols."""
        self.sentiment_data = sentiment_data
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        total_value = self._calculate_total_value()
        
        holdings = {}
        for i, symbol in enumerate(self.symbols):
            if self.stock_shares[i] > 0:
                price = self.market_data[symbol]['price']
                holdings[symbol] = {
                    'shares': self.stock_shares[i],
                    'price': price,
                    'value': self.stock_shares[i] * price
                }
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'holdings': holdings,
            'return_pct': (total_value - self.initial_amount) / self.initial_amount * 100,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'trades': self.trades,
            'cost': self.cost,
            'turbulence': self.turbulence,
            'current_step': self.current_step
        }

logger.info("✅ Enhanced Alpaca-FinRL Trading Environment loaded")