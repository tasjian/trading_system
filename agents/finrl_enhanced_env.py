"""
Enhanced FinRL-based Trading Environment with Short Selling and Limit Orders
Integrates FinRL architecture with ML4T trading system for advanced RL trading.
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio

from agents.llm_rl_integration import LLMStateEnricher, LLMMarketState
from tools.alpaca_client import alpaca_client

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Enhanced order types supporting advanced trading strategies."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side including short selling support."""
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"    # New: explicit short selling
    COVER = "cover"    # New: cover short position


@dataclass
class Position:
    """Enhanced position tracking with short selling support."""
    symbol: str
    quantity: float  # Positive = long, Negative = short
    avg_price: float
    market_value: float
    unrealized_pnl: float
    side: str  # 'long', 'short', or 'flat'
    borrowed_quantity: float = 0.0  # For short positions
    margin_requirement: float = 0.0
    borrowing_cost: float = 0.0


@dataclass
class PendingOrder:
    """Pending limit order structure."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    time_in_force: str = "GTC"  # Good Till Cancelled


@dataclass
class PortfolioState:
    """Enhanced portfolio state with margin and short selling support."""
    cash: float
    equity: float
    buying_power: float
    maintenance_margin: float
    positions: Dict[str, Position] = field(default_factory=dict)
    pending_orders: List[PendingOrder] = field(default_factory=list)
    day_trades_count: int = 0
    short_interest: float = 0.0  # Total borrowing costs


class EnhancedFinRLTradingEnv(gym.Env):
    """
    Enhanced FinRL-based trading environment with short selling and limit orders.
    
    Features:
    - Continuous action space: [-1, 1] per symbol (negative = short, positive = long)
    - Short selling with margin requirements and borrowing costs
    - Limit orders with execution logic
    - Risk management and margin calls
    - LLM state enrichment integration
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self,
                 symbols: List[str],
                 initial_balance: float = 100000,
                 commission_rate: float = 0.001,
                 margin_requirement: float = 0.5,  # 50% margin requirement
                 short_borrow_rate: float = 0.03,  # 3% annual borrow rate
                 enable_short_selling: bool = True,
                 enable_limit_orders: bool = True,
                 max_position_size: float = 0.2,  # Max 20% of portfolio per position
                 risk_free_rate: float = 0.02,  # 2% annual risk-free rate
                 lookback_window: int = 252,  # 1 year of trading days
                 **kwargs):
        
        super().__init__()
        
        # Core configuration
        self.symbols = symbols
        self.symbol_count = len(symbols)
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.margin_requirement = margin_requirement
        self.short_borrow_rate = short_borrow_rate / 252  # Daily rate
        self.enable_short_selling = enable_short_selling
        self.enable_limit_orders = enable_limit_orders
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        self.lookback_window = lookback_window
        
        # Market data
        self.price_data: Optional[pd.DataFrame] = None
        self.current_step = 0
        self.max_steps = 0
        
        # Portfolio state
        self.portfolio = PortfolioState(
            cash=initial_balance,
            equity=initial_balance,
            buying_power=initial_balance * 2,  # 2:1 margin
            maintenance_margin=0.0
        )
        
        # Action space: continuous [-1, 1] for each symbol
        # -1 = max short, 0 = hold, 1 = max long
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.symbol_count,), 
            dtype=np.float32
        )
        
        # State space will be defined after data is loaded
        self.observation_space = None
        
        # LLM integration
        self.llm_enricher = LLMStateEnricher()
        self.llm_state_history = []
        
        # Performance tracking
        self.portfolio_value_history = []
        self.trade_history = []
        self.reward_history = []
        
        # Risk management
        self.margin_call_threshold = 0.25  # 25% maintenance margin
        self.max_leverage = 2.0
        
        logger.info(f"✅ Enhanced FinRL Trading Environment initialized")
        logger.info(f"   Symbols: {len(symbols)}")
        logger.info(f"   Short selling: {'enabled' if enable_short_selling else 'disabled'}")
        logger.info(f"   Limit orders: {'enabled' if enable_limit_orders else 'disabled'}")
    
    def set_data(self, data: pd.DataFrame):
        """Set market data for the environment."""
        self.price_data = data.copy()
        self.max_steps = len(data) - self.lookback_window - 1
        
        # Define observation space based on data
        # State: [cash, equity, positions, prices, technical_indicators, llm_features]
        state_dim = (
            2 +  # cash, equity
            self.symbol_count * 3 +  # position_size, avg_price, unrealized_pnl per symbol
            self.symbol_count * 5 +  # OHLCV per symbol
            self.symbol_count * 4 +  # Technical indicators per symbol
            7  # LLM features
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,), dtype=np.float32
        )
        
        # Validate data has required columns
        required_columns = []
        for symbol in self.symbols:
            required_columns.extend([
                f"{symbol}_open", f"{symbol}_high", f"{symbol}_low", 
                f"{symbol}_close", f"{symbol}_volume"
            ])
        
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            logger.warning(f"Missing columns in data: {missing_columns}")
        
        logger.info(f"✅ Market data loaded: {len(data)} rows, {len(data.columns)} columns")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        if self.price_data is None:
            raise ValueError("No market data loaded. Call set_data() first.")
        
        # Reset portfolio
        self.portfolio = PortfolioState(
            cash=self.initial_balance,
            equity=self.initial_balance,
            buying_power=self.initial_balance * 2,
            maintenance_margin=0.0
        )
        
        # Reset tracking - start from lookback_window but ensure it's within bounds
        self.current_step = min(self.lookback_window, len(self.price_data) - 1)
        self.portfolio_value_history = [self.initial_balance]
        self.trade_history = []
        self.reward_history = []
        self.llm_state_history = []
        
        # Generate initial state
        state = self._get_observation()
        
        logger.info(f"🔄 Environment reset - Step: {self.current_step}/{len(self.price_data)}")
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one environment step."""
        if self.price_data is None:
            raise ValueError("No market data loaded. Call set_data() first.")
        
        # Process limit orders first
        if self.enable_limit_orders:
            self._process_pending_orders()
        
        # Execute market orders based on action
        trade_info = self._execute_actions(action)
        
        # Update portfolio with market movements
        self._update_portfolio_values()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.reward_history.append(reward)
        
        # Check for margin calls and risk management
        margin_call = self._check_margin_requirements()
        if margin_call:
            reward -= 1.0  # Penalty for margin call
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Get new observation
        obs = self._get_observation()
        
        # Info dict
        info = {
            'portfolio_value': self.portfolio.equity,
            'cash': self.portfolio.cash,
            'positions': len([p for p in self.portfolio.positions.values() if abs(p.quantity) > 0]),
            'trades_executed': trade_info['trades_executed'],
            'margin_call': margin_call,
            'step': self.current_step,
            'done': done
        }
        
        return obs, reward, done, info
    
    def _execute_actions(self, actions: np.ndarray) -> Dict:
        """Execute trading actions based on agent decisions."""
        trades_executed = 0
        total_commission = 0.0
        
        current_prices = self._get_current_prices()
        
        for i, action in enumerate(actions):
            symbol = self.symbols[i]
            current_price = current_prices[symbol]
            
            if abs(action) < 0.01:  # Ignore very small actions
                continue
            
            # Calculate target position size as percentage of portfolio
            target_position_value = action * self.portfolio.equity * self.max_position_size
            target_shares = target_position_value / current_price
            
            # Get current position
            current_position = self.portfolio.positions.get(symbol, Position(
                symbol=symbol, quantity=0, avg_price=0, market_value=0, 
                unrealized_pnl=0, side='flat'
            ))
            
            # Calculate required trade
            shares_to_trade = target_shares - current_position.quantity
            
            if abs(shares_to_trade) < 1:  # Minimum 1 share trade
                continue
            
            # Execute trade
            if self._can_execute_trade(symbol, shares_to_trade, current_price):
                self._execute_trade(symbol, shares_to_trade, current_price, OrderType.MARKET)
                trades_executed += 1
                total_commission += abs(shares_to_trade) * current_price * self.commission_rate
        
        return {
            'trades_executed': trades_executed,
            'total_commission': total_commission
        }
    
    def _execute_trade(self, symbol: str, quantity: float, price: float, order_type: OrderType):
        """Execute a single trade with position management."""
        
        # Get or create position
        if symbol not in self.portfolio.positions:
            self.portfolio.positions[symbol] = Position(
                symbol=symbol, quantity=0, avg_price=0, market_value=0,
                unrealized_pnl=0, side='flat'
            )
        
        position = self.portfolio.positions[symbol]
        
        # Calculate trade value and commission
        trade_value = abs(quantity) * price
        commission = trade_value * self.commission_rate
        
        # Handle different trade scenarios
        if quantity > 0:  # Buying (or covering short)
            if position.quantity < 0:  # Covering short position
                self._cover_short_position(position, quantity, price)
            else:  # Adding to long or opening long
                self._add_long_position(position, quantity, price)
        else:  # Selling (or opening short)
            if position.quantity > 0:  # Reducing long position
                self._reduce_long_position(position, abs(quantity), price)
            elif self.enable_short_selling:  # Opening or adding to short
                self._add_short_position(position, abs(quantity), price)
        
        # Update cash and apply commission
        cash_impact = -quantity * price - commission
        self.portfolio.cash += cash_impact
        
        # Log trade
        self.trade_history.append({
            'timestamp': self.current_step,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'side': 'buy' if quantity > 0 else 'sell',
            'cash_impact': cash_impact
        })
        
        logger.debug(f"Trade executed: {symbol} {quantity:+.2f} @ ${price:.2f}")
    
    def _add_long_position(self, position: Position, quantity: float, price: float):
        """Add to long position."""
        if position.quantity == 0:
            position.avg_price = price
        else:
            # Calculate new average price
            total_cost = position.quantity * position.avg_price + quantity * price
            position.avg_price = total_cost / (position.quantity + quantity)
        
        position.quantity += quantity
        position.side = 'long'
    
    def _add_short_position(self, position: Position, quantity: float, price: float):
        """Add to short position."""
        if position.quantity == 0:
            position.avg_price = price
        else:
            # Calculate new average price for short
            total_proceeds = abs(position.quantity) * position.avg_price + quantity * price
            position.avg_price = total_proceeds / (abs(position.quantity) + quantity)
        
        position.quantity -= quantity  # Negative for short
        position.borrowed_quantity += quantity
        position.side = 'short'
        
        # Calculate margin requirement
        short_value = quantity * price
        position.margin_requirement = short_value * self.margin_requirement
        self.portfolio.maintenance_margin += position.margin_requirement
    
    def _reduce_long_position(self, position: Position, quantity: float, price: float):
        """Reduce long position."""
        if quantity >= position.quantity:
            # Closing entire position
            position.quantity = 0
            position.avg_price = 0
            position.side = 'flat'
        else:
            # Partial close
            position.quantity -= quantity
    
    def _cover_short_position(self, position: Position, quantity: float, price: float):
        """Cover short position."""
        quantity_to_cover = min(quantity, position.borrowed_quantity)
        
        # Reduce short position
        position.quantity += quantity_to_cover
        position.borrowed_quantity -= quantity_to_cover
        
        # Reduce margin requirement
        covered_value = quantity_to_cover * price
        margin_reduction = covered_value * self.margin_requirement
        position.margin_requirement -= margin_reduction
        self.portfolio.maintenance_margin -= margin_reduction
        
        if position.borrowed_quantity <= 0:
            position.side = 'long' if position.quantity > 0 else 'flat'
            position.borrowed_quantity = 0
            position.margin_requirement = 0
    
    def _can_execute_trade(self, symbol: str, quantity: float, price: float) -> bool:
        """Check if trade can be executed based on available capital and risk limits."""
        trade_value = abs(quantity) * price
        commission = trade_value * self.commission_rate
        
        if quantity > 0:  # Buying
            required_cash = trade_value + commission
            return self.portfolio.cash >= required_cash
        else:  # Selling or shorting
            if not self.enable_short_selling and symbol not in self.portfolio.positions:
                return False
            
            current_position = self.portfolio.positions.get(symbol)
            if current_position and abs(quantity) > current_position.quantity and quantity < 0:
                # Would create or increase short position
                if not self.enable_short_selling:
                    return False
                
                # Check margin requirements for short selling
                short_value = abs(quantity) * price
                required_margin = short_value * self.margin_requirement
                return self.portfolio.buying_power >= required_margin
            
            return True
    
    def _process_pending_orders(self):
        """Process pending limit orders."""
        if not self.enable_limit_orders:
            return
        
        current_prices = self._get_current_prices()
        executed_orders = []
        
        for order in self.portfolio.pending_orders:
            if order.symbol not in current_prices:
                continue
            
            current_price = current_prices[order.symbol]
            should_execute = False
            
            if order.order_type == OrderType.LIMIT:
                if order.side in [OrderSide.BUY, OrderSide.COVER]:
                    should_execute = current_price <= order.limit_price
                else:  # SELL or SHORT
                    should_execute = current_price >= order.limit_price
            
            if should_execute:
                quantity = order.quantity if order.side in [OrderSide.BUY, OrderSide.COVER] else -order.quantity
                if order.side == OrderSide.SHORT:
                    quantity = -abs(quantity)
                
                if self._can_execute_trade(order.symbol, quantity, current_price):
                    self._execute_trade(order.symbol, quantity, current_price, order.order_type)
                    executed_orders.append(order)
        
        # Remove executed orders
        for order in executed_orders:
            self.portfolio.pending_orders.remove(order)
    
    def _update_portfolio_values(self):
        """Update portfolio values based on current market prices."""
        current_prices = self._get_current_prices()
        total_market_value = 0.0
        total_short_interest = 0.0
        
        for symbol, position in self.portfolio.positions.items():
            if abs(position.quantity) > 0 and symbol in current_prices:
                current_price = current_prices[symbol]
                
                # Update market value
                position.market_value = position.quantity * current_price
                
                # Calculate unrealized P&L
                if position.side == 'long':
                    position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                elif position.side == 'short':
                    position.unrealized_pnl = (position.avg_price - current_price) * abs(position.quantity)
                    # Add borrowing cost for short positions
                    daily_borrow_cost = abs(position.quantity) * current_price * self.short_borrow_rate
                    position.borrowing_cost += daily_borrow_cost
                    total_short_interest += daily_borrow_cost
                
                total_market_value += position.market_value
        
        # Update portfolio equity and short interest
        self.portfolio.equity = self.portfolio.cash + total_market_value
        self.portfolio.short_interest = total_short_interest
        
        # Update buying power
        self.portfolio.buying_power = (
            self.portfolio.cash + 
            sum(pos.market_value for pos in self.portfolio.positions.values() if pos.side == 'long') * 0.5 -
            self.portfolio.maintenance_margin
        )
        
        # Track portfolio value
        self.portfolio_value_history.append(self.portfolio.equity)
    
    def _check_margin_requirements(self) -> bool:
        """Check if portfolio meets margin requirements."""
        if self.portfolio.maintenance_margin == 0:
            return False
        
        # Calculate maintenance margin requirement
        required_margin = self.portfolio.maintenance_margin
        available_equity = self.portfolio.equity
        
        if available_equity < required_margin:
            logger.warning(f"Margin call: Required ${required_margin:.2f}, Available ${available_equity:.2f}")
            # Force close some short positions to meet margin
            self._handle_margin_call()
            return True
        
        return False
    
    def _handle_margin_call(self):
        """Handle margin call by closing short positions."""
        current_prices = self._get_current_prices()
        
        # Sort short positions by loss (worst first)
        short_positions = [(symbol, pos) for symbol, pos in self.portfolio.positions.items() 
                          if pos.side == 'short' and abs(pos.quantity) > 0]
        
        short_positions.sort(key=lambda x: x[1].unrealized_pnl)
        
        for symbol, position in short_positions:
            if self.portfolio.equity >= self.portfolio.maintenance_margin:
                break
            
            if symbol in current_prices:
                # Cover entire short position
                current_price = current_prices[symbol]
                self._execute_trade(symbol, abs(position.quantity), current_price, OrderType.MARKET)
                logger.warning(f"Forced cover of short position: {symbol}")
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current market prices for all symbols."""
        if self.price_data is None or self.current_step >= len(self.price_data):
            return {}
        
        current_row = self.price_data.iloc[self.current_step]
        prices = {}
        
        for symbol in self.symbols:
            close_col = f"{symbol}_close"
            if close_col in current_row:
                prices[symbol] = current_row[close_col]
        
        return prices
    
    def _get_observation(self) -> np.ndarray:
        """Generate current observation state."""
        if self.price_data is None or self.current_step >= len(self.price_data):
            return np.zeros(self.observation_space.shape)
        
        # Portfolio state
        portfolio_state = [
            self.portfolio.cash / self.initial_balance,  # Normalized cash
            self.portfolio.equity / self.initial_balance,  # Normalized equity
        ]
        
        # Position states
        position_state = []
        for symbol in self.symbols:
            if symbol in self.portfolio.positions:
                pos = self.portfolio.positions[symbol]
                position_state.extend([
                    pos.quantity / 1000,  # Normalized position size
                    pos.avg_price / 100,  # Normalized avg price
                    pos.unrealized_pnl / self.initial_balance  # Normalized P&L
                ])
            else:
                position_state.extend([0.0, 0.0, 0.0])
        
        # Market data features
        market_features = []
        current_row = self.price_data.iloc[self.current_step]
        
        for symbol in self.symbols:
            # OHLCV data
            ohlcv = []
            for col_suffix in ['_open', '_high', '_low', '_close', '_volume']:
                col_name = f"{symbol}{col_suffix}"
                value = current_row.get(col_name, 0.0)
                if col_suffix == '_volume':
                    value = value / 1e6  # Normalize volume
                else:
                    value = value / 100  # Normalize prices
                ohlcv.append(value)
            
            market_features.extend(ohlcv)
            
            # Technical indicators (if available)
            tech_indicators = []
            for indicator in ['_rsi', '_macd', '_bb_upper', '_bb_lower']:
                col_name = f"{symbol}{indicator}"
                value = current_row.get(col_name, 0.0)
                if indicator == '_rsi':
                    value = value / 100  # Normalize RSI
                else:
                    value = value / 100  # Normalize other indicators
                tech_indicators.append(value)
            
            market_features.extend(tech_indicators)
        
        # LLM features (simplified - can be enhanced with actual LLM state)
        llm_features = [
            0.0,  # Overall sentiment
            0.0,  # Trend strength
            0.0,  # Volatility forecast
            0.0,  # Risk level
            0.0,  # News impact
            0.0,  # Consensus strength
            0.0   # Technical alignment
        ]
        
        # Combine all features
        observation = np.concatenate([
            portfolio_state,
            position_state,
            market_features,
            llm_features
        ]).astype(np.float32)
        
        # Ensure observation matches expected shape
        if len(observation) != self.observation_space.shape[0]:
            logger.warning(f"Observation shape mismatch: {len(observation)} vs {self.observation_space.shape[0]}")
            # Pad or truncate to match expected shape
            if len(observation) < self.observation_space.shape[0]:
                observation = np.pad(observation, (0, self.observation_space.shape[0] - len(observation)))
            else:
                observation = observation[:self.observation_space.shape[0]]
        
        return observation
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on portfolio performance and risk metrics."""
        if len(self.portfolio_value_history) < 2:
            return 0.0
        
        # Portfolio return
        current_value = self.portfolio_value_history[-1]
        previous_value = self.portfolio_value_history[-2]
        portfolio_return = (current_value - previous_value) / previous_value
        
        # Risk-adjusted return (Sharpe-like)
        if len(self.portfolio_value_history) >= 10:
            recent_returns = np.diff(self.portfolio_value_history[-10:]) / self.portfolio_value_history[-11:-1]
            volatility = np.std(recent_returns) + 1e-8
            risk_adjusted_return = (portfolio_return - self.risk_free_rate) / volatility
        else:
            risk_adjusted_return = portfolio_return
        
        # Base reward
        reward = risk_adjusted_return
        
        # Penalties and bonuses
        
        # 1. Transaction cost penalty
        if self.trade_history:
            recent_trades = [t for t in self.trade_history if t['timestamp'] == self.current_step]
            total_commission = sum(t['commission'] for t in recent_trades)
            reward -= total_commission / self.initial_balance
        
        # 2. Short selling interest cost
        reward -= self.portfolio.short_interest / self.initial_balance
        
        # 3. Diversification bonus (reward for not over-concentrating)
        active_positions = [p for p in self.portfolio.positions.values() if abs(p.quantity) > 0]
        if len(active_positions) > 1:
            position_values = [abs(p.market_value) for p in active_positions]
            total_position_value = sum(position_values)
            if total_position_value > 0:
                concentration = max(position_values) / total_position_value
                diversification_bonus = (1.0 - concentration) * 0.01
                reward += diversification_bonus
        
        # 4. Leverage penalty (discourage excessive leverage)
        leverage = abs(sum(p.market_value for p in self.portfolio.positions.values())) / max(self.portfolio.equity, 1000)
        if leverage > 1.5:
            leverage_penalty = (leverage - 1.5) * 0.05
            reward -= leverage_penalty
        
        # 5. Cash drag penalty (encourage capital utilization)
        cash_ratio = self.portfolio.cash / self.portfolio.equity
        if cash_ratio > 0.2:  # More than 20% cash
            cash_drag = (cash_ratio - 0.2) * 0.02
            reward -= cash_drag
        
        return float(reward)
    
    def place_limit_order(self, symbol: str, side: OrderSide, quantity: float, limit_price: float) -> bool:
        """Place a limit order (if enabled)."""
        if not self.enable_limit_orders:
            logger.warning("Limit orders not enabled")
            return False
        
        order = PendingOrder(
            order_id=f"{symbol}_{side.value}_{len(self.portfolio.pending_orders)}",
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=limit_price
        )
        
        self.portfolio.pending_orders.append(order)
        logger.info(f"Limit order placed: {symbol} {side.value} {quantity} @ ${limit_price}")
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        for i, order in enumerate(self.portfolio.pending_orders):
            if order.order_id == order_id:
                del self.portfolio.pending_orders[i]
                logger.info(f"Order cancelled: {order_id}")
                return True
        return False
    
    def get_portfolio_summary(self) -> Dict:
        """Get detailed portfolio summary."""
        positions_summary = {}
        for symbol, pos in self.portfolio.positions.items():
            if abs(pos.quantity) > 0:
                positions_summary[symbol] = {
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'side': pos.side,
                    'borrowed_quantity': pos.borrowed_quantity
                }
        
        return {
            'cash': self.portfolio.cash,
            'equity': self.portfolio.equity,
            'buying_power': self.portfolio.buying_power,
            'maintenance_margin': self.portfolio.maintenance_margin,
            'short_interest': self.portfolio.short_interest,
            'positions': positions_summary,
            'pending_orders': len(self.portfolio.pending_orders),
            'portfolio_return': (self.portfolio.equity - self.initial_balance) / self.initial_balance,
            'total_trades': len(self.trade_history)
        }
    
    def render(self, mode='human'):
        """Render the environment state."""
        if mode == 'human':
            summary = self.get_portfolio_summary()
            print(f"\n=== Portfolio Summary (Step {self.current_step}) ===")
            print(f"Equity: ${summary['equity']:,.2f}")
            print(f"Cash: ${summary['cash']:,.2f}")
            print(f"Return: {summary['portfolio_return']:.2%}")
            print(f"Active Positions: {len(summary['positions'])}")
            if summary['positions']:
                for symbol, pos in summary['positions'].items():
                    print(f"  {symbol}: {pos['quantity']:+.0f} shares @ ${pos['avg_price']:.2f} "
                          f"({pos['side']}, P&L: ${pos['unrealized_pnl']:+.2f})")
    
    def seed(self, seed=None):
        """Set random seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'llm_enricher') and self.llm_enricher:
            asyncio.create_task(self.llm_enricher.close())
        logger.info("✅ Enhanced FinRL Trading Environment closed")


# Utility functions
def create_enhanced_finrl_env(symbols: List[str], **kwargs) -> EnhancedFinRLTradingEnv:
    """Create enhanced FinRL trading environment with optimal settings."""
    # Set default values, but allow kwargs to override
    defaults = {
        'enable_short_selling': True,
        'enable_limit_orders': True,
        'commission_rate': 0.001,  # 0.1%
        'margin_requirement': 0.5,  # 50%
        'short_borrow_rate': 0.03,  # 3% annual
        'max_position_size': 0.15,  # 15% max per position
    }
    
    # Merge defaults with kwargs, giving precedence to kwargs
    final_kwargs = {**defaults, **kwargs}
    
    return EnhancedFinRLTradingEnv(
        symbols=symbols,
        **final_kwargs
    )


if __name__ == "__main__":
    # Test the enhanced environment
    import yfinance as yf
    
    def test_enhanced_environment():
        # Create test environment
        symbols = ["AAPL", "MSFT", "GOOGL"]
        env = create_enhanced_finrl_env(symbols)
        
        # Generate test data
        data_dict = {}
        for symbol in symbols:
            stock_data = yf.download(symbol, period="1mo", interval="1d")
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                data_dict[f"{symbol}_{col.lower()}"] = stock_data[col].values
        
        # Create DataFrame
        test_data = pd.DataFrame(data_dict)
        test_data.fillna(method='ffill', inplace=True)
        
        # Set data and test
        env.set_data(test_data)
        
        print("Testing Enhanced FinRL Environment")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Test episode
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        for step in range(5):
            # Random action: mix of long, short, and hold positions
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            print(f"\nStep {step + 1}:")
            print(f"  Action: {action}")
            print(f"  Reward: {reward:.6f}")
            print(f"  Portfolio Value: ${info['portfolio_value']:,.2f}")
            print(f"  Trades Executed: {info['trades_executed']}")
            print(f"  Positions: {info['positions']}")
            
            if done:
                break
        
        # Test limit orders
        print("\nTesting limit orders:")
        current_prices = env._get_current_prices()
        if "AAPL" in current_prices:
            limit_price = current_prices["AAPL"] * 0.98  # 2% below current
            success = env.place_limit_order("AAPL", OrderSide.BUY, 10, limit_price)
            print(f"Limit order placed: {success}")
        
        # Portfolio summary
        summary = env.get_portfolio_summary()
        print(f"\nFinal Portfolio Summary:")
        print(f"  Total Return: {summary['portfolio_return']:.2%}")
        print(f"  Cash: ${summary['cash']:,.2f}")
        print(f"  Total Trades: {summary['total_trades']}")
        
        env.close()
        print("✅ Enhanced FinRL Environment test completed")
    
    test_enhanced_environment()