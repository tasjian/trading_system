"""
Realistic Trading Environment with Latency, Slippage, and Order Book Effects
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from enum import Enum
import random
from collections import deque

# Handle optional dependencies
try:
    import torch
    import gym
    from gym import spaces
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from agents.hybrid_data_sources import (
    HybridDataGenerator, MarketRegimeData, SentimentData, L2OrderBookData
)

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"

class OrderStatus(Enum):
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Trading order with realistic properties."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # None for market orders
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    
    # Timestamps
    created_at: datetime = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # Metadata
    latency_ms: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0

@dataclass
class Fill:
    """Individual fill from an order."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float

@dataclass
class MarketImpact:
    """Market impact characteristics."""
    permanent_impact: float  # Long-term price impact
    temporary_impact: float  # Short-term price impact
    impact_decay_rate: float  # How quickly temporary impact decays

class LatencySimulator:
    """Simulates realistic trading latencies."""
    
    def __init__(self):
        # Latency distributions (in milliseconds)
        self.latency_profiles = {
            'market_data': {'mean': 2.0, 'std': 1.0, 'min': 0.1, 'max': 10.0},
            'order_submission': {'mean': 5.0, 'std': 3.0, 'min': 1.0, 'max': 50.0},
            'order_ack': {'mean': 10.0, 'std': 5.0, 'min': 2.0, 'max': 100.0},
            'fill_notification': {'mean': 15.0, 'std': 8.0, 'min': 3.0, 'max': 200.0}
        }
        
        # Network congestion factor (affects all latencies)
        self.congestion_factor = 1.0
        
    def get_latency(self, operation: str) -> float:
        """Get realistic latency for an operation."""
        if operation not in self.latency_profiles:
            return 5.0  # Default latency
        
        profile = self.latency_profiles[operation]
        
        # Generate latency with log-normal distribution (more realistic)
        base_latency = np.random.lognormal(
            mean=np.log(profile['mean']),
            sigma=profile['std'] / profile['mean']
        )
        
        # Apply congestion and clip to bounds
        latency = base_latency * self.congestion_factor
        return np.clip(latency, profile['min'], profile['max'])
    
    def set_congestion_factor(self, factor: float):
        """Set network congestion factor (1.0 = normal, >1.0 = congested)."""
        self.congestion_factor = max(0.1, factor)

class SlippageModel:
    """Models realistic slippage based on market conditions."""
    
    def __init__(self):
        self.base_slippage = 0.0005  # 5 basis points base slippage
        
    def calculate_slippage(self,
                         order: Order,
                         orderbook: L2OrderBookData,
                         regime: MarketRegimeData,
                         market_impact: float = 0.0) -> float:
        """Calculate realistic slippage for an order."""
        
        # Base slippage increases with volatility
        vol_factor = 1.0 + (regime.volatility / 0.02) * 0.5
        
        # Market impact from order size vs available liquidity
        if order.side in [OrderSide.BUY]:
            available_liquidity = sum(orderbook.ask_sizes[:5])  # Top 5 levels
        else:
            available_liquidity = sum(orderbook.bid_sizes[:5])
        
        size_impact = min(2.0, order.quantity / max(100, available_liquidity))
        
        # Spread impact - wider spreads = more slippage
        spread_impact = orderbook.spread / (orderbook.ask_prices[0] - orderbook.bid_prices[0]) if len(orderbook.ask_prices) > 0 else 1.0
        
        # Order type impact
        if order.order_type == OrderType.MARKET:
            urgency_factor = 1.5  # Market orders have higher slippage
        else:
            urgency_factor = 0.5  # Limit orders have less slippage
        
        # Regime-based impact
        regime_factor = {
            'bull': 0.8,
            'bear': 1.2,
            'sideways': 1.0,
            'volatile': 1.8,
            'crisis': 3.0
        }.get(regime.regime_type, 1.0)
        
        # Calculate total slippage
        total_slippage = (
            self.base_slippage * 
            vol_factor * 
            (1 + size_impact) * 
            spread_impact * 
            urgency_factor * 
            regime_factor *
            (1 + market_impact)
        )
        
        # Add some randomness
        noise_factor = np.random.uniform(0.5, 1.5)
        
        return total_slippage * noise_factor

class MarketImpactModel:
    """Models market impact of trades."""
    
    def __init__(self):
        self.impact_decay_rate = 0.95  # How quickly temporary impact decays
        self.permanent_impact_factor = 0.3  # Fraction that becomes permanent
        
    def calculate_market_impact(self,
                              order: Order,
                              orderbook: L2OrderBookData,
                              daily_volume: float) -> MarketImpact:
        """Calculate market impact of a trade."""
        
        # Participation rate - what fraction of daily volume is this order
        participation_rate = order.quantity / max(1000, daily_volume)
        
        # Square root law for temporary impact
        temporary_impact = 0.1 * np.sqrt(participation_rate) * orderbook.spread
        
        # Linear component for large orders
        if participation_rate > 0.05:  # More than 5% of daily volume
            temporary_impact += 0.5 * (participation_rate - 0.05) * orderbook.spread
        
        # Permanent impact is a fraction of temporary
        permanent_impact = temporary_impact * self.permanent_impact_factor
        
        return MarketImpact(
            permanent_impact=permanent_impact,
            temporary_impact=temporary_impact,
            impact_decay_rate=self.impact_decay_rate
        )

class OrderExecutionEngine:
    """Realistic order execution with partial fills, rejections, etc."""
    
    def __init__(self):
        self.latency_simulator = LatencySimulator()
        self.slippage_model = SlippageModel()
        self.impact_model = MarketImpactModel()
        
        # Order tracking
        self.pending_orders: Dict[str, Order] = {}
        self.execution_queue = deque()
        self.fills: List[Fill] = []
        
        # Market state
        self.current_prices: Dict[str, float] = {}
        self.price_impacts: Dict[str, float] = {}  # Temporary price impacts
        
    async def submit_order(self,
                         order: Order,
                         orderbook: L2OrderBookData,
                         regime: MarketRegimeData) -> bool:
        """Submit an order for execution."""
        
        # Add submission latency
        submission_latency = self.latency_simulator.get_latency('order_submission')
        await asyncio.sleep(submission_latency / 1000.0)  # Convert to seconds
        
        order.submitted_at = datetime.now()
        order.latency_ms += submission_latency
        
        # Risk checks
        if not self._validate_order(order, orderbook):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order {order.id} rejected: failed validation")
            return False
        
        # Add to pending orders
        self.pending_orders[order.id] = order
        
        # Add acknowledgment latency
        ack_latency = self.latency_simulator.get_latency('order_ack')
        await asyncio.sleep(ack_latency / 1000.0)
        order.latency_ms += ack_latency
        
        logger.info(f"Order {order.id} submitted: {order.side.value} {order.quantity} {order.symbol}")
        return True
    
    def _validate_order(self, order: Order, orderbook: L2OrderBookData) -> bool:
        """Validate order against risk controls and market conditions."""
        
        # Size validation
        if order.quantity <= 0:
            return False
        
        # Price validation for limit orders
        if order.order_type == OrderType.LIMIT and order.price is None:
            return False
        
        # Market hours check (simplified)
        current_hour = datetime.now().hour
        if current_hour < 9 or current_hour >= 16:
            return False  # Outside market hours
        
        # Liquidity check - don't allow orders larger than available liquidity
        if order.side == OrderSide.BUY:
            available_liquidity = sum(orderbook.ask_sizes[:3])
        else:
            available_liquidity = sum(orderbook.bid_sizes[:3])
        
        if order.quantity > available_liquidity * 2:  # Can't be more than 2x available
            return False
        
        return True
    
    async def process_executions(self,
                               orderbook: L2OrderBookData,
                               regime: MarketRegimeData,
                               daily_volume: float = 1000000):
        """Process pending order executions."""
        
        executed_orders = []
        
        for order_id, order in list(self.pending_orders.items()):
            
            # Check if order should fill
            if self._should_fill_order(order, orderbook):
                
                # Calculate execution details
                fill_price, fill_quantity = self._calculate_fill(order, orderbook, regime, daily_volume)
                
                if fill_quantity > 0:
                    # Create fill
                    commission = self._calculate_commission(order, fill_quantity, fill_price)
                    
                    fill = Fill(
                        order_id=order_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=fill_quantity,
                        price=fill_price,
                        timestamp=datetime.now(),
                        commission=commission
                    )
                    
                    self.fills.append(fill)
                    
                    # Update order
                    order.filled_quantity += fill_quantity
                    order.average_fill_price = (
                        (order.average_fill_price * (order.filled_quantity - fill_quantity) + 
                         fill_price * fill_quantity) / order.filled_quantity
                    )
                    order.commission += commission
                    
                    # Check if order is completely filled
                    if order.filled_quantity >= order.quantity:
                        order.status = OrderStatus.FILLED
                        order.filled_at = datetime.now()
                        executed_orders.append(order_id)
                    else:
                        order.status = OrderStatus.PARTIALLY_FILLED
                    
                    # Add fill notification latency
                    fill_latency = self.latency_simulator.get_latency('fill_notification')
                    order.latency_ms += fill_latency
                    
                    logger.info(f"Order {order_id} filled: {fill_quantity} @ {fill_price:.4f}")
        
        # Remove completed orders
        for order_id in executed_orders:
            del self.pending_orders[order_id]
    
    def _should_fill_order(self, order: Order, orderbook: L2OrderBookData) -> bool:
        """Determine if an order should fill based on market conditions."""
        
        if order.order_type == OrderType.MARKET:
            return True  # Market orders always fill (in liquid markets)
        
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                # Buy limit fills if our price >= ask price
                return order.price >= orderbook.ask_prices[0] if orderbook.ask_prices else False
            else:
                # Sell limit fills if our price <= bid price  
                return order.price <= orderbook.bid_prices[0] if orderbook.bid_prices else False
        
        return False
    
    def _calculate_fill(self,
                       order: Order,
                       orderbook: L2OrderBookData,
                       regime: MarketRegimeData,
                       daily_volume: float) -> Tuple[float, float]:
        """Calculate fill price and quantity."""
        
        if order.order_type == OrderType.MARKET:
            # Market order - fill against orderbook
            if order.side == OrderSide.BUY:
                base_price = orderbook.ask_prices[0] if orderbook.ask_prices else 100.0
            else:
                base_price = orderbook.bid_prices[0] if orderbook.bid_prices else 100.0
        else:
            # Limit order - use limit price
            base_price = order.price
        
        # Calculate market impact
        market_impact = self.impact_model.calculate_market_impact(order, orderbook, daily_volume)
        
        # Calculate slippage
        current_impact = self.price_impacts.get(order.symbol, 0.0)
        slippage = self.slippage_model.calculate_slippage(order, orderbook, regime, current_impact)
        order.slippage = slippage
        
        # Apply slippage to price
        if order.side == OrderSide.BUY:
            fill_price = base_price * (1 + slippage)
        else:
            fill_price = base_price * (1 - slippage)
        
        # Update temporary price impact
        self.price_impacts[order.symbol] = (
            self.price_impacts.get(order.symbol, 0.0) + market_impact.temporary_impact
        )
        
        # Determine fill quantity (partial fills in volatile conditions)
        remaining_quantity = order.quantity - order.filled_quantity
        
        if regime.regime_type in ['volatile', 'crisis']:
            # In volatile conditions, orders might partially fill
            fill_ratio = np.random.uniform(0.3, 1.0)
            fill_quantity = min(remaining_quantity, remaining_quantity * fill_ratio)
        else:
            # Normal conditions - usually full fill
            fill_quantity = remaining_quantity
        
        return fill_price, fill_quantity
    
    def _calculate_commission(self, order: Order, quantity: float, price: float) -> float:
        """Calculate realistic commission/fees."""
        
        # Tiered commission structure
        notional = quantity * price
        
        if notional < 10000:  # < $10k
            commission_rate = 0.0005  # 5 bps
        elif notional < 100000:  # $10k - $100k
            commission_rate = 0.0003  # 3 bps
        else:  # > $100k
            commission_rate = 0.0001  # 1 bp
        
        commission = notional * commission_rate
        
        # Minimum commission
        return max(1.00, commission)
    
    def decay_price_impacts(self):
        """Decay temporary price impacts over time."""
        for symbol in self.price_impacts:
            self.price_impacts[symbol] *= self.impact_model.impact_decay_rate

class RealisticTradingEnvironment:
    """Main realistic trading environment combining all components."""
    
    def __init__(self,
                 symbols: List[str],
                 initial_balance: float = 100000,
                 max_position_size: float = 0.2):
        
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        
        # Components
        self.data_generator = HybridDataGenerator()
        self.execution_engine = OrderExecutionEngine()
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.positions: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
        self.portfolio_value = initial_balance
        
        # Market data
        self.current_market_data = {}
        self.current_regimes = {}
        self.current_orderbooks = {}
        
        # Performance tracking
        self.trade_history = []
        self.portfolio_history = []
        
        # Action space: continuous allocation for each symbol [-1, 1]
        # -1 = max short, 0 = no position, 1 = max long
        if TORCH_AVAILABLE:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, 
                shape=(len(symbols),), 
                dtype=np.float32
            )
            
            # Observation space: [prices, returns, sentiment, orderbook, regime features]
            obs_dim = len(symbols) * 10  # 10 features per symbol
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
    
    async def reset(self, training_data: Dict[str, Any] = None) -> np.ndarray:
        """Reset environment to initial state."""
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_value = self.initial_balance
        
        # Clear tracking
        self.trade_history.clear()
        self.portfolio_history.clear()
        self.execution_engine.fills.clear()
        self.execution_engine.pending_orders.clear()
        
        # Load training data if provided
        if training_data:
            self.training_data = training_data
        else:
            # Generate fresh data
            dataset = await self.data_generator.generate_training_dataset(
                symbols=self.symbols,
                training_days=252,  # 1 year
                validation_split=0.0
            )
            self.training_data = dataset['training']
        
        # Initialize first observation
        await self._update_market_state()
        return self._get_observation()
    
    async def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        
        # Convert action to orders
        orders = self._action_to_orders(action)
        
        # Submit orders
        for order in orders:
            symbol = order.symbol
            if symbol in self.current_orderbooks and symbol in self.current_regimes:
                await self.execution_engine.submit_order(
                    order, 
                    self.current_orderbooks[symbol],
                    self.current_regimes[symbol]
                )
        
        # Process executions
        for symbol in self.symbols:
            if symbol in self.current_orderbooks and symbol in self.current_regimes:
                await self.execution_engine.process_executions(
                    self.current_orderbooks[symbol],
                    self.current_regimes[symbol]
                )
        
        # Update positions from fills
        self._update_positions_from_fills()
        
        # Advance time and update market state
        self.current_step += 1
        await self._update_market_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        done = (self.current_step >= len(self.training_data['market_data']) - 1 or
                self.portfolio_value <= self.initial_balance * 0.2)  # 80% drawdown limit
        
        # Info dict
        info = {
            'portfolio_value': self.portfolio_value,
            'step': self.current_step,
            'positions': self.positions.copy(),
            'recent_fills': len(self.execution_engine.fills),
            'pending_orders': len(self.execution_engine.pending_orders)
        }
        
        # Decay market impacts
        self.execution_engine.decay_price_impacts()
        
        return self._get_observation(), reward, done, info
    
    def _action_to_orders(self, action: np.ndarray) -> List[Order]:
        """Convert RL action to trading orders."""
        orders = []
        
        for i, symbol in enumerate(self.symbols):
            target_allocation = np.clip(action[i], -1.0, 1.0)
            max_position_value = self.portfolio_value * self.max_position_size
            
            if symbol in self.current_market_data:
                current_price = self.current_market_data[symbol]['price']
                current_position = self.positions.get(symbol, 0.0)
                current_position_value = current_position * current_price
                
                # Calculate target position
                target_position_value = target_allocation * max_position_value
                position_change_value = target_position_value - current_position_value
                position_change_shares = position_change_value / current_price
                
                # Only create order if change is significant
                if abs(position_change_shares) > 1.0:  # Minimum 1 share
                    
                    if position_change_shares > 0:
                        side = OrderSide.BUY
                    else:
                        side = OrderSide.SELL if current_position > 0 else OrderSide.SHORT
                    
                    order = Order(
                        id=f"order_{datetime.now().timestamp()}_{i}",
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,  # Use market orders for simplicity
                        quantity=abs(position_change_shares),
                        created_at=datetime.now()
                    )
                    
                    orders.append(order)
        
        return orders
    
    def _update_positions_from_fills(self):
        """Update positions based on recent fills."""
        
        recent_fills = [f for f in self.execution_engine.fills 
                       if f not in getattr(self, '_processed_fills', set())]
        
        for fill in recent_fills:
            if fill.side == OrderSide.BUY:
                self.positions[fill.symbol] += fill.quantity
                self.balance -= fill.quantity * fill.price + fill.commission
            elif fill.side == OrderSide.SELL:
                self.positions[fill.symbol] -= fill.quantity
                self.balance += fill.quantity * fill.price - fill.commission
            elif fill.side == OrderSide.SHORT:
                self.positions[fill.symbol] -= fill.quantity  # Short position is negative
                self.balance += fill.quantity * fill.price - fill.commission
        
        # Track processed fills
        if not hasattr(self, '_processed_fills'):
            self._processed_fills = set()
        self._processed_fills.update(recent_fills)
        
        # Update portfolio value
        portfolio_value = self.balance
        for symbol, position in self.positions.items():
            if symbol in self.current_market_data:
                current_price = self.current_market_data[symbol]['price']
                portfolio_value += position * current_price
        
        self.portfolio_value = portfolio_value
        self.portfolio_history.append({
            'step': self.current_step,
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy()
        })
    
    async def _update_market_state(self):
        """Update current market state from training data."""
        
        if self.current_step < len(self.training_data['market_data']):
            # Get market data for current step
            self.current_market_data = self.training_data['market_data'][self.current_step]
            
            # Get regime data
            if self.current_step < len(self.training_data['regime_data']):
                regime_data = self.training_data['regime_data'][self.current_step]
                regime = MarketRegimeData(**regime_data)
                self.current_regimes = {symbol: regime for symbol in self.symbols}
            
            # Generate order books for each symbol
            self.current_orderbooks = {}
            for symbol in self.symbols:
                if symbol in self.current_market_data:
                    symbol_data = self.current_market_data[symbol]
                    
                    # Create orderbook from data
                    orderbook_data = symbol_data.get('orderbook', {})
                    orderbook = L2OrderBookData(**orderbook_data)
                    self.current_orderbooks[symbol] = orderbook
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation for RL agent."""
        
        obs = []
        
        for symbol in self.symbols:
            if symbol in self.current_market_data:
                data = self.current_market_data[symbol]
                sentiment = data.get('sentiment', {})
                orderbook = data.get('orderbook', {})
                
                symbol_obs = [
                    data.get('price', 0.0) / 100.0,  # Normalized price
                    data.get('return', 0.0) * 100,   # Return in %
                    data.get('volume', 0.0) / 1000000,  # Volume in millions
                    sentiment.get('overall_sentiment', 0.0),
                    sentiment.get('confidence', 0.5),
                    orderbook.get('spread', 0.01) * 1000,  # Spread in bps
                    orderbook.get('imbalance', 0.0),
                    self.positions.get(symbol, 0.0) / 1000,  # Position size normalized
                    (self.portfolio_value - self.initial_balance) / self.initial_balance,  # P&L %
                    self.current_step / 252.0  # Time progress
                ]
            else:
                symbol_obs = [0.0] * 10  # Default observation
            
            obs.extend(symbol_obs)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self) -> float:
        """Calculate reward for RL agent."""
        
        # Portfolio return
        if len(self.portfolio_history) >= 2:
            prev_value = self.portfolio_history[-2]['portfolio_value']
            current_value = self.portfolio_value
            portfolio_return = (current_value - prev_value) / prev_value
        else:
            portfolio_return = 0.0
        
        # Base reward is portfolio return
        reward = portfolio_return * 100  # Scale up
        
        # Risk penalty for excessive positions
        total_exposure = sum(abs(pos) * self.current_market_data.get(sym, {}).get('price', 0) 
                           for sym, pos in self.positions.items())
        exposure_ratio = total_exposure / self.portfolio_value
        
        if exposure_ratio > 1.0:  # Leverage penalty
            reward -= (exposure_ratio - 1.0) * 0.1
        
        # Transaction cost penalty
        recent_fills = [f for f in self.execution_engine.fills[-10:]]  # Last 10 fills
        total_commission = sum(f.commission for f in recent_fills)
        reward -= total_commission / self.portfolio_value * 100
        
        return reward

# Factory function
def create_realistic_trading_env(symbols: List[str], **kwargs) -> RealisticTradingEnvironment:
    """Create a realistic trading environment."""
    return RealisticTradingEnvironment(symbols=symbols, **kwargs)

if __name__ == "__main__":
    # Test the realistic trading environment
    async def test_realistic_env():
        env = create_realistic_trading_env(['AAPL', 'MSFT', 'GOOGL'])
        
        # Reset environment
        obs = await env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial portfolio value: ${env.portfolio_value:,.2f}")
        
        # Take some random actions
        for step in range(10):
            action = np.random.uniform(-0.5, 0.5, size=(3,))  # Conservative actions
            obs, reward, done, info = await env.step(action)
            
            print(f"Step {step}: Reward={reward:.4f}, Portfolio=${info['portfolio_value']:,.2f}")
            
            if done:
                break
        
        print(f"Final portfolio value: ${env.portfolio_value:,.2f}")
        print(f"Total fills: {len(env.execution_engine.fills)}")
    
    asyncio.run(test_realistic_env())