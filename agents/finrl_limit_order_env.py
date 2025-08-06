"""
Enhanced FinRL Trading Environment with Intelligent Limit Order Support
Extends the base FinRL environment to support sophisticated limit order strategies.

Action Space Design:
- For each symbol: [position_action, order_type_action, price_offset_action]
- position_action: [-1, 1] (negative=short, positive=long, magnitude=size)
- order_type_action: [-1, 1] (-1=market, 0=limit, 1=aggressive limit)
- price_offset_action: [-1, 1] (price offset from current market as percentage)

State Representation includes:
- Current bid/ask spreads
- Order book depth indicators
- Open limit order status
- Time-in-force information
- Execution quality metrics
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from agents.finrl_enhanced_env import (
    EnhancedFinRLTradingEnv, Position, PortfolioState, 
    OrderType, OrderSide, PendingOrder
)

logger = logging.getLogger(__name__)


@dataclass 
class LimitOrderState:
    """State information for limit orders."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    limit_price: float
    market_price_at_placement: float
    timestamp: datetime
    time_in_force: str = "GTC"
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    is_active: bool = True
    price_improvement: float = 0.0  # vs market order
    
    @property
    def age_seconds(self) -> float:
        return (datetime.now() - self.timestamp).total_seconds()
    
    @property
    def fill_ratio(self) -> float:
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0.0


@dataclass
class MarketDepth:
    """Market depth information for order book simulation."""
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    spread: float
    mid_price: float
    
    # Level 2 data simulation (simplified)
    bid_levels: List[Tuple[float, float]] = field(default_factory=list)  # (price, size)
    ask_levels: List[Tuple[float, float]] = field(default_factory=list)  # (price, size)
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return (self.spread / self.mid_price) * 10000 if self.mid_price > 0 else 0


class LimitOrderTradingEnv(EnhancedFinRLTradingEnv):
    """
    Enhanced FinRL environment with intelligent limit order support.
    
    Features:
    - 3D action space: [position, order_type, price_offset] per symbol
    - Sophisticated limit order execution simulation
    - Bid/ask spread modeling and order book depth
    - Reward optimization for execution quality
    - Time-in-force and order aging logic
    """
    
    def __init__(self, 
                 symbols: List[str],
                 **kwargs):
        
        # Initialize base environment
        super().__init__(symbols, **kwargs)
        
        # Enhanced action space: 3 dimensions per symbol
        # [position_action, order_type_action, price_offset_action]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.symbol_count * 3,),  # 3 actions per symbol
            dtype=np.float32
        )
        
        # Limit order specific parameters
        self.max_spread_bps = kwargs.get('max_spread_bps', 50)  # 0.5% max spread
        self.order_timeout_minutes = kwargs.get('order_timeout_minutes', 60)  # 1 hour timeout
        self.partial_fill_enabled = kwargs.get('partial_fill_enabled', True)
        self.min_fill_probability = kwargs.get('min_fill_probability', 0.1)
        
        # Market microstructure simulation
        self.market_depth: Dict[str, MarketDepth] = {}
        self.limit_orders: Dict[str, LimitOrderState] = {}  # order_id -> order
        self.execution_history: List[Dict] = []
        
        # Execution quality tracking
        self.execution_metrics = {
            'total_slippage': 0.0,
            'limit_order_fills': 0,
            'market_order_fills': 0,
            'price_improvement_total': 0.0,
            'missed_opportunities': 0
        }
        
        logger.info(f"✅ Limit Order Trading Environment initialized")
        logger.info(f"   Action dimensions: {self.action_space.shape[0]} (3 per symbol)")
        logger.info(f"   Max spread: {self.max_spread_bps} bps")
        logger.info(f"   Order timeout: {self.order_timeout_minutes} minutes")
    
    def set_data(self, data: pd.DataFrame):
        """Enhanced data loading with bid/ask spread simulation."""
        super().set_data(data)
        
        # Simulate market depth for each symbol
        self._simulate_market_depth(data)
        
        # Update observation space to include limit order features
        self._update_observation_space_for_limit_orders()
    
    def _simulate_market_depth(self, data: pd.DataFrame):
        """Simulate realistic bid/ask spreads and market depth."""
        for symbol in self.symbols:
            close_col = f"{symbol}_close"
            volume_col = f"{symbol}_volume"
            
            if close_col in data.columns:
                closes = data[close_col].values
                volumes = data.get(volume_col, pd.Series([1000000] * len(data))).values
                
                # Simulate spreads based on volatility and volume
                spreads = self._calculate_realistic_spreads(closes, volumes)
                
                # Store market depth simulation
                market_depth_series = []
                for i, (close, volume, spread) in enumerate(zip(closes, volumes, spreads)):
                    mid_price = close
                    half_spread = spread / 2
                    
                    depth = MarketDepth(
                        symbol=symbol,
                        bid_price=mid_price - half_spread,
                        ask_price=mid_price + half_spread,
                        bid_size=volume * 0.1,  # 10% of volume at best bid
                        ask_size=volume * 0.1,  # 10% of volume at best ask
                        spread=spread,
                        mid_price=mid_price
                    )
                    
                    # Simulate deeper levels (simplified)
                    depth.bid_levels = [
                        (mid_price - half_spread * 1.5, volume * 0.05),
                        (mid_price - half_spread * 2.0, volume * 0.03),
                    ]
                    depth.ask_levels = [
                        (mid_price + half_spread * 1.5, volume * 0.05),
                        (mid_price + half_spread * 2.0, volume * 0.03),
                    ]
                    
                    market_depth_series.append(depth)
                
                # Store for current step access
                if symbol not in self.market_depth:
                    self.market_depth[symbol] = market_depth_series
    
    def _calculate_realistic_spreads(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Calculate realistic bid/ask spreads based on volatility and volume."""
        
        # Calculate rolling volatility
        returns = np.diff(np.log(prices + 1e-8))
        volatility = np.std(returns) if len(returns) > 1 else 0.01
        
        # Base spread (larger for more volatile stocks)
        base_spread_bps = max(5, min(50, volatility * 10000))  # 5-50 bps
        
        # Adjust for volume (lower volume = wider spreads)
        avg_volume = np.mean(volumes)
        volume_factor = max(0.5, min(2.0, 1e6 / (avg_volume + 1)))
        
        # Calculate spreads
        spreads = []
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            # Volume-adjusted spread
            volume_adj = max(0.7, min(1.5, avg_volume / (volume + 1)))
            spread_bps = base_spread_bps * volume_factor * volume_adj
            
            # Convert to dollar spread
            spread = (spread_bps / 10000) * price
            spreads.append(max(0.01, spread))  # Minimum 1 cent spread
        
        return np.array(spreads)
    
    def _update_observation_space_for_limit_orders(self):
        """Update observation space to include limit order features."""
        if self.observation_space is None:
            return
            
        # Add limit order features per symbol:
        # - bid/ask spread (2 features)
        # - open limit orders (3 features: count, avg_price, avg_age)
        # - execution quality (2 features: recent slippage, fill rate)
        limit_order_features = self.symbol_count * 7
        
        # Original observation size + limit order features
        original_size = self.observation_space.shape[0]
        new_size = original_size + limit_order_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(new_size,), dtype=np.float32
        )
        
        logger.info(f"Updated observation space: {original_size} -> {new_size} features")
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Enhanced step function with limit order processing."""
        
        # Age and process existing limit orders
        self._age_limit_orders()
        self._process_limit_orders()
        
        # Process new actions (position, order_type, price_offset per symbol)
        trade_info = self._execute_limit_order_actions(action)
        
        # Update portfolio with market movements
        self._update_portfolio_values()
        
        # Calculate enhanced reward including execution quality
        reward = self._calculate_limit_order_reward()
        self.reward_history.append(reward)
        
        # Check margin requirements and risk management
        margin_call = self._check_margin_requirements()
        if margin_call:
            reward -= 1.0
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Get enhanced observation
        obs = self._get_limit_order_observation()
        
        # Enhanced info dict
        info = {
            'portfolio_value': self.portfolio.equity,
            'cash': self.portfolio.cash,
            'positions': len([p for p in self.portfolio.positions.values() if abs(p.quantity) > 0]),
            'trades_executed': trade_info['trades_executed'],
            'limit_orders_placed': trade_info['limit_orders_placed'],
            'limit_orders_filled': trade_info['limit_orders_filled'],
            'avg_spread_bps': self._get_avg_spread_bps(),
            'execution_quality': self._get_execution_quality_score(),
            'margin_call': margin_call,
            'step': self.current_step,
            'done': done
        }
        
        return obs, reward, done, info
    
    def _execute_limit_order_actions(self, actions: np.ndarray) -> Dict:
        """Execute limit order actions based on 3D action space."""
        trades_executed = 0
        limit_orders_placed = 0
        limit_orders_filled = 0
        
        # Reshape actions: [symbol_count, 3]
        actions_reshaped = actions.reshape(self.symbol_count, 3)
        
        for i, (position_action, order_type_action, price_offset_action) in enumerate(actions_reshaped):
            if i >= len(self.symbols):
                break
                
            symbol = self.symbols[i]
            
            # Skip very small position actions
            if abs(position_action) < 0.01:
                continue
            
            # Get current market data
            if symbol not in self.market_depth or self.current_step >= len(self.market_depth[symbol]):
                continue
                
            market_data = self.market_depth[symbol][self.current_step]
            
            # Determine order type based on order_type_action
            if order_type_action < -0.3:  # Market order
                result = self._execute_market_order(symbol, position_action, market_data)
                if result:
                    trades_executed += 1
            else:  # Limit order
                result = self._place_limit_order(
                    symbol, position_action, price_offset_action, market_data
                )
                if result:
                    limit_orders_placed += 1
        
        return {
            'trades_executed': trades_executed,
            'limit_orders_placed': limit_orders_placed,
            'limit_orders_filled': limit_orders_filled
        }
    
    def _execute_market_order(self, symbol: str, position_action: float, market_data: MarketDepth) -> bool:
        """Execute immediate market order."""
        try:
            # Calculate quantity
            target_value = position_action * self.portfolio.equity * self.max_position_size
            
            # Use bid/ask instead of mid price for more realistic execution
            if position_action > 0:  # Buy
                execution_price = market_data.ask_price
                slippage = market_data.ask_price - market_data.mid_price
            else:  # Sell/Short
                execution_price = market_data.bid_price
                slippage = market_data.mid_price - market_data.bid_price
            
            quantity = abs(target_value) / execution_price
            
            if quantity < 1:
                return False
            
            # Execute trade using base class method
            order_type = OrderType.MARKET
            self._execute_trade(symbol, quantity if position_action > 0 else -quantity, 
                              execution_price, order_type)
            
            # Track execution metrics
            self.execution_metrics['market_order_fills'] += 1
            self.execution_metrics['total_slippage'] += abs(slippage)
            
            # Record execution
            self.execution_history.append({
                'symbol': symbol,
                'type': 'market',
                'side': 'buy' if position_action > 0 else 'sell',
                'quantity': quantity,
                'price': execution_price,
                'slippage': slippage,
                'timestamp': self.current_step
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Market order execution failed for {symbol}: {e}")
            return False
    
    def _place_limit_order(self, symbol: str, position_action: float, 
                          price_offset_action: float, market_data: MarketDepth) -> bool:
        """Place intelligent limit order."""
        try:
            # Calculate target quantity
            target_value = position_action * self.portfolio.equity * self.max_position_size
            
            # Determine reference price and side
            if position_action > 0:  # Buy limit
                reference_price = market_data.bid_price  # Start from bid
                side = OrderSide.BUY
            else:  # Sell/Short limit
                reference_price = market_data.ask_price  # Start from ask
                side = OrderSide.SHORT if self.enable_short_selling else OrderSide.SELL
            
            # Calculate limit price using price_offset_action
            # price_offset_action: [-1, 1] maps to [aggressive, passive]
            max_offset = market_data.spread * 2  # Max 2x spread offset
            price_offset = price_offset_action * max_offset
            
            if position_action > 0:  # Buy limit
                # Negative offset = more aggressive (higher price)
                # Positive offset = more passive (lower price)  
                limit_price = reference_price - price_offset
                limit_price = max(limit_price, market_data.bid_price * 0.95)  # Sanity check
            else:  # Sell limit
                # Negative offset = more aggressive (lower price)
                # Positive offset = more passive (higher price)
                limit_price = reference_price + price_offset
                limit_price = min(limit_price, market_data.ask_price * 1.05)  # Sanity check
            
            quantity = abs(target_value) / limit_price
            if quantity < 1:
                return False
            
            # Create limit order
            order_id = f"{symbol}_{side.value}_{len(self.limit_orders)}"
            limit_order = LimitOrderState(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                limit_price=limit_price,
                market_price_at_placement=market_data.mid_price,
                timestamp=datetime.now()
            )
            
            self.limit_orders[order_id] = limit_order
            
            logger.debug(f"Limit order placed: {symbol} {side.value} {quantity:.0f} @ ${limit_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Limit order placement failed for {symbol}: {e}")
            return False
    
    def _process_limit_orders(self):
        """Process existing limit orders for potential fills."""
        filled_orders = []
        
        for order_id, order in self.limit_orders.items():
            if not order.is_active:
                continue
            
            symbol = order.symbol
            if symbol not in self.market_depth or self.current_step >= len(self.market_depth[symbol]):
                continue
            
            current_market = self.market_depth[symbol][self.current_step]
            
            # Check if order should be filled
            fill_result = self._check_limit_order_fill(order, current_market)
            
            if fill_result['filled']:
                filled_quantity = fill_result['quantity']
                fill_price = fill_result['price']
                
                # Execute the fill
                trade_quantity = filled_quantity if order.side == OrderSide.BUY else -filled_quantity
                self._execute_trade(order.symbol, trade_quantity, fill_price, OrderType.LIMIT)
                
                # Update order state
                order.filled_quantity += filled_quantity
                order.avg_fill_price = ((order.avg_fill_price * (order.filled_quantity - filled_quantity) + 
                                       fill_price * filled_quantity) / order.filled_quantity)
                
                # Calculate price improvement
                if order.side == OrderSide.BUY:
                    improvement = current_market.ask_price - fill_price
                else:
                    improvement = fill_price - current_market.bid_price
                
                order.price_improvement += improvement * filled_quantity
                
                # Check if order is completely filled
                if order.filled_quantity >= order.quantity:
                    order.is_active = False
                    filled_orders.append(order_id)
                
                # Track metrics
                self.execution_metrics['limit_order_fills'] += 1
                self.execution_metrics['price_improvement_total'] += improvement * filled_quantity
                
                # Record execution
                self.execution_history.append({
                    'symbol': order.symbol,
                    'type': 'limit',
                    'side': order.side.value,
                    'quantity': filled_quantity,
                    'price': fill_price,
                    'improvement': improvement,
                    'age_seconds': order.age_seconds,
                    'timestamp': self.current_step
                })
        
        # Clean up filled orders
        for order_id in filled_orders:
            del self.limit_orders[order_id]
    
    def _check_limit_order_fill(self, order: LimitOrderState, market: MarketDepth) -> Dict:
        """Determine if and how much of a limit order should be filled."""
        
        # Check if price condition is met
        if order.side == OrderSide.BUY:
            price_condition = market.ask_price <= order.limit_price
            reference_price = market.ask_price
        else:  # SELL or SHORT
            price_condition = market.bid_price >= order.limit_price
            reference_price = market.bid_price
        
        if not price_condition:
            return {'filled': False, 'quantity': 0, 'price': 0}
        
        # Calculate fill probability based on order aggressiveness and market conditions
        if order.side == OrderSide.BUY:
            aggressiveness = (order.limit_price - market.bid_price) / market.spread if market.spread > 0 else 1
        else:
            aggressiveness = (market.ask_price - order.limit_price) / market.spread if market.spread > 0 else 1
        
        # Base fill probability
        base_prob = max(self.min_fill_probability, min(0.95, 0.3 + aggressiveness * 0.6))
        
        # Adjust for order age (older orders more likely to fill)
        age_factor = min(1.5, 1.0 + order.age_seconds / 3600)  # Increase with age
        fill_prob = min(0.98, base_prob * age_factor)
        
        # Random fill decision
        if np.random.random() > fill_prob:
            return {'filled': False, 'quantity': 0, 'price': 0}
        
        # Determine fill quantity (support partial fills)
        remaining_quantity = order.quantity - order.filled_quantity
        
        if self.partial_fill_enabled and np.random.random() < 0.3:  # 30% chance of partial fill
            fill_quantity = remaining_quantity * np.random.uniform(0.2, 0.8)
        else:
            fill_quantity = remaining_quantity
        
        fill_quantity = max(1, int(fill_quantity))  # Minimum 1 share
        fill_price = order.limit_price  # Filled at limit price
        
        return {
            'filled': True,
            'quantity': fill_quantity,
            'price': fill_price
        }
    
    def _age_limit_orders(self):
        """Age limit orders and cancel expired ones."""
        expired_orders = []
        
        for order_id, order in self.limit_orders.items():
            if order.age_seconds > self.order_timeout_minutes * 60:
                order.is_active = False
                expired_orders.append(order_id)
                
                # Track missed opportunity
                self.execution_metrics['missed_opportunities'] += 1
        
        # Remove expired orders
        for order_id in expired_orders:
            del self.limit_orders[order_id]
    
    def _get_limit_order_observation(self) -> np.ndarray:
        """Enhanced observation including limit order features."""
        
        # Get base observation
        base_obs = super()._get_observation()
        
        # Add limit order features
        limit_order_features = []
        
        for symbol in self.symbols:
            # Market depth features
            if (symbol in self.market_depth and 
                self.current_step < len(self.market_depth[symbol])):
                market = self.market_depth[symbol][self.current_step]
                spread_bps = market.spread_bps / 100  # Normalize
                spread_ratio = market.spread / market.mid_price if market.mid_price > 0 else 0
            else:
                spread_bps, spread_ratio = 0.0, 0.0
            
            limit_order_features.extend([spread_bps, spread_ratio])
            
            # Open limit orders for this symbol
            symbol_orders = [o for o in self.limit_orders.values() 
                           if o.symbol == symbol and o.is_active]
            
            if symbol_orders:
                order_count = len(symbol_orders) / 10  # Normalize
                avg_price = np.mean([o.limit_price for o in symbol_orders]) / 100  # Normalize
                avg_age = np.mean([o.age_seconds for o in symbol_orders]) / 3600  # Hours
            else:
                order_count, avg_price, avg_age = 0.0, 0.0, 0.0
            
            limit_order_features.extend([order_count, avg_price, avg_age])
            
            # Execution quality metrics
            recent_executions = [e for e in self.execution_history[-10:] if e['symbol'] == symbol]
            if recent_executions:
                avg_slippage = np.mean([abs(e.get('slippage', 0)) for e in recent_executions])
                avg_improvement = np.mean([e.get('improvement', 0) for e in recent_executions])
            else:
                avg_slippage, avg_improvement = 0.0, 0.0
            
            limit_order_features.extend([avg_slippage, avg_improvement])
        
        # Combine base observation with limit order features
        enhanced_obs = np.concatenate([base_obs, limit_order_features]).astype(np.float32)
        
        # Handle NaN values
        enhanced_obs = np.nan_to_num(enhanced_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure observation matches expected shape
        if len(enhanced_obs) != self.observation_space.shape[0]:
            if len(enhanced_obs) < self.observation_space.shape[0]:
                enhanced_obs = np.pad(enhanced_obs, (0, self.observation_space.shape[0] - len(enhanced_obs)))
            else:
                enhanced_obs = enhanced_obs[:self.observation_space.shape[0]]
        
        # Final NaN check and normalization
        enhanced_obs = np.nan_to_num(enhanced_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        enhanced_obs = np.clip(enhanced_obs, -10.0, 10.0)  # Clip extreme values
        
        return enhanced_obs
    
    def _calculate_limit_order_reward(self) -> float:
        """Enhanced reward function incorporating execution quality."""
        
        # Get base reward from parent class
        base_reward = super()._calculate_reward()
        
        # Execution quality bonuses
        execution_bonus = 0.0
        
        # Recent executions for analysis
        recent_executions = self.execution_history[-5:] if self.execution_history else []
        
        if recent_executions:
            # Reward price improvement from limit orders
            total_improvement = sum(e.get('improvement', 0) for e in recent_executions)
            improvement_bonus = total_improvement / self.initial_balance * 10  # Scale up
            
            # Reward successful limit order usage
            limit_fills = sum(1 for e in recent_executions if e['type'] == 'limit')
            limit_ratio = limit_fills / len(recent_executions)
            limit_bonus = limit_ratio * 0.01  # Small bonus for using limit orders
            
            # Penalize excessive slippage from market orders
            total_slippage = sum(abs(e.get('slippage', 0)) for e in recent_executions)
            slippage_penalty = -total_slippage / self.initial_balance * 5
            
            execution_bonus = improvement_bonus + limit_bonus + slippage_penalty
        
        # Penalize too many open limit orders (opportunity cost)
        active_orders = len([o for o in self.limit_orders.values() if o.is_active])
        if active_orders > self.symbol_count:  # More orders than symbols
            order_penalty = -(active_orders - self.symbol_count) * 0.001
            execution_bonus += order_penalty
        
        # Penalize missed opportunities (expired orders)
        if hasattr(self, '_last_missed_opportunities'):
            new_missed = self.execution_metrics['missed_opportunities'] - self._last_missed_opportunities
            if new_missed > 0:
                execution_bonus -= new_missed * 0.005
        
        self._last_missed_opportunities = self.execution_metrics['missed_opportunities']
        
        return base_reward + execution_bonus
    
    def _get_avg_spread_bps(self) -> float:
        """Get average spread in basis points across all symbols."""
        if not self.market_depth:
            return 0.0
        
        spreads = []
        for symbol in self.symbols:
            if (symbol in self.market_depth and 
                self.current_step < len(self.market_depth[symbol])):
                spreads.append(self.market_depth[symbol][self.current_step].spread_bps)
        
        return np.mean(spreads) if spreads else 0.0
    
    def _get_execution_quality_score(self) -> float:
        """Calculate overall execution quality score (0-1)."""
        if not self.execution_history:
            return 0.5  # Neutral
        
        recent_executions = self.execution_history[-20:]  # Last 20 executions
        
        # Calculate metrics
        total_improvement = sum(e.get('improvement', 0) for e in recent_executions)
        total_slippage = sum(abs(e.get('slippage', 0)) for e in recent_executions)
        limit_ratio = sum(1 for e in recent_executions if e['type'] == 'limit') / len(recent_executions)
        
        # Normalize and combine
        improvement_score = max(0, min(1, (total_improvement + 1) / 2))  # Normalize around 0
        slippage_score = max(0, 1 - total_slippage / 10)  # Lower slippage = higher score
        limit_score = limit_ratio  # Higher limit order usage = higher score
        
        # Weighted combination
        quality_score = (improvement_score * 0.4 + slippage_score * 0.4 + limit_score * 0.2)
        
        return max(0.0, min(1.0, quality_score))
    
    def get_limit_order_summary(self) -> Dict[str, Any]:
        """Get comprehensive limit order performance summary."""
        summary = self.get_portfolio_summary()
        
        # Add limit order specific metrics
        active_orders = [o for o in self.limit_orders.values() if o.is_active]
        
        limit_order_stats = {
            'active_limit_orders': len(active_orders),
            'total_limit_orders_placed': len(self.execution_history) + len(active_orders),
            'limit_order_fill_rate': (
                self.execution_metrics['limit_order_fills'] / 
                max(1, self.execution_metrics['limit_order_fills'] + len(active_orders))
            ),
            'avg_spread_bps': self._get_avg_spread_bps(),
            'execution_quality_score': self._get_execution_quality_score(),
            'total_price_improvement': self.execution_metrics['price_improvement_total'],
            'missed_opportunities': self.execution_metrics['missed_opportunities'],
            'market_vs_limit_ratio': (
                self.execution_metrics['market_order_fills'] / 
                max(1, self.execution_metrics['limit_order_fills'])
            )
        }
        
        # Add active orders details
        if active_orders:
            limit_order_stats['active_orders_detail'] = [
                {
                    'symbol': o.symbol,
                    'side': o.side.value,
                    'quantity': o.quantity,
                    'limit_price': o.limit_price,
                    'age_minutes': o.age_seconds / 60,
                    'fill_ratio': o.fill_ratio
                }
                for o in active_orders
            ]
        
        summary.update(limit_order_stats)
        return summary


# Factory function for easy creation
def create_limit_order_finrl_env(symbols: List[str], **kwargs) -> LimitOrderTradingEnv:
    """Create limit order FinRL environment with optimal settings."""
    defaults = {
        'enable_short_selling': True,
        'enable_limit_orders': True,
        'commission_rate': 0.001,
        'margin_requirement': 0.5,
        'short_borrow_rate': 0.03,
        'max_position_size': 0.15,
        'max_spread_bps': 50,
        'order_timeout_minutes': 60,
        'partial_fill_enabled': True,
        'min_fill_probability': 0.1
    }
    
    final_kwargs = {**defaults, **kwargs}
    
    return LimitOrderTradingEnv(symbols=symbols, **final_kwargs)


if __name__ == "__main__":
    # Test the limit order environment
    import yfinance as yf
    
    def test_limit_order_environment():
        # Create test environment
        symbols = ["AAPL", "MSFT", "GOOGL"]
        env = create_limit_order_finrl_env(symbols)
        
        # Generate test data (simplified)
        test_data = {}
        for symbol in symbols:
            prices = np.random.uniform(100, 200, 30)
            volumes = np.random.uniform(1000000, 5000000, 30)
            
            test_data[f"{symbol}_close"] = prices
            test_data[f"{symbol}_volume"] = volumes
        
        df = pd.DataFrame(test_data)
        env.set_data(df)
        
        print("Testing Limit Order FinRL Environment")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Test episode with limit orders
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        for step in range(5):
            # Test actions: [position, order_type, price_offset] per symbol
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            print(f"\nStep {step + 1}:")
            print(f"  Reward: {reward:.6f}")
            print(f"  Portfolio Value: ${info['portfolio_value']:,.2f}")
            print(f"  Limit Orders Placed: {info['limit_orders_placed']}")
            print(f"  Limit Orders Filled: {info['limit_orders_filled']}")
            print(f"  Avg Spread (bps): {info['avg_spread_bps']:.1f}")
            print(f"  Execution Quality: {info['execution_quality']:.3f}")
            
            if done:
                break
        
        # Final summary
        summary = env.get_limit_order_summary()
        print(f"\nFinal Summary:")
        print(f"  Active Limit Orders: {summary['active_limit_orders']}")
        print(f"  Fill Rate: {summary['limit_order_fill_rate']:.2%}")
        print(f"  Price Improvement: ${summary['total_price_improvement']:.2f}")
        print(f"  Execution Quality: {summary['execution_quality_score']:.3f}")
        
        env.close()
        print("✅ Limit Order Environment test completed")
    
    test_limit_order_environment()