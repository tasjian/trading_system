#!/usr/bin/env python3
"""
High Frequency Trading Engine

Core engine for high-frequency trading operations with $25,000 minimum balance
maintenance and unlimited day trading capabilities.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from config.hft_settings import hft_settings
from tools.alpaca_client import alpaca_client
from order_types.advanced_orders import advanced_order_manager, AdvancedOrderRequest

logger = logging.getLogger(__name__)

@dataclass
class HFTSignal:
    """High frequency trading signal."""
    symbol: str
    signal_type: str  # "momentum", "reversal", "breakout", "arbitrage"
    direction: str    # "buy", "sell"
    strength: float   # 0.0 to 1.0
    price_target: float
    stop_loss: float
    confidence: float
    time_horizon: int  # Minutes to hold position
    generated_at: datetime = field(default_factory=datetime.now)
    
@dataclass 
class HFTPosition:
    """High frequency trading position tracking."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    target_exit_time: datetime
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0
    position_id: str = ""

class HighFrequencyTradingEngine:
    """High-frequency trading engine with advanced risk management."""
    
    def __init__(self):
        """Initialize HFT engine."""
        self.active_positions: Dict[str, HFTPosition] = {}
        self.recent_trades: List[Dict] = []
        self.signal_queue: List[HFTSignal] = []
        self.account_balance: float = 0.0
        self.daily_pnl: float = 0.0
        self.trade_count_today: int = 0
        self.last_trade_time: float = 0.0
        self.is_running: bool = False
        
        # Performance tracking
        self.start_balance: float = 0.0
        self.max_balance: float = 0.0
        self.min_balance: float = float('inf')
        self.total_trades: int = 0
        self.winning_trades: int = 0
        
        # Risk management
        self.daily_loss_exceeded: bool = False
        self.drawdown_exceeded: bool = False
        self.balance_below_minimum: bool = False
        
    async def initialize_engine(self) -> bool:
        """Initialize the HFT engine and validate account requirements."""
        try:
            logger.info("🚀 Initializing High Frequency Trading Engine")
            
            # Get current account information
            account_info = alpaca_client.get_account_info()
            self.account_balance = account_info["equity"]
            self.start_balance = self.account_balance
            self.max_balance = self.account_balance
            self.min_balance = self.account_balance
            
            # Validate minimum balance requirement
            if not hft_settings.validate_account_balance(self.account_balance):
                logger.error(f"❌ Account balance ${self.account_balance:,.2f} below minimum ${hft_settings.minimum_account_balance:,.2f}")
                return False
            
            # Check if account is flagged as PDT (Pattern Day Trader)
            if account_info.get("pattern_day_trader", False):
                logger.info("✅ Account has PDT status - unlimited day trading allowed")
            else:
                logger.warning("⚠️ Account not flagged as PDT - will monitor day trade count")
            
            # Initialize positions from current holdings
            positions = alpaca_client.get_positions()
            for pos in positions:
                if abs(float(pos["qty"])) > 0:  # Only track positions with actual holdings
                    position = HFTPosition(
                        symbol=pos["symbol"],
                        quantity=float(pos["qty"]),
                        entry_price=float(pos["avg_entry_price"]),
                        entry_time=datetime.now(),  # Approximate since we don't have original entry time
                        target_exit_time=datetime.now() + timedelta(minutes=30),  # Default 30 min hold
                        stop_loss=float(pos.get("current_price", 0)) * (1 - hft_settings.stop_loss_percent),
                        take_profit=float(pos.get("current_price", 0)) * (1 + hft_settings.take_profit_percent),
                        unrealized_pnl=float(pos["unrealized_pl"]),
                        position_id=f"existing_{pos['symbol']}"
                    )
                    self.active_positions[pos["symbol"]] = position
            
            logger.info(f"✅ HFT Engine initialized successfully")
            logger.info(f"   Account Balance: ${self.account_balance:,.2f}")
            logger.info(f"   Available Capital: ${hft_settings.get_available_trading_capital(self.account_balance):,.2f}")
            logger.info(f"   Active Positions: {len(self.active_positions)}")
            logger.info(f"   Max Position Size: ${hft_settings.get_max_position_value(self.account_balance):,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize HFT engine: {e}")
            return False
    
    async def start_hft_trading(self, duration_minutes: int = 60) -> None:
        """Start high-frequency trading for specified duration."""
        if not await self.initialize_engine():
            logger.error("Failed to initialize HFT engine")
            return
        
        self.is_running = True
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        logger.info(f"🎯 Starting HFT trading for {duration_minutes} minutes")
        logger.info(f"   Will stop at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            while self.is_running and datetime.now() < end_time:
                # Check if market is open
                if not alpaca_client.is_market_open():
                    logger.info("Market closed, pausing HFT trading")
                    await asyncio.sleep(60)  # Check every minute during market close
                    continue
                
                # Perform trading cycle
                await self._hft_trading_cycle()
                
                # Brief pause between cycles (rate limiting)
                await asyncio.sleep(max(1.0, hft_settings.min_time_between_trades))
                
            logger.info("🏁 HFT trading session completed")
            await self._generate_performance_report()
            
        except KeyboardInterrupt:
            logger.info("🛑 HFT trading stopped by user")
        except Exception as e:
            logger.error(f"❌ HFT trading error: {e}")
        finally:
            self.is_running = False
            await self._cleanup_positions()
    
    async def _hft_trading_cycle(self) -> None:
        """Execute one HFT trading cycle."""
        try:
            # Update account balance and risk checks
            await self._update_account_status()
            
            # Check risk limits
            if self._check_risk_limits():
                logger.warning("⚠️ Risk limits exceeded, pausing trading")
                return
            
            # Generate HFT signals
            signals = await self._generate_hft_signals()
            
            # Process signals and execute trades
            for signal in signals:
                if self._should_execute_signal(signal):
                    await self._execute_hft_signal(signal)
            
            # Manage existing positions
            await self._manage_existing_positions()
            
            # Update performance metrics
            self._update_performance_metrics()
            
        except Exception as e:
            logger.error(f"HFT trading cycle error: {e}")
    
    async def _generate_hft_signals(self) -> List[HFTSignal]:
        """Generate high-frequency trading signals."""
        signals = []
        
        try:
            # Analyze each symbol in our universe
            for symbol in hft_settings.tick_data_symbols[:20]:  # Limit for performance
                try:
                    # Get recent market data
                    market_data = alpaca_client.get_market_data(symbol, timeframe="1Min", limit=10)
                    if market_data.empty:
                        continue
                    
                    # Calculate technical indicators quickly
                    signal = await self._analyze_symbol_for_hft(symbol, market_data)
                    if signal:
                        signals.append(signal)
                        
                except Exception as e:
                    logger.debug(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Sort signals by strength and confidence
            signals.sort(key=lambda x: x.strength * x.confidence, reverse=True)
            
            # Return top signals (limit to prevent overtrading)
            return signals[:5]
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return []
    
    async def _analyze_symbol_for_hft(self, symbol: str, market_data) -> Optional[HFTSignal]:
        """Analyze individual symbol for HFT opportunities."""
        try:
            if len(market_data) < 5:
                return None
            
            current_price = float(market_data.iloc[-1]['close'])
            prices = market_data['close'].values
            volumes = market_data['volume'].values
            
            # Calculate quick technical indicators
            price_change = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
            avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[0]
            volume_spike = volumes[-1] / avg_volume if avg_volume > 0 else 1
            
            # Simple RSI calculation
            gains = np.maximum(np.diff(prices), 0)
            losses = np.maximum(-np.diff(prices), 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.001  # Avoid division by zero
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))
            
            # Momentum signal
            if abs(price_change) > hft_settings.price_momentum_threshold and volume_spike > hft_settings.volume_spike_threshold:
                direction = "buy" if price_change > 0 else "sell"
                strength = min(1.0, abs(price_change) / hft_settings.price_momentum_threshold)
                confidence = min(1.0, volume_spike / hft_settings.volume_spike_threshold * 0.5)
                
                # Calculate targets
                if direction == "buy":
                    price_target = current_price * (1 + hft_settings.take_profit_percent)
                    stop_loss = current_price * (1 - hft_settings.stop_loss_percent)
                else:
                    price_target = current_price * (1 - hft_settings.take_profit_percent)
                    stop_loss = current_price * (1 + hft_settings.stop_loss_percent)
                
                return HFTSignal(
                    symbol=symbol,
                    signal_type="momentum",
                    direction=direction,
                    strength=strength,
                    price_target=price_target,
                    stop_loss=stop_loss,
                    confidence=confidence,
                    time_horizon=np.random.randint(5, 30)  # 5-30 minute hold
                )
            
            # Mean reversion signal (RSI-based)
            elif rsi < hft_settings.rsi_oversold and price_change < -hft_settings.price_momentum_threshold:
                # Oversold condition - potential buy
                return HFTSignal(
                    symbol=symbol,
                    signal_type="reversal",
                    direction="buy",
                    strength=0.7,
                    price_target=current_price * (1 + hft_settings.take_profit_percent * 0.5),
                    stop_loss=current_price * (1 - hft_settings.stop_loss_percent),
                    confidence=0.6,
                    time_horizon=np.random.randint(10, 45)
                )
            
            elif rsi > hft_settings.rsi_overbought and price_change > hft_settings.price_momentum_threshold:
                # Overbought condition - potential sell
                return HFTSignal(
                    symbol=symbol,
                    signal_type="reversal", 
                    direction="sell",
                    strength=0.7,
                    price_target=current_price * (1 - hft_settings.take_profit_percent * 0.5),
                    stop_loss=current_price * (1 + hft_settings.stop_loss_percent),
                    confidence=0.6,
                    time_horizon=np.random.randint(10, 45)
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Symbol analysis error for {symbol}: {e}")
            return None
    
    def _should_execute_signal(self, signal: HFTSignal) -> bool:
        """Determine if signal should be executed based on risk management."""
        try:
            # Check trade frequency limits
            if self.trade_count_today >= hft_settings.max_trades_per_day:
                return False
            
            # Check time between trades
            current_time = time.time()
            if current_time - self.last_trade_time < hft_settings.min_time_between_trades:
                return False
            
            # Check if we already have a position in this symbol
            if signal.symbol in self.active_positions:
                return False
            
            # Check signal quality
            if signal.confidence < 0.5 or signal.strength < 0.6:
                return False
            
            # Check position size limits
            max_position_value = hft_settings.get_max_position_value(self.account_balance)
            estimated_position_size = max_position_value
            
            if estimated_position_size < hft_settings.min_position_size:
                return False
            
            # Check available capital
            available_capital = hft_settings.get_available_trading_capital(self.account_balance)
            if estimated_position_size > available_capital:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Signal evaluation error: {e}")
            return False
    
    async def _execute_hft_signal(self, signal: HFTSignal) -> bool:
        """Execute HFT signal with advanced order management."""
        try:
            logger.info(f"🎯 Executing HFT {signal.direction} signal for {signal.symbol} "
                       f"(strength: {signal.strength:.2f}, confidence: {signal.confidence:.2f})")
            
            # Calculate position size
            max_position_value = hft_settings.get_max_position_value(self.account_balance)
            
            # Get current price for position sizing
            market_data = alpaca_client.get_market_data(signal.symbol, limit=1)
            if market_data.empty:
                return False
            
            current_price = float(market_data.iloc[-1]['close'])
            quantity = int(max_position_value / current_price)
            
            if quantity < 1:
                logger.warning(f"Position size too small for {signal.symbol}")
                return False
            
            # Create advanced order request
            order_request = AdvancedOrderRequest(
                symbol=signal.symbol,
                quantity=quantity,
                side=signal.direction,
                order_type="limit",  # Use limit orders for better fills
                limit_price=current_price * (1.001 if signal.direction == "buy" else 0.999),  # Small spread
                stop_loss_price=signal.stop_loss,
                take_profit_price=signal.price_target,
                time_in_force="day",
                reasoning=f"HFT {signal.signal_type} signal",
                confidence=signal.confidence,
                agent_source="hft_engine"
            )
            
            # Execute order
            result = await advanced_order_manager.place_advanced_order(order_request)
            
            if result.success:
                # Track position
                position = HFTPosition(
                    symbol=signal.symbol,
                    quantity=quantity if signal.direction == "buy" else -quantity,
                    entry_price=current_price,
                    entry_time=datetime.now(),
                    target_exit_time=datetime.now() + timedelta(minutes=signal.time_horizon),
                    stop_loss=signal.stop_loss,
                    take_profit=signal.price_target,
                    position_id=result.order_id
                )
                
                self.active_positions[signal.symbol] = position
                
                # Update trade tracking
                self.trade_count_today += 1
                self.last_trade_time = time.time()
                self.total_trades += 1
                
                logger.info(f"✅ HFT order executed: {result.order_id}")
                return True
            else:
                logger.warning(f"❌ HFT order failed: {result.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"HFT signal execution error: {e}")
            return False
    
    async def _manage_existing_positions(self) -> None:
        """Manage existing HFT positions."""
        try:
            positions_to_close = []
            
            for symbol, position in self.active_positions.items():
                # Check if position should be closed
                current_time = datetime.now()
                
                # Time-based exit
                if current_time >= position.target_exit_time:
                    positions_to_close.append(symbol)
                    continue
                
                # Update unrealized P&L
                try:
                    market_data = alpaca_client.get_market_data(symbol, limit=1)
                    if not market_data.empty:
                        current_price = float(market_data.iloc[-1]['close'])
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                        
                        # Check stop loss and take profit
                        if position.quantity > 0:  # Long position
                            if current_price <= position.stop_loss or current_price >= position.take_profit:
                                positions_to_close.append(symbol)
                        else:  # Short position
                            if current_price >= position.stop_loss or current_price <= position.take_profit:
                                positions_to_close.append(symbol)
                                
                except Exception as e:
                    logger.debug(f"Error updating position {symbol}: {e}")
            
            # Close positions that need to be closed
            for symbol in positions_to_close:
                await self._close_hft_position(symbol)
                
        except Exception as e:
            logger.error(f"Position management error: {e}")
    
    async def _close_hft_position(self, symbol: str) -> bool:
        """Close HFT position."""
        try:
            if symbol not in self.active_positions:
                return False
            
            position = self.active_positions[symbol]
            
            # Place closing order
            side = "sell" if position.quantity > 0 else "buy"
            quantity = abs(position.quantity)
            
            result = alpaca_client.place_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                order_type="market"  # Market order for quick exit
            )
            
            if result:
                # Update performance tracking
                if position.unrealized_pnl > 0:
                    self.winning_trades += 1
                
                self.daily_pnl += position.unrealized_pnl
                
                logger.info(f"🏁 Closed HFT position {symbol}: P&L ${position.unrealized_pnl:.2f}")
                
                # Remove from active positions
                del self.active_positions[symbol]
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    async def _update_account_status(self) -> None:
        """Update account balance and status."""
        try:
            account_info = alpaca_client.get_account_info()
            self.account_balance = account_info["equity"]
            
            # Update min/max tracking
            self.max_balance = max(self.max_balance, self.account_balance)
            self.min_balance = min(self.min_balance, self.account_balance)
            
            # Check minimum balance requirement
            self.balance_below_minimum = not hft_settings.validate_account_balance(self.account_balance)
            
        except Exception as e:
            logger.error(f"Account status update error: {e}")
    
    def _check_risk_limits(self) -> bool:
        """Check if risk limits have been exceeded."""
        # Check minimum balance
        if self.balance_below_minimum:
            logger.error(f"❌ Account balance ${self.account_balance:,.2f} below minimum ${hft_settings.minimum_account_balance:,.2f}")
            return True
        
        # Check daily loss limit
        daily_loss_pct = (self.start_balance - self.account_balance) / self.start_balance
        if daily_loss_pct > hft_settings.max_daily_loss:
            self.daily_loss_exceeded = True
            logger.error(f"❌ Daily loss limit exceeded: {daily_loss_pct:.2%}")
            return True
        
        # Check maximum drawdown
        drawdown = (self.max_balance - self.account_balance) / self.max_balance
        if drawdown > hft_settings.max_drawdown:
            self.drawdown_exceeded = True
            logger.error(f"❌ Maximum drawdown exceeded: {drawdown:.2%}")
            return True
        
        return False
    
    def _update_performance_metrics(self) -> None:
        """Update performance tracking metrics."""
        try:
            # Calculate total unrealized P&L
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            
            # Update daily P&L
            self.daily_pnl = (self.account_balance - self.start_balance) + total_unrealized_pnl
            
        except Exception as e:
            logger.error(f"Performance metrics update error: {e}")
    
    async def _cleanup_positions(self) -> None:
        """Clean up all positions at end of session."""
        logger.info("🧹 Cleaning up HFT positions")
        
        for symbol in list(self.active_positions.keys()):
            await self._close_hft_position(symbol)
    
    async def _generate_performance_report(self) -> None:
        """Generate HFT performance report."""
        try:
            win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
            total_return = (self.account_balance - self.start_balance) / self.start_balance * 100
            max_drawdown = (self.max_balance - self.min_balance) / self.max_balance * 100
            
            logger.info("📊 HFT Performance Report")
            logger.info("=" * 50)
            logger.info(f"Starting Balance: ${self.start_balance:,.2f}")
            logger.info(f"Ending Balance:   ${self.account_balance:,.2f}")
            logger.info(f"Total Return:     {total_return:+.2f}%")
            logger.info(f"Daily P&L:        ${self.daily_pnl:+,.2f}")
            logger.info(f"Max Drawdown:     {max_drawdown:.2f}%")
            logger.info(f"Total Trades:     {self.total_trades}")
            logger.info(f"Winning Trades:   {self.winning_trades}")
            logger.info(f"Win Rate:         {win_rate:.1f}%")
            logger.info(f"Active Positions: {len(self.active_positions)}")
            
        except Exception as e:
            logger.error(f"Performance report error: {e}")

# Global HFT engine instance
hft_engine = HighFrequencyTradingEngine()