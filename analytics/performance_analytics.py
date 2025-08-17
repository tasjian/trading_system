#!/usr/bin/env python3
"""
Performance Analytics System

MLOps framework for comprehensive performance tracking, order execution analysis,
slippage monitoring, and fill rate optimization for the trading system.

Features:
- Order execution quality metrics
- Slippage analysis and optimization
- Fill rate tracking and improvement
- Performance attribution analysis
- Trade efficiency metrics
- Execution cost analysis
- Market impact assessment
- Real-time performance monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import json
from pathlib import Path

from tools.alpaca_client import alpaca_client
from monitoring.portfolio_monitoring import portfolio_monitor, PortfolioSnapshot
from config.settings import settings

logger = logging.getLogger(__name__)

class ExecutionQuality(Enum):
    EXCELLENT = "excellent"    # < 5 bps slippage, >95% fill rate
    GOOD = "good"             # < 10 bps slippage, >90% fill rate
    FAIR = "fair"             # < 20 bps slippage, >80% fill rate
    POOR = "poor"             # > 20 bps slippage, <80% fill rate

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

@dataclass
class OrderExecution:
    """Individual order execution record."""
    order_id: str
    symbol: str
    side: str  # buy, sell, sell_short
    order_type: OrderType
    quantity_requested: float
    quantity_filled: float
    limit_price: Optional[float]
    stop_price: Optional[float]
    avg_fill_price: float
    benchmark_price: float  # Price at order submission
    submitted_time: datetime
    filled_time: Optional[datetime]
    execution_time_seconds: Optional[float]
    slippage_bps: float
    is_filled: bool
    is_partial_fill: bool
    market_impact_bps: float = 0.0
    venue: str = "alpaca"
    strategy: str = "unknown"
    reasoning: str = ""

@dataclass
class PerformanceMetrics:
    """Performance metrics for a time period."""
    period_start: datetime
    period_end: datetime
    total_orders: int
    filled_orders: int
    partial_fills: int
    cancelled_orders: int
    rejected_orders: int
    fill_rate: float
    avg_execution_time_seconds: float
    avg_slippage_bps: float
    median_slippage_bps: float
    slippage_std_bps: float
    market_impact_bps: float
    execution_costs_bps: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    total_pnl: float
    gross_pnl: float
    net_pnl: float
    trading_volume: float
    turnover_ratio: float
    order_type_distribution: Dict[str, int]
    side_distribution: Dict[str, int]
    execution_quality: ExecutionQuality

@dataclass
class SlippageAnalysis:
    """Detailed slippage analysis."""
    symbol: str
    order_type: OrderType
    side: str
    avg_slippage_bps: float
    median_slippage_bps: float
    slippage_std_bps: float
    worst_slippage_bps: float
    best_slippage_bps: float
    sample_size: int
    market_impact_bps: float
    time_of_day_analysis: Dict[str, float]
    volatility_correlation: float

class PerformanceAnalytics:
    """Comprehensive performance analytics system."""
    
    def __init__(self, history_days: int = 30):
        """Initialize performance analytics system."""
        self.history_days = history_days
        self.executions: deque = deque(maxlen=10000)  # Keep last 10k executions
        self.daily_metrics: Dict[str, PerformanceMetrics] = {}
        
        # Real-time tracking
        self.current_session_orders = []
        self.session_start_time = datetime.now()
        
        # Benchmarking
        self.benchmark_prices: Dict[str, Dict[datetime, float]] = defaultdict(dict)
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        
        # Analysis parameters
        self.slippage_thresholds = {
            'excellent': 5.0,   # bps
            'good': 10.0,
            'fair': 20.0,
            'poor': float('inf')
        }
        
        self.fill_rate_thresholds = {
            'excellent': 0.95,
            'good': 0.90,
            'fair': 0.80,
            'poor': 0.0
        }
        
        # Data persistence
        self.data_dir = Path("data/analytics")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    async def record_order_submission(self, order_data: Dict) -> str:
        """Record order submission for tracking."""
        try:
            order_id = order_data.get('id', str(datetime.now().timestamp()))
            symbol = order_data.get('symbol', '')
            
            # Capture benchmark price at submission
            benchmark_price = alpaca_client.get_current_price(symbol)
            if benchmark_price:
                self.benchmark_prices[symbol][datetime.now()] = benchmark_price
            
            # Add to session tracking
            self.current_session_orders.append({
                'order_id': order_id,
                'symbol': symbol,
                'submission_time': datetime.now(),
                'benchmark_price': benchmark_price,
                'order_data': order_data
            })
            
            logger.debug(f"Recorded order submission: {order_id} for {symbol}")
            return order_id
            
        except Exception as e:
            logger.error(f"Error recording order submission: {e}")
            return ""
    
    async def record_order_execution(self, execution_data: Dict):
        """Record order execution for analysis."""
        try:
            # Find corresponding submission
            order_id = execution_data.get('id', '')
            submission_record = next(
                (order for order in self.current_session_orders 
                 if order['order_id'] == order_id), None
            )
            
            if not submission_record:
                logger.warning(f"No submission record found for order {order_id}")
                return
            
            # Create execution record
            execution = await self._create_execution_record(execution_data, submission_record)
            
            if execution:
                self.executions.append(execution)
                logger.debug(f"Recorded execution: {order_id}, slippage: {execution.slippage_bps:.1f} bps")
            
        except Exception as e:
            logger.error(f"Error recording order execution: {e}")
    
    async def _create_execution_record(self, execution_data: Dict, 
                                     submission_record: Dict) -> Optional[OrderExecution]:
        """Create detailed execution record."""
        try:
            order_id = execution_data.get('id', '')
            symbol = execution_data.get('symbol', '')
            side = execution_data.get('side', '')
            order_type_str = execution_data.get('type', 'market')
            
            # Parse order type
            order_type = OrderType.MARKET
            try:
                order_type = OrderType(order_type_str)
            except ValueError:
                pass
            
            # Quantities and prices
            quantity_requested = float(execution_data.get('qty', 0))
            quantity_filled = float(execution_data.get('filled_qty', 0))
            avg_fill_price = float(execution_data.get('filled_avg_price', 0))
            limit_price = execution_data.get('limit_price')
            stop_price = execution_data.get('stop_price')
            
            if limit_price:
                limit_price = float(limit_price)
            if stop_price:
                stop_price = float(stop_price)
            
            # Times
            submitted_time = submission_record['submission_time']
            filled_time = None
            execution_time_seconds = None
            
            if execution_data.get('filled_at'):
                try:
                    filled_time = pd.to_datetime(execution_data['filled_at'])
                    execution_time_seconds = (filled_time - submitted_time).total_seconds()
                except:
                    pass
            
            # Status
            status = execution_data.get('status', '')
            is_filled = status == 'filled'
            is_partial_fill = is_filled and quantity_filled < quantity_requested
            
            # Benchmark price and slippage calculation
            benchmark_price = submission_record.get('benchmark_price', avg_fill_price)
            slippage_bps = self._calculate_slippage(
                benchmark_price, avg_fill_price, side
            ) if benchmark_price and avg_fill_price else 0.0
            
            # Market impact estimation
            market_impact_bps = await self._estimate_market_impact(
                symbol, quantity_filled, benchmark_price, avg_fill_price, submitted_time
            )
            
            return OrderExecution(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity_requested=quantity_requested,
                quantity_filled=quantity_filled,
                limit_price=limit_price,
                stop_price=stop_price,
                avg_fill_price=avg_fill_price,
                benchmark_price=benchmark_price,
                submitted_time=submitted_time,
                filled_time=filled_time,
                execution_time_seconds=execution_time_seconds,
                slippage_bps=slippage_bps,
                is_filled=is_filled,
                is_partial_fill=is_partial_fill,
                market_impact_bps=market_impact_bps,
                strategy=submission_record.get('order_data', {}).get('reasoning', '')[:50],
                reasoning=submission_record.get('order_data', {}).get('reasoning', '')
            )
            
        except Exception as e:
            logger.error(f"Error creating execution record: {e}")
            return None
    
    def _calculate_slippage(self, benchmark_price: float, fill_price: float, side: str) -> float:
        """Calculate slippage in basis points."""
        try:
            if benchmark_price <= 0:
                return 0.0
            
            if side.lower() in ['buy', 'buy_cover']:
                # For buys, positive slippage means paying more than benchmark
                slippage = (fill_price - benchmark_price) / benchmark_price
            else:  # sell, sell_short
                # For sells, positive slippage means receiving less than benchmark
                slippage = (benchmark_price - fill_price) / benchmark_price
            
            return slippage * 10000  # Convert to basis points
            
        except Exception as e:
            logger.error(f"Error calculating slippage: {e}")
            return 0.0
    
    async def _estimate_market_impact(self, symbol: str, quantity: float, 
                                    benchmark_price: float, fill_price: float,
                                    submission_time: datetime) -> float:
        """Estimate market impact of the order."""
        try:
            # Get post-execution price movement
            await asyncio.sleep(1)  # Brief delay
            post_price = alpaca_client.get_current_price(symbol)
            
            if not post_price or benchmark_price <= 0:
                return 0.0
            
            # Calculate price movement after execution
            price_movement = (post_price - benchmark_price) / benchmark_price * 10000
            
            # Estimate impact based on order size and price movement
            # This is simplified - real implementation would consider:
            # - Order size relative to average volume
            # - Time of day effects
            # - Market conditions
            # - Bid-ask spread
            
            # Get recent volume data
            market_data = await self._get_market_data(symbol, 5)
            if market_data is not None and len(market_data) > 0:
                avg_volume = market_data['volume'].mean()
                volume_ratio = quantity / (avg_volume / 390)  # Intraday volume estimate
                
                # Impact estimation (simplified)
                estimated_impact = min(abs(price_movement) * volume_ratio * 0.1, 50.0)  # Cap at 50 bps
                return estimated_impact
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error estimating market impact for {symbol}: {e}")
            return 0.0
    
    async def _get_market_data(self, symbol: str, days: int = 5) -> Optional[pd.DataFrame]:
        """Get market data with caching."""
        try:
            cache_key = f"{symbol}_{days}d"
            
            # Check cache
            if cache_key in self.market_data_cache:
                data = self.market_data_cache[cache_key]
                # Check if data is recent (within 1 hour)
                if len(data) > 0 and (datetime.now() - data.index[-1]).total_seconds() < 3600:
                    return data
            
            # Fetch new data
            data = alpaca_client.get_market_data(symbol, limit=days)
            if data is not None:
                self.market_data_cache[cache_key] = data
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def calculate_performance_metrics(self, start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics for a period."""
        try:
            if not start_date:
                start_date = datetime.now() - timedelta(days=1)
            if not end_date:
                end_date = datetime.now()
            
            # Filter executions for the period
            period_executions = [
                ex for ex in self.executions
                if start_date <= ex.submitted_time <= end_date
            ]
            
            if not period_executions:
                return self._empty_metrics(start_date, end_date)
            
            # Basic counts
            total_orders = len(period_executions)
            filled_orders = len([ex for ex in period_executions if ex.is_filled])
            partial_fills = len([ex for ex in period_executions if ex.is_partial_fill])
            cancelled_orders = total_orders - filled_orders
            rejected_orders = 0  # Would need order status tracking
            
            # Fill rate
            fill_rate = filled_orders / total_orders if total_orders > 0 else 0
            
            # Execution time metrics
            execution_times = [ex.execution_time_seconds for ex in period_executions 
                             if ex.execution_time_seconds is not None]
            avg_execution_time = np.mean(execution_times) if execution_times else 0
            
            # Slippage metrics
            slippages = [ex.slippage_bps for ex in period_executions if ex.is_filled]
            avg_slippage = np.mean(slippages) if slippages else 0
            median_slippage = np.median(slippages) if slippages else 0
            slippage_std = np.std(slippages) if slippages else 0
            
            # Market impact
            market_impacts = [ex.market_impact_bps for ex in period_executions if ex.is_filled]
            market_impact = np.mean(market_impacts) if market_impacts else 0
            
            # Execution costs (slippage + impact + commissions)
            execution_costs = avg_slippage + market_impact + 1.0  # Assume 1 bp commission
            
            # PnL calculations (simplified)
            filled_executions = [ex for ex in period_executions if ex.is_filled]
            total_pnl, gross_pnl, net_pnl = self._calculate_pnl(filled_executions)
            
            # Trading volume
            trading_volume = sum(ex.quantity_filled * ex.avg_fill_price 
                               for ex in filled_executions)
            
            # Order type and side distribution
            order_type_dist = defaultdict(int)
            side_dist = defaultdict(int)
            
            for ex in period_executions:
                order_type_dist[ex.order_type.value] += 1
                side_dist[ex.side] += 1
            
            # Performance ratios
            sharpe_ratio, win_rate, profit_factor, max_drawdown = self._calculate_performance_ratios(
                filled_executions
            )
            
            # Determine execution quality
            execution_quality = self._determine_execution_quality(avg_slippage, fill_rate)
            
            return PerformanceMetrics(
                period_start=start_date,
                period_end=end_date,
                total_orders=total_orders,
                filled_orders=filled_orders,
                partial_fills=partial_fills,
                cancelled_orders=cancelled_orders,
                rejected_orders=rejected_orders,
                fill_rate=fill_rate,
                avg_execution_time_seconds=avg_execution_time,
                avg_slippage_bps=avg_slippage,
                median_slippage_bps=median_slippage,
                slippage_std_bps=slippage_std,
                market_impact_bps=market_impact,
                execution_costs_bps=execution_costs,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                total_pnl=total_pnl,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                trading_volume=trading_volume,
                turnover_ratio=0.0,  # Would need portfolio value history
                order_type_distribution=dict(order_type_dist),
                side_distribution=dict(side_dist),
                execution_quality=execution_quality
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._empty_metrics(start_date or datetime.now(), 
                                     end_date or datetime.now())
    
    def analyze_slippage_by_symbol(self, symbols: List[str] = None,
                                 days: int = 7) -> Dict[str, SlippageAnalysis]:
        """Analyze slippage by symbol and order characteristics."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_executions = [
                ex for ex in self.executions
                if ex.submitted_time >= cutoff_date and ex.is_filled
            ]
            
            if symbols:
                recent_executions = [ex for ex in recent_executions if ex.symbol in symbols]
            
            analysis_results = {}
            
            # Group by symbol
            symbol_groups = defaultdict(list)
            for ex in recent_executions:
                symbol_groups[ex.symbol].append(ex)
            
            for symbol, executions in symbol_groups.items():
                if len(executions) < 3:  # Need minimum sample size
                    continue
                
                # Overall slippage metrics
                slippages = [ex.slippage_bps for ex in executions]
                market_impacts = [ex.market_impact_bps for ex in executions]
                
                # Time of day analysis
                time_analysis = self._analyze_time_of_day_slippage(executions)
                
                # Volatility correlation (simplified)
                volatility_corr = 0.0  # Would need volatility data
                
                analysis_results[symbol] = SlippageAnalysis(
                    symbol=symbol,
                    order_type=executions[0].order_type,  # Most common
                    side=executions[0].side,  # Most common
                    avg_slippage_bps=np.mean(slippages),
                    median_slippage_bps=np.median(slippages),
                    slippage_std_bps=np.std(slippages),
                    worst_slippage_bps=max(slippages),
                    best_slippage_bps=min(slippages),
                    sample_size=len(executions),
                    market_impact_bps=np.mean(market_impacts),
                    time_of_day_analysis=time_analysis,
                    volatility_correlation=volatility_corr
                )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing slippage by symbol: {e}")
            return {}
    
    def _analyze_time_of_day_slippage(self, executions: List[OrderExecution]) -> Dict[str, float]:
        """Analyze slippage by time of day."""
        try:
            time_buckets = {
                'market_open': [],    # 9:30-10:30
                'morning': [],        # 10:30-12:00
                'midday': [],         # 12:00-14:00
                'afternoon': [],      # 14:00-15:30
                'market_close': []    # 15:30-16:00
            }
            
            for ex in executions:
                hour = ex.submitted_time.hour
                minute = ex.submitted_time.minute
                time_decimal = hour + minute / 60
                
                if 9.5 <= time_decimal < 10.5:
                    time_buckets['market_open'].append(ex.slippage_bps)
                elif 10.5 <= time_decimal < 12:
                    time_buckets['morning'].append(ex.slippage_bps)
                elif 12 <= time_decimal < 14:
                    time_buckets['midday'].append(ex.slippage_bps)
                elif 14 <= time_decimal < 15.5:
                    time_buckets['afternoon'].append(ex.slippage_bps)
                elif 15.5 <= time_decimal < 16:
                    time_buckets['market_close'].append(ex.slippage_bps)
            
            return {
                period: np.mean(slippages) if slippages else 0
                for period, slippages in time_buckets.items()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing time of day slippage: {e}")
            return {}
    
    def _calculate_pnl(self, executions: List[OrderExecution]) -> Tuple[float, float, float]:
        """Calculate PnL from executions (simplified)."""
        try:
            # This is a simplified calculation
            # Real implementation would track position building/unwinding
            
            total_pnl = 0.0
            gross_pnl = 0.0
            commissions = 0.0
            
            # Group by symbol to track position changes
            symbol_trades = defaultdict(list)
            for ex in executions:
                symbol_trades[ex.symbol].append(ex)
            
            for symbol, trades in symbol_trades.items():
                # Calculate realized PnL for completed round trips
                position = 0.0
                avg_cost = 0.0
                symbol_pnl = 0.0
                
                for trade in sorted(trades, key=lambda x: x.submitted_time):
                    if trade.side in ['buy', 'buy_cover']:
                        if position <= 0:  # Opening or covering
                            position += trade.quantity_filled
                            avg_cost = trade.avg_fill_price
                        else:  # Adding to position
                            total_cost = position * avg_cost + trade.quantity_filled * trade.avg_fill_price
                            position += trade.quantity_filled
                            avg_cost = total_cost / position if position > 0 else 0
                    
                    else:  # sell, sell_short
                        if position > 0:  # Closing long
                            closed_qty = min(position, trade.quantity_filled)
                            symbol_pnl += closed_qty * (trade.avg_fill_price - avg_cost)
                            position -= closed_qty
                        else:  # Opening short
                            position -= trade.quantity_filled
                            avg_cost = trade.avg_fill_price
                    
                    # Add commissions (simplified)
                    commissions += trade.quantity_filled * trade.avg_fill_price * 0.0001  # 1 bp
                
                total_pnl += symbol_pnl
                gross_pnl += symbol_pnl
            
            net_pnl = gross_pnl - commissions
            
            return total_pnl, gross_pnl, net_pnl
            
        except Exception as e:
            logger.error(f"Error calculating PnL: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_performance_ratios(self, executions: List[OrderExecution]) -> Tuple[float, float, float, float]:
        """Calculate performance ratios."""
        try:
            if len(executions) < 2:
                return 0.0, 0.0, 0.0, 0.0
            
            # Calculate trade returns (simplified)
            returns = []
            for ex in executions:
                # Simplified return calculation
                if ex.side in ['sell', 'sell_short']:
                    ret = ex.slippage_bps / 10000  # Convert slippage to return proxy
                    returns.append(-ret)  # Negative slippage is positive return
            
            if not returns:
                return 0.0, 0.0, 0.0, 0.0
            
            # Sharpe ratio
            avg_return = np.mean(returns)
            vol = np.std(returns)
            sharpe_ratio = avg_return / vol if vol > 0 else 0
            
            # Win rate
            positive_returns = [r for r in returns if r > 0]
            win_rate = len(positive_returns) / len(returns)
            
            # Profit factor
            total_gains = sum(r for r in returns if r > 0)
            total_losses = abs(sum(r for r in returns if r < 0))
            profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
            
            # Max drawdown (simplified)
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
            
            return sharpe_ratio, win_rate, profit_factor, max_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating performance ratios: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def _determine_execution_quality(self, avg_slippage: float, fill_rate: float) -> ExecutionQuality:
        """Determine overall execution quality."""
        if (avg_slippage <= self.slippage_thresholds['excellent'] and 
            fill_rate >= self.fill_rate_thresholds['excellent']):
            return ExecutionQuality.EXCELLENT
        elif (avg_slippage <= self.slippage_thresholds['good'] and 
              fill_rate >= self.fill_rate_thresholds['good']):
            return ExecutionQuality.GOOD
        elif (avg_slippage <= self.slippage_thresholds['fair'] and 
              fill_rate >= self.fill_rate_thresholds['fair']):
            return ExecutionQuality.FAIR
        else:
            return ExecutionQuality.POOR
    
    def _empty_metrics(self, start_date: datetime, end_date: datetime) -> PerformanceMetrics:
        """Return empty metrics structure."""
        return PerformanceMetrics(
            period_start=start_date,
            period_end=end_date,
            total_orders=0, filled_orders=0, partial_fills=0,
            cancelled_orders=0, rejected_orders=0, fill_rate=0.0,
            avg_execution_time_seconds=0.0, avg_slippage_bps=0.0,
            median_slippage_bps=0.0, slippage_std_bps=0.0,
            market_impact_bps=0.0, execution_costs_bps=0.0,
            sharpe_ratio=0.0, win_rate=0.0, profit_factor=0.0,
            max_drawdown=0.0, total_pnl=0.0, gross_pnl=0.0, net_pnl=0.0,
            trading_volume=0.0, turnover_ratio=0.0,
            order_type_distribution={}, side_distribution={},
            execution_quality=ExecutionQuality.POOR
        )
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics."""
        try:
            # Current session metrics
            session_duration = (datetime.now() - self.session_start_time).total_seconds() / 3600
            session_orders = len(self.current_session_orders)
            
            # Recent performance (last hour)
            recent_metrics = self.calculate_performance_metrics(
                start_date=datetime.now() - timedelta(hours=1)
            )
            
            # Recent executions for quick stats
            recent_executions = [
                ex for ex in self.executions
                if ex.submitted_time > datetime.now() - timedelta(minutes=15)
            ]
            
            return {
                'timestamp': datetime.now().isoformat(),
                'session': {
                    'duration_hours': session_duration,
                    'orders_submitted': session_orders,
                    'start_time': self.session_start_time.isoformat()
                },
                'recent_15min': {
                    'orders': len(recent_executions),
                    'avg_slippage_bps': np.mean([ex.slippage_bps for ex in recent_executions]) if recent_executions else 0,
                    'fill_rate': len([ex for ex in recent_executions if ex.is_filled]) / len(recent_executions) if recent_executions else 0
                },
                'last_hour': {
                    'total_orders': recent_metrics.total_orders,
                    'fill_rate': recent_metrics.fill_rate,
                    'avg_slippage_bps': recent_metrics.avg_slippage_bps,
                    'execution_quality': recent_metrics.execution_quality.value,
                    'total_pnl': recent_metrics.total_pnl
                },
                'execution_quality_breakdown': {
                    quality.value: len([ex for ex in recent_executions 
                                      if self._determine_execution_quality(ex.slippage_bps, 1.0 if ex.is_filled else 0.0) == quality])
                    for quality in ExecutionQuality
                },
                'order_flow': {
                    'buy_orders': len([ex for ex in recent_executions if ex.side in ['buy', 'buy_cover']]),
                    'sell_orders': len([ex for ex in recent_executions if ex.side in ['sell', 'sell_short']]),
                    'market_orders': len([ex for ex in recent_executions if ex.order_type == OrderType.MARKET]),
                    'limit_orders': len([ex for ex in recent_executions if ex.order_type == OrderType.LIMIT])
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return {'error': str(e)}
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            # Calculate metrics for different periods
            daily_metrics = self.calculate_performance_metrics(
                start_date=datetime.now() - timedelta(days=1)
            )
            weekly_metrics = self.calculate_performance_metrics(
                start_date=datetime.now() - timedelta(days=days)
            )
            
            # Slippage analysis
            slippage_analysis = self.analyze_slippage_by_symbol(days=days)
            
            # Best and worst performing symbols
            symbol_performance = {}
            for symbol, analysis in slippage_analysis.items():
                symbol_performance[symbol] = {
                    'avg_slippage_bps': analysis.avg_slippage_bps,
                    'sample_size': analysis.sample_size,
                    'market_impact_bps': analysis.market_impact_bps
                }
            
            # Sort by performance
            best_symbols = sorted(symbol_performance.items(), 
                                key=lambda x: x[1]['avg_slippage_bps'])[:5]
            worst_symbols = sorted(symbol_performance.items(), 
                                 key=lambda x: x[1]['avg_slippage_bps'], reverse=True)[:5]
            
            return {
                'summary_period_days': days,
                'daily_performance': {
                    'total_orders': daily_metrics.total_orders,
                    'fill_rate': daily_metrics.fill_rate,
                    'avg_slippage_bps': daily_metrics.avg_slippage_bps,
                    'execution_quality': daily_metrics.execution_quality.value,
                    'total_pnl': daily_metrics.total_pnl,
                    'trading_volume': daily_metrics.trading_volume
                },
                'weekly_performance': {
                    'total_orders': weekly_metrics.total_orders,
                    'fill_rate': weekly_metrics.fill_rate,
                    'avg_slippage_bps': weekly_metrics.avg_slippage_bps,
                    'execution_quality': weekly_metrics.execution_quality.value,
                    'sharpe_ratio': weekly_metrics.sharpe_ratio,
                    'win_rate': weekly_metrics.win_rate,
                    'max_drawdown': weekly_metrics.max_drawdown,
                    'total_pnl': weekly_metrics.total_pnl
                },
                'order_distribution': {
                    'by_type': weekly_metrics.order_type_distribution,
                    'by_side': weekly_metrics.side_distribution
                },
                'symbol_analysis': {
                    'total_symbols_traded': len(slippage_analysis),
                    'best_execution': [{'symbol': s[0], 'avg_slippage_bps': s[1]['avg_slippage_bps']} 
                                     for s in best_symbols],
                    'worst_execution': [{'symbol': s[0], 'avg_slippage_bps': s[1]['avg_slippage_bps']} 
                                      for s in worst_symbols]
                },
                'recommendations': self._generate_execution_recommendations(weekly_metrics, slippage_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def _generate_execution_recommendations(self, metrics: PerformanceMetrics,
                                          slippage_analysis: Dict[str, SlippageAnalysis]) -> List[str]:
        """Generate execution improvement recommendations."""
        recommendations = []
        
        try:
            # Slippage recommendations
            if metrics.avg_slippage_bps > 15:
                recommendations.append("Consider using more limit orders to reduce slippage")
            
            if metrics.fill_rate < 0.85:
                recommendations.append("Improve fill rates by adjusting limit order pricing")
            
            if metrics.avg_execution_time_seconds > 60:
                recommendations.append("Consider using market orders for time-sensitive trades")
            
            # Symbol-specific recommendations
            high_slippage_symbols = [
                symbol for symbol, analysis in slippage_analysis.items()
                if analysis.avg_slippage_bps > 20 and analysis.sample_size >= 3
            ]
            
            if high_slippage_symbols:
                recommendations.append(
                    f"Review execution strategy for symbols with high slippage: {', '.join(high_slippage_symbols[:3])}"
                )
            
            # Order type recommendations
            if metrics.order_type_distribution.get('market', 0) > metrics.total_orders * 0.7:
                recommendations.append("Consider more limit orders to reduce execution costs")
            
            if metrics.execution_quality == ExecutionQuality.POOR:
                recommendations.append("Review overall execution strategy - performance is below optimal")
            
            if not recommendations:
                recommendations.append("Execution performance is good - maintain current strategy")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to data error"]
    
    async def save_daily_metrics(self):
        """Save daily metrics to disk."""
        try:
            today = datetime.now().date()
            daily_metrics = self.calculate_performance_metrics(
                start_date=datetime.combine(today, datetime.min.time()),
                end_date=datetime.now()
            )
            
            self.daily_metrics[today.isoformat()] = daily_metrics
            
            # Save to file
            metrics_file = self.data_dir / f"daily_metrics_{today.isoformat()}.json"
            
            # Convert to serializable format
            metrics_data = {
                'date': today.isoformat(),
                'total_orders': daily_metrics.total_orders,
                'fill_rate': daily_metrics.fill_rate,
                'avg_slippage_bps': daily_metrics.avg_slippage_bps,
                'execution_quality': daily_metrics.execution_quality.value,
                'total_pnl': daily_metrics.total_pnl,
                'trading_volume': daily_metrics.trading_volume,
                'order_type_distribution': daily_metrics.order_type_distribution,
                'side_distribution': daily_metrics.side_distribution
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info(f"Saved daily metrics for {today}")
            
        except Exception as e:
            logger.error(f"Error saving daily metrics: {e}")

# Global performance analytics instance
performance_analytics = PerformanceAnalytics()

# Convenience functions
async def record_order_submission(order_data: Dict) -> str:
    """Record order submission."""
    return await performance_analytics.record_order_submission(order_data)

async def record_order_execution(execution_data: Dict):
    """Record order execution."""
    await performance_analytics.record_order_execution(execution_data)

def get_performance_metrics(days: int = 1) -> PerformanceMetrics:
    """Get performance metrics for period."""
    return performance_analytics.calculate_performance_metrics(
        start_date=datetime.now() - timedelta(days=days)
    )

def get_real_time_performance() -> Dict[str, Any]:
    """Get real-time performance metrics."""
    return performance_analytics.get_real_time_metrics()

def get_performance_summary(days: int = 7) -> Dict[str, Any]:
    """Get comprehensive performance summary."""
    return performance_analytics.get_performance_summary(days)

def analyze_symbol_slippage(symbols: List[str] = None, days: int = 7) -> Dict[str, SlippageAnalysis]:
    """Analyze slippage by symbol."""
    return performance_analytics.analyze_slippage_by_symbol(symbols, days)