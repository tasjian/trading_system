#!/usr/bin/env python3
"""
Portfolio Monitoring and Metrics System

MLOps framework for comprehensive portfolio monitoring, order flow tracking,
and real-time risk assessment for the trading system.

Features:
- Real-time portfolio balance monitoring
- Order type distribution analysis  
- Position size and exposure tracking
- Long/short ratio monitoring
- Sector allocation analysis
- Performance metrics calculation
- Risk-adjusted returns analysis
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import json
from pathlib import Path

from tools.alpaca_client import alpaca_client
from config.settings import settings

logger = logging.getLogger(__name__)

class PortfolioHealth(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class OrderMetrics:
    """Order execution metrics."""
    total_orders: int = 0
    buy_orders: int = 0
    sell_orders: int = 0
    short_orders: int = 0
    limit_orders: int = 0
    market_orders: int = 0
    stop_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    avg_fill_time_seconds: float = 0.0
    avg_slippage_bps: float = 0.0
    fill_rate: float = 0.0
    
@dataclass
class PositionMetrics:
    """Position-level metrics."""
    symbol: str
    quantity: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    avg_cost: float
    current_price: float
    position_type: str  # long, short
    sector: Optional[str] = None
    industry: Optional[str] = None
    days_held: int = 0
    risk_score: float = 0.0

@dataclass
class PortfolioSnapshot:
    """Complete portfolio snapshot."""
    timestamp: datetime
    total_value: float
    cash: float
    equity: float
    buying_power: float
    day_pnl: float
    total_pnl: float
    positions: List[PositionMetrics]
    long_exposure: float
    short_exposure: float
    net_exposure: float
    gross_exposure: float
    sector_allocation: Dict[str, float]
    position_count: int
    avg_position_size: float
    max_position_size: float
    concentration_risk: float
    health_score: float
    health_status: PortfolioHealth

class PortfolioMonitor:
    """Comprehensive portfolio monitoring system."""
    
    def __init__(self, history_window_hours: int = 24):
        """Initialize portfolio monitor.
        
        Args:
            history_window_hours: Hours of history to maintain in memory
        """
        self.history_window = timedelta(hours=history_window_hours)
        self.snapshots: deque = deque(maxlen=1000)  # Keep last 1000 snapshots
        self.order_history: List[Dict] = []
        self.alerts: List[Dict] = []
        
        # Performance tracking
        self.performance_metrics = {
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_return": 0.0,
            "volatility": 0.0
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            "max_position_size": settings.max_position_size,
            "max_sector_concentration": 0.30,  # 30% max in any sector
            "max_drawdown": 0.15,  # 15% max drawdown
            "min_diversification": 5,  # Minimum 5 positions
            "max_net_exposure": 0.90,  # 90% max net exposure
            "max_gross_exposure": 1.50,  # 150% max gross exposure (with shorts)
            "min_cash_buffer": 0.05,  # 5% minimum cash
        }
        
        # Data persistence
        self.data_dir = Path("data/monitoring")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    async def capture_portfolio_snapshot(self) -> PortfolioSnapshot:
        """Capture complete portfolio snapshot."""
        try:
            logger.debug("Capturing portfolio snapshot...")
            
            # Get account information
            account_info = alpaca_client.get_account_info()
            positions_data = alpaca_client.get_positions()
            
            # Calculate basic metrics
            total_value = float(account_info.get('portfolio_value', 0))
            cash = float(account_info.get('cash', 0))
            equity = float(account_info.get('equity', 0))
            buying_power = float(account_info.get('buying_power', 0))
            day_pnl = float(account_info.get('unrealized_pl', 0))
            
            # Process positions
            positions = []
            long_exposure = 0.0
            short_exposure = 0.0
            sector_allocation = defaultdict(float)
            
            for pos_data in positions_data:
                position = await self._create_position_metrics(pos_data, total_value)
                positions.append(position)
                
                # Calculate exposures
                if position.quantity > 0:
                    long_exposure += abs(position.market_value)
                else:
                    short_exposure += abs(position.market_value)
                
                # Sector allocation (simplified - would need market data API for real sectors)
                sector = position.sector or "Unknown"
                sector_allocation[sector] += abs(position.market_value) / total_value if total_value > 0 else 0
            
            # Calculate derived metrics
            net_exposure = (long_exposure - short_exposure) / total_value if total_value > 0 else 0
            gross_exposure = (long_exposure + short_exposure) / total_value if total_value > 0 else 0
            
            position_count = len(positions)
            avg_position_size = gross_exposure / position_count if position_count > 0 else 0
            max_position_size = max([abs(p.market_value) / total_value for p in positions]) if positions and total_value > 0 else 0
            
            # Calculate concentration risk (Herfindahl index)
            concentration_risk = sum([(abs(p.market_value) / total_value) ** 2 for p in positions]) if positions and total_value > 0 else 0
            
            # Calculate health score and status
            health_score, health_status = self._calculate_portfolio_health(
                positions, net_exposure, gross_exposure, concentration_risk, 
                cash / total_value if total_value > 0 else 1, sector_allocation
            )
            
            # Calculate total PnL
            total_pnl = sum([p.unrealized_pnl for p in positions])
            
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now(),
                total_value=total_value,
                cash=cash,
                equity=equity,
                buying_power=buying_power,
                day_pnl=day_pnl,
                total_pnl=total_pnl,
                positions=positions,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                net_exposure=net_exposure,
                gross_exposure=gross_exposure,
                sector_allocation=dict(sector_allocation),
                position_count=position_count,
                avg_position_size=avg_position_size,
                max_position_size=max_position_size,
                concentration_risk=concentration_risk,
                health_score=health_score,
                health_status=health_status
            )
            
            # Store snapshot
            self.snapshots.append(snapshot)
            self._cleanup_old_snapshots()
            
            logger.debug(f"Portfolio snapshot captured: {health_status.value} health, "
                        f"{position_count} positions, {net_exposure:.1%} net exposure")
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error capturing portfolio snapshot: {e}")
            # Return empty snapshot on error
            return PortfolioSnapshot(
                timestamp=datetime.now(),
                total_value=0.0, cash=0.0, equity=0.0, buying_power=0.0,
                day_pnl=0.0, total_pnl=0.0, positions=[],
                long_exposure=0.0, short_exposure=0.0, net_exposure=0.0, gross_exposure=0.0,
                sector_allocation={}, position_count=0, avg_position_size=0.0,
                max_position_size=0.0, concentration_risk=0.0,
                health_score=0.0, health_status=PortfolioHealth.CRITICAL
            )
    
    async def _create_position_metrics(self, pos_data: Dict, portfolio_value: float) -> PositionMetrics:
        """Create position metrics from position data."""
        try:
            symbol = pos_data['symbol']
            quantity = float(pos_data['qty'])
            market_value = float(pos_data['market_value'])
            unrealized_pnl = float(pos_data['unrealized_pl'])
            avg_cost = float(pos_data['cost_basis']) / abs(quantity) if quantity != 0 else 0
            current_price = float(pos_data['current_price'])
            
            # Calculate unrealized PnL percentage
            cost_basis = float(pos_data['cost_basis'])
            unrealized_pnl_pct = (unrealized_pnl / abs(cost_basis)) if cost_basis != 0 else 0
            
            # Determine position type
            position_type = "long" if quantity > 0 else "short"
            
            # Simplified sector/industry classification (would need market data API)
            sector, industry = await self._get_sector_industry(symbol)
            
            # Calculate risk score (simplified)
            position_weight = abs(market_value) / portfolio_value if portfolio_value > 0 else 0
            volatility_risk = min(1.0, position_weight / self.risk_thresholds["max_position_size"])
            concentration_risk = position_weight
            risk_score = (volatility_risk + concentration_risk) / 2
            
            return PositionMetrics(
                symbol=symbol,
                quantity=quantity,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                avg_cost=avg_cost,
                current_price=current_price,
                position_type=position_type,
                sector=sector,
                industry=industry,
                days_held=0,  # Would need order history to calculate
                risk_score=risk_score
            )
            
        except Exception as e:
            logger.error(f"Error creating position metrics for {pos_data.get('symbol', 'Unknown')}: {e}")
            return PositionMetrics(
                symbol=pos_data.get('symbol', 'Unknown'),
                quantity=0, market_value=0, unrealized_pnl=0, unrealized_pnl_pct=0,
                avg_cost=0, current_price=0, position_type="long"
            )
    
    async def _get_sector_industry(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """Get sector and industry for symbol (simplified)."""
        # This is a simplified implementation
        # In production, you'd use a market data API like Alpha Vantage, IEX, etc.
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX']
        finance_symbols = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C']
        healthcare_symbols = ['JNJ', 'PFE', 'UNH', 'ABBV', 'BMY', 'MRK']
        
        if symbol in tech_symbols:
            return "Technology", "Software"
        elif symbol in finance_symbols:
            return "Financial Services", "Banks"
        elif symbol in healthcare_symbols:
            return "Healthcare", "Pharmaceuticals"
        else:
            return None, None
    
    def _calculate_portfolio_health(self, positions: List[PositionMetrics], 
                                  net_exposure: float, gross_exposure: float,
                                  concentration_risk: float, cash_ratio: float,
                                  sector_allocation: Dict[str, float]) -> Tuple[float, PortfolioHealth]:
        """Calculate portfolio health score and status."""
        try:
            scores = []
            
            # Diversification score (0-25 points)
            position_count = len(positions)
            if position_count >= 10:
                diversification_score = 25
            elif position_count >= 5:
                diversification_score = 15 + (position_count - 5) * 2
            elif position_count >= 2:
                diversification_score = 5 + (position_count - 2) * 3
            else:
                diversification_score = 0
            scores.append(diversification_score)
            
            # Concentration score (0-25 points)
            max_sector_concentration = max(sector_allocation.values()) if sector_allocation else 0
            if max_sector_concentration <= 0.20:
                concentration_score = 25
            elif max_sector_concentration <= 0.30:
                concentration_score = 15
            elif max_sector_concentration <= 0.50:
                concentration_score = 10
            else:
                concentration_score = 0
            scores.append(concentration_score)
            
            # Exposure score (0-25 points)
            if abs(net_exposure) <= 0.80 and gross_exposure <= 1.20:
                exposure_score = 25
            elif abs(net_exposure) <= 0.90 and gross_exposure <= 1.50:
                exposure_score = 15
            elif abs(net_exposure) <= 1.00 and gross_exposure <= 2.00:
                exposure_score = 10
            else:
                exposure_score = 0
            scores.append(exposure_score)
            
            # Cash management score (0-25 points)
            if cash_ratio >= 0.10:
                cash_score = 25
            elif cash_ratio >= 0.05:
                cash_score = 15
            elif cash_ratio >= 0.02:
                cash_score = 10
            else:
                cash_score = 0
            scores.append(cash_score)
            
            # Calculate overall health score
            health_score = sum(scores)
            
            # Determine health status
            if health_score >= 80:
                health_status = PortfolioHealth.EXCELLENT
            elif health_score >= 60:
                health_status = PortfolioHealth.GOOD
            elif health_score >= 40:
                health_status = PortfolioHealth.WARNING
            else:
                health_status = PortfolioHealth.CRITICAL
            
            return health_score, health_status
            
        except Exception as e:
            logger.error(f"Error calculating portfolio health: {e}")
            return 0.0, PortfolioHealth.CRITICAL
    
    def track_order_execution(self, order_data: Dict):
        """Track order execution for metrics."""
        try:
            order_data['tracked_timestamp'] = datetime.now().isoformat()
            self.order_history.append(order_data)
            
            # Keep only recent order history
            cutoff_time = datetime.now() - timedelta(days=7)
            self.order_history = [
                order for order in self.order_history 
                if datetime.fromisoformat(order['tracked_timestamp']) > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error tracking order execution: {e}")
    
    def calculate_order_metrics(self, time_window_hours: int = 24) -> OrderMetrics:
        """Calculate order execution metrics."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            recent_orders = [
                order for order in self.order_history
                if datetime.fromisoformat(order['tracked_timestamp']) > cutoff_time
            ]
            
            if not recent_orders:
                return OrderMetrics()
            
            # Count order types
            total_orders = len(recent_orders)
            buy_orders = len([o for o in recent_orders if o.get('side') == 'buy'])
            sell_orders = len([o for o in recent_orders if o.get('side') == 'sell'])
            short_orders = len([o for o in recent_orders if o.get('side') == 'sell_short'])
            
            # Count order execution types
            limit_orders = len([o for o in recent_orders if o.get('type') == 'limit'])
            market_orders = len([o for o in recent_orders if o.get('type') == 'market'])
            stop_orders = len([o for o in recent_orders if o.get('type') in ['stop', 'stop_limit']])
            
            # Count order statuses
            filled_orders = len([o for o in recent_orders if o.get('status') == 'filled'])
            cancelled_orders = len([o for o in recent_orders if o.get('status') == 'cancelled'])
            rejected_orders = len([o for o in recent_orders if o.get('status') == 'rejected'])
            
            # Calculate fill rate
            fill_rate = filled_orders / total_orders if total_orders > 0 else 0
            
            # Calculate average fill time and slippage (simplified)
            filled_order_data = [o for o in recent_orders if o.get('status') == 'filled']
            avg_fill_time = 0.0  # Would need order timestamps to calculate
            avg_slippage = 0.0   # Would need execution price vs limit price
            
            return OrderMetrics(
                total_orders=total_orders,
                buy_orders=buy_orders,
                sell_orders=sell_orders,
                short_orders=short_orders,
                limit_orders=limit_orders,
                market_orders=market_orders,
                stop_orders=stop_orders,
                filled_orders=filled_orders,
                cancelled_orders=cancelled_orders,
                rejected_orders=rejected_orders,
                avg_fill_time_seconds=avg_fill_time,
                avg_slippage_bps=avg_slippage,
                fill_rate=fill_rate
            )
            
        except Exception as e:
            logger.error(f"Error calculating order metrics: {e}")
            return OrderMetrics()
    
    def calculate_performance_metrics(self, time_window_days: int = 30) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        try:
            if len(self.snapshots) < 2:
                return self.performance_metrics
            
            # Get snapshots within time window
            cutoff_time = datetime.now() - timedelta(days=time_window_days)
            recent_snapshots = [
                snapshot for snapshot in self.snapshots 
                if snapshot.timestamp > cutoff_time
            ]
            
            if len(recent_snapshots) < 2:
                recent_snapshots = list(self.snapshots)[-min(30, len(self.snapshots)):]
            
            # Calculate returns
            values = [s.total_value for s in recent_snapshots]
            returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values)) if values[i-1] != 0]
            
            if not returns:
                return self.performance_metrics
            
            # Calculate metrics
            avg_return = np.mean(returns)
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            
            # Max drawdown
            peak_value = values[0]
            max_drawdown = 0
            for value in values:
                if value > peak_value:
                    peak_value = value
                drawdown = (peak_value - value) / peak_value if peak_value > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            # Win rate
            positive_returns = [r for r in returns if r > 0]
            win_rate = len(positive_returns) / len(returns) if returns else 0
            
            # Profit factor
            total_gains = sum([r for r in returns if r > 0])
            total_losses = abs(sum([r for r in returns if r < 0]))
            profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
            
            self.performance_metrics.update({
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_return": avg_return,
                "volatility": volatility
            })
            
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self.performance_metrics
    
    def check_risk_thresholds(self, snapshot: PortfolioSnapshot) -> List[Dict]:
        """Check portfolio against risk thresholds and generate alerts."""
        alerts = []
        
        try:
            # Check position concentration
            if snapshot.max_position_size > self.risk_thresholds["max_position_size"]:
                alerts.append({
                    "type": "position_concentration",
                    "severity": "high",
                    "message": f"Position size {snapshot.max_position_size:.1%} exceeds limit {self.risk_thresholds['max_position_size']:.1%}",
                    "value": snapshot.max_position_size,
                    "threshold": self.risk_thresholds["max_position_size"],
                    "timestamp": datetime.now()
                })
            
            # Check sector concentration
            max_sector_allocation = max(snapshot.sector_allocation.values()) if snapshot.sector_allocation else 0
            if max_sector_allocation > self.risk_thresholds["max_sector_concentration"]:
                alerts.append({
                    "type": "sector_concentration",
                    "severity": "medium",
                    "message": f"Sector concentration {max_sector_allocation:.1%} exceeds limit {self.risk_thresholds['max_sector_concentration']:.1%}",
                    "value": max_sector_allocation,
                    "threshold": self.risk_thresholds["max_sector_concentration"],
                    "timestamp": datetime.now()
                })
            
            # Check exposure limits
            if abs(snapshot.net_exposure) > self.risk_thresholds["max_net_exposure"]:
                alerts.append({
                    "type": "net_exposure",
                    "severity": "high",
                    "message": f"Net exposure {snapshot.net_exposure:.1%} exceeds limit {self.risk_thresholds['max_net_exposure']:.1%}",
                    "value": abs(snapshot.net_exposure),
                    "threshold": self.risk_thresholds["max_net_exposure"],
                    "timestamp": datetime.now()
                })
            
            if snapshot.gross_exposure > self.risk_thresholds["max_gross_exposure"]:
                alerts.append({
                    "type": "gross_exposure",
                    "severity": "medium",
                    "message": f"Gross exposure {snapshot.gross_exposure:.1%} exceeds limit {self.risk_thresholds['max_gross_exposure']:.1%}",
                    "value": snapshot.gross_exposure,
                    "threshold": self.risk_thresholds["max_gross_exposure"],
                    "timestamp": datetime.now()
                })
            
            # Check diversification
            if snapshot.position_count < self.risk_thresholds["min_diversification"]:
                alerts.append({
                    "type": "diversification",
                    "severity": "medium",
                    "message": f"Position count {snapshot.position_count} below minimum {self.risk_thresholds['min_diversification']}",
                    "value": snapshot.position_count,
                    "threshold": self.risk_thresholds["min_diversification"],
                    "timestamp": datetime.now()
                })
            
            # Check cash buffer
            cash_ratio = snapshot.cash / snapshot.total_value if snapshot.total_value > 0 else 1
            if cash_ratio < self.risk_thresholds["min_cash_buffer"]:
                alerts.append({
                    "type": "cash_buffer",
                    "severity": "low",
                    "message": f"Cash ratio {cash_ratio:.1%} below minimum {self.risk_thresholds['min_cash_buffer']:.1%}",
                    "value": cash_ratio,
                    "threshold": self.risk_thresholds["min_cash_buffer"],
                    "timestamp": datetime.now()
                })
            
            # Store alerts
            self.alerts.extend(alerts)
            
            # Keep only recent alerts
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.alerts = [
                alert for alert in self.alerts 
                if alert['timestamp'] > cutoff_time
            ]
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking risk thresholds: {e}")
            return []
    
    def _cleanup_old_snapshots(self):
        """Remove old snapshots beyond the history window."""
        cutoff_time = datetime.now() - self.history_window
        while self.snapshots and self.snapshots[0].timestamp < cutoff_time:
            self.snapshots.popleft()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        try:
            if not self.snapshots:
                return {"error": "No portfolio data available"}
            
            latest_snapshot = self.snapshots[-1]
            order_metrics = self.calculate_order_metrics()
            performance_metrics = self.calculate_performance_metrics()
            recent_alerts = [a for a in self.alerts if a['timestamp'] > datetime.now() - timedelta(hours=1)]
            
            return {
                "timestamp": latest_snapshot.timestamp.isoformat(),
                "portfolio": {
                    "total_value": latest_snapshot.total_value,
                    "cash": latest_snapshot.cash,
                    "equity": latest_snapshot.equity,
                    "day_pnl": latest_snapshot.day_pnl,
                    "total_pnl": latest_snapshot.total_pnl,
                    "position_count": latest_snapshot.position_count,
                    "health_score": latest_snapshot.health_score,
                    "health_status": latest_snapshot.health_status.value
                },
                "exposures": {
                    "long_exposure": latest_snapshot.long_exposure,
                    "short_exposure": latest_snapshot.short_exposure,
                    "net_exposure": latest_snapshot.net_exposure,
                    "gross_exposure": latest_snapshot.gross_exposure,
                    "concentration_risk": latest_snapshot.concentration_risk
                },
                "sector_allocation": latest_snapshot.sector_allocation,
                "order_metrics": {
                    "total_orders_24h": order_metrics.total_orders,
                    "buy_orders": order_metrics.buy_orders,
                    "sell_orders": order_metrics.sell_orders,
                    "short_orders": order_metrics.short_orders,
                    "fill_rate": order_metrics.fill_rate
                },
                "performance": performance_metrics,
                "alerts": {
                    "active_alerts": len(recent_alerts),
                    "recent_alerts": recent_alerts[-5:]  # Last 5 alerts
                },
                "risk_status": {
                    "within_limits": len(recent_alerts) == 0,
                    "alert_count": len(recent_alerts)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return {"error": str(e)}
    
    async def save_snapshot_to_disk(self, snapshot: PortfolioSnapshot):
        """Save portfolio snapshot to disk for persistence."""
        try:
            filename = f"portfolio_snapshot_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.data_dir / filename
            
            # Convert snapshot to serializable format
            snapshot_data = {
                "timestamp": snapshot.timestamp.isoformat(),
                "total_value": snapshot.total_value,
                "cash": snapshot.cash,
                "equity": snapshot.equity,
                "buying_power": snapshot.buying_power,
                "day_pnl": snapshot.day_pnl,
                "total_pnl": snapshot.total_pnl,
                "long_exposure": snapshot.long_exposure,
                "short_exposure": snapshot.short_exposure,
                "net_exposure": snapshot.net_exposure,
                "gross_exposure": snapshot.gross_exposure,
                "sector_allocation": snapshot.sector_allocation,
                "position_count": snapshot.position_count,
                "avg_position_size": snapshot.avg_position_size,
                "max_position_size": snapshot.max_position_size,
                "concentration_risk": snapshot.concentration_risk,
                "health_score": snapshot.health_score,
                "health_status": snapshot.health_status.value,
                "positions": [
                    {
                        "symbol": p.symbol,
                        "quantity": p.quantity,
                        "market_value": p.market_value,
                        "unrealized_pnl": p.unrealized_pnl,
                        "unrealized_pnl_pct": p.unrealized_pnl_pct,
                        "position_type": p.position_type,
                        "sector": p.sector,
                        "industry": p.industry,
                        "risk_score": p.risk_score
                    }
                    for p in snapshot.positions
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(snapshot_data, f, indent=2)
            
            # Clean up old files (keep last 100)
            snapshot_files = sorted(self.data_dir.glob("portfolio_snapshot_*.json"))
            if len(snapshot_files) > 100:
                for old_file in snapshot_files[:-100]:
                    old_file.unlink()
                    
        except Exception as e:
            logger.error(f"Error saving snapshot to disk: {e}")

# Global portfolio monitor instance
portfolio_monitor = PortfolioMonitor()

# Convenience functions
async def get_current_portfolio_snapshot() -> PortfolioSnapshot:
    """Get current portfolio snapshot."""
    return await portfolio_monitor.capture_portfolio_snapshot()

async def get_portfolio_summary() -> Dict[str, Any]:
    """Get portfolio summary."""
    return portfolio_monitor.get_portfolio_summary()

def track_order(order_data: Dict):
    """Track order execution."""
    portfolio_monitor.track_order_execution(order_data)

def get_order_metrics(hours: int = 24) -> OrderMetrics:
    """Get order execution metrics."""
    return portfolio_monitor.calculate_order_metrics(hours)

def get_performance_metrics(days: int = 30) -> Dict[str, float]:
    """Get performance metrics."""
    return portfolio_monitor.calculate_performance_metrics(days)

async def check_portfolio_health() -> Tuple[PortfolioSnapshot, List[Dict]]:
    """Check portfolio health and get alerts."""
    snapshot = await portfolio_monitor.capture_portfolio_snapshot()
    alerts = portfolio_monitor.check_risk_thresholds(snapshot)
    return snapshot, alerts