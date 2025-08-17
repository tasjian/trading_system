#!/usr/bin/env python3
"""
Comprehensive Portfolio Monitoring Framework

Real-time monitoring and alerting for portfolio balance, risk metrics,
and trading system performance with detailed analytics and notifications.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
from pathlib import Path

from core.portfolio_balancer import PositionAction, OrderUrgency
from core.position_manager import PositionStatus, RiskLevel, PositionMetrics
from tools.alpaca_client import alpaca_client
from config.settings import settings

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Portfolio metric types."""
    BALANCE = "balance"
    RISK = "risk"
    PERFORMANCE = "performance"
    EXECUTION = "execution"
    SYSTEM = "system"

@dataclass
class Alert:
    """Portfolio monitoring alert."""
    id: str
    level: AlertLevel
    metric_type: MetricType
    symbol: Optional[str]
    title: str
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class PortfolioHealthMetrics:
    """Comprehensive portfolio health metrics."""
    # Balance metrics
    total_value: float
    cash_balance: float
    invested_value: float
    cash_percentage: float
    
    # Diversification metrics
    position_count: int
    sector_concentration: Dict[str, float]
    max_position_weight: float
    avg_position_weight: float
    diversification_score: float
    
    # Risk metrics
    portfolio_beta: float
    value_at_risk_1day: float
    expected_shortfall: float
    max_drawdown: float
    volatility_30day: float
    sharpe_ratio: float
    
    # Performance metrics
    day_pnl: float
    day_pnl_percent: float
    week_pnl: float
    month_pnl: float
    ytd_pnl: float
    total_return: float
    
    # Trading activity
    daily_trades: int
    avg_trade_size: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # System health
    last_rebalance: Optional[datetime]
    rebalance_frequency: float
    signal_processing_latency: float
    order_fill_rate: float
    api_error_rate: float
    
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class BalanceAlert:
    """Portfolio balance alert details."""
    deviation_type: str  # "overweight", "underweight", "missing"
    current_weight: float
    target_weight: float
    deviation_amount: float
    suggested_action: PositionAction
    urgency: OrderUrgency

class PortfolioMonitor:
    """
    Comprehensive portfolio monitoring system with real-time alerts,
    performance tracking, and risk management notifications.
    """
    
    def __init__(self):
        """Initialize portfolio monitor."""
        
        # Alert thresholds
        self.alert_thresholds = {
            'max_position_weight': 0.15,        # 15% max single position
            'max_sector_weight': 0.25,          # 25% max sector concentration
            'min_diversification_score': 0.6,   # Minimum diversification
            'max_drawdown': 0.10,               # 10% max drawdown alert
            'min_cash_percentage': 0.02,        # 2% minimum cash
            'max_daily_loss': 0.05,             # 5% daily loss alert
            'max_volatility': 0.30,             # 30% annual volatility alert
            'min_sharpe_ratio': 0.5,            # Minimum Sharpe ratio
            'max_beta': 1.5,                    # Maximum portfolio beta
            'min_win_rate': 0.45,               # Minimum win rate
            'max_api_error_rate': 0.05          # 5% API error rate alert
        }
        
        # Performance tracking
        self.metrics_history = []
        self.alerts_history = []
        self.max_history_size = 1000
        
        # Alert management
        self.active_alerts = {}
        self.alert_cooldown = timedelta(minutes=30)  # Prevent alert spam
        
        # Monitoring intervals
        self.monitoring_enabled = True
        self.monitor_interval = 60  # seconds
        
        # Data storage
        self.data_dir = Path("data/monitoring")
        self.data_dir.mkdir(exist_ok=True)
    
    async def get_portfolio_health(self) -> PortfolioHealthMetrics:
        """Get comprehensive portfolio health metrics."""
        
        logger.info("📊 Calculating portfolio health metrics...")
        
        try:
            # Get basic portfolio data
            account_info = alpaca_client.get_account_info()
            positions = alpaca_client.get_positions()
            orders = alpaca_client.get_orders(status="all", limit=100)
            
            # Calculate balance metrics
            total_value = float(account_info.get('portfolio_value', 0))
            cash_balance = float(account_info.get('cash', 0))
            invested_value = total_value - cash_balance
            cash_percentage = cash_balance / total_value if total_value > 0 else 0
            
            # Calculate diversification metrics
            diversification_metrics = await self._calculate_diversification_metrics(positions, total_value)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(positions, account_info)
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(account_info, orders)
            
            # Calculate trading activity metrics
            trading_metrics = await self._calculate_trading_metrics(orders)
            
            # Calculate system health metrics
            system_metrics = await self._calculate_system_health_metrics()
            
            return PortfolioHealthMetrics(
                # Balance
                total_value=total_value,
                cash_balance=cash_balance,
                invested_value=invested_value,
                cash_percentage=cash_percentage,
                
                # Diversification
                position_count=diversification_metrics['position_count'],
                sector_concentration=diversification_metrics['sector_concentration'],
                max_position_weight=diversification_metrics['max_position_weight'],
                avg_position_weight=diversification_metrics['avg_position_weight'],
                diversification_score=diversification_metrics['diversification_score'],
                
                # Risk
                portfolio_beta=risk_metrics['portfolio_beta'],
                value_at_risk_1day=risk_metrics['var_1day'],
                expected_shortfall=risk_metrics['expected_shortfall'],
                max_drawdown=risk_metrics['max_drawdown'],
                volatility_30day=risk_metrics['volatility_30day'],
                sharpe_ratio=risk_metrics['sharpe_ratio'],
                
                # Performance
                day_pnl=performance_metrics['day_pnl'],
                day_pnl_percent=performance_metrics['day_pnl_percent'],
                week_pnl=performance_metrics['week_pnl'],
                month_pnl=performance_metrics['month_pnl'],
                ytd_pnl=performance_metrics['ytd_pnl'],
                total_return=performance_metrics['total_return'],
                
                # Trading
                daily_trades=trading_metrics['daily_trades'],
                avg_trade_size=trading_metrics['avg_trade_size'],
                win_rate=trading_metrics['win_rate'],
                avg_win=trading_metrics['avg_win'],
                avg_loss=trading_metrics['avg_loss'],
                profit_factor=trading_metrics['profit_factor'],
                
                # System
                last_rebalance=system_metrics['last_rebalance'],
                rebalance_frequency=system_metrics['rebalance_frequency'],
                signal_processing_latency=system_metrics['signal_latency'],
                order_fill_rate=system_metrics['fill_rate'],
                api_error_rate=system_metrics['api_error_rate']
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio health: {e}")
            return self._get_empty_health_metrics()
    
    async def check_portfolio_balance(self, 
                                    target_allocation: Dict[str, float],
                                    current_positions: Dict[str, Dict]) -> List[Alert]:
        """Check portfolio balance against target allocation and generate alerts."""
        
        logger.info("⚖️ Checking portfolio balance...")
        
        alerts = []
        
        try:
            # Get portfolio value
            account_info = alpaca_client.get_account_info()
            portfolio_value = float(account_info.get('portfolio_value', 100000))
            
            # Check each target position
            for symbol, target_weight in target_allocation.items():
                current_pos = current_positions.get(symbol, {'quantity': 0, 'market_value': 0})
                current_value = float(current_pos.get('market_value', 0))
                current_weight = current_value / portfolio_value if portfolio_value > 0 else 0
                
                deviation = abs(target_weight - current_weight)
                
                # Generate balance alerts
                if deviation > settings.min_rebalance_threshold:
                    alert_level = self._determine_balance_alert_level(deviation)
                    
                    alert = Alert(
                        id=f"balance_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        level=alert_level,
                        metric_type=MetricType.BALANCE,
                        symbol=symbol,
                        title=f"Portfolio Imbalance: {symbol}",
                        message=f"Position weight deviation: {deviation:.1%} (current: {current_weight:.1%}, target: {target_weight:.1%})",
                        current_value=current_weight,
                        threshold_value=target_weight
                    )
                    
                    alerts.append(alert)
            
            # Check for positions not in target allocation
            for symbol, pos_data in current_positions.items():
                if symbol not in target_allocation and abs(pos_data.get('quantity', 0)) > 0:
                    current_value = float(pos_data.get('market_value', 0))
                    current_weight = current_value / portfolio_value if portfolio_value > 0 else 0
                    
                    if current_weight > 0.01:  # More than 1%
                        alert = Alert(
                            id=f"unwanted_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            level=AlertLevel.WARNING,
                            metric_type=MetricType.BALANCE,
                            symbol=symbol,
                            title=f"Unwanted Position: {symbol}",
                            message=f"Position not in target allocation: {current_weight:.1%} of portfolio",
                            current_value=current_weight,
                            threshold_value=0.0
                        )
                        
                        alerts.append(alert)
            
            logger.info(f"📋 Balance check complete: {len(alerts)} alerts generated")
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to check portfolio balance: {e}")
            return []
    
    async def check_risk_alerts(self, health_metrics: PortfolioHealthMetrics) -> List[Alert]:
        """Check risk metrics and generate alerts."""
        
        logger.info("⚠️ Checking risk alerts...")
        
        alerts = []
        
        # Max position weight alert
        if health_metrics.max_position_weight > self.alert_thresholds['max_position_weight']:
            alerts.append(Alert(
                id=f"risk_position_concentration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.WARNING,
                metric_type=MetricType.RISK,
                symbol=None,
                title="Position Concentration Risk",
                message=f"Maximum position weight: {health_metrics.max_position_weight:.1%}",
                current_value=health_metrics.max_position_weight,
                threshold_value=self.alert_thresholds['max_position_weight']
            ))
        
        # Sector concentration alerts
        for sector, weight in health_metrics.sector_concentration.items():
            if weight > self.alert_thresholds['max_sector_weight']:
                alerts.append(Alert(
                    id=f"risk_sector_{sector}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.WARNING,
                    metric_type=MetricType.RISK,
                    symbol=None,
                    title=f"Sector Concentration: {sector}",
                    message=f"Sector weight: {weight:.1%}",
                    current_value=weight,
                    threshold_value=self.alert_thresholds['max_sector_weight']
                ))
        
        # Diversification alert
        if health_metrics.diversification_score < self.alert_thresholds['min_diversification_score']:
            alerts.append(Alert(
                id=f"risk_diversification_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.WARNING,
                metric_type=MetricType.RISK,
                symbol=None,
                title="Low Diversification",
                message=f"Diversification score: {health_metrics.diversification_score:.2f}",
                current_value=health_metrics.diversification_score,
                threshold_value=self.alert_thresholds['min_diversification_score']
            ))
        
        # Drawdown alert
        if health_metrics.max_drawdown > self.alert_thresholds['max_drawdown']:
            alert_level = AlertLevel.CRITICAL if health_metrics.max_drawdown > 0.15 else AlertLevel.ERROR
            alerts.append(Alert(
                id=f"risk_drawdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                level=alert_level,
                metric_type=MetricType.RISK,
                symbol=None,
                title="High Drawdown",
                message=f"Maximum drawdown: {health_metrics.max_drawdown:.1%}",
                current_value=health_metrics.max_drawdown,
                threshold_value=self.alert_thresholds['max_drawdown']
            ))
        
        # Cash level alert
        if health_metrics.cash_percentage < self.alert_thresholds['min_cash_percentage']:
            alerts.append(Alert(
                id=f"risk_cash_low_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.WARNING,
                metric_type=MetricType.RISK,
                symbol=None,
                title="Low Cash Reserves",
                message=f"Cash percentage: {health_metrics.cash_percentage:.1%}",
                current_value=health_metrics.cash_percentage,
                threshold_value=self.alert_thresholds['min_cash_percentage']
            ))
        
        # Daily loss alert
        if health_metrics.day_pnl_percent < -self.alert_thresholds['max_daily_loss']:
            alert_level = AlertLevel.CRITICAL if health_metrics.day_pnl_percent < -0.10 else AlertLevel.ERROR
            alerts.append(Alert(
                id=f"risk_daily_loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                level=alert_level,
                metric_type=MetricType.RISK,
                symbol=None,
                title="High Daily Loss",
                message=f"Daily P&L: {health_metrics.day_pnl_percent:.1%}",
                current_value=health_metrics.day_pnl_percent,
                threshold_value=-self.alert_thresholds['max_daily_loss']
            ))
        
        # Volatility alert
        if health_metrics.volatility_30day > self.alert_thresholds['max_volatility']:
            alerts.append(Alert(
                id=f"risk_volatility_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.WARNING,
                metric_type=MetricType.RISK,
                symbol=None,
                title="High Volatility",
                message=f"30-day volatility: {health_metrics.volatility_30day:.1%}",
                current_value=health_metrics.volatility_30day,
                threshold_value=self.alert_thresholds['max_volatility']
            ))
        
        logger.info(f"⚠️ Risk check complete: {len(alerts)} alerts generated")
        return alerts
    
    async def check_performance_alerts(self, health_metrics: PortfolioHealthMetrics) -> List[Alert]:
        """Check performance metrics and generate alerts."""
        
        logger.info("📈 Checking performance alerts...")
        
        alerts = []
        
        # Sharpe ratio alert
        if health_metrics.sharpe_ratio < self.alert_thresholds['min_sharpe_ratio']:
            alerts.append(Alert(
                id=f"perf_sharpe_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.WARNING,
                metric_type=MetricType.PERFORMANCE,
                symbol=None,
                title="Low Sharpe Ratio",
                message=f"Sharpe ratio: {health_metrics.sharpe_ratio:.2f}",
                current_value=health_metrics.sharpe_ratio,
                threshold_value=self.alert_thresholds['min_sharpe_ratio']
            ))
        
        # Win rate alert
        if health_metrics.win_rate < self.alert_thresholds['min_win_rate']:
            alerts.append(Alert(
                id=f"perf_winrate_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.WARNING,
                metric_type=MetricType.PERFORMANCE,
                symbol=None,
                title="Low Win Rate",
                message=f"Win rate: {health_metrics.win_rate:.1%}",
                current_value=health_metrics.win_rate,
                threshold_value=self.alert_thresholds['min_win_rate']
            ))
        
        # Beta alert
        if health_metrics.portfolio_beta > self.alert_thresholds['max_beta']:
            alerts.append(Alert(
                id=f"perf_beta_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.WARNING,
                metric_type=MetricType.PERFORMANCE,
                symbol=None,
                title="High Portfolio Beta",
                message=f"Portfolio beta: {health_metrics.portfolio_beta:.2f}",
                current_value=health_metrics.portfolio_beta,
                threshold_value=self.alert_thresholds['max_beta']
            ))
        
        logger.info(f"📈 Performance check complete: {len(alerts)} alerts generated")
        return alerts
    
    async def process_alerts(self, alerts: List[Alert]) -> Dict[str, Any]:
        """Process and manage alerts with deduplication and escalation."""
        
        if not alerts:
            return {'new_alerts': 0, 'escalated_alerts': 0, 'summary': 'No alerts'}
        
        new_alerts = []
        escalated_alerts = []
        
        for alert in alerts:
            # Check for duplicate alerts (same type and symbol within cooldown period)
            alert_key = f"{alert.metric_type.value}_{alert.symbol or 'portfolio'}"
            
            if alert_key in self.active_alerts:
                existing_alert = self.active_alerts[alert_key]
                time_since_last = datetime.now() - existing_alert.timestamp
                
                if time_since_last < self.alert_cooldown:
                    continue  # Skip duplicate within cooldown
                
                # Check for escalation
                if alert.level.value > existing_alert.level.value:
                    escalated_alerts.append(alert)
                    self.active_alerts[alert_key] = alert
            else:
                new_alerts.append(alert)
                self.active_alerts[alert_key] = alert
        
        # Add to history
        self.alerts_history.extend(new_alerts + escalated_alerts)
        
        # Limit history size
        if len(self.alerts_history) > self.max_history_size:
            self.alerts_history = self.alerts_history[-self.max_history_size:]
        
        # Send notifications for critical alerts
        critical_alerts = [a for a in new_alerts + escalated_alerts if a.level == AlertLevel.CRITICAL]
        if critical_alerts:
            await self._send_critical_notifications(critical_alerts)
        
        logger.info(f"🚨 Alert processing: {len(new_alerts)} new, {len(escalated_alerts)} escalated")
        
        return {
            'new_alerts': len(new_alerts),
            'escalated_alerts': len(escalated_alerts),
            'critical_alerts': len(critical_alerts),
            'total_active': len(self.active_alerts),
            'summary': f"{len(new_alerts)} new alerts, {len(critical_alerts)} critical"
        }
    
    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        
        logger.info("📊 Generating monitoring dashboard...")
        
        # Get current health metrics
        health_metrics = await self.get_portfolio_health()
        
        # Get recent alerts
        recent_alerts = self.alerts_history[-20:] if self.alerts_history else []
        active_alerts = list(self.active_alerts.values())
        
        # Calculate alert statistics
        alert_stats = self._calculate_alert_statistics()
        
        # Get performance trends
        performance_trends = await self._get_performance_trends()
        
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'health_metrics': health_metrics,
            'alerts': {
                'active_count': len(active_alerts),
                'critical_count': len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
                'recent_alerts': [self._alert_to_dict(a) for a in recent_alerts],
                'statistics': alert_stats
            },
            'performance': {
                'trends': performance_trends,
                'key_metrics': {
                    'total_return': health_metrics.total_return,
                    'sharpe_ratio': health_metrics.sharpe_ratio,
                    'max_drawdown': health_metrics.max_drawdown,
                    'win_rate': health_metrics.win_rate,
                    'daily_pnl': health_metrics.day_pnl_percent
                }
            },
            'risk': {
                'portfolio_beta': health_metrics.portfolio_beta,
                'var_1day': health_metrics.value_at_risk_1day,
                'volatility': health_metrics.volatility_30day,
                'diversification_score': health_metrics.diversification_score,
                'max_position_weight': health_metrics.max_position_weight
            },
            'system': {
                'monitoring_enabled': self.monitoring_enabled,
                'last_rebalance': health_metrics.last_rebalance.isoformat() if health_metrics.last_rebalance else None,
                'api_error_rate': health_metrics.api_error_rate,
                'order_fill_rate': health_metrics.order_fill_rate
            }
        }
        
        # Save dashboard data
        await self._save_dashboard_data(dashboard)
        
        return dashboard
    
    def _determine_balance_alert_level(self, deviation: float) -> AlertLevel:
        """Determine alert level based on balance deviation."""
        if deviation > 0.20:  # 20%+ deviation
            return AlertLevel.ERROR
        elif deviation > 0.10:  # 10%+ deviation
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO
    
    async def _calculate_diversification_metrics(self, positions: List[Dict], total_value: float) -> Dict:
        """Calculate diversification metrics."""
        
        if not positions or total_value <= 0:
            return {
                'position_count': 0,
                'sector_concentration': {},
                'max_position_weight': 0.0,
                'avg_position_weight': 0.0,
                'diversification_score': 0.0
            }
        
        position_weights = []
        sector_weights = {}
        
        for pos in positions:
            weight = abs(float(pos.get('market_value', 0))) / total_value
            position_weights.append(weight)
            
            # Simplified sector mapping (would use actual sector data in production)
            sector = self._get_symbol_sector(pos.get('symbol', ''))
            if sector not in sector_weights:
                sector_weights[sector] = 0
            sector_weights[sector] += weight
        
        # Calculate diversification score (Herfindahl-Hirschman Index inverse)
        hhi = sum(w**2 for w in position_weights)
        diversification_score = 1 - hhi if hhi < 1 else 0
        
        return {
            'position_count': len(positions),
            'sector_concentration': sector_weights,
            'max_position_weight': max(position_weights) if position_weights else 0,
            'avg_position_weight': np.mean(position_weights) if position_weights else 0,
            'diversification_score': diversification_score
        }
    
    async def _calculate_risk_metrics(self, positions: List[Dict], account_info: Dict) -> Dict:
        """Calculate risk metrics."""
        
        # Simplified risk metrics (would use more sophisticated calculations in production)
        return {
            'portfolio_beta': 1.0,  # Placeholder
            'var_1day': 0.02,      # 2% daily VaR
            'expected_shortfall': 0.03,  # 3% expected shortfall
            'max_drawdown': 0.05,   # 5% max drawdown
            'volatility_30day': 0.15,   # 15% annual volatility
            'sharpe_ratio': 1.2     # Sharpe ratio
        }
    
    async def _calculate_performance_metrics(self, account_info: Dict, orders: List[Dict]) -> Dict:
        """Calculate performance metrics."""
        
        day_pnl = float(account_info.get('daychange', 0))
        portfolio_value = float(account_info.get('portfolio_value', 100000))
        day_pnl_percent = day_pnl / portfolio_value if portfolio_value > 0 else 0
        
        return {
            'day_pnl': day_pnl,
            'day_pnl_percent': day_pnl_percent,
            'week_pnl': 0.0,    # Placeholder
            'month_pnl': 0.0,   # Placeholder
            'ytd_pnl': 0.0,     # Placeholder
            'total_return': 0.0  # Placeholder
        }
    
    async def _calculate_trading_metrics(self, orders: List[Dict]) -> Dict:
        """Calculate trading activity metrics."""
        
        # Filter today's filled orders
        today = datetime.now().date()
        filled_orders = [
            o for o in orders 
            if o.get('status') == 'filled' and 
            o.get('filled_at', '').startswith(today.isoformat())
        ]
        
        if not filled_orders:
            return {
                'daily_trades': 0,
                'avg_trade_size': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        trade_sizes = [abs(float(o.get('filled_qty', 0)) * float(o.get('filled_avg_price', 0))) for o in filled_orders]
        
        return {
            'daily_trades': len(filled_orders),
            'avg_trade_size': np.mean(trade_sizes) if trade_sizes else 0,
            'win_rate': 0.6,    # Placeholder
            'avg_win': 0.0,     # Placeholder
            'avg_loss': 0.0,    # Placeholder
            'profit_factor': 1.5 # Placeholder
        }
    
    async def _calculate_system_health_metrics(self) -> Dict:
        """Calculate system health metrics."""
        
        return {
            'last_rebalance': datetime.now() - timedelta(hours=2),  # Placeholder
            'rebalance_frequency': 4.0,     # Times per day
            'signal_latency': 0.5,          # Seconds
            'fill_rate': 0.95,              # 95% fill rate
            'api_error_rate': 0.02          # 2% error rate
        }
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol (simplified mapping)."""
        
        # Simplified sector mapping
        tech_symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'QQQ', 'XLK']
        finance_symbols = ['JPM', 'BAC', 'GS', 'XLF']
        
        if symbol in tech_symbols:
            return 'Technology'
        elif symbol in finance_symbols:
            return 'Financial'
        else:
            return 'Other'
    
    def _calculate_alert_statistics(self) -> Dict:
        """Calculate alert statistics."""
        
        if not self.alerts_history:
            return {'total': 0, 'by_level': {}, 'by_type': {}}
        
        total_alerts = len(self.alerts_history)
        
        # Count by level
        level_counts = {}
        for alert in self.alerts_history:
            level = alert.level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Count by type
        type_counts = {}
        for alert in self.alerts_history:
            metric_type = alert.metric_type.value
            type_counts[metric_type] = type_counts.get(metric_type, 0) + 1
        
        return {
            'total': total_alerts,
            'by_level': level_counts,
            'by_type': type_counts
        }
    
    async def _get_performance_trends(self) -> Dict:
        """Get performance trends data."""
        
        # Placeholder for performance trends
        return {
            'daily_returns': [],
            'cumulative_returns': [],
            'drawdown_series': [],
            'sharpe_evolution': []
        }
    
    def _alert_to_dict(self, alert: Alert) -> Dict:
        """Convert alert to dictionary."""
        
        return {
            'id': alert.id,
            'level': alert.level.value,
            'metric_type': alert.metric_type.value,
            'symbol': alert.symbol,
            'title': alert.title,
            'message': alert.message,
            'current_value': alert.current_value,
            'threshold_value': alert.threshold_value,
            'timestamp': alert.timestamp.isoformat(),
            'acknowledged': alert.acknowledged,
            'resolved': alert.resolved
        }
    
    async def _send_critical_notifications(self, critical_alerts: List[Alert]):
        """Send notifications for critical alerts."""
        
        try:
            # This would integrate with notification systems
            # For now, just log critical alerts
            for alert in critical_alerts:
                logger.critical(f"🚨 CRITICAL ALERT: {alert.title} - {alert.message}")
        
        except Exception as e:
            logger.error(f"Failed to send critical notifications: {e}")
    
    async def _save_dashboard_data(self, dashboard_data: Dict):
        """Save dashboard data to file."""
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.data_dir / f"dashboard_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            # Keep only last 100 files
            dashboard_files = sorted(self.data_dir.glob("dashboard_*.json"))
            if len(dashboard_files) > 100:
                for old_file in dashboard_files[:-100]:
                    old_file.unlink()
        
        except Exception as e:
            logger.error(f"Failed to save dashboard data: {e}")
    
    def _get_empty_health_metrics(self) -> PortfolioHealthMetrics:
        """Get empty health metrics for error cases."""
        
        return PortfolioHealthMetrics(
            total_value=0, cash_balance=0, invested_value=0, cash_percentage=0,
            position_count=0, sector_concentration={}, max_position_weight=0,
            avg_position_weight=0, diversification_score=0, portfolio_beta=0,
            value_at_risk_1day=0, expected_shortfall=0, max_drawdown=0,
            volatility_30day=0, sharpe_ratio=0, day_pnl=0, day_pnl_percent=0,
            week_pnl=0, month_pnl=0, ytd_pnl=0, total_return=0, daily_trades=0,
            avg_trade_size=0, win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
            last_rebalance=None, rebalance_frequency=0, signal_processing_latency=0,
            order_fill_rate=0, api_error_rate=0
        )

# Global instance
portfolio_monitor = PortfolioMonitor()

# Convenience functions
async def get_portfolio_health() -> PortfolioHealthMetrics:
    """Get current portfolio health metrics."""
    return await portfolio_monitor.get_portfolio_health()

async def check_portfolio_balance(target_allocation: Dict[str, float], 
                                current_positions: Dict[str, Dict]) -> List[Alert]:
    """Check portfolio balance and generate alerts."""
    return await portfolio_monitor.check_portfolio_balance(target_allocation, current_positions)

async def get_monitoring_dashboard() -> Dict[str, Any]:
    """Get comprehensive monitoring dashboard."""
    return await portfolio_monitor.get_monitoring_dashboard()

__all__ = [
    'AlertLevel', 'MetricType', 'Alert', 'PortfolioHealthMetrics', 'BalanceAlert',
    'PortfolioMonitor', 'portfolio_monitor',
    'get_portfolio_health', 'check_portfolio_balance', 'get_monitoring_dashboard'
]