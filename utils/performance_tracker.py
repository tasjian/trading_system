#!/usr/bin/env python3
"""
Portfolio-level Performance Tracking for RL Curriculum Training
Comprehensive metrics collection and evaluation for staged learning
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    """Types of performance metrics to track."""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    VAR_95 = "var_95"
    CVAR_95 = "cvar_95"

@dataclass
class TradeRecord:
    """Individual trade record for performance analysis."""
    timestamp: datetime
    symbol: str
    action: str  # "BUY", "SELL", "SHORT", "COVER"
    quantity: float
    price: float
    value: float
    
    # Reward components
    realized_return: float = 0.0
    unrealized_return: float = 0.0
    reward_total: float = 0.0
    reward_profit: float = 0.0
    reward_loss_cutting: float = 0.0
    reward_signal_alignment: float = 0.0
    reward_transaction_cost: float = 0.0
    
    # Market context
    signal_confidence: float = 0.0
    signal_direction: float = 0.0
    volatility: float = 0.0
    sentiment_score: float = 0.0
    
    # Portfolio context
    portfolio_value_before: float = 0.0
    portfolio_value_after: float = 0.0
    cash_before: float = 0.0
    cash_after: float = 0.0
    
    # Performance attribution
    is_profitable: bool = False
    holding_period_days: Optional[int] = None
    
@dataclass
class PerformanceSnapshot:
    """Portfolio performance snapshot at a point in time."""
    timestamp: datetime
    portfolio_value: float
    cash: float
    positions_value: float
    total_return: float
    unrealized_pnl: float
    realized_pnl: float
    
    # Risk metrics
    volatility: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    
    # Performance metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # RL-specific metrics
    avg_episode_reward: float = 0.0
    learning_progress: float = 0.0
    exploration_rate: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    profitable_trades: int = 0
    loss_making_trades: int = 0

class PerformanceTracker:
    """
    Comprehensive performance tracking for RL curriculum training.
    
    Tracks portfolio performance, trade execution, reward components,
    and learning progress across curriculum phases.
    """
    
    def __init__(self, initial_portfolio_value: float = 100000.0,
                 risk_free_rate: float = 0.02,
                 benchmark_return: float = 0.08,
                 tracking_window_days: int = 252):
        """
        Initialize performance tracker.
        
        Args:
            initial_portfolio_value: Starting portfolio value
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            benchmark_return: Benchmark return for comparison
            tracking_window_days: Rolling window for performance calculations
        """
        self.initial_portfolio_value = initial_portfolio_value
        self.risk_free_rate = risk_free_rate
        self.benchmark_return = benchmark_return
        self.tracking_window_days = tracking_window_days
        
        # Trade and performance history
        self.trade_records: List[TradeRecord] = []
        self.performance_snapshots: List[PerformanceSnapshot] = []
        self.daily_returns: List[float] = []
        self.portfolio_values: List[float] = [initial_portfolio_value]
        
        # Real-time tracking
        self.current_portfolio_value = initial_portfolio_value
        self.current_cash = initial_portfolio_value
        self.current_positions_value = 0.0
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        
        # Episode tracking for RL
        self.episode_rewards: List[float] = []
        self.episode_returns: List[float] = []
        self.episode_trades: List[int] = []
        self.current_episode_trades = 0
        
        # Performance metrics cache
        self.cached_metrics: Dict[str, float] = {}
        self.last_metrics_update = datetime.min
        
        logger.info(f"Initialized PerformanceTracker with ${initial_portfolio_value:,.2f}")
    
    def record_trade(self, trade: TradeRecord):
        """Record a trade execution."""
        
        # Add trade to history
        self.trade_records.append(trade)
        self.current_episode_trades += 1
        
        # Update portfolio values
        self.current_portfolio_value = trade.portfolio_value_after
        self.current_cash = trade.cash_after
        self.current_positions_value = self.current_portfolio_value - self.current_cash
        
        # Update P&L
        if trade.realized_return != 0:
            self.total_realized_pnl += trade.realized_return * trade.value
        
        # Add to portfolio value history
        self.portfolio_values.append(self.current_portfolio_value)
        
        # Calculate daily return
        if len(self.portfolio_values) > 1:
            daily_return = (self.current_portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.daily_returns.append(daily_return)
        
        # Log significant trades
        if abs(trade.value) > self.initial_portfolio_value * 0.01:  # > 1% of initial value
            logger.info(f"Recorded {trade.action} trade: {trade.symbol} ${trade.value:,.2f}, "
                       f"Portfolio: ${self.current_portfolio_value:,.2f}")
    
    def record_episode_end(self, episode_reward: float, episode_return: float):
        """Record end of RL training episode."""
        
        self.episode_rewards.append(episode_reward)
        self.episode_returns.append(episode_return)
        self.episode_trades.append(self.current_episode_trades)
        
        # Reset episode counters
        self.current_episode_trades = 0
        
        # Create performance snapshot
        snapshot = self.create_performance_snapshot()
        self.performance_snapshots.append(snapshot)
        
        logger.debug(f"Episode ended: Reward={episode_reward:.4f}, Return={episode_return:.4f}, "
                    f"Trades={self.episode_trades[-1]}")
    
    def create_performance_snapshot(self) -> PerformanceSnapshot:
        """Create current performance snapshot."""
        
        current_time = datetime.now()
        
        # Calculate total return
        total_return = (self.current_portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()
        
        # RL-specific metrics
        avg_episode_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
        
        # Trade statistics
        recent_trades = [t for t in self.trade_records if t.timestamp > current_time - timedelta(days=30)]
        total_recent_trades = len(recent_trades)
        profitable_recent_trades = sum(1 for t in recent_trades if t.is_profitable)
        
        snapshot = PerformanceSnapshot(
            timestamp=current_time,
            portfolio_value=self.current_portfolio_value,
            cash=self.current_cash,
            positions_value=self.current_positions_value,
            total_return=total_return,
            unrealized_pnl=self.total_unrealized_pnl,
            realized_pnl=self.total_realized_pnl,
            
            # Risk metrics
            volatility=metrics.get('volatility', 0.0),
            max_drawdown=metrics.get('max_drawdown', 0.0),
            current_drawdown=metrics.get('current_drawdown', 0.0),
            
            # Performance metrics
            sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
            sortino_ratio=metrics.get('sortino_ratio', 0.0),
            win_rate=metrics.get('win_rate', 0.0),
            profit_factor=metrics.get('profit_factor', 0.0),
            
            # RL metrics
            avg_episode_reward=avg_episode_reward,
            learning_progress=len(self.episode_rewards) / 1000.0,  # Normalized progress
            
            # Trade statistics
            total_trades=total_recent_trades,
            profitable_trades=profitable_recent_trades,
            loss_making_trades=total_recent_trades - profitable_recent_trades
        )
        
        return snapshot
    
    def calculate_performance_metrics(self, window_days: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            window_days: Rolling window for calculations (default: use class setting)
            
        Returns:
            Dictionary of performance metrics
        """
        
        if window_days is None:
            window_days = self.tracking_window_days
        
        # Use cached metrics if recent
        if (datetime.now() - self.last_metrics_update).seconds < 60:  # Cache for 1 minute
            return self.cached_metrics
        
        metrics = {}
        
        # Get recent returns
        recent_returns = np.array(self.daily_returns[-window_days:]) if self.daily_returns else np.array([0.0])
        portfolio_values = np.array(self.portfolio_values[-window_days:]) if len(self.portfolio_values) > window_days else np.array(self.portfolio_values)
        
        if len(recent_returns) == 0:
            return {}
        
        # Total Return
        if len(portfolio_values) > 0:
            metrics['total_return'] = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        else:
            metrics['total_return'] = 0.0
        
        # Volatility (annualized)
        if len(recent_returns) > 1:
            metrics['volatility'] = np.std(recent_returns) * np.sqrt(252)
        else:
            metrics['volatility'] = 0.0
        
        # Sharpe Ratio
        mean_return = np.mean(recent_returns)
        if metrics['volatility'] > 0:
            excess_return = mean_return - (self.risk_free_rate / 252)
            metrics['sharpe_ratio'] = (excess_return * 252) / metrics['volatility']
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Sortino Ratio (downside deviation)
        negative_returns = recent_returns[recent_returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = np.std(negative_returns) * np.sqrt(252)
            if downside_deviation > 0:
                metrics['sortino_ratio'] = (mean_return * 252 - self.risk_free_rate) / downside_deviation
            else:
                metrics['sortino_ratio'] = 0.0
        else:
            metrics['sortino_ratio'] = metrics['sharpe_ratio']
        
        # Maximum Drawdown
        if len(portfolio_values) > 1:
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (portfolio_values - running_max) / running_max
            metrics['max_drawdown'] = abs(np.min(drawdowns))
            metrics['current_drawdown'] = abs(drawdowns[-1])
        else:
            metrics['max_drawdown'] = 0.0
            metrics['current_drawdown'] = 0.0
        
        # Calmar Ratio
        if metrics['max_drawdown'] > 0:
            metrics['calmar_ratio'] = (metrics['total_return'] * 252) / metrics['max_drawdown']
        else:
            metrics['calmar_ratio'] = 0.0
        
        # Win Rate and Profit Factor
        if len(recent_returns) > 0:
            winning_trades = recent_returns[recent_returns > 0]
            losing_trades = recent_returns[recent_returns < 0]
            
            metrics['win_rate'] = len(winning_trades) / len(recent_returns)
            
            gross_profits = np.sum(winning_trades) if len(winning_trades) > 0 else 0
            gross_losses = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 1e-8
            metrics['profit_factor'] = gross_profits / gross_losses
        else:
            metrics['win_rate'] = 0.0
            metrics['profit_factor'] = 0.0
        
        # Value at Risk (95%)
        if len(recent_returns) >= 20:
            metrics['var_95'] = np.percentile(recent_returns, 5)
            # Conditional VaR (Expected Shortfall)
            var_threshold = metrics['var_95']
            tail_losses = recent_returns[recent_returns <= var_threshold]
            metrics['cvar_95'] = np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold
        else:
            metrics['var_95'] = 0.0
            metrics['cvar_95'] = 0.0
        
        # Cache results
        self.cached_metrics = metrics
        self.last_metrics_update = datetime.now()
        
        return metrics
    
    def get_curriculum_phase_metrics(self, phase_start_episode: int) -> Dict[str, Any]:
        """
        Get performance metrics for specific curriculum phase.
        
        Args:
            phase_start_episode: Episode number when phase started
            
        Returns:
            Phase-specific performance metrics
        """
        
        if phase_start_episode >= len(self.episode_rewards):
            return {}
        
        phase_rewards = self.episode_rewards[phase_start_episode:]
        phase_returns = self.episode_returns[phase_start_episode:]
        phase_trades = self.episode_trades[phase_start_episode:]
        
        if len(phase_rewards) == 0:
            return {}
        
        # Calculate phase metrics
        phase_metrics = {
            'episodes_completed': len(phase_rewards),
            'avg_episode_reward': np.mean(phase_rewards),
            'avg_episode_return': np.mean(phase_returns),
            'avg_trades_per_episode': np.mean(phase_trades),
            'reward_volatility': np.std(phase_rewards),
            'return_volatility': np.std(phase_returns),
            'best_episode_reward': np.max(phase_rewards),
            'worst_episode_reward': np.min(phase_rewards),
            'reward_trend': self._calculate_trend(phase_rewards),
            'return_trend': self._calculate_trend(phase_returns)
        }
        
        # Calculate phase-specific Sharpe ratio
        if phase_metrics['return_volatility'] > 0:
            phase_metrics['phase_sharpe'] = phase_metrics['avg_episode_return'] / phase_metrics['return_volatility']
        else:
            phase_metrics['phase_sharpe'] = 0.0
        
        return phase_metrics
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values (positive = improving)."""
        
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression slope
        slope, _ = np.polyfit(x, y, 1)
        return slope
    
    def export_performance_report(self, filepath: str):
        """Export comprehensive performance report to JSON."""
        
        current_metrics = self.calculate_performance_metrics()
        
        report = {
            'summary': {
                'initial_portfolio_value': self.initial_portfolio_value,
                'current_portfolio_value': self.current_portfolio_value,
                'total_return': current_metrics.get('total_return', 0.0),
                'total_trades': len(self.trade_records),
                'total_episodes': len(self.episode_rewards),
                'tracking_period_days': len(self.daily_returns),
                'report_timestamp': datetime.now().isoformat()
            },
            'performance_metrics': current_metrics,
            'rl_training_metrics': {
                'total_episodes': len(self.episode_rewards),
                'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
                'avg_episode_return': np.mean(self.episode_returns) if self.episode_returns else 0.0,
                'reward_improvement_trend': self._calculate_trend(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else 0.0,
                'return_improvement_trend': self._calculate_trend(self.episode_returns[-100:]) if len(self.episode_returns) >= 100 else 0.0
            },
            'recent_performance': [asdict(s) for s in self.performance_snapshots[-10:]],  # Last 10 snapshots
            'trade_summary': {
                'total_trades': len(self.trade_records),
                'profitable_trades': sum(1 for t in self.trade_records if t.is_profitable),
                'avg_trade_value': np.mean([abs(t.value) for t in self.trade_records]) if self.trade_records else 0.0,
                'largest_win': max([t.realized_return * t.value for t in self.trade_records if t.is_profitable], default=0.0),
                'largest_loss': min([t.realized_return * t.value for t in self.trade_records if not t.is_profitable], default=0.0)
            }
        }
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Apply conversion recursively
        def clean_for_json(data):
            if isinstance(data, dict):
                return {k: clean_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_for_json(v) for v in data]
            else:
                return convert_numpy(data)
        
        clean_report = clean_for_json(report)
        
        # Save report
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(clean_report, f, indent=2, default=str)
        
        logger.info(f"Performance report exported to: {filepath}")
    
    def get_phase_advancement_metrics(self) -> Dict[str, float]:
        """Get key metrics for curriculum phase advancement evaluation."""
        
        if len(self.episode_returns) < 50:  # Need sufficient data
            return {
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 1.0,  # Worst possible
                'avg_return': 0.0,
                'return_stability': 0.0,
                'meets_advancement': False
            }
        
        recent_returns = np.array(self.episode_returns[-50:])  # Last 50 episodes
        current_metrics = self.calculate_performance_metrics()
        
        # Calculate return stability (inverse of volatility)
        return_volatility = np.std(recent_returns)
        return_stability = 1.0 / (1.0 + return_volatility) if return_volatility > 0 else 1.0
        
        advancement_metrics = {
            'sharpe_ratio': current_metrics.get('sharpe_ratio', 0.0),
            'win_rate': sum(1 for r in recent_returns if r > 0) / len(recent_returns),
            'max_drawdown': current_metrics.get('max_drawdown', 0.0),
            'avg_return': np.mean(recent_returns),
            'return_stability': return_stability
        }
        
        # Check if meets basic advancement criteria
        advancement_metrics['meets_advancement'] = (
            advancement_metrics['sharpe_ratio'] >= 0.8 and
            advancement_metrics['win_rate'] >= 0.6 and
            advancement_metrics['max_drawdown'] <= 0.15 and
            advancement_metrics['avg_return'] > 0.0
        )
        
        return advancement_metrics
    
    def reset_for_new_phase(self):
        """Reset episode-specific tracking for new curriculum phase."""
        
        logger.info(f"Resetting tracker for new curriculum phase. "
                   f"Completed {len(self.episode_rewards)} episodes in previous phase.")
        
        # Keep performance snapshots and trade records for continuity
        # Reset only episode-specific counters
        self.current_episode_trades = 0
        
        # Clear cached metrics to force recalculation
        self.cached_metrics = {}
        self.last_metrics_update = datetime.min

# Factory function for easy tracker creation
def create_performance_tracker(initial_value: float = 100000.0, **kwargs) -> PerformanceTracker:
    """Create performance tracker with custom configuration."""
    
    return PerformanceTracker(
        initial_portfolio_value=initial_value,
        **kwargs
    )

# Example usage
if __name__ == "__main__":
    # Example performance tracking
    tracker = create_performance_tracker(initial_value=100000.0)
    
    # Simulate some trades and episodes
    for i in range(10):
        # Simulate trade
        trade = TradeRecord(
            timestamp=datetime.now(),
            symbol="AAPL",
            action="BUY",
            quantity=100,
            price=150.0,
            value=15000.0,
            realized_return=0.02,
            reward_total=0.15,
            portfolio_value_before=100000 + i * 1000,
            portfolio_value_after=100000 + (i + 1) * 1000,
            cash_before=50000 - i * 1000,
            cash_after=35000 - i * 1000,
            is_profitable=True
        )
        
        tracker.record_trade(trade)
        tracker.record_episode_end(episode_reward=0.15, episode_return=0.01)
    
    # Get metrics
    metrics = tracker.calculate_performance_metrics()
    advancement_metrics = tracker.get_phase_advancement_metrics()
    
    print("Performance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nPhase Advancement Metrics:")
    for key, value in advancement_metrics.items():
        print(f"  {key}: {value}")
    
    # Export report
    tracker.export_performance_report("./test_performance_report.json")