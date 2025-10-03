#!/usr/bin/env python3
"""
FinRL Performance Tracking System

Production-ready performance tracking for FinRL agents with comprehensive metrics,
persistent storage, historical tracking, and visualization capabilities.

Key Features:
- Sharpe ratio, Win rate, Max drawdown, Sortino ratio, Calmar ratio
- SQLite database for persistent storage
- Historical tracking across training cycles
- Model comparison and ranking
- Automated visualization generation
- Integration with FinRL environments

Usage:
    from utils.performance_tracker import PerformanceTracker

    # Create tracker
    tracker = PerformanceTracker(db_path="data/metrics.db")

    # Evaluate agent
    metrics = tracker.evaluate_agent(
        model=model,
        test_env=env,
        model_name="ppo_20251003",
        agent_type="ppo"
    )

    # Compare models
    comparison = tracker.compare_models(
        model1_name="ppo_base",
        model2_name="ppo_20251003_finetuned"
    )

    # Generate report
    tracker.generate_performance_report(agent_type="ppo")
"""

import os
import sqlite3
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a trading agent."""
    # Identification
    model_name: str
    agent_type: str
    timestamp: str

    # Core Performance Metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    loss_rate: float
    profit_factor: float

    # Risk Metrics
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    downside_deviation: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR (95%)

    # Return Metrics
    total_return: float
    annualized_return: float
    cumulative_return: float
    final_portfolio_value: float
    initial_portfolio_value: float

    # Trade Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float

    # Additional Info
    evaluation_days: int
    test_data_start: str
    test_data_end: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_series(self) -> pd.Series:
        """Convert to pandas Series."""
        return pd.Series(self.to_dict())


class PerformanceTracker:
    """
    Production-ready performance tracking system for FinRL agents.

    Features:
    - Comprehensive metrics calculation (Sharpe, Sortino, Calmar, etc.)
    - SQLite database for persistent storage
    - Historical tracking across training cycles
    - Model comparison and ranking
    - Automated visualization generation
    """

    def __init__(self, db_path: str = "data/performance_metrics.db"):
        """
        Initialize performance tracker.

        Args:
            db_path: Path to SQLite database for storing metrics
        """
        self.db_path = db_path
        self._init_database()
        logger.info(f"PerformanceTracker initialized with database: {db_path}")

    def _init_database(self):
        """Initialize SQLite database with metrics table."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                -- Core Performance Metrics
                sharpe_ratio REAL,
                sortino_ratio REAL,
                calmar_ratio REAL,
                win_rate REAL,
                loss_rate REAL,
                profit_factor REAL,

                -- Risk Metrics
                max_drawdown REAL,
                max_drawdown_duration INTEGER,
                volatility REAL,
                downside_deviation REAL,
                var_95 REAL,
                cvar_95 REAL,

                -- Return Metrics
                total_return REAL,
                annualized_return REAL,
                cumulative_return REAL,
                final_portfolio_value REAL,
                initial_portfolio_value REAL,

                -- Trade Statistics
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                avg_win REAL,
                avg_loss REAL,
                largest_win REAL,
                largest_loss REAL,
                avg_trade_duration REAL,

                -- Additional Info
                evaluation_days INTEGER,
                test_data_start TEXT,
                test_data_end TEXT,
                notes TEXT,

                UNIQUE(model_name, timestamp)
            )
        """)

        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_type
            ON performance_metrics(agent_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON performance_metrics(timestamp)
        """)

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")

    def evaluate_agent(
        self,
        model: Any,
        test_env: Any,
        model_name: str,
        agent_type: str,
        notes: str = ""
    ) -> PerformanceMetrics:
        """
        Evaluate a FinRL agent on test environment and compute all metrics.

        Args:
            model: Trained Stable-Baselines3 model
            test_env: FinRL StockTradingEnv for testing
            model_name: Unique identifier for the model
            agent_type: Type of agent (a2c, ppo, ddpg, sac, td3)
            notes: Additional notes about this evaluation

        Returns:
            PerformanceMetrics object with all computed metrics
        """
        logger.info(f"Evaluating {model_name} ({agent_type}) on test environment")

        # Run episode on test environment
        obs = test_env.reset()
        done = False

        portfolio_values = []
        actions_log = []
        rewards_log = []

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)

            # Log data
            if hasattr(test_env, 'asset_memory'):
                portfolio_values.append(test_env.asset_memory[-1])
            actions_log.append(action)
            rewards_log.append(reward)

        # Extract final data from environment
        if hasattr(test_env, 'asset_memory'):
            portfolio_values = test_env.asset_memory
        else:
            portfolio_values = [test_env.initial_amount]  # Fallback

        # Compute all metrics
        metrics = self._compute_all_metrics(
            portfolio_values=portfolio_values,
            actions_log=actions_log,
            model_name=model_name,
            agent_type=agent_type,
            test_env=test_env,
            notes=notes
        )

        logger.info(f"Evaluation complete. Sharpe: {metrics.sharpe_ratio:.3f}, "
                   f"Win Rate: {metrics.win_rate:.2%}, "
                   f"Max DD: {metrics.max_drawdown:.2%}")

        return metrics

    def _compute_all_metrics(
        self,
        portfolio_values: List[float],
        actions_log: List,
        model_name: str,
        agent_type: str,
        test_env: Any,
        notes: str = ""
    ) -> PerformanceMetrics:
        """
        Compute comprehensive performance metrics from episode data.

        Args:
            portfolio_values: List of portfolio values over time
            actions_log: List of actions taken
            model_name: Model identifier
            agent_type: Agent type
            test_env: Test environment (for metadata)
            notes: Additional notes

        Returns:
            PerformanceMetrics object
        """
        # Convert to numpy arrays
        pv = np.array(portfolio_values)

        # Calculate returns
        daily_returns = np.diff(pv) / pv[:-1]

        # Initial and final values
        initial_value = pv[0]
        final_value = pv[-1]
        total_return = (final_value - initial_value) / initial_value
        cumulative_return = final_value / initial_value - 1

        # Annualized return (assuming 252 trading days per year)
        days = len(pv)
        years = days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility (annualized)
        volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0

        # Sharpe Ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Downside deviation (for Sortino ratio)
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0

        # Sortino Ratio
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0

        # Maximum Drawdown
        cumulative = np.maximum.accumulate(pv)
        drawdowns = (pv - cumulative) / cumulative
        max_drawdown = np.min(drawdowns)

        # Max Drawdown Duration
        in_drawdown = drawdowns < 0
        drawdown_durations = []
        current_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    drawdown_durations.append(current_duration)
                current_duration = 0
        if current_duration > 0:
            drawdown_durations.append(current_duration)
        max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0

        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Value at Risk (95%)
        var_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0

        # Conditional VaR (CVaR/Expected Shortfall at 95%)
        cvar_95 = np.mean(daily_returns[daily_returns <= var_95]) if len(daily_returns) > 0 else 0

        # Trade statistics
        # Assuming actions represent buy/sell signals
        # This is a simplified approach - adjust based on your action space
        trades = self._extract_trade_statistics(actions_log, daily_returns)

        # Create metrics object
        metrics = PerformanceMetrics(
            model_name=model_name,
            agent_type=agent_type,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            # Core Performance
            sharpe_ratio=float(sharpe_ratio),
            sortino_ratio=float(sortino_ratio),
            calmar_ratio=float(calmar_ratio),
            win_rate=trades['win_rate'],
            loss_rate=trades['loss_rate'],
            profit_factor=trades['profit_factor'],

            # Risk Metrics
            max_drawdown=float(max_drawdown),
            max_drawdown_duration=int(max_drawdown_duration),
            volatility=float(volatility),
            downside_deviation=float(downside_deviation),
            var_95=float(var_95),
            cvar_95=float(cvar_95),

            # Return Metrics
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            cumulative_return=float(cumulative_return),
            final_portfolio_value=float(final_value),
            initial_portfolio_value=float(initial_value),

            # Trade Statistics
            total_trades=trades['total_trades'],
            winning_trades=trades['winning_trades'],
            losing_trades=trades['losing_trades'],
            avg_win=trades['avg_win'],
            avg_loss=trades['avg_loss'],
            largest_win=trades['largest_win'],
            largest_loss=trades['largest_loss'],
            avg_trade_duration=trades['avg_trade_duration'],

            # Additional Info
            evaluation_days=days,
            test_data_start=getattr(test_env, 'start_date', 'N/A'),
            test_data_end=getattr(test_env, 'end_date', 'N/A'),
            notes=notes
        )

        return metrics

    def _extract_trade_statistics(
        self,
        actions_log: List,
        daily_returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extract trade statistics from actions and returns.

        This is a simplified approach. Adjust based on your action space.
        """
        # Simplified: count positive/negative returns as wins/losses
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]

        total_trades = len(daily_returns)
        winning_trades = len(positive_returns)
        losing_trades = len(negative_returns)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        loss_rate = losing_trades / total_trades if total_trades > 0 else 0

        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0

        largest_win = np.max(positive_returns) if len(positive_returns) > 0 else 0
        largest_loss = np.min(negative_returns) if len(negative_returns) > 0 else 0

        # Profit factor (gross profit / gross loss)
        gross_profit = np.sum(positive_returns)
        gross_loss = abs(np.sum(negative_returns))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Average trade duration (simplified)
        avg_trade_duration = 1.0  # Assuming daily data

        return {
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'losing_trades': int(losing_trades),
            'win_rate': float(win_rate),
            'loss_rate': float(loss_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'largest_win': float(largest_win),
            'largest_loss': float(largest_loss),
            'profit_factor': float(profit_factor),
            'avg_trade_duration': float(avg_trade_duration)
        }

    def save_metrics(self, metrics: PerformanceMetrics):
        """
        Save metrics to SQLite database.

        Args:
            metrics: PerformanceMetrics object to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO performance_metrics (
                    model_name, agent_type, timestamp,
                    sharpe_ratio, sortino_ratio, calmar_ratio, win_rate, loss_rate, profit_factor,
                    max_drawdown, max_drawdown_duration, volatility, downside_deviation, var_95, cvar_95,
                    total_return, annualized_return, cumulative_return, final_portfolio_value, initial_portfolio_value,
                    total_trades, winning_trades, losing_trades, avg_win, avg_loss, largest_win, largest_loss, avg_trade_duration,
                    evaluation_days, test_data_start, test_data_end, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.model_name, metrics.agent_type, metrics.timestamp,
                metrics.sharpe_ratio, metrics.sortino_ratio, metrics.calmar_ratio,
                metrics.win_rate, metrics.loss_rate, metrics.profit_factor,
                metrics.max_drawdown, metrics.max_drawdown_duration, metrics.volatility,
                metrics.downside_deviation, metrics.var_95, metrics.cvar_95,
                metrics.total_return, metrics.annualized_return, metrics.cumulative_return,
                metrics.final_portfolio_value, metrics.initial_portfolio_value,
                metrics.total_trades, metrics.winning_trades, metrics.losing_trades,
                metrics.avg_win, metrics.avg_loss, metrics.largest_win, metrics.largest_loss,
                metrics.avg_trade_duration,
                metrics.evaluation_days, metrics.test_data_start, metrics.test_data_end, metrics.notes
            ))

            conn.commit()
            logger.info(f"Metrics saved: {metrics.model_name} at {metrics.timestamp}")

        except sqlite3.IntegrityError:
            logger.warning(f"Metrics already exist for {metrics.model_name} at {metrics.timestamp}")

        finally:
            conn.close()

    def get_metrics(
        self,
        model_name: Optional[str] = None,
        agent_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve metrics from database.

        Args:
            model_name: Filter by model name
            agent_type: Filter by agent type
            limit: Limit number of results

        Returns:
            DataFrame with metrics
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM performance_metrics WHERE 1=1"
        params = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        if agent_type:
            query += " AND agent_type = ?"
            params.append(agent_type)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()

        return df

    def compare_models(
        self,
        model1_name: str,
        model2_name: str
    ) -> pd.DataFrame:
        """
        Compare two models side-by-side.

        Args:
            model1_name: Name of first model
            model2_name: Name of second model

        Returns:
            DataFrame with comparison
        """
        conn = sqlite3.connect(self.db_path)

        # Get latest metrics for each model
        query = """
            SELECT * FROM performance_metrics
            WHERE model_name IN (?, ?)
            ORDER BY model_name, timestamp DESC
        """

        df = pd.read_sql_query(query, conn, params=(model1_name, model2_name))
        conn.close()

        if df.empty:
            logger.warning(f"No metrics found for {model1_name} or {model2_name}")
            return pd.DataFrame()

        # Get latest for each model
        df_latest = df.groupby('model_name').first().reset_index()

        # Select key metrics for comparison
        comparison_cols = [
            'model_name', 'agent_type', 'timestamp',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'win_rate', 'max_drawdown', 'annualized_return',
            'total_return', 'volatility', 'total_trades'
        ]

        comparison = df_latest[comparison_cols]

        return comparison

    def get_best_model(
        self,
        agent_type: Optional[str] = None,
        metric: str = 'sharpe_ratio',
        top_n: int = 1
    ) -> pd.DataFrame:
        """
        Get best model(s) ranked by specified metric.

        Args:
            agent_type: Filter by agent type
            metric: Metric to rank by (sharpe_ratio, sortino_ratio, etc.)
            top_n: Number of top models to return

        Returns:
            DataFrame with top models
        """
        df = self.get_metrics(agent_type=agent_type)

        if df.empty:
            logger.warning(f"No metrics found for agent_type={agent_type}")
            return pd.DataFrame()

        # Sort by metric (descending for most metrics, ascending for max_drawdown)
        ascending = metric in ['max_drawdown', 'volatility', 'downside_deviation']
        df_sorted = df.sort_values(metric, ascending=ascending)

        return df_sorted.head(top_n)

    def plot_performance_comparison(
        self,
        agent_type: Optional[str] = None,
        models: Optional[List[str]] = None,
        metrics_to_plot: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Create bar chart comparing model performance.

        Args:
            agent_type: Filter by agent type
            models: List of model names to compare
            metrics_to_plot: List of metrics to plot
            save_path: Path to save figure (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed. Install with: pip install matplotlib")
            return

        # Default metrics to plot
        if metrics_to_plot is None:
            metrics_to_plot = ['sharpe_ratio', 'sortino_ratio', 'win_rate', 'max_drawdown']

        # Get data
        if models:
            df_list = [self.get_metrics(model_name=m, limit=1) for m in models]
            df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
        else:
            df = self.get_metrics(agent_type=agent_type)
            # Get latest for each model
            df = df.groupby('model_name').first().reset_index()

        if df.empty:
            logger.warning("No data to plot")
            return

        # Create subplots
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics_to_plot):
            df_sorted = df.sort_values(metric, ascending=False)
            ax.bar(range(len(df_sorted)), df_sorted[metric])
            ax.set_xticks(range(len(df_sorted)))
            ax.set_xticklabels(df_sorted['model_name'], rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plot_historical_performance(
        self,
        agent_type: str,
        metric: str = 'sharpe_ratio',
        save_path: Optional[str] = None
    ):
        """
        Plot historical performance trend for an agent type.

        Args:
            agent_type: Agent type to plot
            metric: Metric to track over time
            save_path: Path to save figure (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed. Install with: pip install matplotlib")
            return

        df = self.get_metrics(agent_type=agent_type)

        if df.empty:
            logger.warning(f"No data for agent_type={agent_type}")
            return

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Plot
        plt.figure(figsize=(12, 6))

        for model in df['model_name'].unique():
            df_model = df[df['model_name'] == model]
            plt.plot(df_model['timestamp'], df_model[metric],
                    marker='o', label=model, linewidth=2)

        plt.xlabel('Timestamp')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{agent_type.upper()} - {metric.replace("_", " ").title()} Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

    def generate_performance_report(
        self,
        agent_type: Optional[str] = None,
        model_name: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive text report of performance metrics.

        Args:
            agent_type: Filter by agent type
            model_name: Filter by model name
            save_path: Path to save report (optional)

        Returns:
            Report as string
        """
        df = self.get_metrics(agent_type=agent_type, model_name=model_name)

        if df.empty:
            return "No metrics found"

        # Get latest metrics for each model
        df_latest = df.groupby('model_name').first().reset_index()

        # Build report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FINRL PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Models: {len(df_latest)}")

        if agent_type:
            report_lines.append(f"Agent Type: {agent_type.upper()}")

        report_lines.append("")

        for idx, row in df_latest.iterrows():
            report_lines.append("-" * 80)
            report_lines.append(f"Model: {row['model_name']}")
            report_lines.append(f"Agent Type: {row['agent_type']}")
            report_lines.append(f"Evaluated: {row['timestamp']}")
            report_lines.append("")

            report_lines.append("PERFORMANCE METRICS:")
            report_lines.append(f"  Sharpe Ratio:       {row['sharpe_ratio']:>10.3f}")
            report_lines.append(f"  Sortino Ratio:      {row['sortino_ratio']:>10.3f}")
            report_lines.append(f"  Calmar Ratio:       {row['calmar_ratio']:>10.3f}")
            report_lines.append("")

            report_lines.append("RISK METRICS:")
            report_lines.append(f"  Max Drawdown:       {row['max_drawdown']:>10.2%}")
            report_lines.append(f"  Volatility:         {row['volatility']:>10.2%}")
            report_lines.append(f"  Downside Dev:       {row['downside_deviation']:>10.2%}")
            report_lines.append(f"  VaR (95%):          {row['var_95']:>10.2%}")
            report_lines.append(f"  CVaR (95%):         {row['cvar_95']:>10.2%}")
            report_lines.append("")

            report_lines.append("RETURN METRICS:")
            report_lines.append(f"  Total Return:       {row['total_return']:>10.2%}")
            report_lines.append(f"  Annual Return:      {row['annualized_return']:>10.2%}")
            report_lines.append(f"  Cumulative Return:  {row['cumulative_return']:>10.2%}")
            report_lines.append("")

            report_lines.append("TRADE STATISTICS:")
            report_lines.append(f"  Win Rate:           {row['win_rate']:>10.2%}")
            report_lines.append(f"  Loss Rate:          {row['loss_rate']:>10.2%}")
            report_lines.append(f"  Profit Factor:      {row['profit_factor']:>10.3f}")
            report_lines.append(f"  Total Trades:       {row['total_trades']:>10d}")
            report_lines.append(f"  Winning Trades:     {row['winning_trades']:>10d}")
            report_lines.append(f"  Losing Trades:      {row['losing_trades']:>10d}")
            report_lines.append("")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")

        return report


# Convenience functions
def evaluate_and_save(
    model: Any,
    test_env: Any,
    model_name: str,
    agent_type: str,
    db_path: str = "data/performance_metrics.db",
    notes: str = ""
) -> PerformanceMetrics:
    """
    Convenience function to evaluate agent and save metrics in one call.

    Args:
        model: Trained model
        test_env: Test environment
        model_name: Model identifier
        agent_type: Agent type
        db_path: Database path
        notes: Additional notes

    Returns:
        PerformanceMetrics object
    """
    tracker = PerformanceTracker(db_path=db_path)
    metrics = tracker.evaluate_agent(model, test_env, model_name, agent_type, notes)
    tracker.save_metrics(metrics)
    return metrics


def compare_and_plot(
    model_names: List[str],
    db_path: str = "data/performance_metrics.db",
    save_path: Optional[str] = None
):
    """
    Convenience function to compare models and generate plot.

    Args:
        model_names: List of model names to compare
        db_path: Database path
        save_path: Path to save plot
    """
    tracker = PerformanceTracker(db_path=db_path)
    tracker.plot_performance_comparison(models=model_names, save_path=save_path)
