"""
Pairs Trading Backtesting Framework

This module provides comprehensive backtesting capabilities for pairs trading strategies,
including performance metrics, visualization, and risk analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
from pathlib import Path

from ..strategies.pairs_trading import PairsStrategy, PairCandidate, PairSignal

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for pairs trading backtest"""
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    commission_per_share: float = 0.005  # $0.005 per share
    market_impact: float = 0.001  # 0.1% market impact
    short_borrow_rate: float = 0.02  # 2% annual short borrow rate
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    max_positions: int = 5
    position_size: float = 10000.0  # Dollar amount per leg
    
    # Strategy parameters
    lookback_period: int = 252
    signal_window: int = 20
    entry_zscore: float = 2.0
    exit_zscore: float = 0.0
    stop_loss_zscore: float = 3.0


class Trade(NamedTuple):
    """Individual trade record"""
    entry_date: datetime
    exit_date: datetime
    symbol_a: str
    symbol_b: str
    side: str  # 'long_spread' or 'short_spread'
    quantity_a: int
    quantity_b: int
    entry_price_a: float
    entry_price_b: float
    exit_price_a: float
    exit_price_b: float
    entry_spread: float
    exit_spread: float
    entry_zscore: float
    exit_zscore: float
    pnl: float
    commission: float
    duration_days: int
    max_adverse_zscore: float


@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    # Performance metrics
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Pairs-specific metrics
    avg_holding_period: float
    avg_spread_reversion_time: float
    correlation_stability: float
    
    # Time series
    equity_curve: pd.Series
    drawdown_series: pd.Series
    trades: List[Trade]
    portfolio_stats: pd.DataFrame


class PairsBacktester:
    """
    Comprehensive backtesting framework for pairs trading strategies
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.strategy = PairsStrategy(
            lookback_period=config.lookback_period,
            signal_window=config.signal_window,
            entry_zscore=config.entry_zscore,
            exit_zscore=config.exit_zscore,
            stop_loss_zscore=config.stop_loss_zscore,
            max_positions=config.max_positions,
            position_size=config.position_size
        )
        
        # Backtest state
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict] = []
        self.current_positions: Dict[str, Dict] = {}
        self.cash = config.initial_capital
        self.equity_curve: List[float] = []
        self.dates: List[datetime] = []
    
    async def run_backtest(
        self, 
        price_data: Dict[str, pd.Series],
        benchmark_data: Optional[pd.Series] = None
    ) -> BacktestResults:
        """
        Run comprehensive pairs trading backtest
        
        Args:
            price_data: Dictionary of symbol -> price series
            benchmark_data: Optional benchmark for comparison (e.g., SPY)
            
        Returns:
            BacktestResults with comprehensive performance analysis
        """
        try:
            logger.info(f"Starting pairs trading backtest from {self.config.start_date} to {self.config.end_date}")
            
            # Align all price data
            aligned_data = self._align_price_data(price_data)
            
            if aligned_data.empty:
                raise ValueError("No aligned price data available for backtest")
            
            # Filter by date range
            start_date = pd.to_datetime(self.config.start_date)
            end_date = pd.to_datetime(self.config.end_date)
            backtest_data = aligned_data.loc[start_date:end_date]
            
            if backtest_data.empty:
                raise ValueError(f"No data available for date range {start_date} to {end_date}")
            
            # Initialize strategy with initial data
            symbols = list(backtest_data.columns)
            initial_data = {
                symbol: backtest_data[symbol].iloc[:self.config.lookback_period]
                for symbol in symbols
            }
            
            # Find initial pairs
            logger.info("Finding initial pairs...")
            self.strategy.find_pairs(symbols, initial_data)
            logger.info(f"Found {len(self.strategy.pair_candidates)} initial pairs")
            
            # Run day-by-day simulation
            for i, (date, prices) in enumerate(backtest_data.iterrows()):
                if i < self.config.lookback_period:
                    continue  # Need sufficient history
                
                # Get current price data for strategy
                current_data = {
                    symbol: backtest_data[symbol].iloc[:i+1]
                    for symbol in symbols
                }
                
                # Process trading day
                await self._process_trading_day(date, prices, current_data)
                
                # Record portfolio state
                self._record_portfolio_state(date, prices)
            
            # Calculate final results
            results = self._calculate_results(benchmark_data)
            
            logger.info(f"Backtest completed. Total return: {results.total_return:.2%}")
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            raise
    
    async def _process_trading_day(
        self, 
        date: datetime, 
        prices: pd.Series, 
        historical_data: Dict[str, pd.Series]
    ):
        """
        Process a single trading day
        """
        try:
            # Update current positions with latest prices
            self._update_positions(date, prices)
            
            # Generate signals
            signals = self.strategy.generate_signals(historical_data)
            
            # Execute signals
            for signal in signals:
                await self._execute_signal(signal, date, prices)
            
            # Re-scan for pairs periodically (e.g., monthly)
            if self._should_rescan_pairs(date):
                symbols = list(prices.index)
                recent_data = {
                    symbol: historical_data[symbol].tail(self.config.lookback_period)
                    for symbol in symbols
                }
                self.strategy.find_pairs(symbols, recent_data)
                
        except Exception as e:
            logger.error(f"Error processing trading day {date}: {e}")
    
    def _should_rescan_pairs(self, date: datetime) -> bool:
        """
        Determine if we should rescan for new pairs
        """
        # Rescan monthly
        return date.day == 1
    
    async def _execute_signal(self, signal: PairSignal, date: datetime, prices: pd.Series):
        """
        Execute a trading signal with realistic constraints
        """
        try:
            pair_key = f"{signal.symbol_a}-{signal.symbol_b}"
            
            # Get current prices
            price_a = prices[signal.symbol_a]
            price_b = prices[signal.symbol_b]
            
            # Calculate transaction costs
            commission_a = abs(signal.suggested_quantity_a) * self.config.commission_per_share
            commission_b = abs(signal.suggested_quantity_b) * self.config.commission_per_share
            total_commission = commission_a + commission_b
            
            # Market impact (simplified model)
            impact_a = abs(signal.suggested_quantity_a * price_a) * self.config.market_impact
            impact_b = abs(signal.suggested_quantity_b * price_b) * self.config.market_impact
            total_impact = impact_a + impact_b
            
            if signal.action in ['enter_long_spread', 'enter_short_spread']:
                # Opening new position
                if pair_key in self.current_positions:
                    return  # Already have position
                
                # Check cash availability
                required_capital = abs(signal.suggested_quantity_a * price_a) + abs(signal.suggested_quantity_b * price_b)
                if required_capital + total_commission + total_impact > self.cash:
                    logger.warning(f"Insufficient cash for {pair_key}: need ${required_capital:.0f}, have ${self.cash:.0f}")
                    return
                
                # Open position
                self.current_positions[pair_key] = {
                    'symbol_a': signal.symbol_a,
                    'symbol_b': signal.symbol_b,
                    'side': signal.action.replace('enter_', ''),
                    'quantity_a': signal.suggested_quantity_a,
                    'quantity_b': signal.suggested_quantity_b,
                    'entry_date': date,
                    'entry_price_a': price_a,
                    'entry_price_b': price_b,
                    'entry_spread': signal.spread,
                    'entry_zscore': signal.zscore,
                    'commission_paid': total_commission,
                    'max_adverse_zscore': signal.zscore
                }
                
                # Update cash
                self.cash -= total_commission + total_impact
                
                logger.debug(f"Opened position {pair_key}: {signal.action}")
                
            elif signal.action in ['exit', 'stop_loss']:
                # Closing existing position
                if pair_key not in self.current_positions:
                    return  # No position to close
                
                position = self.current_positions[pair_key]
                
                # Calculate P&L
                pnl_a = position['quantity_a'] * (price_a - position['entry_price_a'])
                pnl_b = position['quantity_b'] * (price_b - position['entry_price_b'])
                gross_pnl = pnl_a + pnl_b
                net_pnl = gross_pnl - position['commission_paid'] - total_commission - total_impact
                
                # Calculate short borrow costs (simplified)
                days_held = (date - position['entry_date']).days
                short_cost = 0
                if position['quantity_a'] < 0:
                    short_cost += abs(position['quantity_a'] * position['entry_price_a']) * self.config.short_borrow_rate * days_held / 365
                if position['quantity_b'] < 0:
                    short_cost += abs(position['quantity_b'] * position['entry_price_b']) * self.config.short_borrow_rate * days_held / 365
                
                net_pnl -= short_cost
                
                # Record trade
                trade = Trade(
                    entry_date=position['entry_date'],
                    exit_date=date,
                    symbol_a=position['symbol_a'],
                    symbol_b=position['symbol_b'],
                    side=position['side'],
                    quantity_a=position['quantity_a'],
                    quantity_b=position['quantity_b'],
                    entry_price_a=position['entry_price_a'],
                    entry_price_b=position['entry_price_b'],
                    exit_price_a=price_a,
                    exit_price_b=price_b,
                    entry_spread=position['entry_spread'],
                    exit_spread=signal.spread,
                    entry_zscore=position['entry_zscore'],
                    exit_zscore=signal.zscore,
                    pnl=net_pnl,
                    commission=position['commission_paid'] + total_commission,
                    duration_days=days_held,
                    max_adverse_zscore=position['max_adverse_zscore']
                )
                
                self.trades.append(trade)
                
                # Update cash
                self.cash += net_pnl - total_commission - total_impact
                
                # Remove position
                del self.current_positions[pair_key]
                
                logger.debug(f"Closed position {pair_key}: {signal.action}, P&L: ${net_pnl:.2f}")
                
        except Exception as e:
            logger.error(f"Error executing signal {signal.symbol_a}-{signal.symbol_b}: {e}")
    
    def _update_positions(self, date: datetime, prices: pd.Series):
        """
        Update existing positions with current market prices
        """
        for pair_key, position in self.current_positions.items():
            try:
                # Update max adverse move for risk tracking
                current_spread = prices[position['symbol_a']] - position['entry_price_b'] * prices[position['symbol_b']] / position['entry_price_b']
                # This is a simplified z-score calculation - in practice you'd use the rolling window
                current_zscore = (current_spread - position['entry_spread']) / abs(position['entry_spread'])
                
                if abs(current_zscore) > abs(position['max_adverse_zscore']):
                    position['max_adverse_zscore'] = current_zscore
                    
            except Exception as e:
                logger.warning(f"Error updating position {pair_key}: {e}")
    
    def _record_portfolio_state(self, date: datetime, prices: pd.Series):
        """
        Record current portfolio state for performance tracking
        """
        try:
            # Calculate position values
            position_value = 0
            for position in self.current_positions.values():
                try:
                    value_a = position['quantity_a'] * prices[position['symbol_a']]
                    value_b = position['quantity_b'] * prices[position['symbol_b']]
                    position_value += value_a + value_b
                except KeyError:
                    continue  # Missing price data
            
            total_equity = self.cash + position_value
            
            self.dates.append(date)
            self.equity_curve.append(total_equity)
            
            self.portfolio_history.append({
                'date': date,
                'cash': self.cash,
                'position_value': position_value,
                'total_equity': total_equity,
                'active_positions': len(self.current_positions)
            })
            
        except Exception as e:
            logger.error(f"Error recording portfolio state for {date}: {e}")
    
    def _align_price_data(self, price_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Align price data across all symbols
        """
        try:
            df = pd.DataFrame(price_data)
            return df.dropna()
        except Exception as e:
            logger.error(f"Error aligning price data: {e}")
            return pd.DataFrame()
    
    def _calculate_results(self, benchmark_data: Optional[pd.Series] = None) -> BacktestResults:
        """
        Calculate comprehensive backtest results
        """
        try:
            if not self.equity_curve:
                raise ValueError("No equity curve data available")
            
            # Convert to pandas Series
            equity_series = pd.Series(self.equity_curve, index=self.dates)
            returns = equity_series.pct_change().dropna()
            
            # Basic performance metrics
            total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
            annual_return = (1 + total_return) ** (252 / len(equity_series)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Drawdown analysis
            rolling_max = equity_series.expanding().max()
            drawdown_series = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown_series.min()
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
            
            # Trading metrics
            if self.trades:
                winning_trades = sum(1 for t in self.trades if t.pnl > 0)
                losing_trades = sum(1 for t in self.trades if t.pnl <= 0)
                win_rate = winning_trades / len(self.trades)
                
                wins = [t.pnl for t in self.trades if t.pnl > 0]
                losses = [abs(t.pnl) for t in self.trades if t.pnl <= 0]
                
                avg_win = np.mean(wins) if wins else 0
                avg_loss = np.mean(losses) if losses else 0
                profit_factor = sum(wins) / sum(losses) if losses else float('inf')
                
                avg_holding_period = np.mean([t.duration_days for t in self.trades])
                
                # Pairs-specific metrics
                avg_spread_reversion_time = np.mean([
                    t.duration_days for t in self.trades 
                    if abs(t.exit_zscore) < abs(t.entry_zscore)
                ])
                
            else:
                winning_trades = losing_trades = 0
                win_rate = avg_win = avg_loss = profit_factor = 0
                avg_holding_period = avg_spread_reversion_time = 0
            
            # Portfolio stats DataFrame
            portfolio_df = pd.DataFrame(self.portfolio_history)
            if not portfolio_df.empty:
                portfolio_df.set_index('date', inplace=True)
            
            return BacktestResults(
                total_return=total_return,
                annual_return=annual_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                total_trades=len(self.trades),
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                avg_holding_period=avg_holding_period,
                avg_spread_reversion_time=avg_spread_reversion_time,
                correlation_stability=0.0,  # Would calculate from actual correlation data
                equity_curve=equity_series,
                drawdown_series=drawdown_series,
                trades=self.trades,
                portfolio_stats=portfolio_df
            )
            
        except Exception as e:
            logger.error(f"Error calculating results: {e}")
            raise


def run_pairs_backtest(
    price_data: Dict[str, pd.Series],
    config: BacktestConfig,
    benchmark_data: Optional[pd.Series] = None
) -> BacktestResults:
    """
    Convenience function to run a pairs trading backtest
    """
    backtester = PairsBacktester(config)
    return asyncio.run(backtester.run_backtest(price_data, benchmark_data))


def generate_backtest_report(results: BacktestResults, output_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive backtest report
    """
    report = f"""
PAIRS TRADING BACKTEST REPORT
{'='*50}

PERFORMANCE SUMMARY
-------------------
Total Return:           {results.total_return:>8.2%}
Annualized Return:      {results.annual_return:>8.2%}
Volatility:             {results.volatility:>8.2%}
Sharpe Ratio:           {results.sharpe_ratio:>8.2f}
Maximum Drawdown:       {results.max_drawdown:>8.2%}
Calmar Ratio:           {results.calmar_ratio:>8.2f}

TRADING STATISTICS
------------------
Total Trades:           {results.total_trades:>8d}
Winning Trades:         {results.winning_trades:>8d}
Losing Trades:          {results.losing_trades:>8d}
Win Rate:               {results.win_rate:>8.2%}
Average Win:            ${results.avg_win:>7.2f}
Average Loss:           ${results.avg_loss:>7.2f}
Profit Factor:          {results.profit_factor:>8.2f}

PAIRS-SPECIFIC METRICS
----------------------
Avg Holding Period:     {results.avg_holding_period:>8.1f} days
Avg Reversion Time:     {results.avg_spread_reversion_time:>8.1f} days
Correlation Stability:  {results.correlation_stability:>8.3f}

TRADE ANALYSIS
--------------
"""
    
    if results.trades:
        # Add top 5 best and worst trades
        sorted_trades = sorted(results.trades, key=lambda t: t.pnl, reverse=True)
        
        report += "\nTop 5 Best Trades:\n"
        for i, trade in enumerate(sorted_trades[:5], 1):
            report += f"{i}. {trade.symbol_a}-{trade.symbol_b}: ${trade.pnl:.2f} ({trade.duration_days}d)\n"
        
        report += "\nTop 5 Worst Trades:\n"
        for i, trade in enumerate(sorted_trades[-5:], 1):
            report += f"{i}. {trade.symbol_a}-{trade.symbol_b}: ${trade.pnl:.2f} ({trade.duration_days}d)\n"
    
    if output_path:
        Path(output_path).write_text(report)
    
    return report