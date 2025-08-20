"""
Comprehensive Backtesting Framework for RL Trading Strategies
Provides detailed performance analysis, risk metrics, and visualization
"""

import numpy as np
import pandas as pd
import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import warnings
import pickle
import os
from pathlib import Path

# Initialize logger first
logger = logging.getLogger(__name__)

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("⚠️ Matplotlib/seaborn not available - plotting disabled")

from agents.rl_trading_agent import RLTradingAgent, TrainingConfig, TradingPerformance
from agents.llm_rl_integration import EnhancedLLMTradingEnvironment, LLMStateEnricher
from agents.realistic_trading_env import RealisticTradingEnvironment
# Import proper market data fetcher for backtesting
from tools.market_data_fetcher import MarketDataFetcher

warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Configuration for backtesting runs."""
    # Time period
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"
    
    # Portfolio settings
    initial_capital: float = 100000.0
    position_size_limit: float = 1.0
    transaction_cost: float = 0.001
    
    # Analysis settings
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02
    confidence_interval: float = 0.95
    
    # Output settings
    save_results: bool = True
    save_plots: bool = True
    output_dir: str = "backtest_results"
    
    # Advanced settings
    use_enhanced_environment: bool = True
    regime_detection: bool = True
    monte_carlo_runs: int = 100


@dataclass
class BacktestMetrics:
    """Comprehensive backtesting metrics."""
    # Return metrics
    total_return: float
    annualized_return: float
    excess_return: float
    cumulative_return: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    
    # Performance metrics
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    # Trading metrics
    total_trades: int
    trades_per_year: float
    average_holding_period: float
    turnover_ratio: float
    
    # Benchmark comparison
    benchmark_return: float
    active_return: float
    information_ratio: float
    beta: float
    alpha: float
    correlation: float
    
    # Advanced metrics
    tail_ratio: float
    gain_to_pain_ratio: float
    sterling_ratio: float
    burke_ratio: float


@dataclass
class BacktestResults:
    """Complete backtesting results."""
    config: BacktestConfig
    metrics: BacktestMetrics
    
    # Time series data
    portfolio_values: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    drawdowns: pd.Series
    
    # Benchmark data
    benchmark_values: pd.Series
    benchmark_returns: pd.Series
    
    # Additional analysis
    monthly_returns: pd.DataFrame
    yearly_returns: pd.DataFrame
    rolling_metrics: pd.DataFrame
    
    # Metadata
    backtest_duration: float
    data_quality_score: float
    execution_summary: Dict[str, Any]


class RLBacktestingFramework:
    """
    Comprehensive backtesting framework for RL trading strategies.
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.market_data_fetcher = MarketDataFetcher()
        
        # Results storage
        self.results_history: List[BacktestResults] = []
        
        # Setup output directory
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(exist_ok=True)
        
        logger.info(f"✅ RL Backtesting Framework initialized")
        logger.info(f"📁 Output directory: {self.output_path}")
    
    async def run_comprehensive_backtest(self,
                                       agent: RLTradingAgent,
                                       symbols: List[str],
                                       comparison_strategies: List[str] = None) -> Dict[str, BacktestResults]:
        """
        Run comprehensive backtesting across multiple symbols and strategies.
        """
        
        logger.info(f"🚀 Starting comprehensive backtest")
        logger.info(f"📊 Symbols: {symbols}")
        logger.info(f"📅 Period: {self.config.start_date} to {self.config.end_date}")
        
        all_results = {}
        
        for symbol in symbols:
            logger.info(f"📈 Backtesting {symbol}...")
            
            try:
                # Run single symbol backtest
                results = await self.backtest_single_symbol(agent, symbol)
                all_results[symbol] = results
                
                # Save individual results
                if self.config.save_results:
                    await self._save_results(results, f"{symbol}_backtest")
                
                logger.info(f"✅ {symbol} backtest completed")
                
            except Exception as e:
                logger.error(f"❌ {symbol} backtest failed: {e}")
                continue
        
        # Generate comparative analysis
        if len(all_results) > 1:
            comparative_results = self._generate_comparative_analysis(all_results)
            
            if self.config.save_results:
                await self._save_comparative_analysis(comparative_results)
        
        logger.info(f"🎉 Comprehensive backtest completed for {len(all_results)} symbols")
        return all_results
    
    async def backtest_single_symbol(self,
                                   agent: RLTradingAgent,
                                   symbol: str) -> BacktestResults:
        """
        Run detailed backtesting for a single symbol.
        """
        
        start_time = datetime.now()
        logger.info(f"🔍 Starting detailed backtest for {symbol}")
        
        # Get market data
        market_data = await self._prepare_market_data(symbol)
        benchmark_data = await self._prepare_benchmark_data()
        
        # Split data for walk-forward analysis
        train_data, test_data = self._split_data_for_backtest(market_data)
        
        # Create trading environment
        env = self._create_trading_environment(symbol, test_data)
        
        # Run backtest simulation
        simulation_results = await self._run_backtest_simulation(agent, env)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            simulation_results, benchmark_data, market_data
        )
        
        # Generate additional analysis
        additional_analysis = self._generate_additional_analysis(
            simulation_results, market_data
        )
        
        # Create results object
        results = BacktestResults(
            config=self.config,
            metrics=metrics,
            portfolio_values=simulation_results['portfolio_values'],
            returns=simulation_results['returns'],
            positions=simulation_results['positions'],
            trades=simulation_results['trades'],
            drawdowns=simulation_results['drawdowns'],
            benchmark_values=benchmark_data['values'],
            benchmark_returns=benchmark_data['returns'],
            monthly_returns=additional_analysis['monthly_returns'],
            yearly_returns=additional_analysis['yearly_returns'],
            rolling_metrics=additional_analysis['rolling_metrics'],
            backtest_duration=(datetime.now() - start_time).total_seconds(),
            data_quality_score=self._assess_data_quality(market_data),
            execution_summary=simulation_results['execution_summary']
        )
        
        await env.close()
        
        logger.info(f"✅ Backtest completed for {symbol} in {results.backtest_duration:.1f}s")
        return results
    
    async def _prepare_market_data(self, symbol: str) -> pd.DataFrame:
        """Prepare market data for backtesting."""
        
        logger.info(f"📊 Preparing market data for {symbol}")
        
        # Fetch historical data
        data = await self.market_data_fetcher.get_historical_data(
            symbol=symbol,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            interval="1h"
        )
        
        if data is None or len(data) == 0:
            raise ValueError(f"No market data available for {symbol}")
        
        # Clean and validate data
        data = self._clean_market_data(data)
        
        logger.info(f"📊 Prepared {len(data)} data points for {symbol}")
        return data
    
    async def _prepare_benchmark_data(self) -> Dict[str, pd.Series]:
        """Prepare benchmark data for comparison."""
        
        benchmark_data = await self.market_data_fetcher.get_historical_data(
            symbol=self.config.benchmark_symbol,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            interval="1h"
        )
        
        if benchmark_data is None:
            # Create dummy benchmark
            logger.warning("⚠️ No benchmark data available, using synthetic benchmark")
            dates = pd.date_range(self.config.start_date, self.config.end_date, freq='H')
            benchmark_values = pd.Series(
                self.config.initial_capital * (1 + np.random.normal(0, 0.001, len(dates))).cumprod(),
                index=dates
            )
        else:
            benchmark_values = benchmark_data['close'] / benchmark_data['close'].iloc[0] * self.config.initial_capital
        
        benchmark_returns = benchmark_values.pct_change().fillna(0)
        
        return {
            'values': benchmark_values,
            'returns': benchmark_returns
        }
    
    def _split_data_for_backtest(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing periods."""
        
        # Use 70% for training, 30% for testing
        split_idx = int(len(data) * 0.7)
        
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
        logger.info(f"📊 Split data: {len(train_data)} train, {len(test_data)} test samples")
        return train_data, test_data
    
    def _create_trading_environment(self, symbol: str, data: pd.DataFrame):
        """Create trading environment for backtesting."""
        
        if self.config.use_enhanced_environment:
            # Use enhanced LLM environment
            env = EnhancedLLMTradingEnvironment(
                symbol=symbol,
                initial_balance=self.config.initial_capital,
                transaction_cost=self.config.transaction_cost,
                max_position_size=self.config.position_size_limit
            )
            env.set_data(data)
        else:
            # Use realistic trading environment (generates its own data)
            env = RealisticTradingEnvironment(
                symbols=[symbol],
                initial_balance=self.config.initial_capital,
                max_position_size=self.config.position_size_limit
            )
            # RealisticTradingEnvironment generates its own market data
            logger.warning(f"⚠️ Using RealisticTradingEnvironment with synthetic data for {symbol}")
        
        return env
    
    async def _run_backtest_simulation(self, agent: RLTradingAgent, env) -> Dict[str, Any]:
        """Run the main backtesting simulation."""
        
        logger.info("🎮 Running backtest simulation...")
        
        # Initialize tracking
        portfolio_values = []
        positions = []
        actions_taken = []
        rewards = []
        trade_log = []
        
        # Reset environment
        obs = await env.reset()
        done = False
        step = 0
        
        execution_summary = {
            'total_steps': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'environment_errors': 0
        }
        
        while not done:
            try:
                # Get action from agent
                if agent.is_trained:
                    action, confidence = await agent.predict(obs, deterministic=True)
                else:
                    # Random baseline if agent not trained
                    action = env.action_space.sample()[0] if hasattr(env.action_space, 'sample') else 0.0
                    confidence = 0.5
                
                # Execute action
                obs, reward, done, info = await env.step([action])
                
                # Record data
                portfolio_values.append(info['portfolio_value'])
                positions.append(info['position'])
                actions_taken.append(action)
                rewards.append(reward)
                
                # Log trades
                if 'trade_executed' in info and info['trade_executed']:
                    trade_log.append({
                        'step': step,
                        'action': action,
                        'position': info['position'],
                        'price': info.get('current_price', 0),
                        'portfolio_value': info['portfolio_value'],
                        'confidence': confidence
                    })
                
                execution_summary['successful_actions'] += 1
                step += 1
                
                if step % 100 == 0:
                    logger.debug(f"Step {step}: Portfolio ${info['portfolio_value']:.2f}")
                
            except Exception as e:
                logger.error(f"Error at step {step}: {e}")
                execution_summary['failed_actions'] += 1
                
                if execution_summary['failed_actions'] > 10:
                    logger.error("Too many execution errors, stopping simulation")
                    break
        
        execution_summary['total_steps'] = step
        
        # Convert to pandas objects
        portfolio_values = pd.Series(portfolio_values, name='portfolio_value')
        positions = pd.Series(positions, name='position')
        returns = portfolio_values.pct_change().fillna(0)
        drawdowns = self._calculate_drawdowns(portfolio_values)
        trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
        
        logger.info(f"✅ Simulation completed: {step} steps, {len(trade_log)} trades")
        
        return {
            'portfolio_values': portfolio_values,
            'positions': positions,
            'returns': returns,
            'drawdowns': drawdowns,
            'trades': trades_df,
            'actions': pd.Series(actions_taken),
            'rewards': pd.Series(rewards),
            'execution_summary': execution_summary
        }
    
    def _calculate_comprehensive_metrics(self,
                                       simulation_results: Dict[str, Any],
                                       benchmark_data: Dict[str, pd.Series],
                                       market_data: pd.DataFrame) -> BacktestMetrics:
        """Calculate comprehensive performance metrics."""
        
        portfolio_values = simulation_results['portfolio_values']
        returns = simulation_results['returns']
        trades = simulation_results['trades']
        benchmark_returns = benchmark_data['returns']
        
        # Basic return metrics
        total_return = (portfolio_values.iloc[-1] / self.config.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 * 24 / len(returns)) - 1  # Hourly data
        cumulative_return = total_return
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252 * 24)  # Annualized
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252 * 24) if len(downside_returns) > 0 else volatility  
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Drawdown metrics
        drawdowns = simulation_results['drawdowns']
        max_drawdown = drawdowns.min()
        max_dd_duration = self._calculate_max_drawdown_duration(drawdowns)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, (1 - self.config.confidence_interval) * 100)
        cvar_95 = returns[returns <= var_95].mean() if np.any(returns <= var_95) else var_95
        
        # Trading metrics
        if len(trades) > 0:
            trade_returns = trades['portfolio_value'].pct_change().dropna()
            wins = trade_returns[trade_returns > 0]
            losses = trade_returns[trade_returns < 0]
            
            win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
            average_win = wins.mean() if len(wins) > 0 else 0
            average_loss = losses.mean() if len(losses) > 0 else 0
            largest_win = wins.max() if len(wins) > 0 else 0
            largest_loss = losses.min() if len(losses) > 0 else 0
            profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float('inf')
            
            total_trades = len(trades)
            trades_per_year = total_trades * (252 * 24 / len(returns))
            average_holding_period = len(returns) / total_trades if total_trades > 0 else 0
            
            # Turnover ratio
            position_changes = simulation_results['positions'].diff().abs()
            turnover_ratio = position_changes.sum() / 2  # Divide by 2 for round-trip
        else:
            win_rate = average_win = average_loss = 0.0
            largest_win = largest_loss = 0.0
            profit_factor = 1.0
            total_trades = trades_per_year = average_holding_period = turnover_ratio = 0.0
        
        # Benchmark comparison
        if len(benchmark_returns) > 0:
            # Align returns with benchmark
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            
            if len(aligned_returns) > 10:
                benchmark_return = (1 + aligned_benchmark).prod() - 1
                active_return = annualized_return - benchmark_return
                
                # Calculate beta and alpha
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = aligned_benchmark.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                alpha = annualized_return - (self.config.risk_free_rate + beta * (benchmark_return - self.config.risk_free_rate))
                
                # Information ratio
                active_returns = aligned_returns - aligned_benchmark
                information_ratio = active_returns.mean() / active_returns.std() if active_returns.std() > 0 else 0
                information_ratio *= np.sqrt(252 * 24)  # Annualize
                
                # Correlation
                correlation = aligned_returns.corr(aligned_benchmark)
            else:
                benchmark_return = active_return = alpha = beta = information_ratio = correlation = 0.0
        else:
            benchmark_return = active_return = alpha = beta = information_ratio = correlation = 0.0
        
        # Advanced metrics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        tail_ratio = (np.percentile(positive_returns, 95) / abs(np.percentile(negative_returns, 5))) if len(negative_returns) > 0 else 1.0
        
        # Gain-to-pain ratio
        total_gain = positive_returns.sum() if len(positive_returns) > 0 else 0
        total_pain = abs(negative_returns.sum()) if len(negative_returns) > 0 else 1e-8
        gain_to_pain_ratio = total_gain / total_pain
        
        # Sterling ratio
        sterling_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Burke ratio
        burke_ratio = (annualized_return - self.config.risk_free_rate) / np.sqrt((drawdowns ** 2).sum()) if (drawdowns ** 2).sum() > 0 else 0
        
        excess_return = annualized_return - self.config.risk_free_rate
        
        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            excess_return=excess_return,
            cumulative_return=cumulative_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_trades=total_trades,
            trades_per_year=trades_per_year,
            average_holding_period=average_holding_period,
            turnover_ratio=turnover_ratio,
            benchmark_return=benchmark_return,
            active_return=active_return,
            information_ratio=information_ratio,
            beta=beta,
            alpha=alpha,
            correlation=correlation,
            tail_ratio=tail_ratio,
            gain_to_pain_ratio=gain_to_pain_ratio,
            sterling_ratio=sterling_ratio,
            burke_ratio=burke_ratio
        )
    
    def _generate_additional_analysis(self,
                                    simulation_results: Dict[str, Any],
                                    market_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate additional analysis for the backtest."""
        
        returns = simulation_results['returns']
        
        # Monthly returns
        if len(returns) > 30:
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        else:
            monthly_returns = pd.DataFrame()
        
        # Yearly returns  
        if len(returns) > 252:
            yearly_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        else:
            yearly_returns = pd.DataFrame()
        
        # Rolling metrics
        rolling_window = min(30 * 24, len(returns) // 4)  # 30 days or 1/4 of data
        
        if rolling_window > 10:
            rolling_returns = returns.rolling(rolling_window).mean() * 252 * 24
            rolling_volatility = returns.rolling(rolling_window).std() * np.sqrt(252 * 24)
            rolling_sharpe = (rolling_returns - self.config.risk_free_rate) / rolling_volatility
            
            rolling_metrics = pd.DataFrame({
                'rolling_return': rolling_returns,
                'rolling_volatility': rolling_volatility,  
                'rolling_sharpe': rolling_sharpe
            })
        else:
            rolling_metrics = pd.DataFrame()
        
        return {
            'monthly_returns': monthly_returns,
            'yearly_returns': yearly_returns,
            'rolling_metrics': rolling_metrics
        }
    
    def _calculate_drawdowns(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        running_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - running_max) / running_max
        return drawdowns
    
    def _calculate_max_drawdown_duration(self, drawdowns: pd.Series) -> int:
        """Calculate maximum drawdown duration in periods."""
        is_in_drawdown = drawdowns < 0
        drawdown_periods = []
        current_period = 0
        
        for in_dd in is_in_drawdown:
            if in_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _clean_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate market data."""
        # Remove NaN values
        data = data.dropna()
        
        # Remove zero/negative prices
        data = data[data['close'] > 0]
        
        # Remove extreme outliers (more than 50% daily moves)
        daily_returns = data['close'].pct_change()
        data = data[abs(daily_returns) < 0.5]
        
        return data
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess data quality score (0-1)."""
        quality_score = 1.0
        
        # Penalize for missing data
        if data.isnull().any().any():
            quality_score -= 0.2
        
        # Penalize for gaps
        time_diff = data.index.to_series().diff()
        expected_freq = time_diff.mode()[0] if len(time_diff.mode()) > 0 else pd.Timedelta(hours=1)
        gaps = time_diff > expected_freq * 2
        
        if gaps.sum() > len(data) * 0.01:  # More than 1% gaps
            quality_score -= 0.3
        
        # Penalize for extreme volatility
        returns = data['close'].pct_change()
        if returns.std() > 0.1:  # Very high volatility
            quality_score -= 0.2
        
        return max(0.0, quality_score)
    
    def _generate_comparative_analysis(self, all_results: Dict[str, BacktestResults]) -> pd.DataFrame:
        """Generate comparative analysis across symbols."""
        
        comparison_data = []
        
        for symbol, results in all_results.items():
            metrics = results.metrics
            comparison_data.append({
                'symbol': symbol,
                'total_return': metrics.total_return,
                'annualized_return': metrics.annualized_return,
                'volatility': metrics.volatility,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'win_rate': metrics.win_rate,
                'total_trades': metrics.total_trades,
                'data_quality': results.data_quality_score
            })
        
        return pd.DataFrame(comparison_data).set_index('symbol')
    
    async def _save_results(self, results: BacktestResults, filename: str):
        """Save backtest results to disk."""
        
        # Save metrics summary
        metrics_file = self.output_path / f"{filename}_metrics.json"
        with open(metrics_file, 'w') as f:
            import json
            json.dump(asdict(results.metrics), f, indent=2, default=str)
        
        # Save time series data
        data_file = self.output_path / f"{filename}_data.pkl"
        data_to_save = {
            'portfolio_values': results.portfolio_values,
            'returns': results.returns,
            'positions': results.positions,
            'trades': results.trades,
            'drawdowns': results.drawdowns
        }
        
        with open(data_file, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        # Generate plots if requested
        if self.config.save_plots:
            await self._generate_plots(results, filename)
        
        logger.info(f"💾 Results saved: {filename}")
    
    async def _save_comparative_analysis(self, comparative_df: pd.DataFrame):
        """Save comparative analysis results."""
        
        comp_file = self.output_path / "comparative_analysis.csv"
        comparative_df.to_csv(comp_file)
        
        logger.info("💾 Comparative analysis saved")
    
    async def _generate_plots(self, results: BacktestResults, filename: str):
        """Generate visualization plots for backtest results."""
        
        if not PLOTTING_AVAILABLE:
            logger.warning("⚠️ Plotting disabled - matplotlib/seaborn not available")
            return
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Portfolio value over time
        ax1 = plt.subplot(2, 3, 1)
        results.portfolio_values.plot(ax=ax1, label='Strategy', color='blue')
        results.benchmark_values.plot(ax=ax1, label='Benchmark', color='gray', alpha=0.7)
        ax1.set_title('Portfolio Value Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = plt.subplot(2, 3, 2)
        results.drawdowns.plot(ax=ax2, color='red', alpha=0.7)
        ax2.fill_between(results.drawdowns.index, results.drawdowns, 0, alpha=0.3, color='red')
        ax2.set_title('Drawdown Over Time')
        ax2.set_ylabel('Drawdown')
        ax2.grid(True, alpha=0.3)
        
        # 3. Returns distribution
        ax3 = plt.subplot(2, 3, 3)
        results.returns.hist(bins=50, ax=ax3, alpha=0.7, color='green')
        ax3.axvline(results.returns.mean(), color='red', linestyle='--', label=f'Mean: {results.returns.mean():.4f}')
        ax3.set_title('Returns Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Rolling Sharpe ratio
        if not results.rolling_metrics.empty:
            ax4 = plt.subplot(2, 3, 4)
            results.rolling_metrics['rolling_sharpe'].plot(ax=ax4, color='purple')
            ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax4.set_title('Rolling Sharpe Ratio')
            ax4.grid(True, alpha=0.3)
        
        # 5. Monthly returns heatmap
        if not results.monthly_returns.empty:
            ax5 = plt.subplot(2, 3, 5)
            monthly_data = results.monthly_returns.values.reshape(-1, 1)
            if len(monthly_data) > 1:
                sns.heatmap(monthly_data, ax=ax5, cmap='RdYlGn', center=0, annot=True, fmt='.2%')
            ax5.set_title('Monthly Returns')
        
        # 6. Positions over time
        ax6 = plt.subplot(2, 3, 6)
        results.positions.plot(ax=ax6, color='orange', alpha=0.7)
        ax6.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax6.set_title('Positions Over Time')
        ax6.set_ylabel('Position Size')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_path / f"{filename}_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Plots generated: {filename}_analysis.png")


# Utility functions
async def run_quick_backtest(agent: RLTradingAgent, symbol: str = "AAPL") -> BacktestResults:
    """Run a quick backtest with default settings."""
    
    config = BacktestConfig(
        start_date="2023-06-01",
        end_date="2023-12-01",
        initial_capital=50000,
        use_enhanced_environment=True
    )
    
    framework = RLBacktestingFramework(config)
    results = await framework.backtest_single_symbol(agent, symbol)
    
    return results


def print_backtest_summary(results: BacktestResults):
    """Print a summary of backtest results."""
    
    metrics = results.metrics
    
    print("=" * 60)
    print("📊 BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"📈 Total Return: {metrics.total_return:.2%}")
    print(f"📈 Annualized Return: {metrics.annualized_return:.2%}")
    print(f"📉 Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"📊 Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"📊 Sortino Ratio: {metrics.sortino_ratio:.3f}")
    print(f"🎯 Win Rate: {metrics.win_rate:.2%}")
    print(f"💼 Total Trades: {metrics.total_trades}")
    print(f"📅 Backtest Duration: {results.backtest_duration:.1f}s")
    print(f"🔍 Data Quality Score: {results.data_quality_score:.2f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test the backtesting framework
    async def test_backtesting_framework():
        from agents.rl_trading_agent import RLTradingAgent, create_training_config
        
        # Create a sample agent (for testing)
        config = create_training_config(total_timesteps=1000)
        agent = RLTradingAgent("AAPL", config=config)
        
        # Run quick backtest
        results = await run_quick_backtest(agent, "AAPL")
        
        # Print summary
        print_backtest_summary(results)
        
        await agent.close()
    
    asyncio.run(test_backtesting_framework())