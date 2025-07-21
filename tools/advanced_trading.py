"""Advanced trading tools and strategies based on Alpaca examples and ML4T insights."""

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import math

from tools.alpaca_client import alpaca_client
from agents.market_analysis import enhanced_market_analysis_factory, AnalysisResult
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Results from strategy backtesting."""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_duration: float
    volatility: float
    start_date: datetime
    end_date: datetime
    trades: List[Dict[str, Any]]

class QuantitativeStrategy:
    """Base class for quantitative trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
        self.lookback_period = 252  # 1 year default
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals. Override in subclasses."""
        raise NotImplementedError
    
    def calculate_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Calculate strategy returns given signals."""
        # Shift signals to avoid look-ahead bias
        signals_shifted = signals.shift(1).fillna(0)
        
        # Calculate daily returns
        daily_returns = data['close'].pct_change()
        
        # Strategy returns = signal * daily returns
        strategy_returns = signals_shifted * daily_returns
        
        return strategy_returns.fillna(0)

class MomentumStrategy(QuantitativeStrategy):
    """Momentum-based trading strategy."""
    
    def __init__(self, lookback: int = 20, threshold: float = 0.02):
        super().__init__("Momentum Strategy")
        self.parameters = {
            "lookback": lookback,
            "threshold": threshold
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum signals."""
        # Calculate momentum as percentage change over lookback period
        momentum = data['close'].pct_change(self.parameters["lookback"])
        
        # Generate signals based on momentum threshold
        signals = pd.Series(0, index=data.index)
        signals[momentum > self.parameters["threshold"]] = 1   # Buy signal
        signals[momentum < -self.parameters["threshold"]] = -1  # Sell signal
        
        return signals

class MeanReversionStrategy(QuantitativeStrategy):
    """Mean reversion trading strategy."""
    
    def __init__(self, window: int = 20, std_threshold: float = 2.0):
        super().__init__("Mean Reversion Strategy")
        self.parameters = {
            "window": window,
            "std_threshold": std_threshold
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals using Bollinger Bands."""
        # Calculate moving average and standard deviation
        ma = data['close'].rolling(self.parameters["window"]).mean()
        std = data['close'].rolling(self.parameters["window"]).std()
        
        # Calculate z-score
        z_score = (data['close'] - ma) / std
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[z_score < -self.parameters["std_threshold"]] = 1   # Buy oversold
        signals[z_score > self.parameters["std_threshold"]] = -1   # Sell overbought
        
        return signals

class PairsTradeStrategy(QuantitativeStrategy):
    """Statistical arbitrage pairs trading strategy."""
    
    def __init__(self, symbol_a: str, symbol_b: str, lookback: int = 60, 
                 entry_threshold: float = 2.0, exit_threshold: float = 0.5):
        super().__init__(f"Pairs Trade: {symbol_a}/{symbol_b}")
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.parameters = {
            "lookback": lookback,
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold
        }
    
    def generate_signals(self, data_a: pd.DataFrame, data_b: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate pairs trading signals for both symbols."""
        # Calculate spread
        spread = data_a['close'] - data_b['close']
        
        # Calculate rolling mean and std of spread
        spread_mean = spread.rolling(self.parameters["lookback"]).mean()
        spread_std = spread.rolling(self.parameters["lookback"]).std()
        
        # Calculate z-score of spread
        z_score = (spread - spread_mean) / spread_std
        
        # Generate signals
        signals_a = pd.Series(0, index=data_a.index)
        signals_b = pd.Series(0, index=data_b.index)
        
        # Entry signals
        long_entry = z_score < -self.parameters["entry_threshold"]
        short_entry = z_score > self.parameters["entry_threshold"]
        
        # Exit signals
        exit_condition = abs(z_score) < self.parameters["exit_threshold"]
        
        # Long A, Short B when spread is too low
        signals_a[long_entry] = 1
        signals_b[long_entry] = -1
        
        # Short A, Long B when spread is too high
        signals_a[short_entry] = -1
        signals_b[short_entry] = 1
        
        # Exit both positions
        signals_a[exit_condition] = 0
        signals_b[exit_condition] = 0
        
        return {self.symbol_a: signals_a, self.symbol_b: signals_b}

class PortfolioOptimizer:
    """Modern Portfolio Theory-based optimizer."""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def optimize_portfolio(self, returns_data: Dict[str, pd.Series], 
                         target_return: Optional[float] = None) -> Dict[str, float]:
        """Optimize portfolio weights using mean-variance optimization."""
        try:
            if len(returns_data) < 2:
                return {}
            
            # Convert to DataFrame
            returns_df = pd.DataFrame(returns_data)
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_df.mean() * 252  # Annualized
            cov_matrix = returns_df.cov() * 252  # Annualized
            
            # Simple optimization: maximize Sharpe ratio
            optimal_weights = self._maximize_sharpe_ratio(expected_returns, cov_matrix)
            
            return optimal_weights.to_dict()
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return {}
    
    def _maximize_sharpe_ratio(self, expected_returns: pd.Series, 
                             cov_matrix: pd.DataFrame) -> pd.Series:
        """Find portfolio weights that maximize Sharpe ratio."""
        n_assets = len(expected_returns)
        
        # Equal weights as starting point
        weights = pd.Series([1/n_assets] * n_assets, index=expected_returns.index)
        
        # Simple optimization using volatility inverse weighting
        volatilities = pd.Series(np.diag(cov_matrix) ** 0.5, index=expected_returns.index)
        inv_vol_weights = (1 / volatilities) / (1 / volatilities).sum()
        
        return inv_vol_weights
    
    def calculate_portfolio_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        if len(returns) < 2:
            return {}
        
        # Annualized metrics
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        return {
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": abs(max_drawdown),
            "win_rate": win_rate,
            "total_return": cumulative_returns.iloc[-1] - 1
        }

class AdvancedBacktester:
    """Advanced backtesting engine with transaction costs and slippage."""
    
    def __init__(self, commission: float = 0.001, slippage: float = 0.0005):
        self.commission = commission  # 0.1% commission
        self.slippage = slippage      # 0.05% slippage
        self.portfolio_optimizer = PortfolioOptimizer()
    
    def backtest_strategy(self, strategy: QuantitativeStrategy, 
                         symbol: str, start_date: datetime, 
                         end_date: datetime) -> BacktestResult:
        """Backtest a strategy on historical data."""
        try:
            logger.info(f"Backtesting {strategy.name} on {symbol}")
            
            # Get historical data
            days_diff = (end_date - start_date).days + strategy.lookback_period
            data = alpaca_client.get_market_data(symbol, limit=days_diff)
            
            if len(data) < strategy.lookback_period:
                raise ValueError("Insufficient data for backtesting")
            
            # Filter data to date range
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            # Generate signals
            signals = strategy.generate_signals(data)
            
            # Calculate returns with transaction costs
            returns = self._calculate_returns_with_costs(data, signals)
            
            # Calculate metrics
            metrics = self.portfolio_optimizer.calculate_portfolio_metrics(returns)
            
            # Analyze trades
            trades = self._analyze_trades(data, signals)
            
            return BacktestResult(
                strategy_name=strategy.name,
                total_return=metrics.get("total_return", 0),
                sharpe_ratio=metrics.get("sharpe_ratio", 0),
                max_drawdown=metrics.get("max_drawdown", 0),
                win_rate=metrics.get("win_rate", 0),
                total_trades=len(trades),
                avg_trade_duration=self._calculate_avg_trade_duration(trades),
                volatility=metrics.get("annual_volatility", 0),
                start_date=start_date,
                end_date=end_date,
                trades=trades
            )
            
        except Exception as e:
            logger.error(f"Backtesting error: {e}")
            return BacktestResult(
                strategy_name=strategy.name,
                total_return=0, sharpe_ratio=0, max_drawdown=0,
                win_rate=0, total_trades=0, avg_trade_duration=0,
                volatility=0, start_date=start_date, end_date=end_date,
                trades=[]
            )
    
    def _calculate_returns_with_costs(self, data: pd.DataFrame, 
                                    signals: pd.Series) -> pd.Series:
        """Calculate returns including transaction costs and slippage."""
        # Shift signals to avoid look-ahead bias
        signals_shifted = signals.shift(1).fillna(0)
        
        # Calculate raw returns
        daily_returns = data['close'].pct_change()
        
        # Detect position changes
        position_changes = signals_shifted.diff().abs()
        
        # Apply transaction costs
        transaction_costs = position_changes * (self.commission + self.slippage)
        
        # Strategy returns = signal * returns - transaction costs
        strategy_returns = signals_shifted * daily_returns - transaction_costs
        
        return strategy_returns.fillna(0)
    
    def _analyze_trades(self, data: pd.DataFrame, signals: pd.Series) -> List[Dict[str, Any]]:
        """Analyze individual trades from signals."""
        trades = []
        position = 0
        entry_price = 0
        entry_date = None
        
        for date, signal in signals.items():
            current_price = data.loc[date, 'close']
            
            # Check for position entry
            if position == 0 and signal != 0:
                position = signal
                entry_price = current_price
                entry_date = date
            
            # Check for position exit
            elif position != 0 and (signal == 0 or signal != position):
                exit_price = current_price
                exit_date = date
                
                # Calculate trade metrics
                if position > 0:  # Long trade
                    pnl = (exit_price - entry_price) / entry_price
                else:  # Short trade
                    pnl = (entry_price - exit_price) / entry_price
                
                duration = (exit_date - entry_date).days
                
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "position": "long" if position > 0 else "short",
                    "pnl": pnl,
                    "duration_days": duration
                })
                
                # Update position
                position = signal
                if signal != 0:
                    entry_price = current_price
                    entry_date = date
                else:
                    entry_price = 0
                    entry_date = None
        
        return trades
    
    def _calculate_avg_trade_duration(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate average trade duration in days."""
        if not trades:
            return 0
        
        durations = [trade["duration_days"] for trade in trades]
        return sum(durations) / len(durations)

class RiskManagementEngine:
    """Advanced risk management with position sizing and stop losses."""
    
    def __init__(self):
        self.max_position_risk = settings.max_portfolio_risk
        self.max_portfolio_heat = 0.06  # Maximum 6% portfolio heat
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss: float, portfolio_value: float,
                              signal_strength: float = 1.0) -> float:
        """Calculate position size using fixed fractional risk model."""
        try:
            if stop_loss <= 0 or entry_price <= 0:
                return 0
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            risk_percentage = risk_per_share / entry_price
            
            # Maximum risk amount
            max_risk_amount = portfolio_value * self.max_position_risk
            
            # Adjust for signal strength
            adjusted_risk_amount = max_risk_amount * signal_strength
            
            # Calculate shares
            shares = adjusted_risk_amount / risk_per_share
            
            # Apply position size limits
            max_position_value = portfolio_value * settings.max_position_size
            max_shares_by_value = max_position_value / entry_price
            
            return min(shares, max_shares_by_value)
            
        except Exception as e:
            logger.error(f"Position sizing error for {symbol}: {e}")
            return 0
    
    def calculate_portfolio_heat(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate current portfolio heat (total risk exposure)."""
        total_risk = 0
        portfolio_value = sum(pos.get("market_value", 0) for pos in positions)
        
        if portfolio_value <= 0:
            return 0
        
        for position in positions:
            # Estimate risk as distance to stop loss or volatility-based
            market_value = abs(position.get("market_value", 0))
            # Simplified: assume 5% risk per position
            estimated_risk = market_value * 0.05
            total_risk += estimated_risk
        
        return total_risk / portfolio_value
    
    def should_add_position(self, new_position_risk: float, 
                          current_heat: float) -> bool:
        """Determine if new position should be added based on portfolio heat."""
        projected_heat = current_heat + new_position_risk
        return projected_heat <= self.max_portfolio_heat
    
    def calculate_dynamic_stop_loss(self, symbol: str, entry_price: float,
                                  position_type: str = "long") -> float:
        """Calculate dynamic stop loss based on volatility."""
        try:
            # Get recent volatility data
            data = alpaca_client.get_market_data(symbol, limit=20)
            if len(data) < 10:
                # Fallback to fixed percentage
                multiplier = 0.95 if position_type == "long" else 1.05
                return entry_price * multiplier
            
            # Calculate ATR-based stop loss
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # Set stop loss at 2x ATR
            if position_type == "long":
                stop_loss = entry_price - (2 * atr)
            else:
                stop_loss = entry_price + (2 * atr)
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Dynamic stop loss calculation error for {symbol}: {e}")
            # Fallback to fixed percentage
            multiplier = 0.95 if position_type == "long" else 1.05
            return entry_price * multiplier

class AlgorithmicTradingEngine:
    """Main engine combining all advanced trading components."""
    
    def __init__(self):
        self.backtester = AdvancedBacktester()
        self.risk_manager = RiskManagementEngine()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.strategies = {
            "momentum": MomentumStrategy(),
            "mean_reversion": MeanReversionStrategy()
        }
    
    def run_strategy_comparison(self, symbols: List[str], 
                              lookback_days: int = 252) -> Dict[str, Any]:
        """Compare multiple strategies across symbols."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            results = {}
            
            for strategy_name, strategy in self.strategies.items():
                strategy_results = {}
                
                for symbol in symbols:
                    try:
                        backtest_result = self.backtester.backtest_strategy(
                            strategy, symbol, start_date, end_date
                        )
                        strategy_results[symbol] = backtest_result
                    except Exception as e:
                        logger.warning(f"Backtest failed for {strategy_name}/{symbol}: {e}")
                
                results[strategy_name] = strategy_results
            
            # Find best performing strategies
            best_strategies = self._rank_strategies(results)
            
            return {
                "backtest_results": results,
                "best_strategies": best_strategies,
                "analysis_period": f"{start_date.date()} to {end_date.date()}"
            }
            
        except Exception as e:
            logger.error(f"Strategy comparison error: {e}")
            return {}
    
    def _rank_strategies(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank strategies by risk-adjusted returns."""
        rankings = []
        
        for strategy_name, strategy_results in results.items():
            for symbol, backtest_result in strategy_results.items():
                if isinstance(backtest_result, BacktestResult):
                    # Risk-adjusted score (Sharpe ratio weighted by total return)
                    score = backtest_result.sharpe_ratio * (1 + backtest_result.total_return)
                    
                    rankings.append({
                        "strategy": strategy_name,
                        "symbol": symbol,
                        "score": score,
                        "sharpe_ratio": backtest_result.sharpe_ratio,
                        "total_return": backtest_result.total_return,
                        "max_drawdown": backtest_result.max_drawdown,
                        "win_rate": backtest_result.win_rate
                    })
        
        # Sort by score descending
        rankings.sort(key=lambda x: x["score"], reverse=True)
        
        return rankings[:10]  # Top 10

    async def generate_enhanced_signals(self, symbols: List[str], 
                                      portfolio_value: float) -> List[Dict[str, Any]]:
        """Generate enhanced trading signals using multiple strategies."""
        try:
            signals = []
            
            # Get market analysis from existing factory
            analysis_results = {}
            symbol_analysis = {}
            
            # Get individual analysis for each symbol
            for symbol in symbols:
                try:
                    tech_analysis = await enhanced_market_analysis_factory['technical'].analyze_symbol(symbol)
                    symbol_analysis[symbol] = tech_analysis
                except Exception as e:
                    logger.warning(f"Analysis failed for {symbol}: {e}")
            
            analysis_results = {"symbol_analysis": symbol_analysis}
            
            # Enhance with quantitative strategies
            for symbol in symbols:
                try:
                    # Get technical analysis
                    tech_analysis = analysis_results["symbol_analysis"].get(symbol)
                    if not tech_analysis or tech_analysis.action == "hold":
                        continue
                    
                    # Get current price and calculate stop loss
                    current_price = tech_analysis.indicators.get('current_price', 0)
                    if current_price <= 0:
                        continue
                    
                    # Calculate position size using risk management
                    stop_loss = self.risk_manager.calculate_dynamic_stop_loss(
                        symbol, current_price, tech_analysis.action
                    )
                    
                    position_size = self.risk_manager.calculate_position_size(
                        symbol, current_price, stop_loss, portfolio_value,
                        tech_analysis.confidence
                    )
                    
                    if position_size > 0:
                        signals.append({
                            "symbol": symbol,
                            "action": tech_analysis.action,
                            "quantity": position_size,
                            "entry_price": current_price,
                            "stop_loss": stop_loss,
                            "confidence": tech_analysis.confidence,
                            "strategy": tech_analysis.signal_type,
                            "reasoning": tech_analysis.reasoning
                        })
                
                except Exception as e:
                    logger.warning(f"Signal generation error for {symbol}: {e}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Enhanced signal generation error: {e}")
            return []

# Global engine instance
algorithmic_engine = AlgorithmicTradingEngine()