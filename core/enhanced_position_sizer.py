#!/usr/bin/env python3
"""
Enhanced Position Sizing Engine for Short Selling

This module provides sophisticated position sizing using advanced mathematical models
specifically tailored for short selling strategies. It implements the Kelly Criterion
with uncertainty adjustments, volatility-based sizing, correlation-aware allocation,
and dynamic risk management.

Key Features:
- Kelly Criterion with uncertainty and drawdown protection
- Volatility-adjusted position sizing with regime awareness
- Correlation-aware allocation to prevent over-concentration
- Dynamic position sizing based on market conditions and portfolio state
- Risk parity and factor-based allocation methods
- Integration with short squeeze risk and borrow cost analysis

The engine ensures optimal capital allocation while maintaining proper risk controls
for short selling strategies.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import json
import math
from scipy import stats, optimize
from scipy.linalg import inv
import warnings

from core.short_risk_manager import short_risk_manager, SqueezeRiskMetrics
from core.borrow_cost_monitor import borrow_cost_monitor, ShortabilityAnalysis
from core.enhanced_short_signal_engine import enhanced_short_signal_engine, EnhancedShortSignal
from tools.alpaca_client import alpaca_client
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class KellyParameters:
    """Kelly Criterion parameters with uncertainty adjustments."""
    symbol: str
    win_probability: float  # Probability of profitable trade
    avg_win: float  # Average win amount (as fraction)
    avg_loss: float  # Average loss amount (as fraction)
    kelly_fraction: float  # Raw Kelly fraction
    adjusted_kelly: float  # Kelly with uncertainty adjustment
    confidence_interval: Tuple[float, float]  # 95% confidence interval
    uncertainty_discount: float  # Discount factor for uncertainty
    max_drawdown_protection: float  # Additional protection factor

@dataclass
class VolatilityAdjustment:
    """Volatility-based position sizing adjustments."""
    symbol: str
    historical_volatility: float
    implied_volatility: Optional[float]
    volatility_percentile: float  # Percentile vs historical range
    regime_volatility: float  # Current regime volatility estimate
    vol_target: float  # Target portfolio volatility
    size_multiplier: float  # Volatility-based size adjustment
    risk_budget: float  # Risk budget allocation

@dataclass
class CorrelationContext:
    """Correlation context for portfolio allocation."""
    symbol: str
    correlation_matrix: Optional[np.ndarray]
    portfolio_symbols: List[str]
    max_correlation: float
    correlation_cluster: List[str]  # Highly correlated symbols
    diversification_ratio: float  # Portfolio diversification ratio
    effective_positions: float  # Risk-adjusted position count
    concentration_penalty: float  # Penalty for correlation concentration

@dataclass
class EnhancedPositionSize:
    """Comprehensive position sizing recommendation."""
    symbol: str
    
    # Raw size calculations
    kelly_size: float  # Kelly-based size
    volatility_size: float  # Volatility-adjusted size
    risk_parity_size: float  # Risk parity allocation
    correlation_adjusted_size: float  # Correlation-adjusted size
    
    # Final recommendation
    recommended_size: float  # Final recommended size (% of portfolio)
    max_size: float  # Maximum allowed size
    min_size: float  # Minimum viable size
    
    # Risk metrics
    expected_return: float
    expected_volatility: float
    expected_sharpe: float
    value_at_risk: float  # 95% VaR
    max_drawdown_estimate: float
    
    # Sizing rationale
    primary_method: str  # Primary sizing method used
    adjustments_applied: List[str]
    confidence_score: float  # 0-1 confidence in sizing
    reasoning: str
    
    # Risk controls
    position_limits: Dict[str, float]
    stop_loss_level: Optional[float]
    take_profit_level: Optional[float]
    
    timestamp: datetime

class KellyCriterionCalculator:
    """Advanced Kelly Criterion implementation with uncertainty handling."""
    
    def __init__(self):
        """Initialize Kelly calculator."""
        self.lookback_periods = [30, 60, 120, 252]  # Different analysis periods
        self.confidence_level = 0.95
        
    async def calculate_kelly_parameters(self, symbol: str, 
                                       short_signal: EnhancedShortSignal) -> KellyParameters:
        """Calculate Kelly Criterion parameters with uncertainty adjustments."""
        try:
            # Get historical performance data
            historical_data = await self._get_historical_performance(symbol)
            
            # Calculate base Kelly parameters
            win_prob, avg_win, avg_loss = self._calculate_base_statistics(historical_data, short_signal)
            
            # Calculate raw Kelly fraction
            if avg_loss > 0 and win_prob > 0:
                kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_loss
            else:
                kelly_fraction = 0.0
            
            # Apply uncertainty adjustments
            uncertainty_discount = self._calculate_uncertainty_discount(historical_data, short_signal)
            adjusted_kelly = kelly_fraction * uncertainty_discount
            
            # Calculate confidence intervals
            confidence_interval = self._calculate_confidence_interval(
                historical_data, kelly_fraction, self.confidence_level
            )
            
            # Apply drawdown protection
            drawdown_protection = self._calculate_drawdown_protection(historical_data)
            final_kelly = adjusted_kelly * drawdown_protection
            
            return KellyParameters(
                symbol=symbol,
                win_probability=win_prob,
                avg_win=avg_win,
                avg_loss=avg_loss,
                kelly_fraction=kelly_fraction,
                adjusted_kelly=final_kelly,
                confidence_interval=confidence_interval,
                uncertainty_discount=uncertainty_discount,
                max_drawdown_protection=drawdown_protection
            )
            
        except Exception as e:
            logger.error(f"Error calculating Kelly parameters for {symbol}: {e}")
            return self._create_default_kelly_parameters(symbol)
    
    async def _get_historical_performance(self, symbol: str) -> pd.DataFrame:
        """Get historical price and performance data."""
        try:
            # Get extended historical data
            market_data = alpaca_client.get_market_data(symbol, limit=300)
            
            if market_data is None or len(market_data) < 50:
                # Create minimal synthetic data if no real data available
                dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
                synthetic_data = pd.DataFrame({
                    'close': 100 * np.cumprod(1 + np.random.normal(0, 0.02, 100)),
                    'volume': np.random.randint(100000, 1000000, 100)
                }, index=dates)
                return synthetic_data
            
            # Calculate returns and additional metrics
            market_data['returns'] = market_data['close'].pct_change()
            market_data['cumulative_returns'] = (1 + market_data['returns']).cumprod()
            
            # Add volatility measures
            market_data['rolling_vol'] = market_data['returns'].rolling(20).std()
            market_data['rolling_return'] = market_data['returns'].rolling(20).mean()
            
            return market_data.dropna()
            
        except Exception as e:
            logger.error(f"Error getting historical performance for {symbol}: {e}")
            # Return empty DataFrame
            return pd.DataFrame()
    
    def _calculate_base_statistics(self, historical_data: pd.DataFrame, 
                                 signal: EnhancedShortSignal) -> Tuple[float, float, float]:
        """Calculate base win probability and average win/loss."""
        try:
            if len(historical_data) < 20:
                # Use signal-based estimates when insufficient data
                return self._estimate_from_signal(signal)
            
            returns = historical_data['returns'].dropna()
            
            # For short selling, we want negative returns (stock goes down)
            short_returns = -returns  # Invert for short position perspective
            
            # Calculate win statistics
            wins = short_returns[short_returns > 0]
            losses = short_returns[short_returns <= 0]
            
            win_probability = len(wins) / len(short_returns) if len(short_returns) > 0 else 0.5
            avg_win = wins.mean() if len(wins) > 0 else 0.02
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.02
            
            # Adjust based on signal strength
            signal_adjustment = self._apply_signal_adjustment(
                win_probability, avg_win, avg_loss, signal
            )
            
            return signal_adjustment
            
        except Exception as e:
            logger.error(f"Error calculating base statistics: {e}")
            return 0.45, 0.02, 0.025  # Conservative defaults
    
    def _estimate_from_signal(self, signal: EnhancedShortSignal) -> Tuple[float, float, float]:
        """Estimate statistics from signal characteristics."""
        try:
            # Base estimates adjusted by signal quality
            base_win_prob = 0.45  # Slightly below 50% for conservative short selling
            base_avg_win = 0.025  # 2.5% average win
            base_avg_loss = 0.030  # 3.0% average loss
            
            # Adjust based on signal strength and confidence
            strength_multiplier = 1.0 + (signal.signal_strength - 0.5) * 0.2
            confidence_multiplier = 1.0 + (signal.confidence - 0.5) * 0.1
            
            adjusted_win_prob = min(0.65, base_win_prob * strength_multiplier * confidence_multiplier)
            adjusted_avg_win = base_avg_win * strength_multiplier
            
            # Higher confidence may also reduce average loss (better timing)
            adjusted_avg_loss = base_avg_loss / confidence_multiplier
            
            return adjusted_win_prob, adjusted_avg_win, adjusted_avg_loss
            
        except Exception as e:
            logger.error(f"Error estimating from signal: {e}")
            return 0.45, 0.02, 0.025
    
    def _apply_signal_adjustment(self, win_prob: float, avg_win: float, avg_loss: float,
                               signal: EnhancedShortSignal) -> Tuple[float, float, float]:
        """Apply signal-based adjustments to historical statistics."""
        try:
            # Adjust win probability based on signal strength
            signal_boost = (signal.signal_strength - 0.5) * 0.1  # Max ±5% adjustment
            adjusted_win_prob = np.clip(win_prob + signal_boost, 0.2, 0.8)
            
            # Adjust win/loss ratio based on confidence
            confidence_factor = 1.0 + (signal.confidence - 0.5) * 0.1
            adjusted_avg_win = avg_win * confidence_factor
            adjusted_avg_loss = avg_loss / confidence_factor
            
            return adjusted_win_prob, adjusted_avg_win, adjusted_avg_loss
            
        except Exception as e:
            logger.error(f"Error applying signal adjustment: {e}")
            return win_prob, avg_win, avg_loss
    
    def _calculate_uncertainty_discount(self, historical_data: pd.DataFrame,
                                      signal: EnhancedShortSignal) -> float:
        """Calculate uncertainty discount factor for Kelly fraction."""
        try:
            # Base uncertainty from data quality
            data_quality_discount = 1.0
            
            if len(historical_data) < 50:
                data_quality_discount = 0.6  # Significant discount for limited data
            elif len(historical_data) < 100:
                data_quality_discount = 0.8
            
            # Uncertainty from signal confidence
            signal_confidence_discount = 0.5 + (signal.confidence * 0.5)
            
            # Uncertainty from market conditions
            if len(historical_data) > 20:
                recent_volatility = historical_data['returns'].tail(20).std()
                historical_volatility = historical_data['returns'].std()
                vol_ratio = recent_volatility / max(historical_volatility, 0.01)
                
                if vol_ratio > 1.5:  # High current volatility
                    vol_discount = 0.8
                elif vol_ratio > 1.2:
                    vol_discount = 0.9
                else:
                    vol_discount = 1.0
            else:
                vol_discount = 0.9
            
            # Combined uncertainty discount
            total_discount = data_quality_discount * signal_confidence_discount * vol_discount
            return max(0.3, min(1.0, total_discount))  # Keep between 30%-100%
            
        except Exception as e:
            logger.error(f"Error calculating uncertainty discount: {e}")
            return 0.7  # Conservative default
    
    def _calculate_confidence_interval(self, historical_data: pd.DataFrame,
                                     kelly_fraction: float, confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for Kelly fraction."""
        try:
            if len(historical_data) < 30:
                # Wide confidence interval for limited data
                margin = kelly_fraction * 0.5
                return (kelly_fraction - margin, kelly_fraction + margin)
            
            # Bootstrap confidence interval
            returns = historical_data['returns'].dropna()
            bootstrap_kellys = []
            
            n_bootstrap = 100
            for _ in range(n_bootstrap):
                bootstrap_returns = np.random.choice(returns, size=len(returns), replace=True)
                short_returns = -bootstrap_returns
                
                wins = short_returns[short_returns > 0]
                losses = short_returns[short_returns <= 0]
                
                if len(wins) > 0 and len(losses) > 0:
                    win_prob = len(wins) / len(short_returns)
                    avg_win = wins.mean()
                    avg_loss = abs(losses.mean())
                    
                    if avg_loss > 0:
                        bootstrap_kelly = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_loss
                        bootstrap_kellys.append(bootstrap_kelly)
            
            if bootstrap_kellys:
                alpha = 1 - confidence_level
                lower = np.percentile(bootstrap_kellys, alpha/2 * 100)
                upper = np.percentile(bootstrap_kellys, (1 - alpha/2) * 100)
                return (lower, upper)
            else:
                margin = kelly_fraction * 0.3
                return (kelly_fraction - margin, kelly_fraction + margin)
                
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            margin = kelly_fraction * 0.3
            return (kelly_fraction - margin, kelly_fraction + margin)
    
    def _calculate_drawdown_protection(self, historical_data: pd.DataFrame) -> float:
        """Calculate drawdown protection factor."""
        try:
            if len(historical_data) < 20:
                return 0.8  # Conservative protection
            
            # Calculate historical drawdowns
            cumulative_returns = (1 + historical_data['returns']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            
            max_drawdown = abs(drawdowns.min())
            
            # Protection factor based on max drawdown
            if max_drawdown > 0.30:  # 30%+ drawdown
                return 0.6
            elif max_drawdown > 0.20:  # 20%+ drawdown
                return 0.7
            elif max_drawdown > 0.10:  # 10%+ drawdown
                return 0.8
            else:
                return 0.9
                
        except Exception as e:
            logger.error(f"Error calculating drawdown protection: {e}")
            return 0.8
    
    def _create_default_kelly_parameters(self, symbol: str) -> KellyParameters:
        """Create conservative default Kelly parameters."""
        return KellyParameters(
            symbol=symbol,
            win_probability=0.45,
            avg_win=0.02,
            avg_loss=0.025,
            kelly_fraction=0.02,
            adjusted_kelly=0.01,
            confidence_interval=(0.005, 0.015),
            uncertainty_discount=0.7,
            max_drawdown_protection=0.8
        )

class VolatilityBasedSizer:
    """Volatility-based position sizing with regime awareness."""
    
    def __init__(self):
        """Initialize volatility-based sizer."""
        self.target_portfolio_vol = 0.15  # 15% target annual volatility
        self.lookback_periods = [20, 60, 120]  # Different volatility periods
        
    async def calculate_volatility_adjustment(self, symbol: str) -> VolatilityAdjustment:
        """Calculate volatility-based position sizing adjustment."""
        try:
            # Get market data for volatility calculation
            market_data = alpaca_client.get_market_data(symbol, limit=150)
            
            if market_data is None or len(market_data) < 30:
                return self._create_default_vol_adjustment(symbol)
            
            # Calculate multiple volatility measures
            vol_metrics = self._calculate_volatility_metrics(market_data)
            
            # Estimate regime volatility
            regime_vol = self._estimate_regime_volatility(market_data)
            
            # Calculate volatility percentile
            vol_percentile = self._calculate_volatility_percentile(market_data, vol_metrics['current'])
            
            # Calculate size multiplier based on volatility targeting
            size_multiplier = self._calculate_vol_size_multiplier(vol_metrics['current'], regime_vol)
            
            # Calculate risk budget allocation
            risk_budget = self._calculate_risk_budget(vol_metrics['current'])
            
            return VolatilityAdjustment(
                symbol=symbol,
                historical_volatility=vol_metrics['historical'],
                implied_volatility=None,  # Would integrate with options data
                volatility_percentile=vol_percentile,
                regime_volatility=regime_vol,
                vol_target=self.target_portfolio_vol,
                size_multiplier=size_multiplier,
                risk_budget=risk_budget
            )
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment for {symbol}: {e}")
            return self._create_default_vol_adjustment(symbol)
    
    def _calculate_volatility_metrics(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive volatility metrics."""
        try:
            returns = market_data['close'].pct_change().dropna()
            
            # Current volatility (20-day)
            current_vol = returns.tail(20).std() * np.sqrt(252)
            
            # Historical volatility (full period)
            historical_vol = returns.std() * np.sqrt(252)
            
            # EWMA volatility (exponentially weighted)
            ewma_span = 30
            ewma_vol = returns.ewm(span=ewma_span).std().iloc[-1] * np.sqrt(252)
            
            # GARCH-like volatility (simplified)
            garch_vol = self._simple_garch_vol(returns)
            
            return {
                'current': current_vol,
                'historical': historical_vol,
                'ewma': ewma_vol,
                'garch': garch_vol
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return {'current': 0.25, 'historical': 0.25, 'ewma': 0.25, 'garch': 0.25}
    
    def _simple_garch_vol(self, returns: pd.Series) -> float:
        """Simplified GARCH volatility estimation."""
        try:
            # Simple GARCH(1,1) approximation
            alpha = 0.1
            beta = 0.85
            
            variance = returns.var()  # Initial variance
            garch_variances = []
            
            for ret in returns:
                variance = alpha * ret**2 + beta * variance
                garch_variances.append(variance)
            
            return np.sqrt(garch_variances[-1] * 252)  # Annualized
            
        except Exception:
            return returns.std() * np.sqrt(252)  # Fallback to standard volatility
    
    def _estimate_regime_volatility(self, market_data: pd.DataFrame) -> float:
        """Estimate current market regime volatility."""
        try:
            returns = market_data['close'].pct_change().dropna()
            
            # Use rolling quantiles to identify regime changes
            short_vol = returns.tail(10).std() * np.sqrt(252)
            medium_vol = returns.tail(30).std() * np.sqrt(252)
            long_vol = returns.tail(60).std() * np.sqrt(252)
            
            # Weight recent volatility more heavily
            regime_vol = 0.5 * short_vol + 0.3 * medium_vol + 0.2 * long_vol
            
            return regime_vol
            
        except Exception as e:
            logger.error(f"Error estimating regime volatility: {e}")
            return 0.25  # Default volatility
    
    def _calculate_volatility_percentile(self, market_data: pd.DataFrame, current_vol: float) -> float:
        """Calculate current volatility percentile vs historical range."""
        try:
            returns = market_data['close'].pct_change().dropna()
            
            # Calculate rolling volatilities
            rolling_vols = []
            window = 20
            
            for i in range(window, len(returns)):
                period_returns = returns.iloc[i-window:i]
                vol = period_returns.std() * np.sqrt(252)
                rolling_vols.append(vol)
            
            if not rolling_vols:
                return 0.5
            
            # Calculate percentile
            percentile = stats.percentileofscore(rolling_vols, current_vol) / 100
            return percentile
            
        except Exception as e:
            logger.error(f"Error calculating volatility percentile: {e}")
            return 0.5
    
    def _calculate_vol_size_multiplier(self, current_vol: float, regime_vol: float) -> float:
        """Calculate size multiplier based on volatility targeting."""
        try:
            # Target volatility vs current volatility
            vol_ratio = self.target_portfolio_vol / max(current_vol, 0.05)
            
            # Apply regime adjustment
            regime_adjustment = self.target_portfolio_vol / max(regime_vol, 0.05)
            
            # Combined multiplier
            size_multiplier = (vol_ratio + regime_adjustment) / 2
            
            # Cap the multiplier to reasonable ranges
            return max(0.2, min(3.0, size_multiplier))
            
        except Exception as e:
            logger.error(f"Error calculating vol size multiplier: {e}")
            return 1.0
    
    def _calculate_risk_budget(self, volatility: float) -> float:
        """Calculate risk budget allocation based on volatility."""
        try:
            # Higher volatility gets smaller risk budget
            if volatility > 0.50:  # 50%+ volatility
                return 0.05  # 5% risk budget
            elif volatility > 0.30:  # 30%+ volatility
                return 0.08  # 8% risk budget
            elif volatility > 0.20:  # 20%+ volatility
                return 0.12  # 12% risk budget
            else:  # Low volatility
                return 0.15  # 15% risk budget
                
        except Exception as e:
            logger.error(f"Error calculating risk budget: {e}")
            return 0.10
    
    def _create_default_vol_adjustment(self, symbol: str) -> VolatilityAdjustment:
        """Create default volatility adjustment."""
        return VolatilityAdjustment(
            symbol=symbol,
            historical_volatility=0.25,
            implied_volatility=None,
            volatility_percentile=0.5,
            regime_volatility=0.25,
            vol_target=self.target_portfolio_vol,
            size_multiplier=1.0,
            risk_budget=0.10
        )

class CorrelationAwareAllocator:
    """Correlation-aware position sizing to prevent over-concentration."""
    
    def __init__(self):
        """Initialize correlation-aware allocator."""
        self.max_correlation_threshold = 0.7
        self.min_diversification_ratio = 0.6
        
    async def calculate_correlation_context(self, symbol: str, 
                                          current_portfolio: Dict,
                                          candidate_symbols: List[str] = None) -> CorrelationContext:
        """Calculate correlation context for position sizing."""
        try:
            # Get current portfolio symbols
            portfolio_symbols = list(current_portfolio.get('positions', {}).keys())
            
            # Add candidate symbol if not already in portfolio
            if symbol not in portfolio_symbols:
                portfolio_symbols.append(symbol)
            
            # Get correlation matrix
            correlation_matrix = await self._calculate_correlation_matrix(portfolio_symbols)
            
            # Identify correlation clusters
            correlation_cluster = self._identify_correlation_cluster(symbol, correlation_matrix, portfolio_symbols)
            
            # Calculate diversification metrics
            diversification_ratio = self._calculate_diversification_ratio(correlation_matrix)
            effective_positions = self._calculate_effective_positions(correlation_matrix)
            
            # Calculate concentration penalty
            concentration_penalty = self._calculate_concentration_penalty(
                symbol, current_portfolio, correlation_cluster
            )
            
            # Find maximum correlation with existing positions
            max_correlation = self._find_max_correlation(symbol, correlation_matrix, portfolio_symbols)
            
            return CorrelationContext(
                symbol=symbol,
                correlation_matrix=correlation_matrix,
                portfolio_symbols=portfolio_symbols,
                max_correlation=max_correlation,
                correlation_cluster=correlation_cluster,
                diversification_ratio=diversification_ratio,
                effective_positions=effective_positions,
                concentration_penalty=concentration_penalty
            )
            
        except Exception as e:
            logger.error(f"Error calculating correlation context for {symbol}: {e}")
            return self._create_default_correlation_context(symbol)
    
    async def _calculate_correlation_matrix(self, symbols: List[str]) -> Optional[np.ndarray]:
        """Calculate correlation matrix for given symbols."""
        try:
            if len(symbols) < 2:
                return None
            
            # Get historical data for all symbols
            returns_data = {}
            
            for sym in symbols:
                market_data = alpaca_client.get_market_data(sym, limit=100)
                if market_data is not None and len(market_data) > 50:
                    returns = market_data['close'].pct_change().dropna()
                    returns_data[sym] = returns
            
            if len(returns_data) < 2:
                return None
            
            # Align dates and create correlation matrix
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 20:
                return None
            
            correlation_matrix = returns_df.corr().values
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return None
    
    def _identify_correlation_cluster(self, symbol: str, correlation_matrix: Optional[np.ndarray],
                                    portfolio_symbols: List[str]) -> List[str]:
        """Identify symbols highly correlated with the target symbol."""
        try:
            if correlation_matrix is None or symbol not in portfolio_symbols:
                return []
            
            symbol_index = portfolio_symbols.index(symbol)
            correlations = correlation_matrix[symbol_index]
            
            cluster = []
            for i, corr in enumerate(correlations):
                if i != symbol_index and abs(corr) > self.max_correlation_threshold:
                    cluster.append(portfolio_symbols[i])
            
            return cluster
            
        except Exception as e:
            logger.error(f"Error identifying correlation cluster: {e}")
            return []
    
    def _calculate_diversification_ratio(self, correlation_matrix: Optional[np.ndarray]) -> float:
        """Calculate portfolio diversification ratio."""
        try:
            if correlation_matrix is None:
                return 1.0
            
            n = correlation_matrix.shape[0]
            if n < 2:
                return 1.0
            
            # Diversification ratio = weighted average correlation
            off_diagonal_sum = np.sum(correlation_matrix) - np.trace(correlation_matrix)
            avg_correlation = off_diagonal_sum / (n * (n - 1))
            
            # Convert to diversification ratio (lower correlation = higher diversification)
            diversification_ratio = 1.0 - abs(avg_correlation)
            
            return max(0.0, min(1.0, diversification_ratio))
            
        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {e}")
            return 0.7
    
    def _calculate_effective_positions(self, correlation_matrix: Optional[np.ndarray]) -> float:
        """Calculate effective number of positions considering correlations."""
        try:
            if correlation_matrix is None:
                return 1.0
            
            n = correlation_matrix.shape[0]
            if n < 2:
                return float(n)
            
            # Effective positions based on eigenvalue decomposition
            try:
                eigenvalues = np.linalg.eigvals(correlation_matrix)
                eigenvalues = eigenvalues[eigenvalues > 0]  # Remove negative/zero eigenvalues
                
                if len(eigenvalues) > 0:
                    # Shannon entropy approach
                    eigenvalues = eigenvalues / np.sum(eigenvalues)  # Normalize
                    effective_positions = np.exp(-np.sum(eigenvalues * np.log(eigenvalues + 1e-10)))
                    return min(effective_positions, float(n))
                else:
                    return float(n)
                    
            except Exception:
                # Fallback: use average correlation
                avg_corr = np.mean(correlation_matrix[correlation_matrix != 1.0])
                effective_positions = n / (1 + (n - 1) * abs(avg_corr))
                return max(1.0, effective_positions)
                
        except Exception as e:
            logger.error(f"Error calculating effective positions: {e}")
            return 1.0
    
    def _calculate_concentration_penalty(self, symbol: str, current_portfolio: Dict,
                                       correlation_cluster: List[str]) -> float:
        """Calculate penalty for correlation concentration."""
        try:
            if not correlation_cluster:
                return 1.0  # No penalty
            
            positions = current_portfolio.get('positions', {})
            total_value = current_portfolio.get('equity', 100000)
            
            # Calculate total exposure to correlation cluster
            cluster_exposure = 0.0
            for cluster_symbol in correlation_cluster:
                if cluster_symbol in positions:
                    position = positions[cluster_symbol]
                    market_value = abs(float(position.get('market_value', 0)))
                    cluster_exposure += market_value / total_value
            
            # Penalty increases with cluster concentration
            if cluster_exposure > 0.20:  # 20%+ in correlated positions
                penalty = 0.5  # 50% size reduction
            elif cluster_exposure > 0.15:  # 15%+ in correlated positions
                penalty = 0.7  # 30% size reduction
            elif cluster_exposure > 0.10:  # 10%+ in correlated positions
                penalty = 0.85  # 15% size reduction
            else:
                penalty = 1.0  # No penalty
            
            return penalty
            
        except Exception as e:
            logger.error(f"Error calculating concentration penalty: {e}")
            return 0.8  # Conservative penalty
    
    def _find_max_correlation(self, symbol: str, correlation_matrix: Optional[np.ndarray],
                            portfolio_symbols: List[str]) -> float:
        """Find maximum correlation with existing positions."""
        try:
            if correlation_matrix is None or symbol not in portfolio_symbols:
                return 0.0
            
            symbol_index = portfolio_symbols.index(symbol)
            correlations = correlation_matrix[symbol_index]
            
            # Exclude self-correlation
            other_correlations = np.concatenate([correlations[:symbol_index], correlations[symbol_index+1:]])
            
            if len(other_correlations) > 0:
                return float(np.max(np.abs(other_correlations)))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error finding max correlation: {e}")
            return 0.0
    
    def _create_default_correlation_context(self, symbol: str) -> CorrelationContext:
        """Create default correlation context."""
        return CorrelationContext(
            symbol=symbol,
            correlation_matrix=None,
            portfolio_symbols=[symbol],
            max_correlation=0.0,
            correlation_cluster=[],
            diversification_ratio=1.0,
            effective_positions=1.0,
            concentration_penalty=1.0
        )

class EnhancedPositionSizer:
    """Main enhanced position sizing engine."""
    
    def __init__(self):
        """Initialize enhanced position sizer."""
        self.kelly_calculator = KellyCriterionCalculator()
        self.volatility_sizer = VolatilityBasedSizer()
        self.correlation_allocator = CorrelationAwareAllocator()
        
        # Sizing method weights
        self.method_weights = {
            'kelly': 0.30,
            'volatility': 0.25,
            'risk_parity': 0.20,
            'correlation': 0.25
        }
        
        # Global limits
        self.absolute_max_position = 0.10  # 10% absolute maximum
        self.absolute_min_position = 0.001  # 0.1% absolute minimum
        
        logger.info("Enhanced Position Sizer initialized")
    
    async def calculate_enhanced_position_size(self, symbol: str, 
                                             short_signal: EnhancedShortSignal,
                                             current_portfolio: Dict) -> EnhancedPositionSize:
        """Calculate comprehensive position size using multiple methods."""
        try:
            logger.debug(f"Calculating enhanced position size for {symbol}")
            
            # Get all required analyses in parallel
            tasks = [
                self.kelly_calculator.calculate_kelly_parameters(symbol, short_signal),
                self.volatility_sizer.calculate_volatility_adjustment(symbol),
                self.correlation_allocator.calculate_correlation_context(symbol, current_portfolio),
                short_risk_manager.get_position_recommendations(symbol, current_portfolio)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Extract results (handle exceptions)
            kelly_params = results[0] if not isinstance(results[0], Exception) else None
            vol_adjustment = results[1] if not isinstance(results[1], Exception) else None
            correlation_context = results[2] if not isinstance(results[2], Exception) else None
            risk_recommendations = results[3] if not isinstance(results[3], Exception) else {}
            
            # Calculate individual sizing methods
            kelly_size = self._calculate_kelly_size(kelly_params, short_signal)
            volatility_size = self._calculate_volatility_size(vol_adjustment, short_signal)
            risk_parity_size = self._calculate_risk_parity_size(vol_adjustment, current_portfolio)
            correlation_size = self._calculate_correlation_adjusted_size(
                correlation_context, kelly_size, current_portfolio
            )
            
            # Determine primary method and calculate weighted average
            primary_method, recommended_size = self._calculate_weighted_size(
                kelly_size, volatility_size, risk_parity_size, correlation_size, short_signal
            )
            
            # Apply risk management constraints
            max_size = self._calculate_max_allowed_size(risk_recommendations, short_signal)
            min_size = self._calculate_min_viable_size(short_signal)
            
            # Final size with constraints
            final_size = max(min_size, min(recommended_size, max_size))
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(symbol, final_size, short_signal)
            
            # Generate reasoning and adjustments
            adjustments_applied, reasoning = self._generate_sizing_rationale(
                kelly_size, volatility_size, risk_parity_size, correlation_size,
                final_size, primary_method, risk_recommendations
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                kelly_params, vol_adjustment, correlation_context, short_signal
            )
            
            # Generate risk controls
            position_limits, stop_loss, take_profit = self._generate_risk_controls(
                symbol, final_size, short_signal, risk_recommendations
            )
            
            return EnhancedPositionSize(
                symbol=symbol,
                kelly_size=kelly_size,
                volatility_size=volatility_size,
                risk_parity_size=risk_parity_size,
                correlation_adjusted_size=correlation_size,
                recommended_size=final_size,
                max_size=max_size,
                min_size=min_size,
                expected_return=risk_metrics['expected_return'],
                expected_volatility=risk_metrics['expected_volatility'],
                expected_sharpe=risk_metrics['expected_sharpe'],
                value_at_risk=risk_metrics['value_at_risk'],
                max_drawdown_estimate=risk_metrics['max_drawdown'],
                primary_method=primary_method,
                adjustments_applied=adjustments_applied,
                confidence_score=confidence_score,
                reasoning=reasoning,
                position_limits=position_limits,
                stop_loss_level=stop_loss,
                take_profit_level=take_profit,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating enhanced position size for {symbol}: {e}")
            return self._create_conservative_position_size(symbol, short_signal)
    
    def _calculate_kelly_size(self, kelly_params: Optional[KellyParameters],
                            signal: EnhancedShortSignal) -> float:
        """Calculate Kelly-based position size."""
        try:
            if not kelly_params:
                # Fallback calculation based on signal
                return min(0.02, signal.signal_strength * 0.03)
            
            # Use adjusted Kelly with additional signal-based adjustment
            base_kelly = kelly_params.adjusted_kelly
            signal_adjustment = 1.0 + (signal.confidence - 0.5) * 0.2
            
            kelly_size = base_kelly * signal_adjustment
            
            # Cap Kelly size to reasonable limits
            return max(0.001, min(0.08, kelly_size))
            
        except Exception as e:
            logger.error(f"Error calculating Kelly size: {e}")
            return 0.01
    
    def _calculate_volatility_size(self, vol_adjustment: Optional[VolatilityAdjustment],
                                 signal: EnhancedShortSignal) -> float:
        """Calculate volatility-adjusted position size."""
        try:
            if not vol_adjustment:
                return 0.02  # Default 2%
            
            # Use risk budget as base size
            base_size = vol_adjustment.risk_budget
            
            # Apply volatility multiplier
            vol_adjusted_size = base_size * vol_adjustment.size_multiplier
            
            # Apply signal adjustment
            signal_adjustment = 0.5 + (signal.signal_strength * 0.5)
            final_size = vol_adjusted_size * signal_adjustment
            
            return max(0.001, min(0.08, final_size))
            
        except Exception as e:
            logger.error(f"Error calculating volatility size: {e}")
            return 0.02
    
    def _calculate_risk_parity_size(self, vol_adjustment: Optional[VolatilityAdjustment],
                                  current_portfolio: Dict) -> float:
        """Calculate risk parity allocation."""
        try:
            if not vol_adjustment:
                volatility = 0.25  # Default volatility
            else:
                volatility = vol_adjustment.historical_volatility
            
            # Target equal risk contribution
            target_risk_contribution = 0.05  # 5% risk contribution
            
            # Position size = target risk / volatility
            risk_parity_size = target_risk_contribution / max(volatility, 0.05)
            
            # Adjust for current portfolio concentration
            positions = current_portfolio.get('positions', {})
            if len(positions) > 5:  # Well diversified
                concentration_adjustment = 1.0
            elif len(positions) > 2:  # Moderately diversified
                concentration_adjustment = 0.8
            else:  # Concentrated
                concentration_adjustment = 0.6
            
            adjusted_size = risk_parity_size * concentration_adjustment
            
            return max(0.001, min(0.06, adjusted_size))
            
        except Exception as e:
            logger.error(f"Error calculating risk parity size: {e}")
            return 0.03
    
    def _calculate_correlation_adjusted_size(self, correlation_context: Optional[CorrelationContext],
                                           base_size: float, current_portfolio: Dict) -> float:
        """Calculate correlation-adjusted position size."""
        try:
            if not correlation_context:
                return base_size
            
            # Apply concentration penalty
            correlation_adjusted = base_size * correlation_context.concentration_penalty
            
            # Apply diversification adjustment
            diversification_bonus = 1.0 + (correlation_context.diversification_ratio - 0.5) * 0.2
            final_size = correlation_adjusted * diversification_bonus
            
            return max(0.001, min(0.08, final_size))
            
        except Exception as e:
            logger.error(f"Error calculating correlation adjusted size: {e}")
            return base_size * 0.8  # Conservative adjustment
    
    def _calculate_weighted_size(self, kelly_size: float, volatility_size: float,
                               risk_parity_size: float, correlation_size: float,
                               signal: EnhancedShortSignal) -> Tuple[str, float]:
        """Calculate weighted average size and determine primary method."""
        try:
            # Calculate weighted average
            weighted_size = (
                kelly_size * self.method_weights['kelly'] +
                volatility_size * self.method_weights['volatility'] +
                risk_parity_size * self.method_weights['risk_parity'] +
                correlation_size * self.method_weights['correlation']
            )
            
            # Determine primary method (closest to weighted average)
            sizes = {
                'kelly': kelly_size,
                'volatility': volatility_size,
                'risk_parity': risk_parity_size,
                'correlation': correlation_size
            }
            
            # Find method closest to weighted average
            primary_method = min(sizes, key=lambda k: abs(sizes[k] - weighted_size))
            
            # Apply signal-based final adjustment
            signal_multiplier = 0.7 + (signal.confidence * 0.6)  # 0.7 to 1.3 range
            final_size = weighted_size * signal_multiplier
            
            return primary_method, final_size
            
        except Exception as e:
            logger.error(f"Error calculating weighted size: {e}")
            return 'volatility', 0.02
    
    def _calculate_max_allowed_size(self, risk_recommendations: Dict, signal: EnhancedShortSignal) -> float:
        """Calculate maximum allowed position size."""
        try:
            # Start with absolute maximum
            max_size = self.absolute_max_position
            
            # Apply risk manager constraints
            if risk_recommendations and 'max_position_size' in risk_recommendations:
                risk_max = risk_recommendations['max_position_size']
                max_size = min(max_size, risk_max)
            
            # Apply signal-based constraints
            if signal.signal_type == 'WEAK_SHORT':
                max_size *= 0.5  # Halve size for weak signals
            elif signal.signal_type == 'MONITOR':
                max_size *= 0.3  # Very small size for monitoring
            
            # Apply liquidity constraints
            if signal.liquidity_score < 0.5:
                max_size *= 0.6  # Reduce for low liquidity
            
            return max(self.absolute_min_position, max_size)
            
        except Exception as e:
            logger.error(f"Error calculating max allowed size: {e}")
            return 0.05  # Conservative maximum
    
    def _calculate_min_viable_size(self, signal: EnhancedShortSignal) -> float:
        """Calculate minimum viable position size."""
        try:
            # Base minimum
            min_size = self.absolute_min_position
            
            # Increase minimum for high-confidence signals
            if signal.confidence > 0.8 and signal.signal_strength > 0.7:
                min_size = max(min_size, 0.005)  # 0.5% minimum for high conviction
            
            return min_size
            
        except Exception as e:
            logger.error(f"Error calculating min viable size: {e}")
            return self.absolute_min_position
    
    async def _calculate_risk_metrics(self, symbol: str, position_size: float,
                                    signal: EnhancedShortSignal) -> Dict[str, float]:
        """Calculate expected risk metrics for the position."""
        try:
            # Get historical data for calculations
            market_data = alpaca_client.get_market_data(symbol, limit=100)
            
            if market_data is None or len(market_data) < 20:
                # Default risk metrics
                return {
                    'expected_return': 0.02,
                    'expected_volatility': 0.25,
                    'expected_sharpe': 0.08,
                    'value_at_risk': 0.08,
                    'max_drawdown': 0.15
                }
            
            # Calculate historical metrics
            returns = market_data['close'].pct_change().dropna()
            
            # For short positions, invert returns
            short_returns = -returns
            
            # Expected return (adjusted for signal)
            historical_return = short_returns.mean() * 252  # Annualized
            signal_adjustment = (signal.signal_strength - 0.5) * 0.1
            expected_return = historical_return + signal_adjustment
            
            # Expected volatility
            expected_volatility = short_returns.std() * np.sqrt(252)
            
            # Expected Sharpe ratio
            expected_sharpe = expected_return / max(expected_volatility, 0.01)
            
            # Value at Risk (95% confidence)
            value_at_risk = abs(np.percentile(short_returns, 5)) * position_size
            
            # Maximum drawdown estimate
            cumulative = (1 + short_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdowns.min()) * position_size
            
            return {
                'expected_return': expected_return,
                'expected_volatility': expected_volatility,
                'expected_sharpe': expected_sharpe,
                'value_at_risk': value_at_risk,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics for {symbol}: {e}")
            return {
                'expected_return': 0.02,
                'expected_volatility': 0.25,
                'expected_sharpe': 0.08,
                'value_at_risk': 0.08,
                'max_drawdown': 0.15
            }
    
    def _generate_sizing_rationale(self, kelly_size: float, volatility_size: float,
                                 risk_parity_size: float, correlation_size: float,
                                 final_size: float, primary_method: str,
                                 risk_recommendations: Dict) -> Tuple[List[str], str]:
        """Generate sizing rationale and list of adjustments."""
        try:
            adjustments = []
            reasoning_parts = []
            
            # Primary method
            reasoning_parts.append(f"Primary method: {primary_method}")
            
            # Method comparison
            methods_str = f"Kelly: {kelly_size:.1%}, Vol: {volatility_size:.1%}, Risk Parity: {risk_parity_size:.1%}, Correlation: {correlation_size:.1%}"
            reasoning_parts.append(f"Method sizes - {methods_str}")
            
            # Adjustments applied
            if final_size < kelly_size * 0.8:
                adjustments.append("Kelly size reduced for risk management")
            
            if final_size < volatility_size * 0.8:
                adjustments.append("Volatility adjustment applied")
            
            if correlation_size < kelly_size * 0.9:
                adjustments.append("Correlation penalty applied")
            
            if risk_recommendations and final_size < risk_recommendations.get('max_position_size', 1.0):
                adjustments.append("Risk manager limits applied")
            
            # Final reasoning
            if adjustments:
                reasoning_parts.append(f"Adjustments: {'; '.join(adjustments)}")
            
            reasoning = " | ".join(reasoning_parts)
            
            return adjustments, reasoning
            
        except Exception as e:
            logger.error(f"Error generating sizing rationale: {e}")
            return ["Error in analysis"], "Position sizing with conservative defaults"
    
    def _calculate_confidence_score(self, kelly_params: Optional[KellyParameters],
                                  vol_adjustment: Optional[VolatilityAdjustment],
                                  correlation_context: Optional[CorrelationContext],
                                  signal: EnhancedShortSignal) -> float:
        """Calculate confidence score for the position sizing."""
        try:
            confidence_factors = []
            
            # Signal confidence
            confidence_factors.append(signal.confidence)
            
            # Data quality confidence
            if kelly_params:
                # Kelly confidence based on uncertainty discount
                kelly_confidence = kelly_params.uncertainty_discount
                confidence_factors.append(kelly_confidence)
            
            # Volatility confidence
            if vol_adjustment:
                # Higher confidence for mid-range volatility percentiles
                vol_percentile = vol_adjustment.volatility_percentile
                vol_confidence = 1.0 - abs(vol_percentile - 0.5) * 2  # Peak at 50th percentile
                confidence_factors.append(vol_confidence)
            
            # Diversification confidence
            if correlation_context:
                div_confidence = correlation_context.diversification_ratio
                confidence_factors.append(div_confidence)
            
            # Overall confidence
            overall_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
            
            return max(0.1, min(1.0, overall_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.6  # Moderate confidence default
    
    def _generate_risk_controls(self, symbol: str, position_size: float,
                              signal: EnhancedShortSignal, risk_recommendations: Dict) -> Tuple[Dict[str, float], Optional[float], Optional[float]]:
        """Generate position-specific risk controls."""
        try:
            # Position limits
            position_limits = {
                'max_daily_loss': position_size * 0.5,  # 50% of position
                'max_portfolio_impact': position_size,
                'concentration_limit': min(position_size * 2, 0.1)  # Max 10%
            }
            
            # Stop loss level from risk recommendations or calculate
            stop_loss_level = risk_recommendations.get('stop_loss_level')
            if not stop_loss_level:
                # Calculate based on position size and volatility
                current_price = alpaca_client.get_current_price(symbol)
                if current_price:
                    # Stop loss at 2x the position size above current price
                    stop_loss_level = current_price * (1 + position_size * 2)
            
            # Take profit level (target 2:1 reward:risk ratio)
            take_profit_level = None
            if stop_loss_level:
                current_price = alpaca_client.get_current_price(symbol)
                if current_price:
                    risk_amount = stop_loss_level - current_price
                    take_profit_level = current_price - (risk_amount * 2)  # 2:1 ratio
            
            return position_limits, stop_loss_level, take_profit_level
            
        except Exception as e:
            logger.error(f"Error generating risk controls: {e}")
            return {'max_daily_loss': position_size * 0.5}, None, None
    
    def _create_conservative_position_size(self, symbol: str, signal: EnhancedShortSignal) -> EnhancedPositionSize:
        """Create conservative position size when calculation fails."""
        conservative_size = 0.01  # 1% conservative default
        
        return EnhancedPositionSize(
            symbol=symbol,
            kelly_size=conservative_size,
            volatility_size=conservative_size,
            risk_parity_size=conservative_size,
            correlation_adjusted_size=conservative_size,
            recommended_size=conservative_size,
            max_size=conservative_size * 2,
            min_size=conservative_size * 0.5,
            expected_return=0.02,
            expected_volatility=0.25,
            expected_sharpe=0.08,
            value_at_risk=0.05,
            max_drawdown_estimate=0.10,
            primary_method='conservative',
            adjustments_applied=['Conservative sizing due to analysis failure'],
            confidence_score=0.3,
            reasoning="Conservative position sizing due to insufficient data or analysis failure",
            position_limits={'max_daily_loss': conservative_size * 0.5},
            stop_loss_level=None,
            take_profit_level=None,
            timestamp=datetime.now()
        )
    
    def get_sizer_status(self) -> Dict[str, Any]:
        """Get position sizer status and configuration."""
        return {
            'method_weights': self.method_weights,
            'absolute_max_position': self.absolute_max_position,
            'absolute_min_position': self.absolute_min_position,
            'target_portfolio_vol': self.volatility_sizer.target_portfolio_vol,
            'max_correlation_threshold': self.correlation_allocator.max_correlation_threshold,
            'initialized': True
        }

# Global instance
enhanced_position_sizer = EnhancedPositionSizer()

# Convenience functions
async def calculate_enhanced_position_size(symbol: str, short_signal: EnhancedShortSignal,
                                         current_portfolio: Dict) -> EnhancedPositionSize:
    """Calculate enhanced position size for a short signal."""
    return await enhanced_position_sizer.calculate_enhanced_position_size(symbol, short_signal, current_portfolio)

async def calculate_kelly_parameters(symbol: str, short_signal: EnhancedShortSignal) -> KellyParameters:
    """Calculate Kelly Criterion parameters for a symbol."""
    return await enhanced_position_sizer.kelly_calculator.calculate_kelly_parameters(symbol, short_signal)

async def calculate_volatility_adjustment(symbol: str) -> VolatilityAdjustment:
    """Calculate volatility-based position adjustment."""
    return await enhanced_position_sizer.volatility_sizer.calculate_volatility_adjustment(symbol)

async def calculate_correlation_context(symbol: str, current_portfolio: Dict) -> CorrelationContext:
    """Calculate correlation context for position sizing."""
    return await enhanced_position_sizer.correlation_allocator.calculate_correlation_context(symbol, current_portfolio)

def get_position_sizer_status() -> Dict[str, Any]:
    """Get position sizer status."""
    return enhanced_position_sizer.get_sizer_status()