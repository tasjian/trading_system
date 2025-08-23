#!/usr/bin/env python3
"""
Advanced Risk Management Framework for Short Selling

This module provides comprehensive risk management specifically designed for short selling,
including squeeze risk assessment, position sizing limits, correlation analysis, and
enhanced pre-trade validation.

Key Features:
- Short squeeze risk assessment using options data and technical indicators
- Position concentration limits and correlation-aware risk management
- Dynamic risk limits based on market conditions and portfolio exposure
- Real-time monitoring of margin requirements and portfolio beta
- Integration with borrow cost monitoring for comprehensive risk assessment

The framework ensures proper risk controls while maximizing short selling opportunities
within acceptable risk parameters.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import math
from scipy import stats
from scipy.optimize import minimize

from tools.alpaca_client import alpaca_client
from core.borrow_cost_monitor import borrow_cost_monitor, ShortabilityAnalysis
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class SqueezeRiskMetrics:
    """Short squeeze risk assessment metrics."""
    symbol: str
    squeeze_risk_score: float  # 0-100 (higher = higher squeeze risk)
    short_interest_ratio: float
    days_to_cover: float
    options_skew: Optional[float]  # Put/call ratio and volatility skew
    price_momentum: float
    volume_surge: float
    technical_breakout_risk: float
    
    # Risk factors
    risk_factors: List[str]
    protective_factors: List[str]
    
    # Recommendations
    max_position_size: float  # Maximum recommended position as % of portfolio
    stop_loss_level: Optional[float]  # Recommended stop loss price
    monitoring_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    
    last_updated: datetime

@dataclass
class PositionRiskLimits:
    """Position-specific risk limits and constraints."""
    symbol: str
    max_position_size_percent: float  # Maximum position size as % of portfolio
    max_daily_loss_percent: float  # Maximum daily loss tolerance
    max_correlation_exposure: float  # Maximum correlated position exposure
    margin_requirement: float  # Estimated margin requirement
    
    # Dynamic limits based on volatility and market conditions
    volatility_adjusted_limit: float
    market_regime_adjustment: float
    liquidity_discount: float
    
    effective_limit: float  # Final calculated limit
    reasoning: str

@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk metrics for short positions."""
    total_short_exposure: float  # Total short exposure as % of portfolio
    short_position_count: int
    average_short_position_size: float
    portfolio_beta: float  # Portfolio beta including short positions
    
    # Concentration risks
    max_single_position: float
    sector_concentration: Dict[str, float]
    correlation_risk_score: float
    
    # Margin and liquidity
    total_margin_requirement: float
    available_margin: float
    liquidity_score: float
    
    # Risk limits compliance
    within_risk_limits: bool
    limit_violations: List[str]
    recommended_actions: List[str]

@dataclass
class RiskAlert:
    """Risk alert for monitoring and notifications."""
    alert_id: str
    symbol: str
    alert_type: str  # 'SQUEEZE_RISK', 'POSITION_LIMIT', 'CORRELATION', 'MARGIN'
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message: str
    recommended_action: str
    expires_at: datetime
    created_at: datetime

class SqueezeRiskAnalyzer:
    """Analyze short squeeze risk using multiple indicators."""
    
    def __init__(self):
        """Initialize squeeze risk analyzer."""
        self.risk_cache: Dict[str, SqueezeRiskMetrics] = {}
        self.cache_ttl_minutes = 30
        
    async def assess_squeeze_risk(self, symbol: str) -> SqueezeRiskMetrics:
        """Comprehensive short squeeze risk assessment."""
        try:
            # Check cache
            if symbol in self.risk_cache:
                cached_metrics = self.risk_cache[symbol]
                age_minutes = (datetime.now() - cached_metrics.last_updated).total_seconds() / 60
                if age_minutes < self.cache_ttl_minutes:
                    return cached_metrics
            
            logger.debug(f"Analyzing squeeze risk for {symbol}")
            
            # Gather data for analysis
            market_data = await self._get_market_data(symbol)
            short_interest_data = await self._estimate_short_interest(symbol)
            options_data = await self._analyze_options_activity(symbol)
            technical_data = await self._analyze_technical_indicators(symbol, market_data)
            
            # Calculate individual risk components
            short_interest_risk = self._calculate_short_interest_risk(short_interest_data)
            options_risk = self._calculate_options_risk(options_data)
            momentum_risk = self._calculate_momentum_risk(technical_data)
            volume_risk = self._calculate_volume_risk(market_data)
            breakout_risk = self._calculate_breakout_risk(technical_data)
            
            # Combine into overall squeeze risk score
            risk_weights = {
                'short_interest': 0.30,
                'options': 0.20,
                'momentum': 0.25,
                'volume': 0.15,
                'breakout': 0.10
            }
            
            squeeze_risk_score = (
                short_interest_risk * risk_weights['short_interest'] +
                options_risk * risk_weights['options'] +
                momentum_risk * risk_weights['momentum'] +
                volume_risk * risk_weights['volume'] +
                breakout_risk * risk_weights['breakout']
            )
            
            # Identify risk and protective factors
            risk_factors, protective_factors = self._identify_risk_factors(
                short_interest_data, options_data, technical_data, market_data
            )
            
            # Generate recommendations
            max_position_size, stop_loss_level, monitoring_level = self._generate_risk_recommendations(
                squeeze_risk_score, risk_factors, technical_data
            )
            
            # Create squeeze risk metrics
            metrics = SqueezeRiskMetrics(
                symbol=symbol,
                squeeze_risk_score=squeeze_risk_score,
                short_interest_ratio=short_interest_data.get('short_ratio', 0.0),
                days_to_cover=short_interest_data.get('days_to_cover', 0.0),
                options_skew=options_data.get('put_call_ratio'),
                price_momentum=technical_data.get('momentum', 0.0),
                volume_surge=market_data.get('volume_surge', 0.0),
                technical_breakout_risk=breakout_risk,
                risk_factors=risk_factors,
                protective_factors=protective_factors,
                max_position_size=max_position_size,
                stop_loss_level=stop_loss_level,
                monitoring_level=monitoring_level,
                last_updated=datetime.now()
            )
            
            # Cache the result
            self.risk_cache[symbol] = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error assessing squeeze risk for {symbol}: {e}")
            return self._create_default_squeeze_metrics(symbol)
    
    async def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data for squeeze risk analysis."""
        try:
            # Get extended market data for comprehensive analysis
            market_df = alpaca_client.get_market_data(symbol, limit=50)
            
            if market_df is None or len(market_df) == 0:
                return {}
            
            # Calculate key metrics
            current_price = market_df['close'].iloc[-1]
            volume_data = market_df['volume'].values
            price_data = market_df['close'].values
            
            # Volume analysis
            recent_volume = np.mean(volume_data[-5:])
            historical_volume = np.mean(volume_data[:-5]) if len(volume_data) > 5 else recent_volume
            volume_surge = (recent_volume / max(historical_volume, 1)) - 1
            
            # Price volatility
            returns = np.diff(price_data) / price_data[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            return {
                'price_data': price_data,
                'volume_data': volume_data,
                'current_price': current_price,
                'volatility': volatility,
                'volume_surge': volume_surge,
                'recent_volume': recent_volume,
                'avg_volume': np.mean(volume_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {}
    
    async def _estimate_short_interest(self, symbol: str) -> Dict[str, Any]:
        """Estimate short interest metrics from available data."""
        try:
            # In practice, this would use specialized short interest data sources
            # For now, we'll estimate based on available market indicators
            
            # Get borrow cost data which may include short interest estimates
            shortability = await borrow_cost_monitor.get_borrow_cost_analysis(symbol)
            
            # Estimate short interest ratio based on borrow costs and availability
            if shortability and shortability.borrow_cost:
                borrow_fee = shortability.borrow_cost.borrow_fee_rate
                availability = shortability.borrow_cost.availability
                
                # Higher borrow fees typically indicate higher short interest
                if borrow_fee > 0.20:  # 20%+ annual
                    estimated_short_ratio = 0.25
                elif borrow_fee > 0.10:  # 10%+ annual
                    estimated_short_ratio = 0.15
                elif borrow_fee > 0.05:  # 5%+ annual
                    estimated_short_ratio = 0.08
                else:
                    estimated_short_ratio = 0.03
                
                # Adjust based on availability
                if availability == 'HTB':
                    estimated_short_ratio *= 1.5
                elif availability == 'NOT_AVAILABLE':
                    estimated_short_ratio *= 2.0
                
                # Estimate days to cover (simplified)
                market_data = await self._get_market_data(symbol)
                avg_volume = market_data.get('avg_volume', 1000000)
                
                # Assume short interest is estimated_short_ratio * shares outstanding
                # For simplicity, use volume as proxy for liquidity
                days_to_cover = estimated_short_ratio * 100000000 / max(avg_volume, 100000)  # Rough estimate
                
                return {
                    'short_ratio': min(estimated_short_ratio, 0.50),  # Cap at 50%
                    'days_to_cover': min(days_to_cover, 20.0),  # Cap at 20 days
                    'borrow_fee': borrow_fee,
                    'availability': availability
                }
            
            # Fallback estimation
            return {
                'short_ratio': 0.05,  # Conservative 5%
                'days_to_cover': 2.0,
                'borrow_fee': 0.03,
                'availability': 'UNKNOWN'
            }
            
        except Exception as e:
            logger.error(f"Error estimating short interest for {symbol}: {e}")
            return {'short_ratio': 0.05, 'days_to_cover': 2.0}
    
    async def _analyze_options_activity(self, symbol: str) -> Dict[str, Any]:
        """Analyze options activity for squeeze indicators."""
        try:
            # This would integrate with options data providers in production
            # For now, we'll use price action as a proxy for options activity
            
            market_data = await self._get_market_data(symbol)
            if not market_data:
                return {}
            
            price_data = market_data.get('price_data', [])
            if len(price_data) < 10:
                return {}
            
            # Estimate put/call ratio based on price momentum and volatility
            recent_returns = np.diff(price_data[-10:]) / price_data[-10:-1]
            momentum = np.mean(recent_returns)
            volatility = np.std(recent_returns)
            
            # Higher volatility and negative momentum suggest higher put activity
            estimated_put_call_ratio = 1.0 + (volatility * 5) - (momentum * 2)
            estimated_put_call_ratio = max(0.3, min(3.0, estimated_put_call_ratio))
            
            # Estimate gamma exposure (simplified)
            # High gamma can amplify squeezes
            price_change = abs(price_data[-1] - price_data[-5]) / price_data[-5]
            estimated_gamma_exposure = price_change * volatility * 10
            
            return {
                'put_call_ratio': estimated_put_call_ratio,
                'gamma_exposure': estimated_gamma_exposure,
                'volatility_skew': volatility * 2,  # Simplified skew estimate
                'options_volume_estimate': market_data.get('recent_volume', 0) * 0.1
            }
            
        except Exception as e:
            logger.error(f"Error analyzing options activity for {symbol}: {e}")
            return {}
    
    async def _analyze_technical_indicators(self, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """Analyze technical indicators for squeeze risk."""
        try:
            price_data = market_data.get('price_data', [])
            if len(price_data) < 20:
                return {}
            
            prices = np.array(price_data)
            
            # Momentum indicators
            short_ma = np.mean(prices[-5:])
            long_ma = np.mean(prices[-20:])
            momentum = (short_ma - long_ma) / long_ma
            
            # RSI calculation
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            if len(gains) >= 14:
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                rs = avg_gain / max(avg_loss, 0.001)
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50.0
            
            # Support/resistance levels
            recent_high = np.max(prices[-20:])
            recent_low = np.min(prices[-20:])
            current_price = prices[-1]
            
            # Distance from resistance (breakout risk)
            resistance_distance = (recent_high - current_price) / current_price
            
            # Bollinger Bands
            ma_20 = np.mean(prices[-20:])
            std_20 = np.std(prices[-20:])
            upper_band = ma_20 + (2 * std_20)
            lower_band = ma_20 - (2 * std_20)
            
            bb_position = (current_price - lower_band) / (upper_band - lower_band)
            
            return {
                'momentum': momentum,
                'rsi': rsi,
                'resistance_distance': resistance_distance,
                'bb_position': bb_position,
                'volatility': std_20 / ma_20,
                'trend_strength': abs(momentum),
                'breakout_proximity': 1.0 - resistance_distance
            }
            
        except Exception as e:
            logger.error(f"Error analyzing technical indicators for {symbol}: {e}")
            return {}
    
    def _calculate_short_interest_risk(self, short_data: Dict) -> float:
        """Calculate risk score based on short interest metrics."""
        try:
            short_ratio = short_data.get('short_ratio', 0.0)
            days_to_cover = short_data.get('days_to_cover', 0.0)
            
            # High short interest increases squeeze risk
            short_ratio_score = min(short_ratio * 200, 80)  # Max 80 points
            
            # High days to cover increases squeeze risk
            days_score = min(days_to_cover * 5, 20)  # Max 20 points
            
            return short_ratio_score + days_score
            
        except Exception as e:
            logger.error(f"Error calculating short interest risk: {e}")
            return 20.0  # Conservative default
    
    def _calculate_options_risk(self, options_data: Dict) -> float:
        """Calculate risk score based on options activity."""
        try:
            if not options_data:
                return 10.0  # Neutral when no data
            
            put_call_ratio = options_data.get('put_call_ratio', 1.0)
            gamma_exposure = options_data.get('gamma_exposure', 0.0)
            
            # Extreme put/call ratios can indicate squeeze setup
            if put_call_ratio > 2.0:  # High put activity
                pc_score = 15.0
            elif put_call_ratio < 0.5:  # High call activity (more risky for shorts)
                pc_score = 30.0
            else:
                pc_score = 5.0
            
            # High gamma exposure increases volatility and squeeze risk
            gamma_score = min(gamma_exposure * 50, 15)
            
            return pc_score + gamma_score
            
        except Exception as e:
            logger.error(f"Error calculating options risk: {e}")
            return 10.0
    
    def _calculate_momentum_risk(self, technical_data: Dict) -> float:
        """Calculate risk score based on price momentum."""
        try:
            if not technical_data:
                return 15.0
            
            momentum = technical_data.get('momentum', 0.0)
            rsi = technical_data.get('rsi', 50.0)
            trend_strength = technical_data.get('trend_strength', 0.0)
            
            # Positive momentum increases squeeze risk for shorts
            momentum_score = max(0, momentum * 100)  # Positive momentum = risk
            momentum_score = min(momentum_score, 30)
            
            # Oversold conditions (low RSI) can lead to bounces
            if rsi < 30:
                rsi_score = (30 - rsi) * 0.5
            else:
                rsi_score = 0
            
            # Strong trends can continue and squeeze shorts
            trend_score = trend_strength * 10
            
            return momentum_score + rsi_score + trend_score
            
        except Exception as e:
            logger.error(f"Error calculating momentum risk: {e}")
            return 15.0
    
    def _calculate_volume_risk(self, market_data: Dict) -> float:
        """Calculate risk score based on volume patterns."""
        try:
            volume_surge = market_data.get('volume_surge', 0.0)
            
            # Volume surges can indicate institutional buying or covering
            if volume_surge > 2.0:  # 200%+ volume increase
                return 25.0
            elif volume_surge > 1.0:  # 100%+ volume increase
                return 15.0
            elif volume_surge > 0.5:  # 50%+ volume increase
                return 8.0
            else:
                return 2.0
            
        except Exception as e:
            logger.error(f"Error calculating volume risk: {e}")
            return 5.0
    
    def _calculate_breakout_risk(self, technical_data: Dict) -> float:
        """Calculate risk score based on breakout probability."""
        try:
            if not technical_data:
                return 5.0
            
            resistance_distance = technical_data.get('resistance_distance', 0.1)
            bb_position = technical_data.get('bb_position', 0.5)
            volatility = technical_data.get('volatility', 0.02)
            
            # Close to resistance = higher breakout risk
            resistance_score = max(0, (0.05 - resistance_distance) * 200)
            
            # High Bollinger Band position = breakout risk
            bb_score = max(0, (bb_position - 0.8) * 50) if bb_position > 0.8 else 0
            
            # High volatility increases breakout probability
            vol_score = min(volatility * 100, 10)
            
            return resistance_score + bb_score + vol_score
            
        except Exception as e:
            logger.error(f"Error calculating breakout risk: {e}")
            return 5.0
    
    def _identify_risk_factors(self, short_data: Dict, options_data: Dict, 
                             technical_data: Dict, market_data: Dict) -> Tuple[List[str], List[str]]:
        """Identify specific risk and protective factors."""
        risk_factors = []
        protective_factors = []
        
        try:
            # Short interest factors
            short_ratio = short_data.get('short_ratio', 0.0)
            if short_ratio > 0.20:
                risk_factors.append(f"High short interest ({short_ratio:.1%})")
            elif short_ratio < 0.05:
                protective_factors.append("Low short interest")
            
            days_to_cover = short_data.get('days_to_cover', 0.0)
            if days_to_cover > 5:
                risk_factors.append(f"High days to cover ({days_to_cover:.1f})")
            
            # Technical factors
            if technical_data:
                momentum = technical_data.get('momentum', 0.0)
                if momentum > 0.05:
                    risk_factors.append("Strong upward momentum")
                elif momentum < -0.10:
                    protective_factors.append("Strong downward momentum")
                
                rsi = technical_data.get('rsi', 50.0)
                if rsi < 30:
                    risk_factors.append("Oversold conditions (potential bounce)")
                elif rsi > 70:
                    protective_factors.append("Overbought conditions")
                
                resistance_distance = technical_data.get('resistance_distance', 0.1)
                if resistance_distance < 0.02:
                    risk_factors.append("Near resistance level")
            
            # Volume factors
            volume_surge = market_data.get('volume_surge', 0.0)
            if volume_surge > 1.0:
                risk_factors.append(f"Volume surge ({volume_surge:.0%})")
            
            # Options factors
            if options_data:
                put_call_ratio = options_data.get('put_call_ratio', 1.0)
                if put_call_ratio < 0.5:
                    risk_factors.append("Heavy call buying activity")
                elif put_call_ratio > 2.0:
                    protective_factors.append("Heavy put buying activity")
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
        
        return risk_factors, protective_factors
    
    def _generate_risk_recommendations(self, squeeze_risk_score: float, 
                                     risk_factors: List[str], 
                                     technical_data: Dict) -> Tuple[float, Optional[float], str]:
        """Generate risk management recommendations."""
        try:
            # Maximum position size based on squeeze risk
            if squeeze_risk_score >= 80:
                max_position_size = 0.005  # 0.5% max
                monitoring_level = 'CRITICAL'
            elif squeeze_risk_score >= 60:
                max_position_size = 0.01   # 1% max
                monitoring_level = 'HIGH'
            elif squeeze_risk_score >= 40:
                max_position_size = 0.02   # 2% max
                monitoring_level = 'MEDIUM'
            else:
                max_position_size = 0.03   # 3% max
                monitoring_level = 'LOW'
            
            # Stop loss level based on technical analysis
            stop_loss_level = None
            if technical_data:
                current_price = technical_data.get('current_price')
                resistance_distance = technical_data.get('resistance_distance', 0.05)
                
                if current_price:
                    # Set stop loss above recent resistance with buffer
                    stop_loss_buffer = max(0.02, resistance_distance + 0.01)  # Min 2% buffer
                    stop_loss_level = current_price * (1 + stop_loss_buffer)
            
            return max_position_size, stop_loss_level, monitoring_level
            
        except Exception as e:
            logger.error(f"Error generating risk recommendations: {e}")
            return 0.01, None, 'MEDIUM'  # Conservative defaults
    
    def _create_default_squeeze_metrics(self, symbol: str) -> SqueezeRiskMetrics:
        """Create default squeeze metrics when analysis fails."""
        return SqueezeRiskMetrics(
            symbol=symbol,
            squeeze_risk_score=50.0,  # Medium risk default
            short_interest_ratio=0.05,
            days_to_cover=2.0,
            options_skew=None,
            price_momentum=0.0,
            volume_surge=0.0,
            technical_breakout_risk=20.0,
            risk_factors=["Analysis unavailable"],
            protective_factors=[],
            max_position_size=0.01,  # Conservative 1%
            stop_loss_level=None,
            monitoring_level='MEDIUM',
            last_updated=datetime.now()
        )

class PositionRiskCalculator:
    """Calculate position-specific risk limits and constraints."""
    
    def __init__(self):
        """Initialize position risk calculator."""
        self.base_limits = {
            'max_single_position': 0.05,  # 5% max single position
            'max_sector_exposure': 0.15,  # 15% max sector exposure
            'max_correlation_group': 0.10,  # 10% max correlated positions
            'min_liquidity_score': 0.3,   # Minimum liquidity requirement
        }
    
    async def calculate_position_limits(self, symbol: str, 
                                      current_portfolio: Dict) -> PositionRiskLimits:
        """Calculate comprehensive position limits for a symbol."""
        try:
            # Get squeeze risk assessment
            squeeze_analyzer = SqueezeRiskAnalyzer()
            squeeze_metrics = await squeeze_analyzer.assess_squeeze_risk(symbol)
            
            # Get market data for volatility analysis
            market_data = alpaca_client.get_market_data(symbol, limit=30)
            volatility = self._calculate_volatility(market_data)
            
            # Get current portfolio metrics
            portfolio_metrics = self._analyze_portfolio_risk(current_portfolio)
            
            # Calculate base position limit
            base_limit = self.base_limits['max_single_position']
            
            # Adjust for squeeze risk
            squeeze_adjustment = self._calculate_squeeze_adjustment(squeeze_metrics)
            
            # Adjust for volatility
            volatility_adjustment = self._calculate_volatility_adjustment(volatility)
            
            # Adjust for market regime
            market_adjustment = self._calculate_market_regime_adjustment()
            
            # Adjust for liquidity
            liquidity_score = await self._assess_liquidity(symbol)
            liquidity_adjustment = self._calculate_liquidity_adjustment(liquidity_score)
            
            # Calculate effective limit
            effective_limit = base_limit * squeeze_adjustment * volatility_adjustment * market_adjustment * liquidity_adjustment
            effective_limit = max(0.001, min(effective_limit, 0.05))  # Cap between 0.1% and 5%
            
            # Calculate margin requirement
            margin_requirement = self._estimate_margin_requirement(symbol, volatility)
            
            # Generate reasoning
            reasoning = self._generate_limit_reasoning(
                squeeze_adjustment, volatility_adjustment, market_adjustment, 
                liquidity_adjustment, squeeze_metrics
            )
            
            return PositionRiskLimits(
                symbol=symbol,
                max_position_size_percent=effective_limit,
                max_daily_loss_percent=effective_limit * 0.5,  # 50% of position size
                max_correlation_exposure=self.base_limits['max_correlation_group'],
                margin_requirement=margin_requirement,
                volatility_adjusted_limit=base_limit * volatility_adjustment,
                market_regime_adjustment=market_adjustment,
                liquidity_discount=liquidity_adjustment,
                effective_limit=effective_limit,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error calculating position limits for {symbol}: {e}")
            return self._create_conservative_limits(symbol)
    
    def _calculate_volatility(self, market_data: Optional[pd.DataFrame]) -> float:
        """Calculate annualized volatility."""
        if market_data is None or len(market_data) < 10:
            return 0.30  # Default 30% volatility
        
        try:
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            return min(volatility, 2.0)  # Cap at 200%
        except Exception:
            return 0.30
    
    def _analyze_portfolio_risk(self, portfolio: Dict) -> Dict[str, Any]:
        """Analyze current portfolio risk metrics."""
        try:
            positions = portfolio.get('positions', {})
            total_value = portfolio.get('equity', 100000)
            
            # Calculate current exposures
            total_short_exposure = 0.0
            position_sizes = []
            
            for symbol, position in positions.items():
                qty = float(position.get('qty', 0))
                market_value = float(position.get('market_value', 0))
                
                if qty < 0:  # Short position
                    exposure = abs(market_value) / total_value
                    total_short_exposure += exposure
                    position_sizes.append(exposure)
            
            return {
                'total_short_exposure': total_short_exposure,
                'short_position_count': len([p for p in positions.values() if float(p.get('qty', 0)) < 0]),
                'max_position_size': max(position_sizes) if position_sizes else 0.0,
                'avg_position_size': np.mean(position_sizes) if position_sizes else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio risk: {e}")
            return {'total_short_exposure': 0.0, 'short_position_count': 0}
    
    def _calculate_squeeze_adjustment(self, squeeze_metrics: SqueezeRiskMetrics) -> float:
        """Calculate position limit adjustment based on squeeze risk."""
        risk_score = squeeze_metrics.squeeze_risk_score
        
        if risk_score >= 80:
            return 0.2  # 80% reduction for high squeeze risk
        elif risk_score >= 60:
            return 0.4  # 60% reduction
        elif risk_score >= 40:
            return 0.7  # 30% reduction
        else:
            return 1.0  # No reduction for low risk
    
    def _calculate_volatility_adjustment(self, volatility: float) -> float:
        """Calculate position limit adjustment based on volatility."""
        if volatility > 0.80:  # 80%+ volatility
            return 0.3
        elif volatility > 0.50:  # 50%+ volatility
            return 0.6
        elif volatility > 0.30:  # 30%+ volatility
            return 0.8
        else:
            return 1.0
    
    def _calculate_market_regime_adjustment(self) -> float:
        """Calculate adjustment based on current market regime."""
        # In production, this would analyze market conditions
        # For now, return neutral adjustment
        return 0.9  # Slightly conservative default
    
    async def _assess_liquidity(self, symbol: str) -> float:
        """Assess liquidity score for the symbol."""
        try:
            market_data = alpaca_client.get_market_data(symbol, limit=10)
            if market_data is None or len(market_data) == 0:
                return 0.5
            
            avg_volume = market_data['volume'].mean()
            
            # Simple liquidity scoring based on volume
            if avg_volume > 5000000:
                return 0.9
            elif avg_volume > 1000000:
                return 0.7
            elif avg_volume > 100000:
                return 0.5
            else:
                return 0.3
                
        except Exception:
            return 0.5
    
    def _calculate_liquidity_adjustment(self, liquidity_score: float) -> float:
        """Calculate position limit adjustment based on liquidity."""
        if liquidity_score < 0.3:
            return 0.3  # 70% reduction for illiquid stocks
        elif liquidity_score < 0.5:
            return 0.6  # 40% reduction
        elif liquidity_score < 0.7:
            return 0.8  # 20% reduction
        else:
            return 1.0  # No reduction for liquid stocks
    
    def _estimate_margin_requirement(self, symbol: str, volatility: float) -> float:
        """Estimate margin requirement for short position."""
        # Base margin requirement is typically 150% for shorts
        base_margin = 1.50
        
        # Adjust for volatility
        volatility_adjustment = min(volatility * 0.5, 0.50)  # Max 50% additional
        
        return base_margin + volatility_adjustment
    
    def _generate_limit_reasoning(self, squeeze_adj: float, vol_adj: float,
                                market_adj: float, liquidity_adj: float,
                                squeeze_metrics: SqueezeRiskMetrics) -> str:
        """Generate reasoning for position limits."""
        reasons = []
        
        if squeeze_adj < 0.8:
            reasons.append(f"Squeeze risk adjustment ({squeeze_metrics.squeeze_risk_score:.0f}/100)")
        
        if vol_adj < 0.8:
            reasons.append("High volatility discount")
        
        if market_adj < 0.9:
            reasons.append("Market regime adjustment")
        
        if liquidity_adj < 0.8:
            reasons.append("Low liquidity discount")
        
        if not reasons:
            reasons.append("Standard risk parameters")
        
        return "; ".join(reasons)
    
    def _create_conservative_limits(self, symbol: str) -> PositionRiskLimits:
        """Create conservative limits when calculation fails."""
        return PositionRiskLimits(
            symbol=symbol,
            max_position_size_percent=0.01,  # Very conservative 1%
            max_daily_loss_percent=0.005,   # 0.5% daily loss limit
            max_correlation_exposure=0.05,
            margin_requirement=1.8,  # Conservative margin
            volatility_adjusted_limit=0.01,
            market_regime_adjustment=0.8,
            liquidity_discount=0.8,
            effective_limit=0.01,
            reasoning="Conservative limits due to analysis failure"
        )

class PortfolioRiskAnalyzer:
    """Analyze portfolio-level risk for short positions."""
    
    def __init__(self):
        """Initialize portfolio risk analyzer."""
        self.risk_limits = {
            'max_total_short_exposure': 0.30,  # 30% max short exposure
            'max_sector_concentration': 0.15,  # 15% max in any sector
            'min_position_count': 3,           # Minimum diversification
            'max_correlation_risk': 0.25       # 25% max correlated exposure
        }
    
    async def analyze_portfolio_risk(self, current_portfolio: Dict, 
                                   pending_orders: List[Dict] = None) -> PortfolioRiskMetrics:
        """Analyze comprehensive portfolio risk including pending orders."""
        try:
            positions = current_portfolio.get('positions', {})
            total_value = current_portfolio.get('equity', 100000)
            
            if pending_orders is None:
                pending_orders = []
            
            # Analyze current short positions
            short_positions = self._extract_short_positions(positions, total_value)
            
            # Project impact of pending orders
            projected_positions = self._project_pending_orders(short_positions, pending_orders, total_value)
            
            # Calculate portfolio metrics
            portfolio_metrics = await self._calculate_portfolio_metrics(projected_positions, total_value)
            
            # Check compliance with risk limits
            compliance_check = self._check_risk_compliance(portfolio_metrics)
            
            return portfolio_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio risk: {e}")
            return self._create_default_portfolio_metrics()
    
    def _extract_short_positions(self, positions: Dict, total_value: float) -> List[Dict]:
        """Extract and analyze short positions."""
        short_positions = []
        
        for symbol, position in positions.items():
            qty = float(position.get('qty', 0))
            if qty < 0:  # Short position
                market_value = abs(float(position.get('market_value', 0)))
                position_size = market_value / total_value
                
                short_positions.append({
                    'symbol': symbol,
                    'quantity': abs(qty),
                    'market_value': market_value,
                    'position_size': position_size,
                    'unrealized_pnl': float(position.get('unrealized_pl', 0)),
                    'current_price': float(position.get('current_price', 0))
                })
        
        return short_positions
    
    def _project_pending_orders(self, current_positions: List[Dict], 
                              pending_orders: List[Dict], total_value: float) -> List[Dict]:
        """Project portfolio state including pending orders."""
        projected = current_positions.copy()
        
        for order in pending_orders:
            if order.get('side') == 'sell_short':
                symbol = order.get('symbol', '')
                quantity = float(order.get('qty', 0))
                estimated_price = float(order.get('limit_price', 0)) or float(order.get('estimated_price', 100))
                
                estimated_value = quantity * estimated_price
                position_size = estimated_value / total_value
                
                # Check if position already exists
                existing_position = next((p for p in projected if p['symbol'] == symbol), None)
                
                if existing_position:
                    # Add to existing position
                    existing_position['quantity'] += quantity
                    existing_position['market_value'] += estimated_value
                    existing_position['position_size'] += position_size
                else:
                    # Create new position
                    projected.append({
                        'symbol': symbol,
                        'quantity': quantity,
                        'market_value': estimated_value,
                        'position_size': position_size,
                        'unrealized_pnl': 0.0,
                        'current_price': estimated_price,
                        'pending': True
                    })
        
        return projected
    
    async def _calculate_portfolio_metrics(self, positions: List[Dict], 
                                         total_value: float) -> PortfolioRiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        try:
            if not positions:
                return self._create_default_portfolio_metrics()
            
            # Basic metrics
            total_short_exposure = sum(p['position_size'] for p in positions)
            position_count = len(positions)
            avg_position_size = total_short_exposure / position_count if position_count > 0 else 0
            max_single_position = max((p['position_size'] for p in positions), default=0)
            
            # Sector analysis (simplified - would use real sector data in production)
            sector_exposure = await self._analyze_sector_exposure(positions)
            
            # Correlation analysis
            correlation_risk = await self._analyze_correlation_risk(positions)
            
            # Portfolio beta calculation
            portfolio_beta = await self._calculate_portfolio_beta(positions)
            
            # Margin requirements
            total_margin = sum(p['market_value'] * 1.5 for p in positions)  # 150% margin
            available_margin = total_value * 2  # Simplified available margin
            
            # Liquidity assessment
            liquidity_score = await self._assess_portfolio_liquidity(positions)
            
            # Risk compliance check
            limit_violations = []
            recommended_actions = []
            
            if total_short_exposure > self.risk_limits['max_total_short_exposure']:
                limit_violations.append(f"Total short exposure ({total_short_exposure:.1%}) exceeds limit ({self.risk_limits['max_total_short_exposure']:.1%})")
                recommended_actions.append("Reduce overall short exposure")
            
            if max_single_position > 0.05:  # 5% single position limit
                limit_violations.append(f"Single position too large ({max_single_position:.1%})")
                recommended_actions.append("Reduce largest position")
            
            within_limits = len(limit_violations) == 0
            
            return PortfolioRiskMetrics(
                total_short_exposure=total_short_exposure,
                short_position_count=position_count,
                average_short_position_size=avg_position_size,
                portfolio_beta=portfolio_beta,
                max_single_position=max_single_position,
                sector_concentration=sector_exposure,
                correlation_risk_score=correlation_risk,
                total_margin_requirement=total_margin,
                available_margin=available_margin,
                liquidity_score=liquidity_score,
                within_risk_limits=within_limits,
                limit_violations=limit_violations,
                recommended_actions=recommended_actions
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return self._create_default_portfolio_metrics()
    
    async def _analyze_sector_exposure(self, positions: List[Dict]) -> Dict[str, float]:
        """Analyze sector concentration (simplified)."""
        # In production, this would use real sector classification
        sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'AMZN': 'Consumer', 'TSLA': 'Consumer', 'META': 'Technology',
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare'
        }
        
        sector_exposure = defaultdict(float)
        
        for position in positions:
            symbol = position['symbol']
            sector = sector_map.get(symbol, 'Other')
            sector_exposure[sector] += position['position_size']
        
        return dict(sector_exposure)
    
    async def _analyze_correlation_risk(self, positions: List[Dict]) -> float:
        """Analyze correlation risk across positions."""
        # Simplified correlation analysis
        # In production, would use real correlation matrix
        
        if len(positions) < 2:
            return 0.0
        
        # Estimate correlation based on sector and size
        tech_exposure = 0.0
        finance_exposure = 0.0
        
        tech_symbols = {'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN'}
        finance_symbols = {'JPM', 'BAC', 'WFC', 'GS', 'MS'}
        
        for position in positions:
            symbol = position['symbol']
            size = position['position_size']
            
            if symbol in tech_symbols:
                tech_exposure += size
            elif symbol in finance_symbols:
                finance_exposure += size
        
        # High concentration in correlated sectors increases risk
        max_sector_exposure = max(tech_exposure, finance_exposure)
        correlation_risk = min(max_sector_exposure / 0.15, 1.0)  # Normalize to 0-1
        
        return correlation_risk
    
    async def _calculate_portfolio_beta(self, positions: List[Dict]) -> float:
        """Calculate portfolio beta for short positions."""
        # Simplified beta calculation
        # In production, would use real beta data
        
        beta_estimates = {
            'AAPL': 1.2, 'MSFT': 0.9, 'GOOGL': 1.1, 'AMZN': 1.3,
            'TSLA': 2.0, 'META': 1.4, 'NVDA': 1.8, 'JPM': 1.1,
            'BAC': 1.3, 'WFC': 1.2, 'JNJ': 0.7, 'PFE': 0.8
        }
        
        weighted_beta = 0.0
        total_weight = 0.0
        
        for position in positions:
            symbol = position['symbol']
            weight = position['position_size']
            beta = beta_estimates.get(symbol, 1.0)
            
            # For short positions, beta effect is inverted
            weighted_beta += -beta * weight  # Negative because short
            total_weight += weight
        
        if total_weight > 0:
            return weighted_beta / total_weight
        else:
            return 0.0
    
    async def _assess_portfolio_liquidity(self, positions: List[Dict]) -> float:
        """Assess overall portfolio liquidity."""
        if not positions:
            return 1.0
        
        liquidity_scores = []
        
        for position in positions:
            # Simple liquidity assessment based on position size
            # In production, would use real liquidity metrics
            position_size = position['position_size']
            
            if position_size > 0.03:  # Large position
                liquidity_scores.append(0.3)
            elif position_size > 0.02:  # Medium position
                liquidity_scores.append(0.6)
            else:  # Small position
                liquidity_scores.append(0.9)
        
        return np.mean(liquidity_scores)
    
    def _check_risk_compliance(self, metrics: PortfolioRiskMetrics) -> bool:
        """Check if portfolio complies with risk limits."""
        violations = []
        
        if metrics.total_short_exposure > self.risk_limits['max_total_short_exposure']:
            violations.append("Excessive short exposure")
        
        if metrics.max_single_position > 0.05:
            violations.append("Position too large")
        
        if any(exposure > self.risk_limits['max_sector_concentration'] 
               for exposure in metrics.sector_concentration.values()):
            violations.append("Sector concentration too high")
        
        if metrics.correlation_risk_score > self.risk_limits['max_correlation_risk']:
            violations.append("Correlation risk too high")
        
        return len(violations) == 0
    
    def _create_default_portfolio_metrics(self) -> PortfolioRiskMetrics:
        """Create default portfolio metrics."""
        return PortfolioRiskMetrics(
            total_short_exposure=0.0,
            short_position_count=0,
            average_short_position_size=0.0,
            portfolio_beta=0.0,
            max_single_position=0.0,
            sector_concentration={},
            correlation_risk_score=0.0,
            total_margin_requirement=0.0,
            available_margin=100000.0,
            liquidity_score=1.0,
            within_risk_limits=True,
            limit_violations=[],
            recommended_actions=[]
        )

class ShortRiskManager:
    """Main short selling risk management system."""
    
    def __init__(self):
        """Initialize short risk manager."""
        self.squeeze_analyzer = SqueezeRiskAnalyzer()
        self.position_calculator = PositionRiskCalculator()
        self.portfolio_analyzer = PortfolioRiskAnalyzer()
        
        # Active alerts
        self.active_alerts: Dict[str, RiskAlert] = {}
        
        logger.info("Short Risk Manager initialized")
    
    async def validate_short_trade(self, symbol: str, quantity: int, 
                                 current_portfolio: Dict) -> Tuple[bool, str, Dict[str, Any]]:
        """Comprehensive validation of a short trade."""
        try:
            logger.info(f"Validating short trade: {symbol} {quantity} shares")
            
            # Get position limits
            position_limits = await self.position_calculator.calculate_position_limits(
                symbol, current_portfolio
            )
            
            # Get squeeze risk assessment
            squeeze_metrics = await self.squeeze_analyzer.assess_squeeze_risk(symbol)
            
            # Get portfolio risk analysis
            portfolio_metrics = await self.portfolio_analyzer.analyze_portfolio_risk(
                current_portfolio, [{'symbol': symbol, 'qty': quantity, 'side': 'sell_short'}]
            )
            
            # Calculate trade size as percentage of portfolio
            portfolio_value = current_portfolio.get('equity', 100000)
            current_price = alpaca_client.get_current_price(symbol) or 100
            trade_value = quantity * current_price
            trade_size_percent = trade_value / portfolio_value
            
            # Validation checks
            validation_results = []
            
            # Position size check
            if trade_size_percent > position_limits.max_position_size_percent:
                return False, f"Position size ({trade_size_percent:.1%}) exceeds limit ({position_limits.max_position_size_percent:.1%})", {
                    'position_limits': asdict(position_limits),
                    'squeeze_metrics': asdict(squeeze_metrics),
                    'portfolio_metrics': asdict(portfolio_metrics)
                }
            
            # Squeeze risk check
            if squeeze_metrics.squeeze_risk_score > 80:
                return False, f"High squeeze risk ({squeeze_metrics.squeeze_risk_score:.0f}/100)", {
                    'position_limits': asdict(position_limits),
                    'squeeze_metrics': asdict(squeeze_metrics),
                    'portfolio_metrics': asdict(portfolio_metrics)
                }
            
            # Portfolio limits check
            if not portfolio_metrics.within_risk_limits:
                violations = "; ".join(portfolio_metrics.limit_violations)
                return False, f"Portfolio risk violations: {violations}", {
                    'position_limits': asdict(position_limits),
                    'squeeze_metrics': asdict(squeeze_metrics),
                    'portfolio_metrics': asdict(portfolio_metrics)
                }
            
            # All checks passed
            return True, f"Trade validated (squeeze risk: {squeeze_metrics.squeeze_risk_score:.0f}/100, position: {trade_size_percent:.1%})", {
                'position_limits': asdict(position_limits),
                'squeeze_metrics': asdict(squeeze_metrics),
                'portfolio_metrics': asdict(portfolio_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error validating short trade for {symbol}: {e}")
            return False, f"Validation error: {e}", {}
    
    async def monitor_short_positions(self, current_portfolio: Dict) -> List[RiskAlert]:
        """Monitor existing short positions for risk alerts."""
        try:
            alerts = []
            positions = current_portfolio.get('positions', {})
            
            # Analyze each short position
            for symbol, position in positions.items():
                qty = float(position.get('qty', 0))
                if qty < 0:  # Short position
                    # Check for squeeze risk
                    squeeze_metrics = await self.squeeze_analyzer.assess_squeeze_risk(symbol)
                    
                    if squeeze_metrics.squeeze_risk_score > 70:
                        alert = RiskAlert(
                            alert_id=f"squeeze_{symbol}_{int(datetime.now().timestamp())}",
                            symbol=symbol,
                            alert_type='SQUEEZE_RISK',
                            severity='HIGH' if squeeze_metrics.squeeze_risk_score > 80 else 'MEDIUM',
                            message=f"High squeeze risk detected ({squeeze_metrics.squeeze_risk_score:.0f}/100)",
                            recommended_action=f"Consider reducing position or setting tight stop loss at {squeeze_metrics.stop_loss_level}",
                            expires_at=datetime.now() + timedelta(hours=4),
                            created_at=datetime.now()
                        )
                        alerts.append(alert)
                        self.active_alerts[alert.alert_id] = alert
            
            # Portfolio-level checks
            portfolio_metrics = await self.portfolio_analyzer.analyze_portfolio_risk(current_portfolio)
            
            if not portfolio_metrics.within_risk_limits:
                alert = RiskAlert(
                    alert_id=f"portfolio_{int(datetime.now().timestamp())}",
                    symbol='PORTFOLIO',
                    alert_type='POSITION_LIMIT',
                    severity='HIGH',
                    message="Portfolio risk limits exceeded",
                    recommended_action="; ".join(portfolio_metrics.recommended_actions),
                    expires_at=datetime.now() + timedelta(hours=6),
                    created_at=datetime.now()
                )
                alerts.append(alert)
                self.active_alerts[alert.alert_id] = alert
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring short positions: {e}")
            return []
    
    async def get_position_recommendations(self, symbol: str, 
                                         current_portfolio: Dict) -> Dict[str, Any]:
        """Get comprehensive position recommendations for a symbol."""
        try:
            # Get all risk assessments
            position_limits = await self.position_calculator.calculate_position_limits(
                symbol, current_portfolio
            )
            squeeze_metrics = await self.squeeze_analyzer.assess_squeeze_risk(symbol)
            
            # Generate recommendations
            recommendations = {
                'symbol': symbol,
                'max_position_size': position_limits.effective_limit,
                'recommended_size': position_limits.effective_limit * 0.8,  # 80% of max
                'stop_loss_level': squeeze_metrics.stop_loss_level,
                'monitoring_level': squeeze_metrics.monitoring_level,
                'squeeze_risk_score': squeeze_metrics.squeeze_risk_score,
                'risk_factors': squeeze_metrics.risk_factors,
                'protective_factors': squeeze_metrics.protective_factors,
                'margin_requirement': position_limits.margin_requirement,
                'reasoning': position_limits.reasoning
            }
            
            # Add specific recommendations based on risk level
            if squeeze_metrics.squeeze_risk_score > 70:
                recommendations['specific_advice'] = "High squeeze risk - consider avoiding or using very small position"
            elif squeeze_metrics.squeeze_risk_score > 40:
                recommendations['specific_advice'] = "Moderate squeeze risk - use reduced position size and tight stops"
            else:
                recommendations['specific_advice'] = "Acceptable risk level for short position"
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_active_alerts(self) -> List[RiskAlert]:
        """Get currently active risk alerts."""
        now = datetime.now()
        active = []
        
        # Clean up expired alerts
        expired_ids = []
        for alert_id, alert in self.active_alerts.items():
            if alert.expires_at < now:
                expired_ids.append(alert_id)
            else:
                active.append(alert)
        
        # Remove expired alerts
        for alert_id in expired_ids:
            del self.active_alerts[alert_id]
        
        return active
    
    def get_risk_manager_status(self) -> Dict[str, Any]:
        """Get risk manager status and statistics."""
        return {
            'squeeze_cache_size': len(self.squeeze_analyzer.risk_cache),
            'active_alerts': len(self.active_alerts),
            'risk_limits': self.portfolio_analyzer.risk_limits,
            'initialized': True
        }

# Global instance
short_risk_manager = ShortRiskManager()

# Convenience functions
async def validate_short_trade(symbol: str, quantity: int, current_portfolio: Dict) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate a short trade using comprehensive risk management."""
    return await short_risk_manager.validate_short_trade(symbol, quantity, current_portfolio)

async def assess_squeeze_risk(symbol: str) -> SqueezeRiskMetrics:
    """Assess short squeeze risk for a symbol."""
    return await short_risk_manager.squeeze_analyzer.assess_squeeze_risk(symbol)

async def get_position_recommendations(symbol: str, current_portfolio: Dict) -> Dict[str, Any]:
    """Get position recommendations for a symbol."""
    return await short_risk_manager.get_position_recommendations(symbol, current_portfolio)

async def monitor_short_positions(current_portfolio: Dict) -> List[RiskAlert]:
    """Monitor short positions and generate risk alerts."""
    return await short_risk_manager.monitor_short_positions(current_portfolio)

def get_short_risk_status() -> Dict[str, Any]:
    """Get short risk manager status."""
    return short_risk_manager.get_risk_manager_status()