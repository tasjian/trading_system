"""Multi-Agent Portfolio Management System for Intelligent Portfolio Construction.

This system uses multiple specialized LLM-powered agents to collaboratively construct 
and manage a diversified portfolio. Each agent contributes insights based on:
- Sectors and asset classes
- Risk preferences and constraints
- Market conditions and regime analysis
- Quantitative metrics and narrative data
- Technical and fundamental analysis

The goal is to maximize portfolio value through intelligent diversification,
risk management, and adaptive allocation strategies.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

warnings.filterwarnings('ignore')

# Import our enhanced analysis systems
from agents.market_analysis import (
    multi_source_data, comprehensive_analyst, enhanced_market_analysis_factory,
    DataSource, AnalysisResult, SignalStrength
)
# H2O prediction agent removed - using alternative analysis
from tools.alpaca_client import alpaca_client
from tools.advanced_trading import algorithmic_engine, RiskManagementEngine
from config.settings import settings

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classification for adaptive portfolio management."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS_MODE = "crisis_mode"

class AssetClass(Enum):
    """Asset class categorization for diversification."""
    LARGE_CAP_GROWTH = "large_cap_growth"
    LARGE_CAP_VALUE = "large_cap_value"
    MID_CAP = "mid_cap"
    SMALL_CAP = "small_cap"
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    ENERGY = "energy"
    CONSUMER = "consumer"
    INDUSTRIALS = "industrials"
    MATERIALS = "materials"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    DEFENSIVE = "defensive"
    CYCLICAL = "cyclical"

class RiskProfile(Enum):
    """Risk profile for portfolio construction."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    TACTICAL = "tactical"

@dataclass
class PortfolioAllocation:
    """Individual portfolio allocation recommendation."""
    symbol: str
    asset_class: AssetClass
    target_weight: float
    current_weight: float
    recommended_action: str  # "buy", "sell", "hold", "rebalance"
    confidence: float
    reasoning: str
    risk_score: float
    expected_return: float
    correlation_score: float
    agent_source: str
    priority: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PortfolioRecommendation:
    """Complete portfolio recommendation from the multi-agent system."""
    allocations: List[PortfolioAllocation]
    target_risk_level: float
    expected_return: float
    expected_volatility: float
    diversification_score: float
    market_regime: MarketRegime
    confidence: float
    cash_allocation: float
    rebalance_urgency: float
    reasoning: str
    agents_consensus: Dict[str, float]
    timestamp: datetime

class MarketRegimeAnalyst:
    """Agent specialized in identifying current market regime."""
    
    def __init__(self):
        self.name = "Market Regime Analyst"
        self.lookback_period = 63  # 3 months
        
    async def analyze_market_regime(self, benchmark_symbols: List[str] = None) -> MarketRegime:
        """Analyze current market regime using multiple indicators."""
        try:
            if not benchmark_symbols:
                benchmark_symbols = ["SPY", "QQQ", "IWM"]  # Large, tech, small cap
            
            logger.info("Analyzing market regime...")
            
            # Get market data for analysis
            market_data = {}
            for symbol in benchmark_symbols:
                try:
                    data_result = await multi_source_data.get_stock_data_multi_source(
                        symbol, f"{self.lookback_period}d"
                    )
                    if data_result.confidence > 0.5 and not data_result.value.empty:
                        market_data[symbol] = data_result.value
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
            
            if not market_data:
                logger.warning("No market data available for regime analysis")
                return MarketRegime.SIDEWAYS_MARKET
            
            # Analyze regime indicators
            regime_scores = {
                MarketRegime.BULL_MARKET: 0,
                MarketRegime.BEAR_MARKET: 0,
                MarketRegime.SIDEWAYS_MARKET: 0,
                MarketRegime.HIGH_VOLATILITY: 0,
                MarketRegime.LOW_VOLATILITY: 0,
                MarketRegime.CRISIS_MODE: 0
            }
            
            for symbol, data in market_data.items():
                if len(data) < 20:
                    continue
                    
                # Calculate metrics
                returns = data['Close'].pct_change().dropna()
                
                # Trend analysis
                sma_20 = data['Close'].rolling(20).mean()
                sma_50 = data['Close'].rolling(50).mean() if len(data) >= 50 else sma_20
                current_price = data['Close'].iloc[-1]
                
                # Volatility analysis
                volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
                
                # Trend strength
                if current_price > sma_20.iloc[-1] * 1.02:  # 2% above SMA
                    if sma_20.iloc[-1] > sma_50.iloc[-1]:
                        regime_scores[MarketRegime.BULL_MARKET] += 1
                elif current_price < sma_20.iloc[-1] * 0.98:  # 2% below SMA
                    if sma_20.iloc[-1] < sma_50.iloc[-1]:
                        regime_scores[MarketRegime.BEAR_MARKET] += 1
                else:
                    regime_scores[MarketRegime.SIDEWAYS_MARKET] += 1
                
                # Volatility regime
                if volatility > 0.25:  # High volatility (>25% annualized)
                    regime_scores[MarketRegime.HIGH_VOLATILITY] += 1
                    if volatility > 0.40:  # Crisis level volatility
                        regime_scores[MarketRegime.CRISIS_MODE] += 1
                elif volatility < 0.15:  # Low volatility
                    regime_scores[MarketRegime.LOW_VOLATILITY] += 1
            
            # Determine dominant regime
            dominant_regime = max(regime_scores, key=regime_scores.get)
            
            logger.info(f"Market regime identified: {dominant_regime.value}")
            logger.info(f"Regime scores: {[(r.value, s) for r, s in regime_scores.items() if s > 0]}")
            
            return dominant_regime
            
        except Exception as e:
            logger.error(f"Market regime analysis error: {e}")
            return MarketRegime.SIDEWAYS_MARKET

class SectorRotationAgent:
    """Agent specialized in sector rotation strategies."""
    
    def __init__(self):
        self.name = "Sector Rotation Agent"
        self.sector_etfs = {
            AssetClass.TECHNOLOGY: "XLK",
            AssetClass.HEALTHCARE: "XLV", 
            AssetClass.FINANCIALS: "XLF",
            AssetClass.ENERGY: "XLE",
            AssetClass.CONSUMER: "XLY",
            AssetClass.INDUSTRIALS: "XLI",
            AssetClass.MATERIALS: "XLB",
            AssetClass.UTILITIES: "XLU",
            AssetClass.REAL_ESTATE: "XLRE"
        }
        
    async def analyze_sector_rotation(self, market_regime: MarketRegime) -> Dict[AssetClass, float]:
        """Analyze sector rotation opportunities based on market regime."""
        try:
            logger.info(f"Analyzing sector rotation for {market_regime.value} regime")
            
            sector_scores = {}
            
            # Get sector performance data
            for asset_class, etf_symbol in self.sector_etfs.items():
                try:
                    # Get comprehensive analysis
                    analysis = await comprehensive_analyst.comprehensive_analysis(etf_symbol)
                    
                    # Calculate sector score based on multiple factors
                    score = 0.0
                    
                    # Technical score
                    if analysis.action == "buy":
                        score += analysis.confidence * 0.4
                    elif analysis.action == "sell":
                        score -= analysis.confidence * 0.4
                    
                    # Momentum and relative strength
                    if analysis.indicators:
                        rsi = analysis.indicators.get('rsi', 50)
                        current_price = analysis.indicators.get('current_price', 0)
                        
                        # RSI momentum (prefer 40-70 range for entries)
                        if 40 <= rsi <= 70:
                            score += 0.2
                        elif rsi < 30:  # Oversold - potential opportunity
                            score += 0.15
                        elif rsi > 80:  # Overbought - caution
                            score -= 0.15
                    
                    # Regime-specific adjustments
                    score = self._adjust_for_regime(asset_class, score, market_regime)
                    
                    sector_scores[asset_class] = max(0.0, min(1.0, score))
                    
                except Exception as e:
                    logger.warning(f"Sector analysis failed for {asset_class.value}: {e}")
                    sector_scores[asset_class] = 0.5  # Neutral score
            
            logger.info(f"Sector scores: {[(s.value, f'{score:.3f}') for s, score in sector_scores.items()]}")
            return sector_scores
            
        except Exception as e:
            logger.error(f"Sector rotation analysis error: {e}")
            return {asset_class: 0.5 for asset_class in self.sector_etfs.keys()}
    
    def _adjust_for_regime(self, asset_class: AssetClass, base_score: float, 
                          regime: MarketRegime) -> float:
        """Adjust sector scores based on market regime preferences."""
        regime_preferences = {
            MarketRegime.BULL_MARKET: {
                AssetClass.TECHNOLOGY: 0.2,
                AssetClass.FINANCIALS: 0.15,
                AssetClass.CONSUMER: 0.1,
                AssetClass.INDUSTRIALS: 0.1,
                AssetClass.UTILITIES: -0.1,
                AssetClass.DEFENSIVE: -0.05
            },
            MarketRegime.BEAR_MARKET: {
                AssetClass.UTILITIES: 0.2,
                AssetClass.HEALTHCARE: 0.15,
                AssetClass.CONSUMER: 0.1,  # Defensive consumer
                AssetClass.TECHNOLOGY: -0.15,
                AssetClass.FINANCIALS: -0.1,
                AssetClass.ENERGY: -0.1
            },
            MarketRegime.HIGH_VOLATILITY: {
                AssetClass.UTILITIES: 0.15,
                AssetClass.HEALTHCARE: 0.1,
                AssetClass.TECHNOLOGY: -0.1,
                AssetClass.FINANCIALS: -0.1,
                AssetClass.ENERGY: -0.15
            },
            MarketRegime.LOW_VOLATILITY: {
                AssetClass.TECHNOLOGY: 0.1,
                AssetClass.FINANCIALS: 0.1,
                AssetClass.REAL_ESTATE: 0.05,
                AssetClass.UTILITIES: -0.05
            },
            MarketRegime.CRISIS_MODE: {
                AssetClass.UTILITIES: 0.3,
                AssetClass.HEALTHCARE: 0.2,
                AssetClass.CONSUMER: 0.1,
                AssetClass.TECHNOLOGY: -0.2,
                AssetClass.FINANCIALS: -0.2,
                AssetClass.ENERGY: -0.2
            }
        }
        
        adjustment = regime_preferences.get(regime, {}).get(asset_class, 0.0)
        return base_score + adjustment

class RiskParityAgent:
    """Agent specialized in risk parity and diversification strategies."""
    
    def __init__(self):
        self.name = "Risk Parity Agent"
        self.lookback_days = 252  # 1 year for risk calculations
        
    async def calculate_risk_parity_weights(self, symbols: List[str]) -> Dict[str, float]:
        """Calculate risk parity weights for given symbols."""
        try:
            logger.info("Calculating risk parity weights...")
            
            # Get historical returns data
            returns_data = {}
            for symbol in symbols:
                try:
                    data_result = await multi_source_data.get_stock_data_multi_source(
                        symbol, f"{self.lookback_days}d"
                    )
                    if data_result.confidence > 0.5 and len(data_result.value) > 50:
                        returns = data_result.value['Close'].pct_change().dropna()
                        if len(returns) > 20:
                            returns_data[symbol] = returns
                except Exception as e:
                    logger.warning(f"Failed to get returns for {symbol}: {e}")
            
            if len(returns_data) < 2:
                logger.warning("Insufficient data for risk parity calculation")
                equal_weight = 1.0 / len(symbols)
                return {symbol: equal_weight for symbol in symbols}
            
            # Calculate volatilities
            volatilities = {}
            for symbol, returns in returns_data.items():
                vol = returns.std() * np.sqrt(252)  # Annualized volatility
                volatilities[symbol] = vol
            
            # Calculate inverse volatility weights
            inv_vol_sum = sum(1/vol for vol in volatilities.values())
            weights = {symbol: (1/vol) / inv_vol_sum for symbol, vol in volatilities.items()}
            
            # Add remaining symbols with equal weight if missing data
            missing_symbols = set(symbols) - set(weights.keys())
            if missing_symbols:
                remaining_weight = 0.2  # Reserve 20% for missing symbols
                current_total = sum(weights.values())
                # Scale down existing weights
                for symbol in weights:
                    weights[symbol] *= (1 - remaining_weight)
                
                # Assign equal weight to missing symbols
                missing_weight = remaining_weight / len(missing_symbols)
                for symbol in missing_symbols:
                    weights[symbol] = missing_weight
            
            logger.info(f"Risk parity weights: {[(s, f'{w:.3f}') for s, w in weights.items()]}")
            return weights
            
        except Exception as e:
            logger.error(f"Risk parity calculation error: {e}")
            equal_weight = 1.0 / len(symbols)
            return {symbol: equal_weight for symbol in symbols}
    
    async def calculate_diversification_score(self, symbols: List[str], 
                                            weights: Dict[str, float]) -> float:
        """Calculate portfolio diversification score."""
        try:
            if len(symbols) < 2:
                return 0.0
            
            # Get correlation matrix
            returns_data = {}
            for symbol in symbols:
                try:
                    data_result = await multi_source_data.get_stock_data_multi_source(
                        symbol, "180d"
                    )
                    if data_result.confidence > 0.5 and len(data_result.value) > 50:
                        returns = data_result.value['Close'].pct_change().dropna()
                        if len(returns) > 20:
                            returns_data[symbol] = returns
                except Exception as e:
                    continue
            
            if len(returns_data) < 2:
                return 0.5  # Medium diversification score
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data).fillna(0)
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
            
            # Calculate weighted average correlation
            total_weight_product = 0
            weighted_corr_sum = 0
            
            for i, symbol1 in enumerate(corr_matrix.columns):
                for j, symbol2 in enumerate(corr_matrix.columns):
                    if i != j and symbol1 in weights and symbol2 in weights:
                        weight_product = weights[symbol1] * weights[symbol2]
                        correlation = abs(corr_matrix.loc[symbol1, symbol2])
                        weighted_corr_sum += weight_product * correlation
                        total_weight_product += weight_product
            
            if total_weight_product == 0:
                return 0.5
            
            avg_correlation = weighted_corr_sum / total_weight_product
            
            # Diversification score = 1 - average correlation
            diversification_score = max(0.0, min(1.0, 1.0 - avg_correlation))
            
            logger.info(f"Diversification score: {diversification_score:.3f} (avg corr: {avg_correlation:.3f})")
            return diversification_score
            
        except Exception as e:
            logger.error(f"Diversification calculation error: {e}")
            return 0.5

class MomentumFactorAgent:
    """Agent specialized in momentum factor investing."""
    
    def __init__(self):
        self.name = "Momentum Factor Agent"
        self.lookback_periods = [21, 63, 126, 252]  # 1M, 3M, 6M, 1Y
        
    async def analyze_momentum_signals(self, symbols: List[str]) -> Dict[str, float]:
        """Analyze momentum signals across multiple timeframes."""
        try:
            logger.info("Analyzing momentum signals...")
            
            momentum_scores = {}
            
            for symbol in symbols:
                try:
                    # Get historical data
                    data_result = await multi_source_data.get_stock_data_multi_source(
                        symbol, "400d"  # Get extra data for longer lookbacks
                    )
                    
                    if data_result.confidence < 0.5 or data_result.value.empty:
                        momentum_scores[symbol] = 0.5  # Neutral
                        continue
                    
                    data = data_result.value
                    current_price = data['Close'].iloc[-1]
                    
                    # Calculate momentum across multiple timeframes
                    momentum_values = []
                    weights = [0.4, 0.3, 0.2, 0.1]  # Weight recent periods more
                    
                    for i, period in enumerate(self.lookback_periods):
                        if len(data) > period:
                            past_price = data['Close'].iloc[-period]
                            momentum = (current_price - past_price) / past_price
                            momentum_values.append(momentum * weights[i])
                    
                    if not momentum_values:
                        momentum_scores[symbol] = 0.5
                        continue
                    
                    # Weighted momentum score
                    weighted_momentum = sum(momentum_values) / sum(weights[:len(momentum_values)])
                    
                    # Convert to 0-1 score (sigmoid function)
                    momentum_score = 1 / (1 + np.exp(-weighted_momentum * 10))
                    
                    # Add volatility adjustment (prefer lower volatility for similar momentum)
                    returns = data['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252) if len(returns) > 20 else 0.3
                    vol_adjustment = max(0.8, min(1.2, 1.0 - (volatility - 0.2) * 0.5))
                    
                    final_score = momentum_score * vol_adjustment
                    momentum_scores[symbol] = max(0.0, min(1.0, final_score))
                    
                except Exception as e:
                    logger.warning(f"Momentum analysis failed for {symbol}: {e}")
                    momentum_scores[symbol] = 0.5
            
            logger.info(f"Momentum scores: {[(s, f'{score:.3f}') for s, score in momentum_scores.items()]}")
            return momentum_scores
            
        except Exception as e:
            logger.error(f"Momentum analysis error: {e}")
            return {symbol: 0.5 for symbol in symbols}

class ValueFactorAgent:
    """Agent specialized in value factor investing."""
    
    def __init__(self):
        self.name = "Value Factor Agent"
        
    async def analyze_value_signals(self, symbols: List[str]) -> Dict[str, float]:
        """Analyze value signals using fundamental metrics."""
        try:
            logger.info("Analyzing value signals...")
            
            value_scores = {}
            
            for symbol in symbols:
                try:
                    # Get fundamental analysis
                    fundamental_result = await enhanced_market_analysis_factory['fundamental'].analyze_symbol(symbol)
                    
                    if not fundamental_result or fundamental_result.confidence < 0.3:
                        value_scores[symbol] = 0.5  # Neutral
                        continue
                    
                    score = 0.5  # Base neutral score
                    
                    # Use fundamental analysis confidence as base
                    if fundamental_result.action == "buy":
                        score += fundamental_result.confidence * 0.3
                    elif fundamental_result.action == "sell":
                        score -= fundamental_result.confidence * 0.3
                    
                    # Adjust based on valuation metrics
                    if fundamental_result.indicators:
                        pe_ratio = fundamental_result.indicators.get('pe_ratio', 20)
                        market_cap = fundamental_result.indicators.get('market_cap', 0)
                        
                        # PE ratio scoring (lower is better for value)
                        if pe_ratio > 0:
                            if pe_ratio < 15:  # Low PE - good value
                                score += 0.2
                            elif pe_ratio > 30:  # High PE - expensive
                                score -= 0.1
                        
                        # Size factor (slight preference for larger caps in value)
                        if market_cap > 50e9:  # $50B+ market cap
                            score += 0.05
                    
                    # Use data confidence as quality indicator
                    score *= fundamental_result.data_confidence
                    
                    value_scores[symbol] = max(0.0, min(1.0, score))
                    
                except Exception as e:
                    logger.warning(f"Value analysis failed for {symbol}: {e}")
                    value_scores[symbol] = 0.5
            
            logger.info(f"Value scores: {[(s, f'{score:.3f}') for s, score in value_scores.items()]}")
            return value_scores
            
        except Exception as e:
            logger.error(f"Value analysis error: {e}")
            return {symbol: 0.5 for symbol in symbols}

class MLPredictiveAgent:
    """Agent that uses H2O.ai machine learning for predictive insights."""
    
    def __init__(self):
        self.name = "ML Predictive Agent"
        
    async def analyze_ml_predictions(self, symbols: List[str]) -> Dict[str, float]:
        """Get ML-based predictions and convert to portfolio scores."""
        try:
            logger.info("Analyzing ML predictions...")
            
            ml_scores = {}
            
            # H2O.ai removed - using comprehensive analysis fallback
            logger.info("Using comprehensive analysis for ML predictions")
            
            for symbol in symbols:
                try:
                    # Get comprehensive analysis as ML substitute
                    ml_analysis = await comprehensive_analyst.comprehensive_analysis(symbol)
                    
                    if not ml_analysis or ml_analysis.data_confidence < 0.3:
                        ml_scores[symbol] = 0.5  # Neutral
                        continue
                    
                    # Convert ML prediction to portfolio score
                    score = 0.5  # Base neutral
                    
                    if ml_analysis.action == "buy":
                        score += ml_analysis.confidence * 0.4
                    elif ml_analysis.action == "sell":
                        score -= ml_analysis.confidence * 0.4
                    
                    # Factor in prediction accuracy
                    score *= ml_analysis.data_confidence
                    
                    # Use expected return from indicators if available
                    if ml_analysis.indicators:
                        expected_return = ml_analysis.indicators.get('expected_return_7d', 0)
                        if abs(expected_return) > 0.02:  # Significant prediction
                            score += expected_return * 2  # Scale return to score
                    
                    ml_scores[symbol] = max(0.0, min(1.0, score))
                    
                except Exception as e:
                    logger.warning(f"ML analysis failed for {symbol}: {e}")
                    ml_scores[symbol] = 0.5
            
            logger.info(f"ML scores: {[(s, f'{score:.3f}') for s, score in ml_scores.items()]}")
            return ml_scores
            
        except Exception as e:
            logger.error(f"ML analysis error: {e}")
            return {symbol: 0.5 for symbol in symbols}

class PortfolioConstructionEngine:
    """Main engine that coordinates all agents to construct optimal portfolios."""
    
    def __init__(self):
        self.name = "Portfolio Construction Engine"
        
        # Initialize specialized agents
        self.regime_analyst = MarketRegimeAnalyst()
        self.sector_agent = SectorRotationAgent()
        self.risk_parity_agent = RiskParityAgent()
        self.momentum_agent = MomentumFactorAgent()
        self.value_agent = ValueFactorAgent()
        self.ml_agent = MLPredictiveAgent()
        self.risk_manager = RiskManagementEngine()
        
        # Agent weights for different risk profiles
        self.agent_weights = {
            RiskProfile.CONSERVATIVE: {
                'risk_parity': 0.3,
                'value': 0.25,
                'sector_rotation': 0.2,
                'momentum': 0.15,
                'ml_predictive': 0.1
            },
            RiskProfile.MODERATE: {
                'risk_parity': 0.25,
                'sector_rotation': 0.25,
                'momentum': 0.2,
                'value': 0.15,
                'ml_predictive': 0.15
            },
            RiskProfile.AGGRESSIVE: {
                'momentum': 0.3,
                'ml_predictive': 0.25,
                'sector_rotation': 0.2,
                'risk_parity': 0.15,
                'value': 0.1
            },
            RiskProfile.TACTICAL: {
                'sector_rotation': 0.3,
                'momentum': 0.25,
                'ml_predictive': 0.2,
                'risk_parity': 0.15,
                'value': 0.1
            }
        }
    
    async def construct_portfolio(self, candidate_symbols: List[str], 
                                portfolio_value: float,
                                risk_profile: RiskProfile = RiskProfile.MODERATE,
                                max_positions: int = 10) -> PortfolioRecommendation:
        """Main method to construct optimal portfolio using multi-agent approach."""
        try:
            logger.info(f"Constructing portfolio with {len(candidate_symbols)} candidates")
            logger.info(f"Portfolio value: ${portfolio_value:,.2f}, Risk profile: {risk_profile.value}")
            
            # Step 1: Analyze market regime
            market_regime = await self.regime_analyst.analyze_market_regime()
            
            # Step 2: Run all agents in parallel
            agent_tasks = [
                self.sector_agent.analyze_sector_rotation(market_regime),
                self.risk_parity_agent.calculate_risk_parity_weights(candidate_symbols),
                self.momentum_agent.analyze_momentum_signals(candidate_symbols),
                self.value_agent.analyze_value_signals(candidate_symbols),
                self.ml_agent.analyze_ml_predictions(candidate_symbols)
            ]
            
            logger.info("Running multi-agent analysis...")
            agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            # Unpack results
            sector_scores = agent_results[0] if not isinstance(agent_results[0], Exception) else {}
            risk_parity_weights = agent_results[1] if not isinstance(agent_results[1], Exception) else {}
            momentum_scores = agent_results[2] if not isinstance(agent_results[2], Exception) else {}
            value_scores = agent_results[3] if not isinstance(agent_results[3], Exception) else {}
            ml_scores = agent_results[4] if not isinstance(agent_results[4], Exception) else {}
            
            # Step 3: Combine agent recommendations
            combined_scores = await self._combine_agent_scores(
                candidate_symbols, risk_profile,
                sector_scores, risk_parity_weights, momentum_scores, 
                value_scores, ml_scores, market_regime
            )
            
            # Step 4: Select top positions
            selected_symbols = self._select_top_positions(combined_scores, max_positions)
            
            # Step 5: Calculate final weights and allocations
            portfolio_allocations = await self._calculate_portfolio_allocations(
                selected_symbols, combined_scores, portfolio_value, market_regime
            )
            
            # Step 6: Calculate portfolio metrics
            portfolio_metrics = await self._calculate_portfolio_metrics(
                portfolio_allocations, market_regime
            )
            
            # Step 7: Determine rebalancing urgency
            rebalance_urgency = self._calculate_rebalance_urgency(
                portfolio_allocations, market_regime
            )
            
            # Step 8: Generate reasoning and consensus
            agents_consensus = self._calculate_agent_consensus(
                risk_profile, sector_scores, momentum_scores, value_scores, ml_scores
            )
            
            reasoning = self._generate_portfolio_reasoning(
                market_regime, risk_profile, selected_symbols, 
                portfolio_metrics, agents_consensus
            )
            
            recommendation = PortfolioRecommendation(
                allocations=portfolio_allocations,
                target_risk_level=portfolio_metrics['risk_level'],
                expected_return=portfolio_metrics['expected_return'],
                expected_volatility=portfolio_metrics['expected_volatility'],
                diversification_score=portfolio_metrics['diversification_score'],
                market_regime=market_regime,
                confidence=portfolio_metrics['confidence'],
                cash_allocation=portfolio_metrics['cash_allocation'],
                rebalance_urgency=rebalance_urgency,
                reasoning=reasoning,
                agents_consensus=agents_consensus,
                timestamp=datetime.now()
            )
            
            logger.info(f"Portfolio construction complete: {len(portfolio_allocations)} positions")
            logger.info(f"Expected return: {portfolio_metrics['expected_return']:.1%}")
            logger.info(f"Diversification score: {portfolio_metrics['diversification_score']:.3f}")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Portfolio construction error: {e}")
            # Return minimal fallback recommendation
            return PortfolioRecommendation(
                allocations=[],
                target_risk_level=0.5,
                expected_return=0.08,
                expected_volatility=0.15,
                diversification_score=0.5,
                market_regime=MarketRegime.SIDEWAYS_MARKET,
                confidence=0.0,
                cash_allocation=1.0,
                rebalance_urgency=0.0,
                reasoning="Portfolio construction failed - maintaining cash position",
                agents_consensus={},
                timestamp=datetime.now()
            )
    
    async def _combine_agent_scores(self, symbols: List[str], risk_profile: RiskProfile,
                                  sector_scores: Dict[AssetClass, float],
                                  risk_parity_weights: Dict[str, float],
                                  momentum_scores: Dict[str, float],
                                  value_scores: Dict[str, float],
                                  ml_scores: Dict[str, float],
                                  market_regime: MarketRegime) -> Dict[str, float]:
        """Combine scores from all agents using risk profile weights."""
        
        agent_weights = self.agent_weights[risk_profile]
        combined_scores = {}
        
        for symbol in symbols:
            score = 0.0
            
            # Risk parity component
            rp_weight = risk_parity_weights.get(symbol, 1.0 / len(symbols))
            rp_score = min(1.0, rp_weight * len(symbols))  # Normalize to 0-1
            score += rp_score * agent_weights['risk_parity']
            
            # Momentum component
            momentum_score = momentum_scores.get(symbol, 0.5)
            score += momentum_score * agent_weights['momentum']
            
            # Value component  
            value_score = value_scores.get(symbol, 0.5)
            score += value_score * agent_weights['value']
            
            # ML predictive component
            ml_score = ml_scores.get(symbol, 0.5)
            score += ml_score * agent_weights['ml_predictive']
            
            # Sector rotation component (need to map symbol to sector)
            sector_score = await self._get_sector_score_for_symbol(symbol, sector_scores)
            score += sector_score * agent_weights['sector_rotation']
            
            # Market regime adjustments
            score = self._adjust_for_market_regime(score, market_regime)
            
            combined_scores[symbol] = max(0.0, min(1.0, score))
        
        return combined_scores
    
    async def _get_sector_score_for_symbol(self, symbol: str, 
                                         sector_scores: Dict[AssetClass, float]) -> float:
        """Map symbol to its sector and get the sector score."""
        try:
            # Get fundamental analysis to determine sector
            analysis = await enhanced_market_analysis_factory['fundamental'].analyze_symbol(symbol)
            
            # Simple sector mapping based on symbol patterns and analysis
            if 'tech' in symbol.lower() or symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:
                return sector_scores.get(AssetClass.TECHNOLOGY, 0.5)
            elif symbol in ['JPM', 'BAC', 'WFC', 'GS']:
                return sector_scores.get(AssetClass.FINANCIALS, 0.5)
            elif symbol in ['JNJ', 'PFE', 'UNH', 'ABBV']:
                return sector_scores.get(AssetClass.HEALTHCARE, 0.5)
            elif symbol in ['XOM', 'CVX', 'SLB']:
                return sector_scores.get(AssetClass.ENERGY, 0.5)
            else:
                # Default to average of all sector scores
                if sector_scores:
                    return sum(sector_scores.values()) / len(sector_scores)
                return 0.5
                
        except Exception as e:
            logger.warning(f"Sector mapping error for {symbol}: {e}")
            return 0.5
    
    def _adjust_for_market_regime(self, score: float, regime: MarketRegime) -> float:
        """Apply market regime adjustments to combined scores."""
        adjustments = {
            MarketRegime.BULL_MARKET: 0.05,      # Slightly more aggressive
            MarketRegime.BEAR_MARKET: -0.1,     # More defensive
            MarketRegime.HIGH_VOLATILITY: -0.05, # Reduce risk
            MarketRegime.LOW_VOLATILITY: 0.03,   # Slightly more risk
            MarketRegime.CRISIS_MODE: -0.15,     # Very defensive
            MarketRegime.SIDEWAYS_MARKET: 0.0    # No adjustment
        }
        
        adjustment = adjustments.get(regime, 0.0)
        return max(0.0, min(1.0, score + adjustment))
    
    def _select_top_positions(self, combined_scores: Dict[str, float], 
                            max_positions: int) -> List[str]:
        """Select top positions based on combined scores."""
        # Sort symbols by score (descending)
        sorted_symbols = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top positions, but ensure minimum score threshold
        min_score_threshold = 0.4
        selected = []
        
        for symbol, score in sorted_symbols:
            if len(selected) >= max_positions:
                break
            if score >= min_score_threshold:
                selected.append(symbol)
        
        # If we don't have enough positions, lower the threshold slightly
        if len(selected) < max_positions // 2:
            min_score_threshold = 0.3
            for symbol, score in sorted_symbols:
                if len(selected) >= max_positions:
                    break
                if symbol not in selected and score >= min_score_threshold:
                    selected.append(symbol)
        
        logger.info(f"Selected {len(selected)} positions from {len(combined_scores)} candidates")
        return selected
    
    async def _calculate_portfolio_allocations(self, selected_symbols: List[str],
                                             combined_scores: Dict[str, float],
                                             portfolio_value: float,
                                             market_regime: MarketRegime) -> List[PortfolioAllocation]:
        """Calculate individual position allocations."""
        allocations = []
        
        if not selected_symbols:
            return allocations
        
        # Calculate weights based on scores
        total_score = sum(combined_scores[symbol] for symbol in selected_symbols)
        base_weights = {symbol: combined_scores[symbol] / total_score 
                       for symbol in selected_symbols}
        
        # Apply diversification constraints
        max_single_position = 0.15  # Maximum 15% in any single position
        min_single_position = 0.02  # Minimum 2% to be meaningful
        
        # Normalize weights within constraints
        final_weights = {}
        remaining_weight = 1.0
        
        # First pass: assign minimum weights
        for symbol in selected_symbols:
            final_weights[symbol] = max(min_single_position, 
                                      min(max_single_position, base_weights[symbol]))
            remaining_weight -= final_weights[symbol]
        
        # Second pass: distribute remaining weight proportionally
        if remaining_weight > 0:
            scalable_symbols = [s for s in selected_symbols 
                              if final_weights[s] < max_single_position]
            if scalable_symbols:
                scale_factor = remaining_weight / len(scalable_symbols)
                for symbol in scalable_symbols:
                    addition = min(scale_factor, max_single_position - final_weights[symbol])
                    final_weights[symbol] += addition
        
        # Create allocations
        for symbol in selected_symbols:
            try:
                weight = final_weights[symbol]
                
                # Get current market data
                analysis = await enhanced_market_analysis_factory['technical'].analyze_symbol(symbol)
                current_price = analysis.indicators.get('current_price', 0) if analysis.indicators else 0
                
                # Determine asset class (simplified)
                asset_class = self._classify_asset(symbol)
                
                # Calculate risk score based on analysis
                risk_score = self._calculate_risk_score(analysis) if analysis else 0.5
                
                # Estimate expected return
                expected_return = self._estimate_expected_return(analysis) if analysis else 0.08
                
                allocation = PortfolioAllocation(
                    symbol=symbol,
                    asset_class=asset_class,
                    target_weight=weight,
                    current_weight=0.0,  # Assume starting fresh
                    recommended_action="buy" if weight > 0.01 else "hold",
                    confidence=combined_scores[symbol],
                    reasoning=analysis.reasoning if analysis and analysis.reasoning else f"Multi-agent score: {combined_scores[symbol]:.3f}",
                    risk_score=risk_score,
                    expected_return=expected_return,
                    correlation_score=0.5,  # Will calculate later if needed
                    agent_source="multi_agent_ensemble",
                    priority=len(allocations) + 1,
                    metadata={
                        'current_price': current_price,
                        'market_regime': market_regime.value,
                        'position_value': portfolio_value * weight
                    }
                )
                
                allocations.append(allocation)
                
            except Exception as e:
                logger.warning(f"Failed to create allocation for {symbol}: {e}")
        
        return allocations
    
    def _classify_asset(self, symbol: str) -> AssetClass:
        """Simple asset classification based on symbol."""
        # Simplified classification - in production, would use more sophisticated mapping
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        healthcare_symbols = ['JNJ', 'PFE', 'UNH', 'ABBV', 'BMY', 'MRK']
        financial_symbols = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C']
        energy_symbols = ['XOM', 'CVX', 'SLB', 'COP']
        
        if symbol in tech_symbols:
            return AssetClass.TECHNOLOGY
        elif symbol in healthcare_symbols:
            return AssetClass.HEALTHCARE
        elif symbol in financial_symbols:
            return AssetClass.FINANCIALS
        elif symbol in energy_symbols:
            return AssetClass.ENERGY
        else:
            return AssetClass.LARGE_CAP_GROWTH  # Default
    
    def _calculate_risk_score(self, analysis: AnalysisResult) -> float:
        """Calculate risk score from analysis."""
        if not analysis or not analysis.indicators:
            return 0.5
        
        # Use volatility and other risk indicators
        risk_score = 0.5
        
        # RSI as momentum risk indicator
        rsi = analysis.indicators.get('rsi', 50)
        if rsi > 70:  # Overbought
            risk_score += 0.2
        elif rsi < 30:  # Oversold
            risk_score += 0.1
        
        # Use confidence inversely for risk
        risk_score = max(0.1, min(0.9, risk_score + (1 - analysis.confidence) * 0.3))
        
        return risk_score
    
    def _estimate_expected_return(self, analysis: AnalysisResult) -> float:
        """Estimate expected return from analysis."""
        if not analysis:
            return 0.08  # Market average
        
        base_return = 0.08
        
        # Adjust based on signal
        if analysis.action == "buy":
            base_return += analysis.confidence * 0.05
        elif analysis.action == "sell":
            base_return -= analysis.confidence * 0.05
        
        # Add strength adjustment
        strength_multiplier = {
            SignalStrength.VERY_STRONG: 1.3,
            SignalStrength.STRONG: 1.15,
            SignalStrength.MODERATE: 1.0,
            SignalStrength.WEAK: 0.9,
            SignalStrength.VERY_WEAK: 0.8
        }
        
        multiplier = strength_multiplier.get(analysis.strength, 1.0)
        return max(0.02, min(0.25, base_return * multiplier))
    
    async def _calculate_portfolio_metrics(self, allocations: List[PortfolioAllocation],
                                         market_regime: MarketRegime) -> Dict[str, float]:
        """Calculate portfolio-level metrics."""
        if not allocations:
            return {
                'expected_return': 0.08,
                'expected_volatility': 0.15,
                'risk_level': 0.5,
                'diversification_score': 0.0,
                'confidence': 0.0,
                'cash_allocation': 1.0
            }
        
        # Calculate weighted metrics
        total_weight = sum(alloc.target_weight for alloc in allocations)
        cash_allocation = max(0.0, 1.0 - total_weight)
        
        weighted_return = sum(alloc.expected_return * alloc.target_weight 
                            for alloc in allocations)
        weighted_risk = sum(alloc.risk_score * alloc.target_weight 
                          for alloc in allocations)
        weighted_confidence = sum(alloc.confidence * alloc.target_weight 
                                for alloc in allocations)
        
        # Estimate portfolio volatility (simplified)
        expected_volatility = 0.15  # Market average
        if market_regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS_MODE]:
            expected_volatility *= 1.3
        elif market_regime == MarketRegime.LOW_VOLATILITY:
            expected_volatility *= 0.8
        
        # Calculate diversification score
        symbols = [alloc.symbol for alloc in allocations]
        weights = {alloc.symbol: alloc.target_weight for alloc in allocations}
        diversification_score = await self.risk_parity_agent.calculate_diversification_score(symbols, weights)
        
        return {
            'expected_return': weighted_return,
            'expected_volatility': expected_volatility * (1 - cash_allocation * 0.5),
            'risk_level': weighted_risk,
            'diversification_score': diversification_score,
            'confidence': weighted_confidence,
            'cash_allocation': cash_allocation
        }
    
    def _calculate_rebalance_urgency(self, allocations: List[PortfolioAllocation],
                                   market_regime: MarketRegime) -> float:
        """Calculate how urgently the portfolio needs rebalancing."""
        if not allocations:
            return 0.0
        
        urgency = 0.0
        
        # Base urgency on weight deviations (assumed to be zero for new portfolio)
        for alloc in allocations:
            weight_deviation = abs(alloc.target_weight - alloc.current_weight)
            urgency += weight_deviation
        
        # Adjust for market regime
        regime_adjustments = {
            MarketRegime.CRISIS_MODE: 0.3,    # High urgency in crisis
            MarketRegime.HIGH_VOLATILITY: 0.2, # Higher urgency in volatility
            MarketRegime.BEAR_MARKET: 0.1,    # Some urgency in bear market
            MarketRegime.BULL_MARKET: -0.05,  # Less urgency in bull market
            MarketRegime.LOW_VOLATILITY: -0.1, # Lower urgency in low vol
            MarketRegime.SIDEWAYS_MARKET: 0.0  # No adjustment
        }
        
        urgency += regime_adjustments.get(market_regime, 0.0)
        
        return max(0.0, min(1.0, urgency))
    
    def _calculate_agent_consensus(self, risk_profile: RiskProfile,
                                 sector_scores: Dict[AssetClass, float],
                                 momentum_scores: Dict[str, float],
                                 value_scores: Dict[str, float],
                                 ml_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate consensus scores across agents."""
        consensus = {}
        
        # Agent agreement scores
        if momentum_scores:
            consensus['momentum_strength'] = np.mean(list(momentum_scores.values()))
        
        if value_scores:
            consensus['value_opportunity'] = np.mean(list(value_scores.values()))
        
        if ml_scores:
            consensus['ml_confidence'] = np.mean(list(ml_scores.values()))
        
        if sector_scores:
            consensus['sector_rotation'] = np.mean(list(sector_scores.values()))
        
        # Overall consensus
        all_scores = []
        for scores in [momentum_scores, value_scores, ml_scores]:
            if scores:
                all_scores.extend(scores.values())
        
        if all_scores:
            consensus['overall_consensus'] = np.mean(all_scores)
            consensus['consensus_std'] = np.std(all_scores)
        
        return consensus
    
    def _generate_portfolio_reasoning(self, market_regime: MarketRegime,
                                    risk_profile: RiskProfile,
                                    selected_symbols: List[str],
                                    portfolio_metrics: Dict[str, float],
                                    agents_consensus: Dict[str, float]) -> str:
        """Generate human-readable reasoning for the portfolio recommendation."""
        
        reasoning_parts = []
        
        # Market regime context
        reasoning_parts.append(f"Market regime: {market_regime.value.replace('_', ' ').title()}")
        
        # Risk profile context
        reasoning_parts.append(f"Risk profile: {risk_profile.value.title()}")
        
        # Portfolio composition
        reasoning_parts.append(f"Selected {len(selected_symbols)} positions")
        
        # Expected performance
        expected_return = portfolio_metrics.get('expected_return', 0) * 100
        expected_volatility = portfolio_metrics.get('expected_volatility', 0) * 100
        reasoning_parts.append(f"Expected return: {expected_return:.1f}%, volatility: {expected_volatility:.1f}%")
        
        # Diversification
        div_score = portfolio_metrics.get('diversification_score', 0) * 100
        reasoning_parts.append(f"Diversification score: {div_score:.1f}%")
        
        # Agent consensus
        if 'overall_consensus' in agents_consensus:
            consensus = agents_consensus['overall_consensus'] * 100
            reasoning_parts.append(f"Agent consensus: {consensus:.1f}%")
        
        # Cash allocation
        cash_alloc = portfolio_metrics.get('cash_allocation', 0) * 100
        if cash_alloc > 5:
            reasoning_parts.append(f"Cash reserve: {cash_alloc:.1f}%")
        
        return ". ".join(reasoning_parts) + "."

# Global portfolio construction engine
portfolio_engine = PortfolioConstructionEngine()

async def construct_optimal_portfolio(candidate_symbols: List[str],
                                    portfolio_value: float,
                                    risk_profile: str = "moderate",
                                    max_positions: int = 10) -> PortfolioRecommendation:
    """
    Main function to construct an optimal portfolio using multi-agent system.
    
    Args:
        candidate_symbols: List of stock symbols to consider
        portfolio_value: Total portfolio value in USD
        risk_profile: "conservative", "moderate", "aggressive", or "tactical"
        max_positions: Maximum number of positions in the portfolio
    
    Returns:
        PortfolioRecommendation with detailed allocations and analysis
    """
    try:
        # Convert risk profile string to enum
        risk_profile_enum = RiskProfile(risk_profile.lower())
        
        # Use the global portfolio engine
        recommendation = await portfolio_engine.construct_portfolio(
            candidate_symbols=candidate_symbols,
            portfolio_value=portfolio_value,
            risk_profile=risk_profile_enum,
            max_positions=max_positions
        )
        
        return recommendation
        
    except Exception as e:
        logger.error(f"Portfolio construction failed: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    async def main():
        candidate_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
            "JPM", "JNJ", "UNH", "XOM", "PFE"
        ]
        
        recommendation = await construct_optimal_portfolio(
            candidate_symbols=candidate_symbols,
            portfolio_value=100000.0,
            risk_profile="moderate",
            max_positions=8
        )
        
        print("\n🎯 PORTFOLIO RECOMMENDATION")
        print("=" * 50)
        print(f"Market Regime: {recommendation.market_regime.value}")
        print(f"Expected Return: {recommendation.expected_return:.1%}")
        print(f"Expected Volatility: {recommendation.expected_volatility:.1%}")
        print(f"Diversification Score: {recommendation.diversification_score:.3f}")
        print(f"Confidence: {recommendation.confidence:.1%}")
        print(f"Cash Allocation: {recommendation.cash_allocation:.1%}")
        print(f"\nReasoning: {recommendation.reasoning}")
        
        print(f"\n📊 PORTFOLIO ALLOCATIONS ({len(recommendation.allocations)} positions)")
        print("-" * 60)
        for alloc in recommendation.allocations:
            print(f"{alloc.symbol:>6}: {alloc.target_weight:>6.1%} | "
                  f"{alloc.asset_class.value:>15} | "
                  f"Confidence: {alloc.confidence:>5.1%}")
        
        print(f"\n🤖 AGENT CONSENSUS")
        print("-" * 30)
        for agent, score in recommendation.agents_consensus.items():
            print(f"{agent:>20}: {score:.3f}")
    
    # Run the example
    asyncio.run(main())