"""
LLM-RL Integration Layer
Enriches RL environment state with comprehensive LLM-based market analysis
"""

import torch
import numpy as np
import pandas as pd
import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

from agents.sentiment_agent import SentimentAgent, ComprehensiveSentiment
from agents.rl_trading_env import TradingState, MarketRegime, LLMTradingEnvironment
# Use existing tools instead of separate market data fetcher
from tools.alpaca_client import alpaca_client

logger = logging.getLogger(__name__)


class AnalysisDepth(Enum):
    """Analysis depth levels for LLM integration."""
    FAST = "fast"      # Basic sentiment only
    STANDARD = "standard"  # Full sentiment + news analysis
    DEEP = "deep"      # All sources + trend analysis


@dataclass 
class LLMMarketState:
    """Enhanced market state with LLM-derived insights."""
    # Core sentiment data
    overall_sentiment: str
    overall_score: float
    confidence: float
    
    # Detailed sentiment breakdown
    news_sentiment: float
    social_sentiment: float
    earnings_sentiment: float
    sec_filings_sentiment: float
    
    # Market insights
    trend_direction: str  # "bullish", "bearish", "neutral"
    trend_strength: float  # 0.0 to 1.0
    volatility_forecast: float
    regime_prediction: MarketRegime
    regime_confidence: float
    
    # Risk assessment
    risk_level: str  # "low", "medium", "high"
    risk_factors: List[str]
    opportunities: List[str]
    
    # Key themes and events
    key_themes: List[str]
    upcoming_events: List[str]
    market_catalysts: List[str]
    
    # Metadata
    analysis_timestamp: datetime
    data_freshness: float  # Hours since latest data
    analysis_depth: AnalysisDepth


@dataclass
class EnhancedTradingState:
    """Trading state enhanced with LLM market analysis."""
    # Original trading state components
    base_state: TradingState
    
    # LLM-enhanced market state
    llm_market_state: LLMMarketState
    
    # Derived features for RL
    sentiment_momentum: float
    consensus_strength: float
    news_impact_score: float
    fundamental_score: float
    technical_alignment: float


class LLMStateEnricher:
    """
    Enriches RL environment states with comprehensive LLM-based market analysis.
    """
    
    def __init__(self,
                 analysis_depth: AnalysisDepth = AnalysisDepth.STANDARD,
                 cache_duration_minutes: int = 15,
                 max_concurrent_requests: int = 3):
        
        self.analysis_depth = analysis_depth
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.max_concurrent = max_concurrent_requests
        
        # Initialize components
        self.sentiment_agent = SentimentAgent()
        # Market data fetcher is optional for basic RL integration
        try:
            from agents.online_learning_orchestrator import MarketDataFetcher
            self.market_data_fetcher = MarketDataFetcher()
        except ImportError:
            self.market_data_fetcher = None
        
        # State caching
        self.state_cache: Dict[str, Tuple[LLMMarketState, datetime]] = {}
        self.analysis_history: List[LLMMarketState] = []
        
        # Performance tracking
        self.enrichment_stats = {
            'total_enrichments': 0,
            'cache_hits': 0,
            'analysis_times': [],
            'error_count': 0
        }
        
        logger.info(f"✅ LLM State Enricher initialized with {analysis_depth.value} depth")
    
    async def enrich_trading_state(self,
                                 base_state: TradingState,
                                 symbol: str,
                                 force_refresh: bool = False) -> EnhancedTradingState:
        """Enrich base trading state with LLM analysis."""
        
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = f"{symbol}_{self.analysis_depth.value}"
            
            if not force_refresh and cache_key in self.state_cache:
                cached_state, cached_time = self.state_cache[cache_key]
                if datetime.now() - cached_time < self.cache_duration:
                    self.enrichment_stats['cache_hits'] += 1
                    logger.debug(f"🔄 Using cached LLM state for {symbol}")
                    
                    return self._create_enhanced_state(base_state, cached_state)
            
            # Perform fresh LLM analysis
            llm_market_state = await self._analyze_market_state(symbol)
            
            # Cache the result
            self.state_cache[cache_key] = (llm_market_state, datetime.now())
            self.analysis_history.append(llm_market_state)
            
            # Track performance
            analysis_time = (datetime.now() - start_time).total_seconds()
            self.enrichment_stats['total_enrichments'] += 1
            self.enrichment_stats['analysis_times'].append(analysis_time)
            
            logger.debug(f"✅ Enhanced state for {symbol} in {analysis_time:.2f}s")
            
            return self._create_enhanced_state(base_state, llm_market_state)
            
        except Exception as e:
            self.enrichment_stats['error_count'] += 1
            logger.error(f"❌ State enrichment failed for {symbol}: {e}")
            
            # Return base state with minimal LLM enhancement
            return self._create_fallback_enhanced_state(base_state, symbol)
    
    async def _analyze_market_state(self, symbol: str) -> LLMMarketState:
        """Perform comprehensive LLM-based market analysis."""
        
        # Get comprehensive sentiment analysis
        sentiment_result = await self.sentiment_agent.analyze_comprehensive_sentiment(symbol)
        
        # Analyze market trends and patterns
        trend_analysis = await self._analyze_market_trends(symbol, sentiment_result)
        
        # Assess risk factors and opportunities
        risk_assessment = await self._assess_market_risks(symbol, sentiment_result)
        
        # Extract key themes and events
        market_themes = await self._extract_market_themes(symbol, sentiment_result)
        
        # Predict market regime
        regime_prediction = self._predict_market_regime(sentiment_result, trend_analysis)
        
        return LLMMarketState(
            # Core sentiment
            overall_sentiment=sentiment_result.overall_sentiment,
            overall_score=sentiment_result.overall_score,
            confidence=sentiment_result.confidence,
            
            # Detailed sentiment breakdown
            news_sentiment=sentiment_result.news_sentiment.score if sentiment_result.news_sentiment else 0.0,
            social_sentiment=np.mean([s.score for s in sentiment_result.social_sentiment.values()]) if sentiment_result.social_sentiment else 0.0,
            earnings_sentiment=np.mean([s.score for s in sentiment_result.earnings_sentiment.values()]) if sentiment_result.earnings_sentiment else 0.0,
            sec_filings_sentiment=np.mean([s.score for s in sentiment_result.sec_filings_sentiment.values()]) if sentiment_result.sec_filings_sentiment else 0.0,
            
            # Market insights
            trend_direction=trend_analysis['direction'],
            trend_strength=trend_analysis['strength'],
            volatility_forecast=trend_analysis['volatility_forecast'],
            regime_prediction=regime_prediction['regime'],
            regime_confidence=regime_prediction['confidence'],
            
            # Risk assessment
            risk_level=risk_assessment['level'],
            risk_factors=risk_assessment['factors'],
            opportunities=risk_assessment['opportunities'],
            
            # Market themes
            key_themes=market_themes['themes'],
            upcoming_events=market_themes['events'],
            market_catalysts=market_themes['catalysts'],
            
            # Metadata
            analysis_timestamp=datetime.now(),
            data_freshness=self._calculate_data_freshness(sentiment_result),
            analysis_depth=self.analysis_depth
        )
    
    async def _analyze_market_trends(self, 
                                   symbol: str, 
                                   sentiment: ComprehensiveSentiment) -> Dict[str, Any]:
        """Analyze market trends using LLM and technical indicators."""
        
        try:
            # Get recent market data
            market_data = await self.market_data_fetcher.get_realtime_data(symbol)
            
            # Calculate trend indicators
            price_momentum = self._calculate_price_momentum(market_data)
            volume_trend = self._calculate_volume_trend(market_data)
            
            # Combine with sentiment for trend direction
            sentiment_bias = sentiment.overall_score
            
            if sentiment_bias > 0.2 and price_momentum > 0.1:
                direction = "bullish"
                strength = min(0.9, (sentiment_bias + price_momentum) / 2)
            elif sentiment_bias < -0.2 and price_momentum < -0.1:
                direction = "bearish"
                strength = min(0.9, abs(sentiment_bias + price_momentum) / 2)
            else:
                direction = "neutral"
                strength = 0.3
            
            # Forecast volatility based on sentiment dispersion
            volatility_forecast = self._forecast_volatility(sentiment, market_data)
            
            return {
                'direction': direction,
                'strength': strength,
                'volatility_forecast': volatility_forecast,
                'price_momentum': price_momentum,
                'volume_trend': volume_trend
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market trends: {e}")
            return {
                'direction': 'neutral',
                'strength': 0.5,
                'volatility_forecast': 0.2,
                'price_momentum': 0.0,
                'volume_trend': 0.0
            }
    
    async def _assess_market_risks(self,
                                 symbol: str,
                                 sentiment: ComprehensiveSentiment) -> Dict[str, Any]:
        """Assess market risks and opportunities using LLM analysis."""
        
        try:
            risk_factors = []
            opportunities = []
            
            # Analyze sentiment for risk indicators
            if sentiment.overall_score < -0.3:
                risk_factors.append("Negative sentiment trend")
            
            if sentiment.confidence < 0.6:
                risk_factors.append("High sentiment uncertainty")
            
            # Check for positive opportunities
            if sentiment.overall_score > 0.3 and sentiment.confidence > 0.7:
                opportunities.append("Strong positive sentiment with high confidence")
            
            # Extract insights from individual sources
            if sentiment.earnings_sentiment:
                avg_earnings_score = np.mean([s.score for s in sentiment.earnings_sentiment.values()])
                if avg_earnings_score > 0.4:
                    opportunities.append("Positive earnings outlook")
                elif avg_earnings_score < -0.4:
                    risk_factors.append("Concerning earnings developments")
            
            if sentiment.sec_filings_sentiment:
                avg_sec_score = np.mean([s.score for s in sentiment.sec_filings_sentiment.values()])
                if avg_sec_score < -0.3:
                    risk_factors.append("Regulatory filing concerns")
            
            # Determine overall risk level
            risk_score = len(risk_factors) - len(opportunities)
            
            if risk_score >= 2:
                risk_level = "high"
            elif risk_score >= 0:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                'level': risk_level,
                'factors': risk_factors[:5],  # Limit to top 5
                'opportunities': opportunities[:5]
            }
            
        except Exception as e:
            logger.error(f"Error assessing market risks: {e}")
            return {
                'level': 'medium',
                'factors': ['Analysis unavailable'],
                'opportunities': []
            }
    
    async def _extract_market_themes(self,
                                   symbol: str,
                                   sentiment: ComprehensiveSentiment) -> Dict[str, List[str]]:
        """Extract key market themes and events."""
        
        try:
            themes = []
            events = []
            catalysts = []
            
            # Extract from sentiment insights
            if hasattr(sentiment, 'insights') and sentiment.insights:
                for insight in sentiment.insights[:3]:
                    themes.append(insight)
            
            # Add common market themes based on sentiment
            if sentiment.overall_score > 0.3:
                themes.append("Positive market momentum")
                catalysts.append("Strong investor sentiment")
            elif sentiment.overall_score < -0.3:
                themes.append("Market uncertainty")
                catalysts.append("Risk-off sentiment")
            
            # Extract from earnings sentiment
            if sentiment.earnings_sentiment:
                themes.append("Earnings focus")
                events.append("Earnings season activity")
            
            # Extract from SEC filings
            if sentiment.sec_filings_sentiment:
                themes.append("Regulatory developments")
                events.append("SEC filing updates")
            
            return {
                'themes': themes[:5],
                'events': events[:5],
                'catalysts': catalysts[:5]
            }
            
        except Exception as e:
            logger.error(f"Error extracting market themes: {e}")
            return {
                'themes': ['Market analysis'],
                'events': [],
                'catalysts': []
            }
    
    def _predict_market_regime(self,
                             sentiment: ComprehensiveSentiment,
                             trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Predict market regime using sentiment and trend data."""
        
        try:
            sentiment_score = sentiment.overall_score
            trend_strength = trend_analysis.get('strength', 0.5)
            volatility = trend_analysis.get('volatility_forecast', 0.2)
            
            # Regime prediction logic
            if sentiment_score > 0.2 and trend_strength > 0.6 and volatility < 0.25:
                regime = MarketRegime.BULL
                confidence = min(0.9, sentiment.confidence * trend_strength)
            elif sentiment_score < -0.2 and trend_strength > 0.6 and volatility < 0.25:
                regime = MarketRegime.BEAR
                confidence = min(0.9, sentiment.confidence * trend_strength)
            elif volatility > 0.4:
                regime = MarketRegime.VOLATILE
                confidence = min(0.8, volatility)
            else:
                regime = MarketRegime.SIDEWAYS
                confidence = 0.6
            
            return {
                'regime': regime,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error predicting market regime: {e}")
            return {
                'regime': MarketRegime.SIDEWAYS,
                'confidence': 0.5
            }
    
    def _create_enhanced_state(self,
                             base_state: TradingState,
                             llm_state: LLMMarketState) -> EnhancedTradingState:
        """Create enhanced trading state with derived features."""
        
        # Calculate derived features for RL
        sentiment_momentum = self._calculate_sentiment_momentum(llm_state)
        consensus_strength = self._calculate_consensus_strength(llm_state)
        news_impact_score = self._calculate_news_impact(llm_state)
        fundamental_score = self._calculate_fundamental_score(llm_state)
        technical_alignment = self._calculate_technical_alignment(base_state, llm_state)
        
        return EnhancedTradingState(
            base_state=base_state,
            llm_market_state=llm_state,
            sentiment_momentum=sentiment_momentum,
            consensus_strength=consensus_strength,
            news_impact_score=news_impact_score,
            fundamental_score=fundamental_score,
            technical_alignment=technical_alignment
        )
    
    def _create_fallback_enhanced_state(self,
                                      base_state: TradingState,
                                      symbol: str) -> EnhancedTradingState:
        """Create minimal enhanced state when LLM analysis fails."""
        
        fallback_llm_state = LLMMarketState(
            overall_sentiment="neutral",
            overall_score=0.0,
            confidence=0.5,
            news_sentiment=0.0,
            social_sentiment=0.0,
            earnings_sentiment=0.0,
            sec_filings_sentiment=0.0,
            trend_direction="neutral",
            trend_strength=0.5,
            volatility_forecast=0.2,
            regime_prediction=MarketRegime.SIDEWAYS,
            regime_confidence=0.5,
            risk_level="medium",
            risk_factors=["Analysis unavailable"],
            opportunities=[],
            key_themes=["Market analysis unavailable"],
            upcoming_events=[],
            market_catalysts=[],
            analysis_timestamp=datetime.now(),
            data_freshness=24.0,  # Stale data
            analysis_depth=self.analysis_depth
        )
        
        return EnhancedTradingState(
            base_state=base_state,
            llm_market_state=fallback_llm_state,
            sentiment_momentum=0.0,
            consensus_strength=0.5,
            news_impact_score=0.0,
            fundamental_score=0.0,
            technical_alignment=0.0
        )
    
    # Helper methods for feature calculation
    def _calculate_price_momentum(self, market_data) -> float:
        """Calculate price momentum from market data."""
        try:
            if len(market_data) < 5:
                return 0.0
            
            recent_prices = market_data['close'].tail(5).values
            returns = np.diff(recent_prices) / recent_prices[:-1]
            return np.tanh(np.mean(returns) * 100)  # Normalize to [-1, 1]
            
        except Exception:
            return 0.0
    
    def _calculate_volume_trend(self, market_data) -> float:
        """Calculate volume trend from market data."""
        try:
            if len(market_data) < 10:
                return 0.0
            
            recent_volume = market_data['volume'].tail(5).mean()
            historical_volume = market_data['volume'].tail(20).head(15).mean()
            
            volume_ratio = recent_volume / (historical_volume + 1e-8)
            return np.tanh((volume_ratio - 1) * 2)  # Normalize
            
        except Exception:
            return 0.0
    
    def _forecast_volatility(self, sentiment, market_data) -> float:
        """Forecast volatility using sentiment dispersion."""
        try:
            # Base volatility from market data
            base_vol = 0.2
            
            if len(market_data) >= 20:
                returns = market_data['close'].pct_change().tail(20)
                base_vol = returns.std() * np.sqrt(252)
            
            # Adjust based on sentiment uncertainty
            sentiment_vol_factor = 1.0 - sentiment.confidence
            adjusted_vol = base_vol * (1 + sentiment_vol_factor)
            
            return min(1.0, adjusted_vol)
            
        except Exception:
            return 0.2
    
    def _calculate_data_freshness(self, sentiment: ComprehensiveSentiment) -> float:
        """Calculate data freshness score."""
        # This would typically check timestamps of various data sources
        # For now, return a reasonable default
        return 2.0  # 2 hours
    
    def _calculate_sentiment_momentum(self, llm_state: LLMMarketState) -> float:
        """Calculate sentiment momentum from historical data."""
        # Compare current sentiment to recent history
        if len(self.analysis_history) < 2:
            return 0.0
        
        current_score = llm_state.overall_score
        prev_score = self.analysis_history[-2].overall_score if len(self.analysis_history) >= 2 else 0.0
        
        return np.tanh((current_score - prev_score) * 5)
    
    def _calculate_consensus_strength(self, llm_state: LLMMarketState) -> float:
        """Calculate strength of sentiment consensus across sources."""
        sentiment_scores = [
            llm_state.news_sentiment,
            llm_state.social_sentiment,
            llm_state.earnings_sentiment,
            llm_state.sec_filings_sentiment
        ]
        
        # Remove zero scores (missing data)
        valid_scores = [s for s in sentiment_scores if abs(s) > 0.01]
        
        if len(valid_scores) < 2:
            return 0.5
        
        # Calculate agreement strength
        mean_sentiment = np.mean(valid_scores)
        std_sentiment = np.std(valid_scores)
        
        # High consensus = low standard deviation
        consensus = 1.0 - min(1.0, std_sentiment * 2)
        return consensus
    
    def _calculate_news_impact(self, llm_state: LLMMarketState) -> float:
        """Calculate potential news impact score."""
        news_score = abs(llm_state.news_sentiment) * llm_state.confidence
        
        # Boost if there are key themes or catalysts
        theme_boost = min(0.3, len(llm_state.key_themes) * 0.1)
        catalyst_boost = min(0.2, len(llm_state.market_catalysts) * 0.1)
        
        return min(1.0, news_score + theme_boost + catalyst_boost)
    
    def _calculate_fundamental_score(self, llm_state: LLMMarketState) -> float:
        """Calculate fundamental analysis score."""
        # Weight SEC filings and earnings higher for fundamentals
        fundamental_score = (
            llm_state.sec_filings_sentiment * 0.6 +
            llm_state.earnings_sentiment * 0.4
        )
        
        # Adjust for opportunities vs risks
        opportunity_factor = len(llm_state.opportunities) * 0.1
        risk_factor = len(llm_state.risk_factors) * -0.05
        
        return np.tanh(fundamental_score + opportunity_factor + risk_factor)
    
    def _calculate_technical_alignment(self,
                                     base_state: TradingState,
                                     llm_state: LLMMarketState) -> float:
        """Calculate alignment between technical and sentiment signals."""
        
        # Technical signal from RSI and price position
        rsi_signal = (base_state.rsi - 50) / 50  # Normalize RSI to [-1, 1]
        
        bollinger_position = (base_state.price - base_state.bollinger_lower) / \
                           (base_state.bollinger_upper - base_state.bollinger_lower + 1e-8)
        bollinger_signal = (bollinger_position - 0.5) * 2  # Normalize to [-1, 1]
        
        technical_signal = (rsi_signal + bollinger_signal) / 2
        
        # Sentiment signal
        sentiment_signal = llm_state.overall_score
        
        # Calculate alignment (correlation)
        alignment = 1.0 - abs(technical_signal - sentiment_signal) / 2
        return max(0.0, alignment)
    
    def get_enrichment_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the enricher."""
        avg_analysis_time = np.mean(self.enrichment_stats['analysis_times']) if self.enrichment_stats['analysis_times'] else 0.0
        cache_hit_rate = self.enrichment_stats['cache_hits'] / max(1, self.enrichment_stats['total_enrichments'])
        
        return {
            'total_enrichments': self.enrichment_stats['total_enrichments'],
            'cache_hit_rate': cache_hit_rate,
            'average_analysis_time': avg_analysis_time,
            'error_rate': self.enrichment_stats['error_count'] / max(1, self.enrichment_stats['total_enrichments']),
            'cache_size': len(self.state_cache)
        }
    
    async def close(self):
        """Clean up resources."""
        if self.sentiment_agent:
            await self.sentiment_agent.close()
        
        logger.info("✅ LLM State Enricher closed")
    
    async def enrich_state_from_llm_analysis(self,
                                           sentiment_data: Dict[str, Any],
                                           market_signals: List[Dict[str, Any]],
                                           portfolio_state: Dict[str, Any],
                                           market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create enhanced state from LLM analysis results for RL decision making.
        
        This bridges the LLM Analysis Layer output to RL-compatible state representation.
        """
        try:
            logger.info("🔄 Creating enhanced state from LLM analysis...")
            
            # Aggregate sentiment scores from LLM analysis
            overall_sentiment = 0.0
            sentiment_confidence = 0.5
            sentiment_count = 0
            
            for symbol, sentiment in sentiment_data.items():
                if hasattr(sentiment, 'overall_score'):
                    overall_sentiment += sentiment.overall_score
                    sentiment_confidence += sentiment.confidence
                    sentiment_count += 1
            
            if sentiment_count > 0:
                overall_sentiment /= sentiment_count
                sentiment_confidence /= sentiment_count
            
            # Extract market signal strength
            buy_signals = [s for s in market_signals if s.get('signal') == 'BUY']
            signal_strength = len(buy_signals) / max(len(market_signals), 1) if market_signals else 0.0
            
            # Create enhanced state representation
            enhanced_state = {
                'sentiment_metrics': {
                    'overall_sentiment': overall_sentiment,
                    'sentiment_confidence': sentiment_confidence,
                    'signal_strength': signal_strength,
                    'total_signals': len(market_signals),
                    'buy_signals': len(buy_signals)
                },
                'portfolio_metrics': {
                    'total_value': portfolio_state.get('equity', 50000),
                    'cash_balance': portfolio_state.get('cash', 5000),
                    'utilization': 1.0 - (portfolio_state.get('cash', 5000) / portfolio_state.get('equity', 50000))
                },
                'market_context': {
                    'available_symbols': len(market_signals),
                    'data_freshness': 1.0,  # Assume fresh data
                    'market_regime': 'normal'  # Simplified
                },
                'risk_factors': {
                    'portfolio_concentration': min(1.0, len(buy_signals) / 10.0),
                    'sentiment_volatility': abs(overall_sentiment) * (1.0 - sentiment_confidence),
                    'signal_quality': sentiment_confidence
                }
            }
            
            logger.info(f"✅ Enhanced state: {overall_sentiment:.2f} sentiment, {len(buy_signals)} BUY signals")
            return enhanced_state
            
        except Exception as e:
            logger.error(f"Error enriching state from LLM analysis: {e}")
            # Return minimal fallback state
            return {
                'sentiment_metrics': {'overall_sentiment': 0.0, 'sentiment_confidence': 0.3},
                'portfolio_metrics': {'total_value': 50000, 'cash_balance': 5000},
                'market_context': {'available_symbols': 0},
                'risk_factors': {'portfolio_concentration': 0.0}
            }


class EnhancedLLMTradingEnvironment(LLMTradingEnvironment):
    """
    Enhanced trading environment that uses LLM state enrichment.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize state enricher
        self.state_enricher = LLMStateEnricher(
            analysis_depth=AnalysisDepth.STANDARD,
            cache_duration_minutes=15
        )
        
        # Enhanced state tracking
        self.enhanced_states_history: List[EnhancedTradingState] = []
        
        logger.info("✅ Enhanced LLM Trading Environment initialized")
    
    async def _get_enhanced_current_state(self) -> EnhancedTradingState:
        """Get current enhanced trading state with LLM enrichment."""
        
        # Get base trading state
        base_state = await self._get_current_state()
        
        # Enrich with LLM analysis
        enhanced_state = await self.state_enricher.enrich_trading_state(
            base_state, self.symbol
        )
        
        return enhanced_state
    
    async def reset(self) -> np.ndarray:
        """Reset environment with enhanced state."""
        # Reset base environment
        base_obs = await super().reset()
        
        # Get enhanced state
        enhanced_state = await self._get_enhanced_current_state()
        self.enhanced_states_history = [enhanced_state]
        
        # Return enhanced observation
        return self._enhanced_state_to_array(enhanced_state)
    
    async def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step with enhanced state and LLM-aware rewards."""
        
        # Execute base step
        _, base_reward, done, info = await super().step(action)
        
        # Get enhanced current state
        enhanced_state = await self._get_enhanced_current_state()
        self.enhanced_states_history.append(enhanced_state)
        
        # Calculate enhanced reward
        if len(self.enhanced_states_history) >= 2:
            prev_enhanced_state = self.enhanced_states_history[-2]
            enhanced_reward = self._calculate_enhanced_reward(
                action, prev_enhanced_state, enhanced_state
            )
        else:
            enhanced_reward = base_reward
        
        # Add LLM insights to info
        info.update({
            'llm_sentiment': enhanced_state.llm_market_state.overall_score,
            'llm_confidence': enhanced_state.llm_market_state.confidence,
            'trend_direction': enhanced_state.llm_market_state.trend_direction,
            'risk_level': enhanced_state.llm_market_state.risk_level,
            'sentiment_momentum': enhanced_state.sentiment_momentum,
            'consensus_strength': enhanced_state.consensus_strength
        })
        
        return self._enhanced_state_to_array(enhanced_state), enhanced_reward, done, info
    
    def _enhanced_state_to_array(self, enhanced_state: EnhancedTradingState) -> np.ndarray:
        """Convert enhanced state to array for RL agent."""
        
        # Get base state array
        base_array = self._state_to_array(enhanced_state.base_state)
        
        # Add enhanced features
        llm_features = np.array([
            enhanced_state.sentiment_momentum,
            enhanced_state.consensus_strength,
            enhanced_state.news_impact_score,
            enhanced_state.fundamental_score,
            enhanced_state.technical_alignment,
            enhanced_state.llm_market_state.trend_strength,
            1.0 if enhanced_state.llm_market_state.risk_level == "high" else 0.5 if enhanced_state.llm_market_state.risk_level == "medium" else 0.0
        ], dtype=np.float32)
        
        # Combine base and enhanced features
        return np.concatenate([base_array, llm_features])
    
    def _calculate_enhanced_reward(self,
                                 action: float,
                                 prev_state: EnhancedTradingState,
                                 current_state: EnhancedTradingState) -> float:
        """Calculate reward with LLM enhancement."""
        
        # Get base reward
        base_reward_components = self._calculate_reward(
            action, prev_state.base_state, current_state.base_state
        )
        
        # LLM-based reward enhancements
        llm_enhancements = 0.0
        
        # Sentiment alignment bonus
        sentiment_score = current_state.llm_market_state.overall_score
        position = current_state.base_state.position
        
        if abs(sentiment_score) > 0.1:  # Only if sentiment is significant
            if (position > 0 and sentiment_score > 0) or (position < 0 and sentiment_score < 0):
                alignment_bonus = abs(sentiment_score) * current_state.llm_market_state.confidence * 0.05
                llm_enhancements += alignment_bonus
        
        # Consensus strength bonus
        if current_state.consensus_strength > 0.7:
            consensus_bonus = (current_state.consensus_strength - 0.7) * 0.02
            llm_enhancements += consensus_bonus
        
        # Technical-fundamental alignment bonus
        if current_state.technical_alignment > 0.8:
            alignment_bonus = (current_state.technical_alignment - 0.8) * 0.03
            llm_enhancements += alignment_bonus
        
        # Risk penalty enhancement
        if current_state.llm_market_state.risk_level == "high" and abs(position) > 0.5:
            risk_penalty = -0.01 * abs(position)
            llm_enhancements += risk_penalty
        
        return base_reward_components.total_reward + llm_enhancements
    
    async def close(self):
        """Clean up enhanced environment."""
        await super().close()
        await self.state_enricher.close()


# Utility functions
def create_enhanced_environment(symbol: str = "SPY", **kwargs) -> EnhancedLLMTradingEnvironment:
    """Create enhanced trading environment with optimal settings."""
    
    return EnhancedLLMTradingEnvironment(
        symbol=symbol,
        sentiment_weight=0.4,  # Higher weight for LLM insights
        **kwargs
    )


if __name__ == "__main__":
    # Test the enhanced environment
    async def test_enhanced_environment():
        env = create_enhanced_environment("SPY")
        
        # Create sample data
        import yfinance as yf
        data = yf.download("SPY", period="1mo", interval="1h")
        env.set_data(data)
        
        # Test enhanced reset and steps
        obs = await env.reset()
        print(f"Enhanced observation shape: {obs.shape}")
        
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, info = await env.step(action)
            
            print(f"\nStep {i+1}:")
            print(f"  LLM Sentiment: {info.get('llm_sentiment', 'N/A')}")
            print(f"  Trend Direction: {info.get('trend_direction', 'N/A')}")
            print(f"  Risk Level: {info.get('risk_level', 'N/A')}")
            print(f"  Enhanced Reward: {reward:.6f}")
            
            if done:
                break
        
        # Show enricher stats
        stats = env.state_enricher.get_enrichment_stats()
        print(f"\nEnrichment Stats: {stats}")
        
        await env.close()
    
    asyncio.run(test_enhanced_environment())