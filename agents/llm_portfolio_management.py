#!/usr/bin/env python3
"""
LLM-Enhanced Portfolio Management with Sentiment Integration

Creates diversified portfolios using sentiment analysis from multiple sources
combined with fundamental analysis and risk management.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from agents.sentiment_agent import sentiment_agent, ComprehensiveSentiment
from enhanced_market_screener import enhanced_screener
from tools.llm_client import llm_client
from config.settings import settings

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime classifications."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    VOLATILE = "volatile"
    CONSOLIDATING = "consolidating"
    UNCERTAIN = "uncertain"


@dataclass
class StockAllocation:
    """Individual stock allocation in portfolio."""
    symbol: str
    target_weight: float
    confidence: float
    recommended_action: str  # buy, hold, sell
    reasoning: str
    sentiment_score: float
    industry: str
    risk_level: str  # low, medium, high


@dataclass
class PortfolioRecommendation:
    """Complete portfolio recommendation."""
    allocations: List[StockAllocation]
    market_regime: MarketRegime
    expected_return: float
    confidence: float
    cash_allocation: float
    total_risk_score: float
    diversification_score: float
    rationale: str


class LLMPortfolioManager:
    """LLM-enhanced portfolio manager with sentiment integration."""
    
    def __init__(self):
        self.max_position_size = settings.max_position_size
        self.max_sector_allocation = settings.max_sector_allocation
        self.min_cash_reserve = settings.min_cash_reserve
        
        # Risk weightings for sentiment integration
        self.sentiment_weight = 0.4  # 40% weight to sentiment
        self.fundamental_weight = 0.4  # 40% weight to fundamentals
        self.technical_weight = 0.2   # 20% weight to technicals
    
    async def construct_llm_portfolio(
        self, 
        candidate_symbols: List[str], 
        portfolio_value: float, 
        risk_profile: str = "moderate",
        max_positions: int = 35,
        sentiment_data: Optional[Dict] = None
    ) -> PortfolioRecommendation:
        """Construct portfolio using LLM analysis with sentiment integration."""
        
        logger.info(f"Constructing LLM portfolio with {len(candidate_symbols)} candidates")
        logger.info(f"Portfolio value: ${portfolio_value:,.2f}, Risk profile: {risk_profile}")
        
        # Step 1: Get diversified stock selection from enhanced screener
        diversified_selection = await enhanced_screener.get_diversified_stock_selection(max_positions)
        
        # Flatten the selection into a candidate list
        screener_symbols = []
        for industry_stocks in diversified_selection.values():
            screener_symbols.extend(industry_stocks)
        
        # Include symbols with sentiment data alongside screener picks for comprehensive analysis
        all_candidate_symbols = set(screener_symbols)
        
        # Also include provided candidate symbols (from universe filter)
        all_candidate_symbols.update(candidate_symbols)
        
        if sentiment_data:
            # Add any symbols with comprehensive sentiment data to the analysis
            sentiment_symbols = list(sentiment_data.keys())
            all_candidate_symbols.update(sentiment_symbols)
            logger.info(f"Including {len(sentiment_symbols)} symbols with sentiment data: {sentiment_symbols}")
        
        # Create final candidates list (prioritize provided candidates, then screener picks)
        final_candidates = []
        
        # First, add provided candidates (from universe filter - these have trading signals)
        for symbol in candidate_symbols[:max_positions//2]:
            if symbol not in final_candidates:
                final_candidates.append(symbol)
        
        # Then add screener picks to fill remaining slots
        for symbol in screener_symbols:
            if len(final_candidates) >= max_positions:
                break
            if symbol not in final_candidates:
                final_candidates.append(symbol)
        
        # Ensure we have at least some candidates
        if not final_candidates and candidate_symbols:
            final_candidates = candidate_symbols[:max_positions]
        elif not final_candidates and screener_symbols:
            final_candidates = screener_symbols[:max_positions]
        
        logger.info(f"Selected {len(final_candidates)} candidates for analysis")
        
        # Step 2: Use provided sentiment data or analyze sentiment for all candidates
        # Handle case where sentiment_data might be a coroutine
        import inspect
        if inspect.iscoroutine(sentiment_data):
            logger.info("Sentiment data is a coroutine, awaiting it...")
            sentiment_data = await sentiment_data
        
        if sentiment_data and isinstance(sentiment_data, dict):
            logger.info(f"Using provided sentiment data for {len(sentiment_data)} symbols")
            sentiment_results = {}
            for symbol in final_candidates:
                if symbol in sentiment_data:
                    sentiment_results[symbol] = sentiment_data[symbol]
                    data = sentiment_data[symbol]
                    # Handle both object and dictionary formats
                    if isinstance(data, dict):
                        sentiment_str = data.get('overall_sentiment', 'N/A')
                        has_earnings = data.get('has_recent_earnings', False)
                    else:
                        sentiment_str = getattr(data, 'overall_sentiment', 'N/A')
                        has_earnings = getattr(data, 'has_recent_earnings', False)
                    logger.info(f"Using sentiment for {symbol}: {sentiment_str} (earnings: {'Yes' if has_earnings else 'No'})")
            
            # Fill in missing sentiment data with fresh analysis for key candidates
            missing_sentiment = [s for s in final_candidates if s not in sentiment_results]
            if missing_sentiment:
                logger.info(f"Analyzing sentiment for {len(missing_sentiment)} candidates without data")
                additional_sentiment = await self._analyze_candidate_sentiments(missing_sentiment)
                sentiment_results.update(additional_sentiment)
        else:
            logger.info("No pre-computed sentiment data provided, analyzing all candidates")
            sentiment_results = await self._analyze_candidate_sentiments(final_candidates)
        
        # Step 3: Get market data for candidates
        market_data = await enhanced_screener.get_bulk_stock_data(final_candidates)
        
        # Step 4: Score and rank candidates
        scored_candidates = await self._score_candidates(
            final_candidates, sentiment_results, market_data, risk_profile
        )
        
        # Step 5: Determine market regime
        market_regime = await self._assess_market_regime(sentiment_results)
        
        # Step 6: Create portfolio allocation
        allocations = self._create_portfolio_allocation(
            scored_candidates, portfolio_value, market_regime, risk_profile
        )
        
        # Step 7: Calculate portfolio metrics
        diversification_score = self._calculate_diversification_score(allocations)
        expected_return, confidence, total_risk = self._calculate_portfolio_metrics(
            allocations, sentiment_results, market_regime
        )
        
        # Step 8: Generate rationale
        rationale = await self._generate_portfolio_rationale(
            allocations, market_regime, sentiment_results
        )
        
        return PortfolioRecommendation(
            allocations=allocations,
            market_regime=market_regime,
            expected_return=expected_return,
            confidence=confidence,
            cash_allocation=self.min_cash_reserve,
            total_risk_score=total_risk,
            diversification_score=diversification_score,
            rationale=rationale
        )
    
    async def _analyze_candidate_sentiments(self, symbols: List[str]) -> Dict[str, ComprehensiveSentiment]:
        """Analyze sentiment for all candidate stocks."""
        sentiment_results = {}
        
        # Analyze in batches to avoid overwhelming APIs
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            tasks = [sentiment_agent.analyze_comprehensive_sentiment(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.warning(f"Sentiment analysis failed for {symbol}: {result}")
                    # Create neutral sentiment as fallback
                    sentiment_results[symbol] = self._create_neutral_sentiment(symbol)
                else:
                    sentiment_results[symbol] = result
            
            # Brief pause between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(2)
        
        logger.info(f"Completed sentiment analysis for {len(sentiment_results)} stocks")
        return sentiment_results
    
    def _create_neutral_sentiment(self, symbol: str) -> ComprehensiveSentiment:
        """Create neutral sentiment fallback."""
        from agents.sentiment_agent import ComprehensiveSentiment
        
        return ComprehensiveSentiment(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_score=0.0,
            overall_sentiment="neutral",
            confidence=0.3,
            data_sources_count=0,
            news_articles_count=0,
            social_posts_count=0,
            has_recent_earnings=False,
            key_themes=[],
            risk_factors=[],
            opportunities=[]
        )
    
    async def _score_candidates(
        self, 
        symbols: List[str], 
        sentiment_data: Dict[str, ComprehensiveSentiment],
        market_data: Dict[str, Dict],
        risk_profile: str
    ) -> List[Tuple[str, float]]:
        """Score and rank candidate stocks."""
        
        scored_stocks = []
        
        for symbol in symbols:
            sentiment = sentiment_data.get(symbol)
            market_info = market_data.get(symbol)
            
            if not sentiment or not market_info:
                continue
            
            # Calculate composite score - handle both object and dictionary formats
            if isinstance(sentiment, dict):
                sentiment_score = sentiment.get('overall_score', 0.0)
                has_recent_earnings = sentiment.get('has_recent_earnings', False)
                sentiment_confidence = sentiment.get('confidence', 0.5)
            else:
                sentiment_score = sentiment.overall_score
                has_recent_earnings = getattr(sentiment, 'has_recent_earnings', False)
                sentiment_confidence = sentiment.confidence
            
            fundamental_score = self._calculate_fundamental_score(market_info)
            technical_score = self._calculate_technical_score(market_info)
            
            # Apply risk profile adjustments
            risk_adjustment = self._get_risk_adjustment(risk_profile, sentiment, market_info)
            
            # Calculate comprehensive score including earnings as one factor
            # Earnings data provides additional confidence and sentiment precision
            earnings_quality_bonus = 0.0
            if has_recent_earnings:
                # Modest bonus for having fresh earnings data (better sentiment accuracy)
                earnings_quality_bonus = 0.1 * sentiment_confidence  # Up to 0.1 bonus based on confidence
                logger.debug(f"Earnings data quality bonus for {symbol}: +{earnings_quality_bonus:.3f}")
            
            # Weighted composite score balancing all data sources
            composite_score = (
                sentiment_score * self.sentiment_weight +
                fundamental_score * self.fundamental_weight +
                technical_score * self.technical_weight
            ) * risk_adjustment + earnings_quality_bonus
            
            scored_stocks.append((symbol, composite_score))
        
        # Sort by score (highest first)
        scored_stocks.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Scored {len(scored_stocks)} candidates")
        for symbol, score in scored_stocks[:10]:  # Log top 10
            logger.info(f"  {symbol}: {score:.3f}")
        
        return scored_stocks
    
    def _calculate_fundamental_score(self, market_info: Dict) -> float:
        """Calculate fundamental analysis score."""
        score = 0.0
        
        # Price-based scoring (prefer reasonably priced stocks)
        price = market_info.get('price', 0)
        if 20 <= price <= 200:
            score += 0.3
        elif 10 <= price <= 300:
            score += 0.1
        
        # Market cap scoring (prefer mid to large cap)
        market_cap = market_info.get('market_cap', 0)
        if market_cap > 10_000_000_000:  # >$10B
            score += 0.3
        elif market_cap > 2_000_000_000:  # >$2B
            score += 0.2
        
        # PE ratio scoring (reasonable valuations)
        pe_ratio = market_info.get('pe_ratio', 0)
        if 10 <= pe_ratio <= 25:
            score += 0.2
        elif 5 <= pe_ratio <= 35:
            score += 0.1
        
        # Recent performance
        change_pct = market_info.get('change_pct', 0)
        if -2 <= change_pct <= 5:  # Stable to moderate growth
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_technical_score(self, market_info: Dict) -> float:
        """Calculate technical analysis score."""
        score = 0.0
        
        # Volume-based scoring
        volume = market_info.get('volume', 0)
        if volume > 1_000_000:  # High volume
            score += 0.3
        elif volume > 100_000:  # Moderate volume
            score += 0.1
        
        # Price momentum
        change_pct = market_info.get('change_pct', 0)
        if change_pct > 0:  # Positive momentum
            score += 0.3
        elif change_pct > -5:  # Not too negative
            score += 0.1
        
        # Volatility consideration (prefer moderate volatility)
        if abs(change_pct) < 10:  # Not too volatile
            score += 0.4
        
        return min(1.0, score)
    
    def _get_risk_adjustment(
        self, 
        risk_profile: str, 
        sentiment, 
        market_info: Dict
    ) -> float:
        """Get risk adjustment factor based on profile."""
        
        base_adjustment = 1.0
        
        # Handle both object and dictionary formats
        if isinstance(sentiment, dict):
            sentiment_score = sentiment.get('overall_score', 0.0)
            confidence = sentiment.get('confidence', 0.5)
        else:
            sentiment_score = sentiment.overall_score
            confidence = sentiment.confidence
        
        if risk_profile == "conservative":
            # Prefer stocks with high confidence and low volatility
            if confidence > 0.7:
                base_adjustment += 0.2
            if abs(market_info.get('change_pct', 0)) < 3:
                base_adjustment += 0.1
        
        elif risk_profile == "aggressive":
            # Prefer higher growth potential
            if sentiment_score > 0.3:
                base_adjustment += 0.3
            if market_info.get('change_pct', 0) > 3:
                base_adjustment += 0.2
        
        # Default is moderate (no major adjustments)
        
        return base_adjustment
    
    async def _assess_market_regime(self, sentiment_data: Dict) -> MarketRegime:
        """Assess overall market regime from sentiment data."""
        
        if not sentiment_data:
            return MarketRegime.UNCERTAIN
        
        # Handle case where sentiment_data might be a coroutine
        import inspect
        if inspect.iscoroutine(sentiment_data):
            logger.warning("Sentiment data is a coroutine, awaiting it...")
            sentiment_data = await sentiment_data
        
        if not sentiment_data or not isinstance(sentiment_data, dict):
            logger.warning("Invalid sentiment data format, returning UNCERTAIN regime")
            return MarketRegime.UNCERTAIN
        
        # Calculate average sentiment across all stocks - handle both object and dictionary formats
        sentiment_scores = []
        confidences = []
        
        for s in sentiment_data.values():
            if isinstance(s, dict):
                sentiment_scores.append(s.get('overall_score', 0.0))
                confidences.append(s.get('confidence', 0.5))
            else:
                sentiment_scores.append(getattr(s, 'overall_score', 0.0))
                confidences.append(getattr(s, 'confidence', 0.5))
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        avg_confidence = sum(confidences) / len(confidences)
        
        # Determine regime
        if avg_sentiment > 0.3 and avg_confidence > 0.6:
            return MarketRegime.BULL_MARKET
        elif avg_sentiment < -0.3 and avg_confidence > 0.6:
            return MarketRegime.BEAR_MARKET
        elif avg_confidence < 0.4:
            return MarketRegime.UNCERTAIN
        elif max(sentiment_scores) - min(sentiment_scores) > 1.0:
            return MarketRegime.VOLATILE
        else:
            return MarketRegime.CONSOLIDATING
    
    def _create_portfolio_allocation(
        self, 
        scored_candidates: List[Tuple[str, float]], 
        portfolio_value: float,
        market_regime: MarketRegime,
        risk_profile: str
    ) -> List[StockAllocation]:
        """Create portfolio allocation from scored candidates."""
        
        allocations = []
        total_weight = 0.0
        
        # Determine number of positions based on regime and risk profile
        if market_regime == MarketRegime.BULL_MARKET:
            target_positions = min(25, len(scored_candidates))
        elif market_regime == MarketRegime.BEAR_MARKET:
            target_positions = min(15, len(scored_candidates))  # More concentrated
        else:
            target_positions = min(20, len(scored_candidates))
        
        # Calculate base weight per position
        available_weight = 1.0 - self.min_cash_reserve
        base_weight = available_weight / target_positions
        
        # Create allocations with score-based weighting
        for i, (symbol, score) in enumerate(scored_candidates[:target_positions]):
            # Weight adjustment based on score and position
            position_multiplier = max(0.5, min(2.0, score))  # 0.5x to 2.0x base weight
            position_decay = max(0.5, 1.0 - (i * 0.02))  # Slight decay for lower-ranked positions
            
            target_weight = base_weight * position_multiplier * position_decay
            target_weight = min(target_weight, self.max_position_size)
            
            # Get industry for the stock
            industry = enhanced_screener.get_industry_for_symbol(symbol)
            
            # Determine risk level
            risk_level = self._determine_risk_level(score, market_regime)
            
            allocation = StockAllocation(
                symbol=symbol,
                target_weight=target_weight,
                confidence=min(0.9, score),
                recommended_action="buy",
                reasoning=f"Score: {score:.3f}, Industry: {industry}",
                sentiment_score=score,
                industry=industry,
                risk_level=risk_level
            )
            
            allocations.append(allocation)
            total_weight += target_weight
        
        # Normalize weights if needed
        if total_weight > available_weight:
            scale_factor = available_weight / total_weight
            for allocation in allocations:
                allocation.target_weight *= scale_factor
        
        logger.info(f"Created {len(allocations)} portfolio allocations")
        return allocations
    
    def _determine_risk_level(self, score: float, market_regime: MarketRegime) -> str:
        """Determine risk level for allocation."""
        
        if market_regime == MarketRegime.BEAR_MARKET:
            if score > 0.7:
                return "medium"
            else:
                return "high"
        elif market_regime == MarketRegime.BULL_MARKET:
            if score > 0.8:
                return "low"
            elif score > 0.5:
                return "medium"
            else:
                return "high"
        else:  # Volatile, consolidating, uncertain
            if score > 0.7:
                return "medium"
            else:
                return "high"
    
    def _calculate_diversification_score(self, allocations: List[StockAllocation]) -> float:
        """Calculate portfolio diversification score."""
        
        if not allocations:
            return 0.0
        
        # Group by industry
        industry_weights = {}
        for allocation in allocations:
            industry = allocation.industry
            industry_weights[industry] = industry_weights.get(industry, 0) + allocation.target_weight
        
        # Calculate diversification metrics
        num_industries = len(industry_weights)
        max_industry_weight = max(industry_weights.values()) if industry_weights else 0
        
        # Ideal would be equal weights across many industries
        ideal_weight = 1.0 / max(1, num_industries)
        concentration_penalty = sum(abs(w - ideal_weight) for w in industry_weights.values()) / 2.0
        
        diversification_score = max(0.0, 1.0 - concentration_penalty)
        
        return diversification_score
    
    def _calculate_portfolio_metrics(
        self, 
        allocations: List[StockAllocation], 
        sentiment_data: Dict,
        market_regime: MarketRegime
    ) -> Tuple[float, float, float]:
        """Calculate expected return, confidence, and risk."""
        
        if not allocations:
            return 0.0, 0.0, 0.0
        
        # Expected return based on sentiment and market regime
        total_expected_return = 0.0
        total_confidence = 0.0
        total_risk = 0.0
        
        regime_multipliers = {
            MarketRegime.BULL_MARKET: 1.2,
            MarketRegime.BEAR_MARKET: 0.7,
            MarketRegime.VOLATILE: 0.9,
            MarketRegime.CONSOLIDATING: 1.0,
            MarketRegime.UNCERTAIN: 0.8
        }
        
        regime_multiplier = regime_multipliers[market_regime]
        
        for allocation in allocations:
            sentiment = sentiment_data.get(allocation.symbol)
            if sentiment:
                # Handle both object and dictionary formats
                if isinstance(sentiment, dict):
                    sentiment_score = sentiment.get('overall_score', 0.0)
                    confidence = sentiment.get('confidence', 0.5)
                else:
                    sentiment_score = sentiment.overall_score
                    confidence = sentiment.confidence
                
                # Expected return based on sentiment score
                position_expected_return = sentiment_score * 0.15 * regime_multiplier  # 15% max expected return
                total_expected_return += position_expected_return * allocation.target_weight
                
                # Confidence weighted by position size
                total_confidence += confidence * allocation.target_weight
                
                # Risk based on sentiment confidence (lower confidence = higher risk)
                position_risk = (1.0 - confidence) * allocation.target_weight
                total_risk += position_risk
        
        return total_expected_return, total_confidence, total_risk
    
    async def _generate_portfolio_rationale(
        self, 
        allocations: List[StockAllocation], 
        market_regime: MarketRegime,
        sentiment_data: Dict
    ) -> str:
        """Generate detailed rationale for portfolio construction."""
        
        # Calculate summary statistics - handle both object and dictionary formats
        num_positions = len(allocations)
        sentiment_scores = []
        for a in allocations:
            sentiment = sentiment_data.get(a.symbol)
            if sentiment:
                if isinstance(sentiment, dict):
                    sentiment_scores.append(sentiment.get('overall_score', 0.0))
                else:
                    sentiment_scores.append(sentiment.overall_score)
        
        avg_sentiment = sum(sentiment_scores) / max(1, len(sentiment_scores)) if sentiment_scores else 0.0
        
        # Industry breakdown
        industries = {}
        for allocation in allocations:
            industry = allocation.industry
            industries[industry] = industries.get(industry, 0) + 1
        
        # Create rationale using LLM
        analysis_data = {
            "portfolio_summary": {
                "num_positions": num_positions,
                "market_regime": market_regime.value,
                "avg_sentiment": avg_sentiment,
                "industries": dict(list(industries.items())[:5])  # Top 5 industries
            },
            "top_holdings": [
                {
                    "symbol": a.symbol,
                    "weight": a.target_weight,
                    "sentiment": (sentiment_data[a.symbol].get('overall_score', 0.0) 
                                if isinstance(sentiment_data[a.symbol], dict) 
                                else sentiment_data[a.symbol].overall_score) 
                               if a.symbol in sentiment_data else 0.0,
                    "industry": a.industry
                }
                for a in allocations[:10]  # Top 10 holdings
            ]
        }
        
        try:
            response = await llm_client.analyze_financial_data(
                agent_name="portfolio_manager",
                analysis_data=analysis_data,
                system_prompt="You are a portfolio manager. Explain this portfolio construction in 2-3 sentences focusing on diversification, sentiment, and market conditions."
            )
            return response.content
        except Exception as e:
            logger.warning(f"Failed to generate LLM rationale: {e}")
            return f"Diversified portfolio of {num_positions} positions across {len(industries)} industries, optimized for {market_regime.value} conditions with average sentiment of {avg_sentiment:.2f}."


# Global instance and convenience function
llm_portfolio_manager = LLMPortfolioManager()

async def construct_llm_portfolio(
    candidate_symbols: List[str], 
    portfolio_value: float, 
    risk_profile: str = "moderate",
    max_positions: int = 35,
    sentiment_data: Optional[Dict] = None
) -> PortfolioRecommendation:
    """Convenience function for LLM portfolio construction."""
    return await llm_portfolio_manager.construct_llm_portfolio(
        candidate_symbols, portfolio_value, risk_profile, max_positions, sentiment_data
    )