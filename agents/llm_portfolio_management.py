#!/usr/bin/env python3
"""
LLM-Enhanced Portfolio Management System

This refactored system uses LLM-based agents for decision making while leveraging
quantitative analysis as supporting data. Each agent uses specialized prompts and
has final decision-making authority enhanced by AI reasoning.
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

# Import our enhanced systems
from agents.market_analysis import (
    multi_source_data, comprehensive_analyst, enhanced_market_analysis_factory,
    DataSource, AnalysisResult, SignalStrength
)
# H2O prediction agent removed - using alternative analysis
from tools.alpaca_client import alpaca_client
from tools.llm_client import llm_client, LLMResponse
from prompts.system_prompts import get_system_prompt, get_agent_weights, PROMPTS
from config.settings import settings

# Import original classes for compatibility
from agents.portfolio_management import (
    MarketRegime, AssetClass, RiskProfile, PortfolioAllocation, 
    PortfolioRecommendation
)

logger = logging.getLogger(__name__)

@dataclass
class LLMAgentDecision:
    """LLM agent decision with supporting data."""
    agent_name: str
    decision: Dict[str, Any]
    confidence: float
    reasoning: str
    supporting_data: Dict[str, Any]
    llm_response: LLMResponse
    timestamp: datetime

class LLMMarketRegimeAnalyst:
    """LLM-enhanced Market Regime Analyst."""
    
    def __init__(self):
        self.name = "LLM Market Regime Analyst"
        self.system_prompt = get_system_prompt("market_regime")
        
    async def analyze_market_regime(self, benchmark_symbols: List[str] = None) -> LLMAgentDecision:
        """Analyze market regime using LLM with quantitative support."""
        try:
            if not benchmark_symbols:
                benchmark_symbols = ["SPY", "QQQ", "IWM"]
            
            logger.info("🧠 LLM Market Regime Analysis starting...")
            
            # Step 1: Gather quantitative data
            quantitative_data = await self._gather_quantitative_data(benchmark_symbols)
            
            # Step 2: Prepare LLM analysis request
            analysis_data = {
                "symbols": benchmark_symbols,
                "quantitative_analysis": quantitative_data,
                "market_data": quantitative_data.get("market_metrics", {}),
                "instructions": """
                Analyze the provided market data and determine the current market regime.
                
                Required outputs:
                1. Primary market regime (bull_market, bear_market, sideways_market, high_volatility, low_volatility, crisis_mode)
                2. Confidence level (0-100%)
                3. Key supporting evidence (3-5 bullet points)
                4. Regime transition risks and timeline
                5. Portfolio positioning implications
                
                Format your response as a structured analysis with clear regime classification.
                """
            }
            
            # Step 3: Get LLM analysis
            llm_response = await llm_client.analyze_financial_data(
                agent_name=self.name,
                analysis_data=analysis_data,
                system_prompt=self.system_prompt.prompt,
                temperature=self.system_prompt.temperature
            )
            
            # Step 4: Parse LLM response and extract decision
            regime_decision = self._parse_regime_response(llm_response.content)
            
            decision = LLMAgentDecision(
                agent_name=self.name,
                decision=regime_decision,
                confidence=llm_response.confidence,
                reasoning=llm_response.content,
                supporting_data=quantitative_data,
                llm_response=llm_response,
                timestamp=datetime.now()
            )
            
            logger.info(f"📊 Market regime: {regime_decision.get('regime', 'unknown')} "
                       f"({regime_decision.get('confidence', 0):.1f}% confidence)")
            
            return decision
            
        except Exception as e:
            logger.error(f"LLM Market Regime Analysis failed: {e}")
            # Return fallback decision
            return self._create_fallback_decision(str(e))
    
    async def _gather_quantitative_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Gather quantitative data to support LLM analysis."""
        data = {
            "market_metrics": {},
            "technical_indicators": {},
            "volatility_analysis": {},
            "breadth_indicators": {}
        }
        
        try:
            for symbol in symbols:
                try:
                    # Get market data
                    market_data = await multi_source_data.get_stock_data_multi_source(
                        symbol, "180d"
                    )
                    
                    if market_data.confidence > 0.5 and not market_data.value.empty:
                        df = market_data.value
                        current_price = df['Close'].iloc[-1]
                        
                        # Calculate key metrics
                        returns = df['Close'].pct_change().dropna()
                        volatility = returns.std() * np.sqrt(252)
                        
                        # Moving averages
                        sma_20 = df['Close'].rolling(20).mean().iloc[-1]
                        sma_50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
                        
                        # Performance metrics
                        perf_1m = (current_price / df['Close'].iloc[-21] - 1) if len(df) >= 21 else 0
                        perf_3m = (current_price / df['Close'].iloc[-63] - 1) if len(df) >= 63 else 0
                        perf_6m = (current_price / df['Close'].iloc[-126] - 1) if len(df) >= 126 else 0
                        
                        data["market_metrics"][symbol] = {
                            "current_price": current_price,
                            "volatility_annualized": volatility,
                            "sma_20": sma_20,
                            "sma_50": sma_50,
                            "price_vs_sma20": (current_price / sma_20 - 1),
                            "price_vs_sma50": (current_price / sma_50 - 1),
                            "performance_1m": perf_1m,
                            "performance_3m": perf_3m,
                            "performance_6m": perf_6m
                        }
                        
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
                    
        except Exception as e:
            logger.warning(f"Quantitative data gathering failed: {e}")
        
        return data
    
    def _parse_regime_response(self, llm_content: str) -> Dict[str, Any]:
        """Parse LLM response to extract regime decision."""
        # Simple parsing - in production, could use more sophisticated NLP
        regime_mapping = {
            "bull": MarketRegime.BULL_MARKET,
            "bear": MarketRegime.BEAR_MARKET,
            "sideways": MarketRegime.SIDEWAYS_MARKET,
            "high_volatility": MarketRegime.HIGH_VOLATILITY,
            "low_volatility": MarketRegime.LOW_VOLATILITY,
            "crisis": MarketRegime.CRISIS_MODE
        }
        
        # Extract regime from content
        content_lower = llm_content.lower()
        detected_regime = MarketRegime.SIDEWAYS_MARKET  # Default
        
        for keyword, regime in regime_mapping.items():
            if keyword in content_lower:
                detected_regime = regime
                break
        
        # Extract confidence (look for percentage)
        import re
        confidence_match = re.search(r'(\d+)%', llm_content)
        confidence = float(confidence_match.group(1)) / 100 if confidence_match else 0.7
        
        return {
            "regime": detected_regime,
            "confidence": confidence,
            "raw_analysis": llm_content
        }
    
    def _create_fallback_decision(self, error: str) -> LLMAgentDecision:
        """Create fallback decision when LLM fails."""
        return LLMAgentDecision(
            agent_name=self.name,
            decision={
                "regime": MarketRegime.SIDEWAYS_MARKET,
                "confidence": 0.5,
                "raw_analysis": f"Fallback analysis due to error: {error}"
            },
            confidence=0.5,
            reasoning=f"LLM analysis failed, using fallback: {error}",
            supporting_data={},
            llm_response=None,
            timestamp=datetime.now()
        )

class LLMSectorRotationAgent:
    """LLM-enhanced Sector Rotation Agent."""
    
    def __init__(self):
        self.name = "LLM Sector Rotation Agent"
        self.system_prompt = get_system_prompt("sector_rotation")
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
    
    async def analyze_sector_rotation(self, market_regime: MarketRegime) -> LLMAgentDecision:
        """Analyze sector rotation using LLM with quantitative support."""
        try:
            logger.info("🧠 LLM Sector Rotation Analysis starting...")
            
            # Step 1: Gather sector performance data
            sector_data = await self._gather_sector_data()
            
            # Step 2: Prepare LLM analysis
            analysis_data = {
                "market_regime": market_regime.value,
                "quantitative_analysis": sector_data,
                "instructions": f"""
                Based on the current market regime ({market_regime.value}) and sector performance data,
                provide sector rotation recommendations.
                
                Required outputs:
                1. Sector rankings (Overweight/Neutral/Underweight) for each major sector
                2. Confidence level for each sector recommendation
                3. Key catalysts and risks for top 3 sectors
                4. Recommended sector allocation percentages
                5. Timeline for sector rotation themes (short/medium/long-term)
                
                Consider the current market regime and how it impacts sector performance.
                Provide specific, actionable recommendations.
                """
            }
            
            # Step 3: Get LLM analysis
            llm_response = await llm_client.analyze_financial_data(
                agent_name=self.name,
                analysis_data=analysis_data,
                system_prompt=self.system_prompt.prompt,
                temperature=self.system_prompt.temperature
            )
            
            # Step 4: Parse response
            sector_decision = self._parse_sector_response(llm_response.content)
            
            decision = LLMAgentDecision(
                agent_name=self.name,
                decision=sector_decision,
                confidence=llm_response.confidence,
                reasoning=llm_response.content,
                supporting_data=sector_data,
                llm_response=llm_response,
                timestamp=datetime.now()
            )
            
            logger.info(f"📊 Sector analysis complete: {len(sector_decision.get('sector_scores', {}))} sectors analyzed")
            
            return decision
            
        except Exception as e:
            logger.error(f"LLM Sector Rotation Analysis failed: {e}")
            return self._create_fallback_sector_decision(str(e))
    
    async def _gather_sector_data(self) -> Dict[str, Any]:
        """Gather sector performance data."""
        sector_data = {"sector_performance": {}, "relative_strength": {}}
        
        try:
            for asset_class, etf_symbol in self.sector_etfs.items():
                try:
                    # Get sector ETF analysis
                    analysis = await comprehensive_analyst.comprehensive_analysis(etf_symbol)
                    
                    if analysis:
                        sector_data["sector_performance"][asset_class.value] = {
                            "action": analysis.action,
                            "confidence": analysis.confidence,
                            "indicators": analysis.indicators or {},
                            "strength": analysis.strength.value if analysis.strength else "MODERATE"
                        }
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze sector {asset_class.value}: {e}")
                    
        except Exception as e:
            logger.warning(f"Sector data gathering failed: {e}")
            
        return sector_data
    
    def _parse_sector_response(self, llm_content: str) -> Dict[str, Any]:
        """Parse LLM sector analysis response."""
        # Simple parsing - could be enhanced with more sophisticated extraction
        sector_scores = {}
        
        # Default neutral scores for all sectors
        for asset_class in self.sector_etfs.keys():
            sector_scores[asset_class] = 0.5  # Neutral
        
        # Try to extract specific recommendations from content
        content_lower = llm_content.lower()
        
        # Look for overweight/underweight mentions
        if "technology" in content_lower and "overweight" in content_lower:
            sector_scores[AssetClass.TECHNOLOGY] = 0.8
        if "healthcare" in content_lower and "overweight" in content_lower:
            sector_scores[AssetClass.HEALTHCARE] = 0.8
            
        # Add more parsing logic as needed
        
        return {
            "sector_scores": sector_scores,
            "raw_analysis": llm_content
        }
    
    def _create_fallback_sector_decision(self, error: str) -> LLMAgentDecision:
        """Create fallback sector decision."""
        return LLMAgentDecision(
            agent_name=self.name,
            decision={
                "sector_scores": {asset_class: 0.5 for asset_class in self.sector_etfs.keys()},
                "raw_analysis": f"Fallback analysis due to error: {error}"
            },
            confidence=0.5,
            reasoning=f"LLM analysis failed, using fallback: {error}",
            supporting_data={},
            llm_response=None,
            timestamp=datetime.now()
        )

class LLMPortfolioConstructionEngine:
    """Main LLM-enhanced portfolio construction engine."""
    
    def __init__(self):
        self.name = "LLM Portfolio Construction Engine"
        self.system_prompt = get_system_prompt("portfolio_construction")
        
        # Initialize LLM agents
        self.regime_analyst = LLMMarketRegimeAnalyst()
        self.sector_agent = LLMSectorRotationAgent()
        # Add other agents as they are refactored
        
    async def construct_portfolio(
        self,
        candidate_symbols: List[str],
        portfolio_value: float,
        risk_profile: RiskProfile = RiskProfile.MODERATE,
        max_positions: int = 10
    ) -> PortfolioRecommendation:
        """Construct portfolio using LLM-enhanced multi-agent approach."""
        
        try:
            logger.info(f"🚀 LLM Portfolio Construction starting...")
            logger.info(f"Candidates: {len(candidate_symbols)}, Value: ${portfolio_value:,.2f}")
            logger.info(f"Risk Profile: {risk_profile.value}, Max Positions: {max_positions}")
            
            # Step 1: Market Regime Analysis
            regime_decision = await self.regime_analyst.analyze_market_regime()
            market_regime = regime_decision.decision.get("regime", MarketRegime.SIDEWAYS_MARKET)
            
            # Step 2: Sector Rotation Analysis
            sector_decision = await self.sector_agent.analyze_sector_rotation(market_regime)
            
            # Step 3: Gather all quantitative analysis for symbols
            symbol_analysis = await self._analyze_candidate_symbols(candidate_symbols)
            
            # Step 4: LLM Portfolio Construction Decision
            portfolio_decision = await self._get_llm_portfolio_decision(
                candidate_symbols, portfolio_value, risk_profile, max_positions,
                regime_decision, sector_decision, symbol_analysis
            )
            
            # Step 5: Convert LLM decision to portfolio recommendation
            recommendation = await self._create_portfolio_recommendation(
                portfolio_decision, portfolio_value, market_regime
            )
            
            logger.info(f"✅ Portfolio construction complete: {len(recommendation.allocations)} positions")
            logger.info(f"Expected return: {recommendation.expected_return:.1%}")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"LLM Portfolio Construction failed: {e}")
            return self._create_fallback_recommendation(portfolio_value)
    
    async def _analyze_candidate_symbols(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze all candidate symbols with quantitative methods."""
        analysis = {"symbol_data": {}, "market_overview": {}}
        
        try:
            for symbol in symbols:
                try:
                    # Get comprehensive analysis
                    result = await comprehensive_analyst.comprehensive_analysis(symbol)
                    
                    if result:
                        analysis["symbol_data"][symbol] = {
                            "action": result.action,
                            "confidence": result.confidence,
                            "indicators": result.indicators or {},
                            "reasoning": result.reasoning or "",
                            "strength": result.strength.value if result.strength else "MODERATE"
                        }
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze {symbol}: {e}")
                    
        except Exception as e:
            logger.warning(f"Symbol analysis failed: {e}")
            
        return analysis
    
    async def _get_llm_portfolio_decision(
        self,
        symbols: List[str],
        portfolio_value: float,
        risk_profile: RiskProfile,
        max_positions: int,
        regime_decision: LLMAgentDecision,
        sector_decision: LLMAgentDecision,
        symbol_analysis: Dict[str, Any]
    ) -> LLMAgentDecision:
        """Get final portfolio construction decision from LLM."""
        
        try:
            # Prepare comprehensive analysis data
            analysis_data = {
                "symbols": symbols,
                "portfolio_value": portfolio_value,
                "risk_profile": risk_profile.value,
                "max_positions": max_positions,
                "market_regime": regime_decision.decision,
                "sector_analysis": sector_decision.decision,
                "symbol_analysis": symbol_analysis,
                "instructions": f"""
                Based on all the provided analysis, construct an optimal portfolio with the following requirements:
                
                1. Select the best {max_positions} positions from the candidate symbols
                2. Assign specific portfolio weights (percentages) for each selected position
                3. Ensure total weights sum to no more than 95% (maintain 5%+ cash)
                4. Consider the market regime: {regime_decision.decision.get('regime', 'unknown')}
                5. Apply sector rotation insights from the sector analysis
                6. Match the {risk_profile.value} risk profile
                7. Provide clear reasoning for each position selection and weight
                8. Estimate expected portfolio return and risk metrics
                
                Format your response with:
                - Selected positions with specific weights
                - Reasoning for each position
                - Portfolio-level metrics and expectations
                - Risk warnings and monitoring points
                
                Be specific with numbers and percentages. This is a real portfolio construction decision.
                """
            }
            
            # Get LLM decision
            llm_response = await llm_client.analyze_financial_data(
                agent_name=self.name,
                analysis_data=analysis_data,
                system_prompt=self.system_prompt.prompt,
                temperature=self.system_prompt.temperature
            )
            
            # Parse the portfolio decision
            portfolio_decision = self._parse_portfolio_response(llm_response.content, symbols)
            
            return LLMAgentDecision(
                agent_name=self.name,
                decision=portfolio_decision,
                confidence=llm_response.confidence,
                reasoning=llm_response.content,
                supporting_data={
                    "regime_analysis": regime_decision.decision,
                    "sector_analysis": sector_decision.decision,
                    "symbol_analysis": symbol_analysis
                },
                llm_response=llm_response,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"LLM portfolio decision failed: {e}")
            return self._create_fallback_portfolio_decision(symbols, str(e))
    
    def _parse_portfolio_response(self, llm_content: str, symbols: List[str]) -> Dict[str, Any]:
        """Parse LLM portfolio construction response."""
        import re
        
        # Extract positions and weights from LLM response
        selected_positions = {}
        
        # Look for percentage patterns in the content
        for symbol in symbols:
            # Look for patterns like "AAPL: 12%" or "AAPL - 12.5%"
            pattern = rf"{symbol}[:\-\s]*(\d+\.?\d*)%"
            match = re.search(pattern, llm_content, re.IGNORECASE)
            if match:
                weight = float(match.group(1)) / 100
                if weight > 0.01:  # Only include meaningful weights
                    selected_positions[symbol] = weight
        
        # If no positions found, create default allocation
        if not selected_positions and symbols:
            # Default equal weight for top symbols
            top_symbols = symbols[:min(8, len(symbols))]
            equal_weight = 0.8 / len(top_symbols)  # 80% invested, 20% cash
            selected_positions = {symbol: equal_weight for symbol in top_symbols}
        
        # Normalize weights to ensure they don't exceed 95%
        total_weight = sum(selected_positions.values())
        if total_weight > 0.95:
            scale_factor = 0.95 / total_weight
            selected_positions = {symbol: weight * scale_factor 
                                for symbol, weight in selected_positions.items()}
        
        return {
            "selected_positions": selected_positions,
            "total_weight": sum(selected_positions.values()),
            "cash_allocation": 1.0 - sum(selected_positions.values()),
            "raw_analysis": llm_content
        }
    
    def _create_fallback_portfolio_decision(self, symbols: List[str], error: str) -> LLMAgentDecision:
        """Create fallback portfolio decision."""
        # Simple equal weight allocation
        top_symbols = symbols[:8]
        equal_weight = 0.10  # 10% each for 8 positions = 80% invested
        
        fallback_positions = {symbol: equal_weight for symbol in top_symbols}
        
        return LLMAgentDecision(
            agent_name=self.name,
            decision={
                "selected_positions": fallback_positions,
                "total_weight": sum(fallback_positions.values()),
                "cash_allocation": 1.0 - sum(fallback_positions.values()),
                "raw_analysis": f"Fallback equal-weight allocation due to error: {error}"
            },
            confidence=0.5,
            reasoning=f"LLM analysis failed, using fallback equal-weight: {error}",
            supporting_data={},
            llm_response=None,
            timestamp=datetime.now()
        )
    
    async def _create_portfolio_recommendation(
        self,
        portfolio_decision: LLMAgentDecision,
        portfolio_value: float,
        market_regime: MarketRegime
    ) -> PortfolioRecommendation:
        """Create final portfolio recommendation from LLM decision."""
        
        decision_data = portfolio_decision.decision
        selected_positions = decision_data.get("selected_positions", {})
        
        # Create portfolio allocations
        allocations = []
        for symbol, weight in selected_positions.items():
            try:
                allocation = PortfolioAllocation(
                    symbol=symbol,
                    asset_class=self._classify_asset(symbol),
                    target_weight=weight,
                    current_weight=0.0,
                    recommended_action="buy" if weight > 0.01 else "hold",
                    confidence=portfolio_decision.confidence,
                    reasoning=f"LLM Portfolio Construction: {weight:.1%} allocation",
                    risk_score=0.5,  # Default risk score
                    expected_return=0.08,  # Default expected return
                    correlation_score=0.5,
                    agent_source="llm_portfolio_construction",
                    priority=len(allocations) + 1,
                    metadata={
                        "llm_reasoning": portfolio_decision.reasoning,
                        "position_value": portfolio_value * weight
                    }
                )
                allocations.append(allocation)
            except Exception as e:
                logger.warning(f"Failed to create allocation for {symbol}: {e}")
        
        # Calculate portfolio metrics
        total_weight = sum(alloc.target_weight for alloc in allocations)
        cash_allocation = max(0.0, 1.0 - total_weight)
        
        # Create recommendation
        return PortfolioRecommendation(
            allocations=allocations,
            target_risk_level=0.5,  # Default
            expected_return=0.08,   # Default 8%
            expected_volatility=0.15,  # Default 15%
            diversification_score=0.7,  # Assume good diversification
            market_regime=market_regime,
            confidence=portfolio_decision.confidence,
            cash_allocation=cash_allocation,
            rebalance_urgency=0.3,  # Moderate urgency
            reasoning=portfolio_decision.reasoning,
            agents_consensus={"llm_construction": portfolio_decision.confidence},
            timestamp=datetime.now()
        )
    
    def _classify_asset(self, symbol: str) -> AssetClass:
        """Simple asset classification."""
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        if symbol in tech_symbols:
            return AssetClass.TECHNOLOGY
        return AssetClass.LARGE_CAP_GROWTH  # Default
    
    def _create_fallback_recommendation(self, portfolio_value: float) -> PortfolioRecommendation:
        """Create fallback recommendation when everything fails."""
        return PortfolioRecommendation(
            allocations=[],
            target_risk_level=0.5,
            expected_return=0.08,
            expected_volatility=0.15,
            diversification_score=0.0,
            market_regime=MarketRegime.SIDEWAYS_MARKET,
            confidence=0.0,
            cash_allocation=1.0,
            rebalance_urgency=0.0,
            reasoning="LLM portfolio construction failed - maintaining cash position",
            agents_consensus={},
            timestamp=datetime.now()
        )

# Global LLM portfolio engine
llm_portfolio_engine = LLMPortfolioConstructionEngine()

async def construct_llm_portfolio(
    candidate_symbols: List[str],
    portfolio_value: float,
    risk_profile: str = "moderate",
    max_positions: int = 10
) -> PortfolioRecommendation:
    """
    Main function to construct portfolio using LLM-enhanced agents.
    
    Args:
        candidate_symbols: List of stock symbols to consider
        portfolio_value: Total portfolio value in USD
        risk_profile: "conservative", "moderate", "aggressive", or "tactical"
        max_positions: Maximum number of positions
        
    Returns:
        PortfolioRecommendation with LLM-enhanced analysis
    """
    try:
        risk_profile_enum = RiskProfile(risk_profile.lower())
        
        recommendation = await llm_portfolio_engine.construct_portfolio(
            candidate_symbols=candidate_symbols,
            portfolio_value=portfolio_value,
            risk_profile=risk_profile_enum,
            max_positions=max_positions
        )
        
        return recommendation
        
    except Exception as e:
        logger.error(f"LLM portfolio construction failed: {e}")
        raise

if __name__ == "__main__":
    # Test the LLM portfolio construction
    async def test_llm_portfolio():
        test_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "NVDA", "META", "JPM", "JNJ", "UNH"
        ]
        
        print("🚀 Testing LLM Portfolio Construction")
        print("=" * 60)
        
        recommendation = await construct_llm_portfolio(
            candidate_symbols=test_symbols,
            portfolio_value=100000.0,
            risk_profile="aggressive",
            max_positions=8
        )
        
        print(f"\n🎯 LLM PORTFOLIO RECOMMENDATION")
        print(f"Market Regime: {recommendation.market_regime.value}")
        print(f"Expected Return: {recommendation.expected_return:.1%}")
        print(f"Confidence: {recommendation.confidence:.1%}")
        print(f"Cash Allocation: {recommendation.cash_allocation:.1%}")
        
        print(f"\n📊 PORTFOLIO POSITIONS ({len(recommendation.allocations)} positions)")
        for alloc in recommendation.allocations:
            print(f"{alloc.symbol:>6}: {alloc.target_weight:>6.1%} | "
                  f"Confidence: {alloc.confidence:>5.1%}")
        
        print(f"\n💭 LLM REASONING:")
        print(recommendation.reasoning[:500] + "..." if len(recommendation.reasoning) > 500 
              else recommendation.reasoning)
    
    asyncio.run(test_llm_portfolio())