#!/usr/bin/env python3
"""
LLM-Enhanced Specialized Trading Agents

This module contains the remaining specialized agents refactored to use LLM decision-making
with quantitative analysis as supporting data.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import re

# Import systems and utilities
from agents.market_analysis import (
    multi_source_data, comprehensive_analyst, enhanced_market_analysis_factory
)
# H2O prediction agent removed - using fallback analysis
from tools.llm_client import llm_client, LLMResponse
from prompts.system_prompts import get_system_prompt
from agents.llm_portfolio_management import LLMAgentDecision

# Import original classes for compatibility
from agents.portfolio_management import MarketRegime, AssetClass, RiskProfile

logger = logging.getLogger(__name__)

class LLMRiskParityAgent:
    """LLM-enhanced Risk Parity and Risk Management Agent."""
    
    def __init__(self):
        self.name = "LLM Risk Parity Agent"
        self.system_prompt = get_system_prompt("risk_parity")
        
    async def analyze_risk_management(
        self,
        symbols: List[str],
        current_portfolio: Dict[str, float] = None
    ) -> LLMAgentDecision:
        """Analyze portfolio risk and provide risk parity recommendations."""
        
        try:
            logger.info("🧠 LLM Risk Management Analysis starting...")
            
            # Step 1: Gather risk data
            risk_data = await self._gather_risk_data(symbols, current_portfolio)
            
            # Step 2: Calculate quantitative risk metrics
            risk_metrics = await self._calculate_risk_metrics(symbols)
            
            # Step 3: Prepare LLM analysis
            analysis_data = {
                "symbols": symbols,
                "current_portfolio": current_portfolio or {},
                "risk_data": risk_data,
                "risk_metrics": risk_metrics,
                "instructions": """
                Analyze the portfolio risk characteristics and provide risk management recommendations.
                
                Required outputs:
                1. Risk-adjusted position sizes for each asset (as percentages)
                2. Portfolio risk assessment (concentration, correlation, volatility risks)
                3. Risk warnings and red flags
                4. Hedging recommendations if needed
                5. Maximum position size limits for each asset
                6. Overall portfolio risk score (1-10 scale)
                
                Focus on:
                - Diversification and correlation analysis
                - Volatility-based position sizing
                - Concentration risk management
                - Downside protection strategies
                
                Provide specific position size recommendations with clear risk rationale.
                """
            }
            
            # Step 4: Get LLM analysis
            llm_response = await llm_client.analyze_financial_data(
                agent_name=self.name,
                analysis_data=analysis_data,
                system_prompt=self.system_prompt.prompt,
                temperature=self.system_prompt.temperature
            )
            
            # Step 5: Parse risk recommendations
            risk_decision = self._parse_risk_response(llm_response.content, symbols)
            
            decision = LLMAgentDecision(
                agent_name=self.name,
                decision=risk_decision,
                confidence=llm_response.confidence,
                reasoning=llm_response.content,
                supporting_data={
                    "risk_data": risk_data,
                    "risk_metrics": risk_metrics
                },
                llm_response=llm_response,
                timestamp=datetime.now()
            )
            
            logger.info(f"📊 Risk analysis complete: {len(risk_decision.get('position_sizes', {}))} positions analyzed")
            
            return decision
            
        except Exception as e:
            logger.error(f"LLM Risk Analysis failed: {e}")
            return self._create_fallback_risk_decision(symbols, str(e))
    
    async def _gather_risk_data(
        self,
        symbols: List[str],
        current_portfolio: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Gather risk-related data for analysis."""
        
        risk_data = {
            "volatilities": {},
            "correlations": {},
            "returns_data": {},
            "portfolio_metrics": {}
        }
        
        try:
            returns_series = {}
            
            # Get historical data for volatility and correlation analysis
            for symbol in symbols:
                try:
                    data_result = await multi_source_data.get_stock_data_multi_source(
                        symbol, "252d"  # 1 year of data
                    )
                    
                    if data_result.confidence > 0.5 and not data_result.value.empty:
                        df = data_result.value
                        returns = df['Close'].pct_change().dropna()
                        
                        if len(returns) > 50:  # Need sufficient data
                            returns_series[symbol] = returns
                            
                            # Calculate volatility
                            volatility = returns.std() * np.sqrt(252)
                            risk_data["volatilities"][symbol] = volatility
                            
                            # Store returns for correlation calculation
                            risk_data["returns_data"][symbol] = returns.tail(126).tolist()  # Last 6 months
                            
                except Exception as e:
                    logger.warning(f"Failed to get risk data for {symbol}: {e}")
            
            # Calculate correlation matrix
            if len(returns_series) >= 2:
                # Align returns data
                returns_df = pd.DataFrame(returns_series).dropna()
                if not returns_df.empty:
                    corr_matrix = returns_df.corr()
                    
                    # Store correlation data
                    for symbol1 in symbols:
                        if symbol1 in corr_matrix.columns:
                            risk_data["correlations"][symbol1] = {}
                            for symbol2 in symbols:
                                if symbol2 in corr_matrix.columns and symbol1 != symbol2:
                                    corr = corr_matrix.loc[symbol1, symbol2]
                                    risk_data["correlations"][symbol1][symbol2] = corr
            
            # Calculate portfolio-level metrics if current portfolio provided
            if current_portfolio and returns_series:
                portfolio_return = 0.0
                portfolio_variance = 0.0
                
                for symbol1, weight1 in current_portfolio.items():
                    if symbol1 in returns_series:
                        portfolio_return += weight1 * returns_series[symbol1].mean() * 252
                        
                        for symbol2, weight2 in current_portfolio.items():
                            if symbol1 in returns_series and symbol2 in returns_series:
                                if symbol1 == symbol2:
                                    variance_contribution = weight1 * weight2 * (returns_series[symbol1].std() ** 2) * 252
                                else:
                                    corr = risk_data["correlations"].get(symbol1, {}).get(symbol2, 0.0)
                                    vol1 = risk_data["volatilities"].get(symbol1, 0.2)
                                    vol2 = risk_data["volatilities"].get(symbol2, 0.2)
                                    variance_contribution = weight1 * weight2 * vol1 * vol2 * corr * 252
                                
                                portfolio_variance += variance_contribution
                
                risk_data["portfolio_metrics"] = {
                    "expected_return": portfolio_return,
                    "portfolio_volatility": np.sqrt(max(0, portfolio_variance)),
                    "sharpe_estimate": portfolio_return / np.sqrt(max(0.0001, portfolio_variance))
                }
                
        except Exception as e:
            logger.warning(f"Risk data gathering failed: {e}")
            
        return risk_data
    
    async def _calculate_risk_metrics(self, symbols: List[str]) -> Dict[str, Any]:
        """Calculate additional risk metrics."""
        
        risk_metrics = {
            "var_estimates": {},
            "max_drawdown_estimates": {},
            "risk_scores": {}
        }
        
        try:
            for symbol in symbols:
                try:
                    # Get comprehensive analysis for risk assessment
                    analysis = await comprehensive_analyst.comprehensive_analysis(symbol)
                    
                    if analysis and analysis.indicators:
                        # Extract risk-relevant indicators
                        rsi = analysis.indicators.get('rsi', 50)
                        current_price = analysis.indicators.get('current_price', 0)
                        
                        # Simple risk scoring based on technical indicators
                        risk_score = 0.5  # Base risk score
                        
                        # Adjust risk based on RSI (extreme values = higher risk)
                        if rsi > 80:
                            risk_score += 0.2  # Overbought = higher risk
                        elif rsi < 20:
                            risk_score += 0.3  # Oversold = higher risk
                        elif 40 <= rsi <= 60:
                            risk_score -= 0.1  # Neutral = lower risk
                        
                        # Adjust based on analysis confidence
                        if analysis.action == "sell":
                            risk_score += analysis.confidence * 0.3
                        elif analysis.action == "buy":
                            risk_score -= analysis.confidence * 0.1
                        
                        risk_metrics["risk_scores"][symbol] = max(0.1, min(1.0, risk_score))
                        
                        # Estimate VaR (5% daily VaR)
                        if symbol in self._volatilities:
                            daily_vol = self._volatilities[symbol] / np.sqrt(252)
                            var_5pct = current_price * daily_vol * 1.645  # 5% VaR
                            risk_metrics["var_estimates"][symbol] = var_5pct
                            
                except Exception as e:
                    logger.warning(f"Risk metrics calculation failed for {symbol}: {e}")
                    
        except Exception as e:
            logger.warning(f"Risk metrics calculation failed: {e}")
            
        return risk_metrics
    
    def _parse_risk_response(self, llm_content: str, symbols: List[str]) -> Dict[str, Any]:
        """Parse LLM risk management response."""
        
        # Extract position sizes from LLM response
        position_sizes = {}
        
        # Look for percentage patterns for each symbol
        for symbol in symbols:
            pattern = rf"{symbol}[:\-\s]*(\d+\.?\d*)%"
            match = re.search(pattern, llm_content, re.IGNORECASE)
            if match:
                size = float(match.group(1)) / 100
                position_sizes[symbol] = min(0.15, max(0.01, size))  # Limit between 1-15%
        
        # If no specific sizes found, use equal weighting with risk adjustment
        if not position_sizes and symbols:
            base_size = 0.8 / len(symbols)  # Equal weight baseline
            position_sizes = {symbol: base_size for symbol in symbols}
        
        # Extract risk score (look for X/10 or X out of 10 patterns)
        risk_score_match = re.search(r'(\d+\.?\d*)[/\s](?:out of\s)?10', llm_content, re.IGNORECASE)
        risk_score = float(risk_score_match.group(1)) / 10 if risk_score_match else 0.5
        
        return {
            "position_sizes": position_sizes,
            "portfolio_risk_score": risk_score,
            "risk_warnings": self._extract_risk_warnings(llm_content),
            "raw_analysis": llm_content
        }
    
    def _extract_risk_warnings(self, content: str) -> List[str]:
        """Extract risk warnings from LLM content."""
        warnings = []
        
        # Look for common risk warning patterns
        warning_patterns = [
            r"warning[:\-\s]+([^\.]+)",
            r"risk[:\-\s]+([^\.]+)",
            r"caution[:\-\s]+([^\.]+)",
            r"concern[:\-\s]+([^\.]+)"
        ]
        
        for pattern in warning_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            warnings.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        return warnings[:5]  # Return top 5 warnings
    
    def _create_fallback_risk_decision(self, symbols: List[str], error: str) -> LLMAgentDecision:
        """Create fallback risk decision."""
        
        # Simple equal weight with conservative sizing
        equal_weight = min(0.12, 0.8 / len(symbols)) if symbols else 0.1
        fallback_sizes = {symbol: equal_weight for symbol in symbols}
        
        return LLMAgentDecision(
            agent_name=self.name,
            decision={
                "position_sizes": fallback_sizes,
                "portfolio_risk_score": 0.6,  # Moderate risk
                "risk_warnings": [f"Risk analysis failed: {error}"],
                "raw_analysis": f"Fallback equal-weight risk allocation: {error}"
            },
            confidence=0.5,
            reasoning=f"LLM risk analysis failed, using conservative fallback: {error}",
            supporting_data={},
            llm_response=None,
            timestamp=datetime.now()
        )

class LLMMomentumFactorAgent:
    """LLM-enhanced Momentum Factor Agent."""
    
    def __init__(self):
        self.name = "LLM Momentum Factor Agent"
        self.system_prompt = get_system_prompt("momentum")
        
    async def analyze_momentum_signals(self, symbols: List[str]) -> LLMAgentDecision:
        """Analyze momentum using LLM with quantitative support."""
        
        try:
            logger.info("🧠 LLM Momentum Analysis starting...")
            
            # Step 1: Calculate quantitative momentum metrics
            momentum_data = await self._calculate_momentum_metrics(symbols)
            
            # Step 2: Prepare LLM analysis
            analysis_data = {
                "symbols": symbols,
                "momentum_data": momentum_data,
                "instructions": """
                Analyze momentum signals for the provided securities across multiple timeframes.
                
                Required outputs:
                1. Momentum scores for each symbol (0-100 scale)
                2. Timeframe analysis (short, medium, long-term momentum)
                3. Momentum quality assessment (sustainable vs. exhausted)
                4. Reversal risk warnings for overbought/oversold securities
                5. Top momentum picks with conviction levels
                6. Momentum strategy recommendations (continuation vs. mean reversion)
                
                Consider:
                - Multi-timeframe momentum consistency
                - Volume confirmation and momentum quality
                - Relative strength vs. market and peers
                - Risk of momentum reversals and crowding
                
                Provide specific momentum scores and actionable recommendations.
                """
            }
            
            # Step 3: Get LLM analysis
            llm_response = await llm_client.analyze_financial_data(
                agent_name=self.name,
                analysis_data=analysis_data,
                system_prompt=self.system_prompt.prompt,
                temperature=self.system_prompt.temperature
            )
            
            # Step 4: Parse momentum decision
            momentum_decision = self._parse_momentum_response(llm_response.content, symbols)
            
            decision = LLMAgentDecision(
                agent_name=self.name,
                decision=momentum_decision,
                confidence=llm_response.confidence,
                reasoning=llm_response.content,
                supporting_data=momentum_data,
                llm_response=llm_response,
                timestamp=datetime.now()
            )
            
            logger.info(f"📊 Momentum analysis complete: {len(momentum_decision.get('momentum_scores', {}))} symbols scored")
            
            return decision
            
        except Exception as e:
            logger.error(f"LLM Momentum Analysis failed: {e}")
            return self._create_fallback_momentum_decision(symbols, str(e))
    
    async def _calculate_momentum_metrics(self, symbols: List[str]) -> Dict[str, Any]:
        """Calculate quantitative momentum metrics."""
        
        momentum_data = {
            "price_momentum": {},
            "volume_momentum": {},
            "relative_strength": {},
            "momentum_quality": {}
        }
        
        try:
            for symbol in symbols:
                try:
                    # Get historical data
                    data_result = await multi_source_data.get_stock_data_multi_source(
                        symbol, "300d"  # 10+ months for longer momentum
                    )
                    
                    if data_result.confidence > 0.5 and not data_result.value.empty:
                        df = data_result.value
                        current_price = df['Close'].iloc[-1]
                        
                        # Calculate momentum across timeframes
                        momentum_periods = [21, 63, 126, 252]  # 1M, 3M, 6M, 1Y
                        momentum_values = {}
                        
                        for period in momentum_periods:
                            if len(df) > period:
                                past_price = df['Close'].iloc[-period]
                                momentum = (current_price - past_price) / past_price
                                momentum_values[f"{period}d"] = momentum
                        
                        momentum_data["price_momentum"][symbol] = momentum_values
                        
                        # Volume-weighted momentum (if volume available)
                        if 'Volume' in df.columns:
                            recent_volume = df['Volume'].tail(21).mean()
                            older_volume = df['Volume'].tail(63).head(21).mean()
                            volume_trend = (recent_volume / older_volume - 1) if older_volume > 0 else 0
                            momentum_data["volume_momentum"][symbol] = volume_trend
                        
                        # Momentum quality (volatility-adjusted)
                        returns = df['Close'].pct_change().dropna()
                        if len(returns) > 21:
                            volatility = returns.tail(63).std() * np.sqrt(252)
                            momentum_3m = momentum_values.get("63d", 0)
                            quality_score = momentum_3m / max(volatility, 0.01)  # Risk-adjusted momentum
                            momentum_data["momentum_quality"][symbol] = quality_score
                        
                except Exception as e:
                    logger.warning(f"Momentum calculation failed for {symbol}: {e}")
                    
        except Exception as e:
            logger.warning(f"Momentum data calculation failed: {e}")
            
        return momentum_data
    
    def _parse_momentum_response(self, llm_content: str, symbols: List[str]) -> Dict[str, Any]:
        """Parse LLM momentum analysis response."""
        
        momentum_scores = {}
        
        # Extract momentum scores (0-100 scale)
        for symbol in symbols:
            # Look for patterns like "AAPL: 85/100" or "AAPL momentum: 75"
            patterns = [
                rf"{symbol}[:\-\s]*(\d+\.?\d*)[/\s](?:out of\s)?100",
                rf"{symbol}[:\-\s]*momentum[:\-\s]*(\d+\.?\d*)",
                rf"{symbol}[:\-\s]*score[:\-\s]*(\d+\.?\d*)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, llm_content, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    if score <= 100:  # Assume 0-100 scale
                        momentum_scores[symbol] = score / 100
                        break
                    elif score <= 1:  # Assume 0-1 scale
                        momentum_scores[symbol] = score
                        break
        
        # If no specific scores found, assign default based on sentiment
        if not momentum_scores and symbols:
            for symbol in symbols:
                if f"{symbol}" in llm_content and "strong" in llm_content.lower():
                    momentum_scores[symbol] = 0.8
                elif f"{symbol}" in llm_content and ("weak" in llm_content.lower() or "poor" in llm_content.lower()):
                    momentum_scores[symbol] = 0.3
                else:
                    momentum_scores[symbol] = 0.5  # Neutral
        
        # Extract top momentum picks
        top_picks = []
        for symbol, score in sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
            top_picks.append({
                "symbol": symbol,
                "momentum_score": score,
                "conviction": "high" if score > 0.7 else "medium" if score > 0.5 else "low"
            })
        
        return {
            "momentum_scores": momentum_scores,
            "top_momentum_picks": top_picks,
            "reversal_warnings": self._extract_reversal_warnings(llm_content),
            "raw_analysis": llm_content
        }
    
    def _extract_reversal_warnings(self, content: str) -> List[str]:
        """Extract momentum reversal warnings."""
        warnings = []
        
        # Look for reversal-related terms
        reversal_patterns = [
            r"reversal[:\-\s]+([^\.]+)",
            r"overbought[:\-\s]+([^\.]+)",
            r"oversold[:\-\s]+([^\.]+)",
            r"exhausted[:\-\s]+([^\.]+)"
        ]
        
        for pattern in reversal_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            warnings.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        return warnings[:3]
    
    def _create_fallback_momentum_decision(self, symbols: List[str], error: str) -> LLMAgentDecision:
        """Create fallback momentum decision."""
        
        # Neutral momentum scores for all symbols
        neutral_scores = {symbol: 0.5 for symbol in symbols}
        
        return LLMAgentDecision(
            agent_name=self.name,
            decision={
                "momentum_scores": neutral_scores,
                "top_momentum_picks": [],
                "reversal_warnings": [f"Momentum analysis failed: {error}"],
                "raw_analysis": f"Fallback neutral momentum scores: {error}"
            },
            confidence=0.5,
            reasoning=f"LLM momentum analysis failed, using neutral scores: {error}",
            supporting_data={},
            llm_response=None,
            timestamp=datetime.now()
        )

class LLMValueFactorAgent:
    """LLM-enhanced Value Factor Agent."""
    
    def __init__(self):
        self.name = "LLM Value Factor Agent"
        self.system_prompt = get_system_prompt("value")
        
    async def analyze_value_signals(self, symbols: List[str]) -> LLMAgentDecision:
        """Analyze value opportunities using LLM with quantitative support."""
        
        try:
            logger.info("🧠 LLM Value Analysis starting...")
            
            # Step 1: Gather fundamental data
            value_data = await self._gather_value_data(symbols)
            
            # Step 2: Prepare LLM analysis
            analysis_data = {
                "symbols": symbols,
                "value_data": value_data,
                "instructions": """
                Analyze value investment opportunities in the provided securities.
                
                Required outputs:
                1. Value scores for each symbol (0-100 scale)
                2. Quality assessment (financial strength, competitive position)
                3. Value trap warnings and red flags
                4. Catalyst identification for value realization
                5. Intrinsic value estimates vs. current prices
                6. Value investment conviction levels and timelines
                
                Focus on:
                - Traditional valuation metrics (P/E, P/B, EV/EBITDA, FCF yield)
                - Quality factors (ROE, debt levels, profit margins)
                - Competitive moats and business sustainability
                - Management quality and capital allocation
                - Contrarian opportunities with improving fundamentals
                
                Distinguish between quality value and value traps. Provide specific recommendations.
                """
            }
            
            # Step 3: Get LLM analysis
            llm_response = await llm_client.analyze_financial_data(
                agent_name=self.name,
                analysis_data=analysis_data,
                system_prompt=self.system_prompt.prompt,
                temperature=self.system_prompt.temperature
            )
            
            # Step 4: Parse value decision
            value_decision = self._parse_value_response(llm_response.content, symbols)
            
            decision = LLMAgentDecision(
                agent_name=self.name,
                decision=value_decision,
                confidence=llm_response.confidence,
                reasoning=llm_response.content,
                supporting_data=value_data,
                llm_response=llm_response,
                timestamp=datetime.now()
            )
            
            logger.info(f"📊 Value analysis complete: {len(value_decision.get('value_scores', {}))} symbols analyzed")
            
            return decision
            
        except Exception as e:
            logger.error(f"LLM Value Analysis failed: {e}")
            return self._create_fallback_value_decision(symbols, str(e))
    
    async def _gather_value_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Gather fundamental value data."""
        
        value_data = {
            "valuation_metrics": {},
            "quality_metrics": {},
            "fundamental_analysis": {}
        }
        
        try:
            for symbol in symbols:
                try:
                    # Get fundamental analysis
                    fundamental_result = await enhanced_market_analysis_factory['fundamental'].analyze_symbol(symbol)
                    
                    if fundamental_result and fundamental_result.confidence > 0.3:
                        # Store fundamental analysis
                        value_data["fundamental_analysis"][symbol] = {
                            "action": fundamental_result.action,
                            "confidence": fundamental_result.confidence,
                            "reasoning": fundamental_result.reasoning or "",
                            "indicators": fundamental_result.indicators or {}
                        }
                        
                        # Extract valuation metrics if available
                        if fundamental_result.indicators:
                            indicators = fundamental_result.indicators
                            value_data["valuation_metrics"][symbol] = {
                                "pe_ratio": indicators.get("pe_ratio"),
                                "price_to_book": indicators.get("price_to_book"),
                                "market_cap": indicators.get("market_cap"),
                                "enterprise_value": indicators.get("enterprise_value"),
                                "current_price": indicators.get("current_price")
                            }
                        
                except Exception as e:
                    logger.warning(f"Value data gathering failed for {symbol}: {e}")
                    
        except Exception as e:
            logger.warning(f"Value data gathering failed: {e}")
            
        return value_data
    
    def _parse_value_response(self, llm_content: str, symbols: List[str]) -> Dict[str, Any]:
        """Parse LLM value analysis response."""
        
        value_scores = {}
        value_traps = []
        
        # Extract value scores
        for symbol in symbols:
            patterns = [
                rf"{symbol}[:\-\s]*(\d+\.?\d*)[/\s](?:out of\s)?100",
                rf"{symbol}[:\-\s]*value[:\-\s]*score[:\-\s]*(\d+\.?\d*)",
                rf"{symbol}[:\-\s]*score[:\-\s]*(\d+\.?\d*)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, llm_content, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    if score <= 100:
                        value_scores[symbol] = score / 100
                        break
                    elif score <= 1:
                        value_scores[symbol] = score
                        break
        
        # Default scoring based on sentiment if no explicit scores
        if not value_scores and symbols:
            for symbol in symbols:
                symbol_context = self._extract_symbol_context(llm_content, symbol)
                if "attractive" in symbol_context or "undervalued" in symbol_context:
                    value_scores[symbol] = 0.75
                elif "overvalued" in symbol_context or "expensive" in symbol_context:
                    value_scores[symbol] = 0.25
                elif "trap" in symbol_context:
                    value_scores[symbol] = 0.2
                    value_traps.append(symbol)
                else:
                    value_scores[symbol] = 0.5
        
        # Extract value traps
        trap_patterns = [
            r"trap[:\-\s]+([A-Z]{2,5})",
            r"avoid[:\-\s]+([A-Z]{2,5})",
            r"declining[:\-\s]+([A-Z]{2,5})"
        ]
        
        for pattern in trap_patterns:
            matches = re.findall(pattern, llm_content)
            for match in matches:
                if match in symbols and match not in value_traps:
                    value_traps.append(match)
        
        return {
            "value_scores": value_scores,
            "value_traps": value_traps,
            "quality_picks": [symbol for symbol, score in value_scores.items() if score > 0.7][:3],
            "raw_analysis": llm_content
        }
    
    def _extract_symbol_context(self, content: str, symbol: str) -> str:
        """Extract context around a specific symbol mention."""
        import re
        
        # Find sentences containing the symbol
        sentences = re.split(r'[.!?]+', content)
        symbol_context = []
        
        for sentence in sentences:
            if symbol in sentence:
                symbol_context.append(sentence.strip())
        
        return " ".join(symbol_context).lower()
    
    def _create_fallback_value_decision(self, symbols: List[str], error: str) -> LLMAgentDecision:
        """Create fallback value decision."""
        
        neutral_scores = {symbol: 0.5 for symbol in symbols}
        
        return LLMAgentDecision(
            agent_name=self.name,
            decision={
                "value_scores": neutral_scores,
                "value_traps": [],
                "quality_picks": [],
                "raw_analysis": f"Fallback neutral value scores: {error}"
            },
            confidence=0.5,
            reasoning=f"LLM value analysis failed, using neutral scores: {error}",
            supporting_data={},
            llm_response=None,
            timestamp=datetime.now()
        )

class LLMPredictiveAgent:
    """LLM-enhanced ML Predictive Agent."""
    
    def __init__(self):
        self.name = "LLM ML Predictive Agent"
        self.system_prompt = get_system_prompt("ml_predictive")
        
    async def analyze_ml_predictions(self, symbols: List[str]) -> LLMAgentDecision:
        """Analyze ML predictions using LLM interpretation."""
        
        try:
            logger.info("🧠 LLM ML Predictive Analysis starting...")
            
            # Step 1: Gather ML predictions and alternative data
            ml_data = await self._gather_ml_predictions(symbols)
            
            # Step 2: Prepare LLM analysis
            analysis_data = {
                "symbols": symbols,
                "ml_predictions": ml_data,
                "instructions": """
                Interpret machine learning predictions and alternative data signals for trading insights.
                
                Required outputs:
                1. ML prediction scores for each symbol (0-100 scale)
                2. Model confidence assessment and reliability
                3. Pattern recognition insights and anomaly detection
                4. Short-term vs. long-term prediction alignment
                5. Alternative data signal confirmation
                6. Model limitation warnings and uncertainty quantification
                
                Consider:
                - Model accuracy and historical performance
                - Feature importance and signal explanation
                - Regime sensitivity and model stability
                - Alternative data quality and relevance
                - Ensemble prediction consistency
                
                Provide actionable ML-driven insights with appropriate uncertainty bounds.
                """
            }
            
            # Step 3: Get LLM analysis
            llm_response = await llm_client.analyze_financial_data(
                agent_name=self.name,
                analysis_data=analysis_data,
                system_prompt=self.system_prompt.prompt,
                temperature=self.system_prompt.temperature
            )
            
            # Step 4: Parse ML decision
            ml_decision = self._parse_ml_response(llm_response.content, symbols)
            
            decision = LLMAgentDecision(
                agent_name=self.name,
                decision=ml_decision,
                confidence=llm_response.confidence,
                reasoning=llm_response.content,
                supporting_data=ml_data,
                llm_response=llm_response,
                timestamp=datetime.now()
            )
            
            logger.info(f"📊 ML Predictive analysis complete: {len(ml_decision.get('ml_scores', {}))} predictions generated")
            
            return decision
            
        except Exception as e:
            logger.error(f"LLM ML Predictive Analysis failed: {e}")
            return self._create_fallback_ml_decision(symbols, str(e))
    
    async def _gather_ml_predictions(self, symbols: List[str]) -> Dict[str, Any]:
        """Gather ML predictions and alternative data."""
        
        ml_data = {
            "h2o_predictions": {},
            "model_metrics": {},
            "alternative_signals": {}
        }
        
        try:
            # H2O predictions removed - using alternative ML signals only
            logger.info("Using alternative ML signals (H2O removed from system)")
            
            # Get comprehensive analysis as alternative ML signal
            for symbol in symbols:
                try:
                    analysis = await comprehensive_analyst.comprehensive_analysis(symbol)
                    
                    if analysis:
                        ml_data["alternative_signals"][symbol] = {
                            "technical_signal": analysis.action,
                            "confidence": analysis.confidence,
                            "strength": analysis.strength.value if analysis.strength else "MODERATE",
                            "indicators": analysis.indicators or {}
                        }
                        
                except Exception as e:
                    logger.warning(f"Alternative ML signal failed for {symbol}: {e}")
                    
        except Exception as e:
            logger.warning(f"ML prediction gathering failed: {e}")
            
        return ml_data
    
    def _parse_ml_response(self, llm_content: str, symbols: List[str]) -> Dict[str, Any]:
        """Parse LLM ML prediction response."""
        
        ml_scores = {}
        model_warnings = []
        
        # Extract ML prediction scores
        for symbol in symbols:
            patterns = [
                rf"{symbol}[:\-\s]*prediction[:\-\s]*(\d+\.?\d*)",
                rf"{symbol}[:\-\s]*ml[:\-\s]*score[:\-\s]*(\d+\.?\d*)",
                rf"{symbol}[:\-\s]*(\d+\.?\d*)[/\s](?:out of\s)?100"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, llm_content, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    if score <= 100:
                        ml_scores[symbol] = score / 100
                        break
                    elif score <= 1:
                        ml_scores[symbol] = score
                        break
        
        # Default scoring if no explicit scores found
        if not ml_scores and symbols:
            for symbol in symbols:
                # Use sentiment analysis on symbol mentions
                symbol_mentions = [line for line in llm_content.split('\n') if symbol in line]
                positive_words = ['strong', 'positive', 'bullish', 'buy', 'outperform']
                negative_words = ['weak', 'negative', 'bearish', 'sell', 'underperform']
                
                score = 0.5  # Neutral baseline
                for mention in symbol_mentions:
                    mention_lower = mention.lower()
                    for word in positive_words:
                        if word in mention_lower:
                            score += 0.1
                    for word in negative_words:
                        if word in mention_lower:
                            score -= 0.1
                
                ml_scores[symbol] = max(0.1, min(0.9, score))
        
        # Extract model warnings
        warning_patterns = [
            r"warning[:\-\s]+([^\.]+)",
            r"limitation[:\-\s]+([^\.]+)",
            r"uncertainty[:\-\s]+([^\.]+)",
            r"caution[:\-\s]+([^\.]+)"
        ]
        
        for pattern in warning_patterns:
            matches = re.findall(pattern, llm_content, re.IGNORECASE)
            model_warnings.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        return {
            "ml_scores": ml_scores,
            "model_warnings": model_warnings[:3],
            "high_confidence_picks": [symbol for symbol, score in ml_scores.items() if score > 0.75][:3],
            "raw_analysis": llm_content
        }
    
    def _create_fallback_ml_decision(self, symbols: List[str], error: str) -> LLMAgentDecision:
        """Create fallback ML decision."""
        
        neutral_scores = {symbol: 0.5 for symbol in symbols}
        
        return LLMAgentDecision(
            agent_name=self.name,
            decision={
                "ml_scores": neutral_scores,
                "model_warnings": [f"ML analysis failed: {error}"],
                "high_confidence_picks": [],
                "raw_analysis": f"Fallback neutral ML scores: {error}"
            },
            confidence=0.5,
            reasoning=f"LLM ML analysis failed, using neutral scores: {error}",
            supporting_data={},
            llm_response=None,
            timestamp=datetime.now()
        )

# Export all LLM agents for use
__all__ = [
    'LLMRiskParityAgent',
    'LLMMomentumFactorAgent', 
    'LLMValueFactorAgent',
    'LLMPredictiveAgent',
    'LLMAgentDecision'
]