"""
Specialized tools for pairs trading operations

These tools provide LangChain-compatible interfaces for pairs trading
functionality within the enhanced trading system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime, timedelta

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..strategies.pairs_trading import PairsStrategy, create_pairs_strategy
from ..tools.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


class PairAnalysisInput(BaseModel):
    """Input schema for pair analysis tool"""
    symbols: List[str] = Field(description="List of symbols to analyze for pairs")
    lookback_days: int = Field(default=252, description="Number of days of historical data")
    min_correlation: float = Field(default=0.7, description="Minimum correlation threshold")


class PairAnalysisTool(BaseTool):
    """Tool for analyzing potential trading pairs"""
    
    name: str = "analyze_trading_pairs"
    description: str = """
    Analyze a universe of stocks to find statistically significant trading pairs.
    Returns cointegrated pairs ranked by quality score including correlation,
    p-values, half-life, and other statistical measures.
    """
    args_schema = PairAnalysisInput
    
    def __init__(self, alpaca_client: AlpacaClient, **kwargs):
        super().__init__(**kwargs)
        self.alpaca_client = alpaca_client
        self.strategy = create_pairs_strategy({})
    
    async def _arun(self, symbols: List[str], lookback_days: int = 252, min_correlation: float = 0.7) -> str:
        """Analyze symbols for trading pairs"""
        try:
            # Fetch price data
            price_data = await self._fetch_price_data(symbols, lookback_days)
            
            if len(price_data) < 2:
                return "Insufficient price data to analyze pairs"
            
            # Update strategy parameters
            self.strategy.correlation_threshold = min_correlation
            
            # Find pairs
            pairs = self.strategy.find_pairs(symbols, price_data)
            
            if not pairs:
                return f"No cointegrated pairs found from {len(symbols)} symbols with correlation >= {min_correlation}"
            
            # Format results
            results = []
            for i, pair in enumerate(pairs[:10]):  # Top 10 pairs
                results.append(
                    f"{i+1}. {pair.symbol_a}-{pair.symbol_b}: "
                    f"corr={pair.correlation:.3f}, p-val={pair.p_value:.4f}, "
                    f"half-life={pair.half_life:.1f}d, score={pair.score:.3f}"
                )
            
            return f"Found {len(pairs)} cointegrated pairs:\n" + "\n".join(results)
            
        except Exception as e:
            logger.error(f"Error in pair analysis: {e}")
            return f"Error analyzing pairs: {str(e)}"
    
    def _run(self, symbols: List[str], lookback_days: int = 252, min_correlation: float = 0.7) -> str:
        """Synchronous wrapper"""
        return asyncio.run(self._arun(symbols, lookback_days, min_correlation))
    
    async def _fetch_price_data(self, symbols: List[str], lookback_days: int) -> Dict[str, pd.Series]:
        """Fetch historical price data"""
        price_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        for symbol in symbols:
            try:
                bars = await self.alpaca_client.get_historical_bars(
                    symbol, 
                    start_date.isoformat(), 
                    end_date.isoformat(),
                    timeframe='1Day'
                )
                
                if bars and len(bars) > 0:
                    prices = pd.Series(
                        [bar.close for bar in bars],
                        index=[bar.timestamp for bar in bars]
                    )
                    price_data[symbol] = prices
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        return price_data


class SpreadAnalysisInput(BaseModel):
    """Input schema for spread analysis tool"""
    symbol_a: str = Field(description="First symbol in the pair")
    symbol_b: str = Field(description="Second symbol in the pair")
    lookback_days: int = Field(default=60, description="Days of data for spread analysis")


class SpreadAnalysisTool(BaseTool):
    """Tool for analyzing the spread between two securities"""
    
    name: str = "analyze_spread"
    description: str = """
    Analyze the spread between two securities including current z-score,
    recent spread behavior, and trading signal recommendations.
    """
    args_schema = SpreadAnalysisInput
    
    def __init__(self, alpaca_client: AlpacaClient, **kwargs):
        super().__init__(**kwargs)
        self.alpaca_client = alpaca_client
        self.strategy = create_pairs_strategy({})
    
    async def _arun(self, symbol_a: str, symbol_b: str, lookback_days: int = 60) -> str:
        """Analyze spread between two symbols"""
        try:
            # Fetch price data
            price_data = await self._fetch_price_data([symbol_a, symbol_b], lookback_days)
            
            if symbol_a not in price_data or symbol_b not in price_data:
                return f"Insufficient price data for {symbol_a} or {symbol_b}"
            
            # Test cointegration to get beta
            pairs = self.strategy._test_pair_cointegration(
                symbol_a, symbol_b, price_data[symbol_a], price_data[symbol_b]
            )
            
            if not pairs:
                return f"No cointegration found between {symbol_a} and {symbol_b}"
            
            # Calculate current spread and z-score
            current_spread, current_zscore = self.strategy.calculate_spread_zscore(
                symbol_a, symbol_b, pairs.beta, price_data
            )
            
            # Calculate spread statistics
            spread = price_data[symbol_a] - pairs.beta * price_data[symbol_b]
            spread_mean = spread.mean()
            spread_std = spread.std()
            
            # Generate signal recommendation
            signal_rec = "HOLD"
            if current_zscore >= 2.0:
                signal_rec = f"SHORT SPREAD (short {symbol_a}, long {symbol_b})"
            elif current_zscore <= -2.0:
                signal_rec = f"LONG SPREAD (long {symbol_a}, short {symbol_b})"
            elif abs(current_zscore) <= 0.5:
                signal_rec = "CONSIDER EXIT if in position"
            
            return f"""Spread Analysis for {symbol_a}-{symbol_b}:
• Beta (hedge ratio): {pairs.beta:.4f}
• Current spread: {current_spread:.4f}
• Current z-score: {current_zscore:.2f}
• Spread mean: {spread_mean:.4f}
• Spread volatility: {spread_std:.4f}
• Cointegration p-value: {pairs.p_value:.4f}
• Half-life: {pairs.half_life:.1f} days
• Signal: {signal_rec}"""
            
        except Exception as e:
            logger.error(f"Error in spread analysis: {e}")
            return f"Error analyzing spread: {str(e)}"
    
    def _run(self, symbol_a: str, symbol_b: str, lookback_days: int = 60) -> str:
        """Synchronous wrapper"""
        return asyncio.run(self._arun(symbol_a, symbol_b, lookback_days))
    
    async def _fetch_price_data(self, symbols: List[str], lookback_days: int) -> Dict[str, pd.Series]:
        """Fetch historical price data"""
        price_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        for symbol in symbols:
            try:
                bars = await self.alpaca_client.get_historical_bars(
                    symbol, 
                    start_date.isoformat(), 
                    end_date.isoformat(),
                    timeframe='1Day'
                )
                
                if bars and len(bars) > 0:
                    prices = pd.Series(
                        [bar.close for bar in bars],
                        index=[bar.timestamp for bar in bars]
                    )
                    price_data[symbol] = prices
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        return price_data


class PairsPositionInput(BaseModel):
    """Input schema for pairs position tool"""
    action: str = Field(description="Action: 'status', 'list', or 'performance'")


class PairsPositionTool(BaseTool):
    """Tool for monitoring pairs trading positions"""
    
    name: str = "pairs_position_monitor"
    description: str = """
    Monitor active pairs trading positions including current spreads,
    z-scores, unrealized P&L, and position status.
    """
    args_schema = PairsPositionInput
    
    def __init__(self, pairs_strategy: PairsStrategy, **kwargs):
        super().__init__(**kwargs)
        self.strategy = pairs_strategy
    
    async def _arun(self, action: str = "status") -> str:
        """Monitor pairs positions"""
        try:
            if action == "status":
                status = self.strategy.get_portfolio_status()
                return f"""Pairs Trading Portfolio Status:
• Active pairs: {status['active_pairs']}/{status['max_pairs']}
• Total exposure: ${status['total_exposure']:,.2f}
• Available capacity: {status['max_pairs'] - status['active_pairs']} pairs"""
            
            elif action == "list":
                status = self.strategy.get_portfolio_status()
                if not status['pairs_detail']:
                    return "No active pairs positions"
                
                pairs_list = []
                for i, pair in enumerate(status['pairs_detail']):
                    pairs_list.append(
                        f"{i+1}. {pair['pair']} ({pair['side']}): "
                        f"entry_z={pair['entry_zscore']:.2f}, "
                        f"current_z={pair['current_zscore']:.2f}, "
                        f"pnl=${pair['unrealized_pnl']:,.2f}"
                    )
                
                return "Active Pairs Positions:\n" + "\n".join(pairs_list)
            
            elif action == "performance":
                # This would calculate actual performance metrics
                return """Pairs Trading Performance:
• Total pairs traded: 0
• Win rate: 0%
• Average holding period: 0 days
• Sharpe ratio: 0.0
• Maximum drawdown: 0%"""
            
            else:
                return f"Unknown action: {action}. Use 'status', 'list', or 'performance'"
                
        except Exception as e:
            logger.error(f"Error monitoring pairs positions: {e}")
            return f"Error monitoring positions: {str(e)}"
    
    def _run(self, action: str = "status") -> str:
        """Synchronous wrapper"""
        return asyncio.run(self._arun(action))


class PairsRiskInput(BaseModel):
    """Input schema for pairs risk assessment tool"""
    symbol_a: str = Field(description="First symbol in the pair")
    symbol_b: str = Field(description="Second symbol in the pair")
    position_size: float = Field(description="Proposed position size per leg")


class PairsRiskAssessmentTool(BaseTool):
    """Tool for assessing pairs trading risk"""
    
    name: str = "assess_pairs_risk"
    description: str = """
    Assess the risk of a proposed pairs trade including correlation stability,
    volatility analysis, and portfolio impact.
    """
    args_schema = PairsRiskInput
    
    def __init__(self, alpaca_client: AlpacaClient, **kwargs):
        super().__init__(**kwargs)
        self.alpaca_client = alpaca_client
    
    async def _arun(self, symbol_a: str, symbol_b: str, position_size: float) -> str:
        """Assess pairs trading risk"""
        try:
            # Fetch recent price data for volatility analysis
            price_data = await self._fetch_price_data([symbol_a, symbol_b], 60)
            
            if symbol_a not in price_data or symbol_b not in price_data:
                return f"Insufficient data for risk assessment of {symbol_a}-{symbol_b}"
            
            # Calculate volatilities
            returns_a = price_data[symbol_a].pct_change().dropna()
            returns_b = price_data[symbol_b].pct_change().dropna()
            
            vol_a = returns_a.std() * np.sqrt(252)  # Annualized
            vol_b = returns_b.std() * np.sqrt(252)
            
            # Calculate correlation stability (rolling 30-day correlation)
            aligned_returns = pd.DataFrame({
                'A': returns_a,
                'B': returns_b
            }).dropna()
            
            rolling_corr = aligned_returns['A'].rolling(30).corr(aligned_returns['B'])
            corr_stability = rolling_corr.std()
            current_corr = aligned_returns['A'].corr(aligned_returns['B'])
            
            # Estimate position risk
            current_price_a = price_data[symbol_a].iloc[-1]
            current_price_b = price_data[symbol_b].iloc[-1]
            
            shares_a = int(position_size / current_price_a)
            shares_b = int(position_size / current_price_b)
            
            # Daily VaR estimation (95% confidence)
            daily_var_a = shares_a * current_price_a * vol_a / np.sqrt(252) * 1.645
            daily_var_b = shares_b * current_price_b * vol_b / np.sqrt(252) * 1.645
            
            risk_rating = "LOW"
            if corr_stability > 0.3 or vol_a > 0.4 or vol_b > 0.4:
                risk_rating = "HIGH"
            elif corr_stability > 0.2 or vol_a > 0.3 or vol_b > 0.3:
                risk_rating = "MEDIUM"
            
            return f"""Pairs Risk Assessment for {symbol_a}-{symbol_b}:
• Current correlation: {current_corr:.3f}
• Correlation stability (std): {corr_stability:.3f}
• {symbol_a} volatility: {vol_a:.1%} annually
• {symbol_b} volatility: {vol_b:.1%} annually
• Position sizes: {shares_a} shares {symbol_a}, {shares_b} shares {symbol_b}
• Daily VaR (95%): ${daily_var_a:.0f} ({symbol_a}), ${daily_var_b:.0f} ({symbol_b})
• Risk Rating: {risk_rating}"""
            
        except Exception as e:
            logger.error(f"Error in pairs risk assessment: {e}")
            return f"Error assessing risk: {str(e)}"
    
    def _run(self, symbol_a: str, symbol_b: str, position_size: float) -> str:
        """Synchronous wrapper"""
        return asyncio.run(self._arun(symbol_a, symbol_b, position_size))
    
    async def _fetch_price_data(self, symbols: List[str], lookback_days: int) -> Dict[str, pd.Series]:
        """Fetch historical price data"""
        price_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        for symbol in symbols:
            try:
                bars = await self.alpaca_client.get_historical_bars(
                    symbol, 
                    start_date.isoformat(), 
                    end_date.isoformat(),
                    timeframe='1Day'
                )
                
                if bars and len(bars) > 0:
                    prices = pd.Series(
                        [bar.close for bar in bars],
                        index=[bar.timestamp for bar in bars]
                    )
                    price_data[symbol] = prices
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        return price_data


def create_pairs_trading_tools(alpaca_client: AlpacaClient, pairs_strategy: Optional[PairsStrategy] = None) -> List[BaseTool]:
    """
    Create all pairs trading tools for use in the LangChain agent system
    """
    tools = [
        PairAnalysisTool(alpaca_client=alpaca_client),
        SpreadAnalysisTool(alpaca_client=alpaca_client),
        PairsRiskAssessmentTool(alpaca_client=alpaca_client)
    ]
    
    if pairs_strategy:
        tools.append(PairsPositionTool(pairs_strategy=pairs_strategy))
    
    return tools