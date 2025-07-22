"""Simplified market analysis with essential data sources only.

Streamlined version focusing on:
- Yahoo Finance (primary data source)
- Alpaca (trading data)
- Basic news sentiment
- Eliminates complex multi-source aggregation for better performance
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
import yfinance as yf
import yahoo_fin.stock_info as si

warnings.filterwarnings('ignore')

from agents.state import TradingSignal, MarketCondition
from config.settings import settings

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Essential data sources only."""
    YAHOO_FINANCE = "yahoo_finance"
    ALPACA = "alpaca"
    NEWS_SENTIMENT = "news_sentiment"

class SignalStrength(Enum):
    """Signal strength classification."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"  
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class AnalysisResult:
    """Simplified analysis result."""
    symbol: str
    signal: SignalStrength
    confidence: float
    price: float
    expected_return: float
    risk_score: float
    reasoning: str
    timestamp: datetime

class SimplifiedDataProvider:
    """Streamlined data provider using only essential sources."""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
    async def get_stock_data(self, symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
        """Get stock data from Yahoo Finance (primary source)."""
        cache_key = f"{symbol}_{period}"
        
        # Check cache
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_duration:
                return data
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if not data.empty:
                # Cache the result
                self.cache[cache_key] = (data, datetime.now())
                return data
            
        except Exception as e:
            logger.warning(f"Failed to get data for {symbol}: {e}")
            
        return None
    
    async def get_basic_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get basic fundamental data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'market_cap': info.get('marketCap', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'profit_margin': info.get('profitMargins', 0)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get fundamentals for {symbol}: {e}")
            return {}
    
    async def get_market_sentiment(self, symbol: str) -> float:
        """Simple market sentiment based on recent price action."""
        try:
            data = await self.get_stock_data(symbol, "1mo")
            if data is None or len(data) < 5:
                return 0.0
            
            # Simple momentum-based sentiment
            recent_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
            volatility = data['Close'].pct_change().std()
            
            # Normalize to -1 to 1 scale
            sentiment = np.tanh(recent_return * 5)  # Scale factor
            
            # Adjust for volatility (high volatility = lower confidence)
            if volatility > 0.05:  # High volatility threshold
                sentiment *= 0.7
                
            return float(sentiment)
            
        except Exception as e:
            logger.warning(f"Failed to get sentiment for {symbol}: {e}")
            return 0.0

class SimplifiedMarketAnalyst:
    """Streamlined market analyst with essential analysis only."""
    
    def __init__(self):
        self.data_provider = SimplifiedDataProvider()
        
    async def analyze_stock(self, symbol: str) -> AnalysisResult:
        """Comprehensive but simplified stock analysis."""
        try:
            logger.info(f"📈 Analyzing {symbol}...")
            
            # Get essential data
            price_data = await self.data_provider.get_stock_data(symbol, "3mo")
            fundamentals = await self.data_provider.get_basic_fundamentals(symbol)
            sentiment = await self.data_provider.get_market_sentiment(symbol)
            
            if price_data is None or price_data.empty:
                return self._create_neutral_result(symbol, "No price data available")
            
            current_price = float(price_data['Close'].iloc[-1])
            
            # Technical analysis
            technical_score = self._analyze_technical_indicators(price_data)
            
            # Fundamental analysis  
            fundamental_score = self._analyze_fundamentals(fundamentals)
            
            # Combine scores
            total_score = (technical_score * 0.4 + 
                          fundamental_score * 0.4 + 
                          sentiment * 0.2)
            
            # Convert to signal
            signal = self._score_to_signal(total_score)
            confidence = abs(total_score)
            
            # Expected return (simplified)
            expected_return = total_score * 0.15  # Max 15% expected return
            
            # Risk score (based on volatility)
            volatility = price_data['Close'].pct_change().std()
            risk_score = min(volatility * 10, 1.0)  # Normalize to 0-1
            
            reasoning = f"Technical: {technical_score:.2f}, Fundamental: {fundamental_score:.2f}, Sentiment: {sentiment:.2f}"
            
            return AnalysisResult(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                price=current_price,
                expected_return=expected_return,
                risk_score=risk_score,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return self._create_neutral_result(symbol, f"Analysis error: {e}")
    
    def _analyze_technical_indicators(self, data: pd.DataFrame) -> float:
        """Simplified technical analysis."""
        try:
            # Moving averages
            ma_20 = data['Close'].rolling(20).mean()
            ma_50 = data['Close'].rolling(50).mean()
            current_price = data['Close'].iloc[-1]
            
            # Price vs moving averages
            ma_score = 0.0
            if current_price > ma_20.iloc[-1]:
                ma_score += 0.3
            if current_price > ma_50.iloc[-1]:
                ma_score += 0.3
            if ma_20.iloc[-1] > ma_50.iloc[-1]:  # Golden cross
                ma_score += 0.2
            
            # RSI (simplified)
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # RSI score
            rsi_score = 0.0
            if current_rsi < 30:  # Oversold
                rsi_score = 0.2
            elif current_rsi > 70:  # Overbought
                rsi_score = -0.2
            
            return ma_score + rsi_score - 0.3  # Center around 0
            
        except Exception as e:
            logger.warning(f"Technical analysis error: {e}")
            return 0.0
    
    def _analyze_fundamentals(self, fundamentals: Dict[str, Any]) -> float:
        """Simplified fundamental analysis."""
        try:
            score = 0.0
            
            # P/E ratio
            pe = fundamentals.get('pe_ratio', 0)
            if 0 < pe < 15:
                score += 0.2
            elif 15 <= pe < 25:
                score += 0.1
            elif pe >= 25:
                score -= 0.1
            
            # P/B ratio
            pb = fundamentals.get('pb_ratio', 0)
            if 0 < pb < 1.5:
                score += 0.1
            elif pb >= 3:
                score -= 0.1
            
            # Revenue growth
            growth = fundamentals.get('revenue_growth', 0)
            if growth > 0.1:  # 10% growth
                score += 0.2
            elif growth < 0:  # Declining revenue
                score -= 0.2
            
            # Profit margin
            margin = fundamentals.get('profit_margin', 0)
            if margin > 0.15:  # Good margins
                score += 0.1
            elif margin < 0:  # Losses
                score -= 0.2
            
            return score
            
        except Exception as e:
            logger.warning(f"Fundamental analysis error: {e}")
            return 0.0
    
    def _score_to_signal(self, score: float) -> SignalStrength:
        """Convert numeric score to signal strength."""
        if score > 0.4:
            return SignalStrength.STRONG_BUY
        elif score > 0.1:
            return SignalStrength.BUY
        elif score > -0.1:
            return SignalStrength.HOLD
        elif score > -0.4:
            return SignalStrength.SELL
        else:
            return SignalStrength.STRONG_SELL
    
    def _create_neutral_result(self, symbol: str, reason: str) -> AnalysisResult:
        """Create neutral analysis result for errors."""
        return AnalysisResult(
            symbol=symbol,
            signal=SignalStrength.HOLD,
            confidence=0.0,
            price=0.0,
            expected_return=0.0,
            risk_score=0.5,
            reasoning=reason,
            timestamp=datetime.now()
        )
    
    async def analyze_multiple_stocks(self, symbols: List[str]) -> List[AnalysisResult]:
        """Analyze multiple stocks efficiently."""
        tasks = [self.analyze_stock(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, AnalysisResult):
                valid_results.append(result)
            else:
                logger.error(f"Analysis task failed: {result}")
        
        return valid_results

# Global instances for backward compatibility
simplified_data_provider = SimplifiedDataProvider()
simplified_market_analyst = SimplifiedMarketAnalyst()

# Aliases for existing code compatibility
multi_source_data = simplified_data_provider
comprehensive_analyst = simplified_market_analyst

def enhanced_market_analysis_factory():
    """Factory function for creating market analysis instances."""
    return simplified_market_analyst

# Export main components
__all__ = [
    'SimplifiedDataProvider',
    'SimplifiedMarketAnalyst', 
    'AnalysisResult',
    'SignalStrength',
    'DataSource',
    'simplified_data_provider',
    'simplified_market_analyst',
    'multi_source_data',
    'comprehensive_analyst',
    'enhanced_market_analysis_factory'
]