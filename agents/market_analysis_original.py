"""Specialized market analysis agents for informed trading decisions."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Yahoo Finance libraries for reliable market data
import yfinance as yf
import yahoo_fin.stock_info as si
from yahoo_fin import news

# Yahoo Finance data provider replaces alpaca_client for market data
from agents.state import TradingSignal, MarketCondition
from config.settings import settings

logger = logging.getLogger(__name__)

class YahooFinanceDataProvider:
    """Yahoo Finance data provider for reliable market data access."""
    
    def __init__(self):
        self.cache = {}  # Simple caching to avoid redundant requests
        self.cache_duration = 300  # 5 minutes cache
    
    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Get historical stock data from Yahoo Finance."""
        try:
            cache_key = f"{symbol}_{period}_{interval}"
            current_time = datetime.now()
            
            # Check cache
            if cache_key in self.cache:
                cached_data, cached_time = self.cache[cache_key]
                if (current_time - cached_time).seconds < self.cache_duration:
                    return cached_data
            
            # Fetch fresh data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            # Clean and prepare data
            if not data.empty:
                data.columns = [col.lower() for col in data.columns]
                data = data.dropna()
                
                # Cache the data
                self.cache[cache_key] = (data, current_time)
                
                logger.info(f"Retrieved {len(data)} data points for {symbol}")
                return data
            else:
                logger.warning(f"No data available for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to get Yahoo Finance data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_real_time_price(self, symbol: str) -> Dict[str, Any]:
        """Get real-time price and basic info with fallback methods."""
        try:
            # First try yahoo_fin for real-time data
            try:
                current_price = si.get_live_price(symbol)
                info = si.get_quote_table(symbol)
                
                return {
                    "symbol": symbol,
                    "price": float(current_price),
                    "previous_close": float(info.get("Previous Close", current_price)),
                    "open": float(info.get("Open", current_price)),
                    "day_range": info.get("Day's Range", "N/A"),
                    "volume": info.get("Volume", "N/A"),
                    "market_cap": info.get("Market Cap", "N/A"),
                    "pe_ratio": info.get("PE Ratio (TTM)", "N/A"),
                    "timestamp": datetime.now()
                }
            except Exception:
                # Fallback to yfinance for current price
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d", interval="1m")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    previous_close = hist['Close'].iloc[0] if len(hist) > 1 else current_price
                    
                    return {
                        "symbol": symbol,
                        "price": float(current_price),
                        "previous_close": float(previous_close),
                        "open": float(hist['Open'].iloc[0] if not hist.empty else current_price),
                        "day_range": f"{hist['Low'].min():.2f} - {hist['High'].max():.2f}",
                        "volume": int(hist['Volume'].sum()) if 'Volume' in hist.columns else 0,
                        "timestamp": datetime.now()
                    }
                else:
                    raise ValueError("No current price data available")
            
        except Exception as e:
            logger.error(f"Failed to get real-time data for {symbol}: {e}")
            return {"symbol": symbol, "price": 0, "error": str(e)}
    
    def get_financial_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get key financial metrics for fundamental analysis."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key metrics
            metrics = {
                "symbol": symbol,
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "price_to_book": info.get("priceToBook", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "roe": info.get("returnOnEquity", 0),
                "roa": info.get("returnOnAssets", 0),
                "profit_margin": info.get("profitMargins", 0),
                "revenue_growth": info.get("revenueGrowth", 0),
                "earnings_growth": info.get("earningsGrowth", 0),
                "beta": info.get("beta", 1.0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "avg_volume": info.get("averageVolume", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown")
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get financial metrics for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    def get_news_sentiment(self, symbol: str, num_articles: int = 10) -> Dict[str, Any]:
        """Get recent news and basic sentiment analysis."""
        try:
            # Get news for the symbol
            ticker = yf.Ticker(symbol)
            news_data = ticker.news
            
            if not news_data:
                return {"symbol": symbol, "sentiment_score": 0.5, "news_count": 0}
            
            # Basic sentiment analysis based on title keywords
            positive_keywords = ["beats", "exceeds", "growth", "strong", "positive", "upgrade", "buy", "bullish"]
            negative_keywords = ["misses", "disappoints", "weak", "decline", "negative", "downgrade", "sell", "bearish"]
            
            sentiment_scores = []
            processed_articles = []
            
            for article in news_data[:num_articles]:
                title = article.get("title", "").lower()
                
                positive_count = sum(1 for keyword in positive_keywords if keyword in title)
                negative_count = sum(1 for keyword in negative_keywords if keyword in title)
                
                # Simple sentiment scoring
                if positive_count > negative_count:
                    sentiment = 0.7
                elif negative_count > positive_count:
                    sentiment = 0.3
                else:
                    sentiment = 0.5
                
                sentiment_scores.append(sentiment)
                processed_articles.append({
                    "title": article.get("title", ""),
                    "publisher": article.get("publisher", ""),
                    "link": article.get("link", ""),
                    "sentiment": sentiment
                })
            
            # Average sentiment
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.5
            
            return {
                "symbol": symbol,
                "sentiment_score": avg_sentiment,
                "news_count": len(processed_articles),
                "articles": processed_articles[:5],  # Return top 5 articles
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get news sentiment for {symbol}: {e}")
            return {"symbol": symbol, "sentiment_score": 0.5, "news_count": 0, "error": str(e)}
    
    def get_market_indices(self) -> Dict[str, Dict[str, Any]]:
        """Get major market indices data."""
        indices = {
            "SPY": "S&P 500",
            "QQQ": "NASDAQ 100", 
            "DIA": "Dow Jones",
            "IWM": "Russell 2000",
            "VIX": "Volatility Index"
        }
        
        indices_data = {}
        
        for symbol, name in indices.items():
            try:
                price_data = self.get_real_time_price(symbol)
                if "error" not in price_data:
                    # Get 1-month historical data for trend
                    hist_data = self.get_stock_data(symbol, period="1mo", interval="1d")
                    
                    if not hist_data.empty:
                        current_price = price_data["price"]
                        month_ago_price = hist_data.iloc[0]["close"]
                        month_change = (current_price - month_ago_price) / month_ago_price
                        
                        # Calculate short-term trend
                        recent_5d = hist_data.tail(5)["close"]
                        trend = "up" if recent_5d.iloc[-1] > recent_5d.iloc[0] else "down"
                        
                        indices_data[symbol] = {
                            "name": name,
                            "price": current_price,
                            "month_change": month_change,
                            "trend_5d": trend,
                            "volatility": hist_data["close"].pct_change().std() * np.sqrt(252)
                        }
                
            except Exception as e:
                logger.warning(f"Failed to get data for index {symbol}: {e}")
        
        return indices_data

# Global Yahoo Finance data provider
yahoo_data = YahooFinanceDataProvider()

class SignalStrength(Enum):
    VERY_WEAK = 0.2
    WEAK = 0.4
    MODERATE = 0.6
    STRONG = 0.8
    VERY_STRONG = 1.0

class MarketRegime(Enum):
    BULL = "bullish"
    BEAR = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

@dataclass
class AnalysisResult:
    """Result from market analysis."""
    symbol: str
    signal_type: str  # "momentum", "mean_reversion", "breakout"
    action: str  # "buy", "sell", "hold"
    confidence: float
    strength: SignalStrength
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: str = ""
    indicators: Dict[str, float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.indicators is None:
            self.indicators = {}

class TechnicalAnalysisAgent:
    """Agent specializing in technical analysis for trading signals."""
    
    def __init__(self):
        self.name = "Technical Analysis Agent"
        self.lookback_period = 100  # Days of historical data
        self.min_data_points = 50   # Minimum data for reliable analysis
    
    def analyze_symbol(self, symbol: str) -> AnalysisResult:
        """Perform comprehensive technical analysis on a symbol using Yahoo Finance data."""
        try:
            logger.info(f"Technical analysis for {symbol}")
            
            # Get historical data from Yahoo Finance
            df = yahoo_data.get_stock_data(symbol, period="1y", interval="1d")
            if len(df) < self.min_data_points:
                return AnalysisResult(
                    symbol=symbol,
                    signal_type="insufficient_data",
                    action="hold",
                    confidence=0.0,
                    strength=SignalStrength.VERY_WEAK,
                    reasoning="Insufficient historical data for analysis"
                )
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(df)
            
            # Generate signals from different strategies
            momentum_signal = self._momentum_analysis(df, indicators)
            mean_reversion_signal = self._mean_reversion_analysis(df, indicators)
            breakout_signal = self._breakout_analysis(df, indicators)
            
            # Combine signals using ensemble approach
            final_signal = self._ensemble_decision([
                momentum_signal,
                mean_reversion_signal,
                breakout_signal
            ])
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Technical analysis error for {symbol}: {e}")
            return AnalysisResult(
                symbol=symbol,
                signal_type="error",
                action="hold",
                confidence=0.0,
                strength=SignalStrength.VERY_WEAK,
                reasoning=f"Analysis error: {e}"
            )
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators."""
        indicators = {}
        
        # Price-based indicators
        indicators['current_price'] = float(df.iloc[-1]['close'])
        indicators['sma_20'] = df['close'].rolling(20).mean().iloc[-1]
        indicators['sma_50'] = df['close'].rolling(50).mean().iloc[-1]
        indicators['ema_12'] = df['close'].ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = df['close'].ewm(span=26).mean().iloc[-1]
        
        # Volatility indicators
        indicators['volatility_20'] = df['close'].rolling(20).std().iloc[-1]
        indicators['atr'] = self._calculate_atr(df).iloc[-1]
        
        # Momentum indicators
        indicators['rsi'] = self._calculate_rsi(df['close']).iloc[-1]
        indicators['macd'], indicators['macd_signal'] = self._calculate_macd(df['close'])
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # Volume indicators
        if 'volume' in df.columns:
            indicators['volume_sma'] = df['volume'].rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = df.iloc[-1]['volume'] / indicators['volume_sma']
        
        # Support/Resistance levels
        indicators['resistance'] = df['high'].rolling(20).max().iloc[-1]
        indicators['support'] = df['low'].rolling(20).min().iloc[-1]
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(df['close'])
        indicators['bb_upper'] = bb_upper.iloc[-1]
        indicators['bb_lower'] = bb_lower.iloc[-1]
        indicators['bb_position'] = (indicators['current_price'] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[float, float]:
        """Calculate MACD and signal line."""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return float(macd.iloc[-1]), float(signal.iloc[-1])
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def _momentum_analysis(self, df: pd.DataFrame, indicators: Dict) -> AnalysisResult:
        """Analyze momentum signals."""
        confidence = 0.0
        action = "hold"
        reasoning_parts = []
        
        # MACD momentum
        if indicators['macd'] > indicators['macd_signal']:
            confidence += 0.3
            reasoning_parts.append("MACD bullish crossover")
        elif indicators['macd'] < indicators['macd_signal']:
            confidence -= 0.3
            reasoning_parts.append("MACD bearish crossover")
        
        # RSI momentum
        rsi = indicators['rsi']
        if rsi < 30:
            confidence += 0.2
            reasoning_parts.append("RSI oversold")
        elif rsi > 70:
            confidence -= 0.2
            reasoning_parts.append("RSI overbought")
        elif 40 < rsi < 60:
            confidence += 0.1
            reasoning_parts.append("RSI neutral momentum")
        
        # Price vs. moving averages
        current_price = indicators['current_price']
        if current_price > indicators['sma_20'] > indicators['sma_50']:
            confidence += 0.3
            reasoning_parts.append("Price above SMAs")
        elif current_price < indicators['sma_20'] < indicators['sma_50']:
            confidence -= 0.3
            reasoning_parts.append("Price below SMAs")
        
        # Volume confirmation
        if indicators.get('volume_ratio', 1.0) > 1.5:
            confidence += 0.2
            reasoning_parts.append("High volume confirmation")
        
        # Determine action
        if confidence > 0.4:
            action = "buy"
        elif confidence < -0.4:
            action = "sell"
        
        return AnalysisResult(
            symbol=df.iloc[-1].get('symbol', 'UNKNOWN'),
            signal_type="momentum",
            action=action,
            confidence=abs(confidence),
            strength=self._confidence_to_strength(abs(confidence)),
            reasoning=f"Momentum: {', '.join(reasoning_parts)}",
            indicators=indicators
        )
    
    def _mean_reversion_analysis(self, df: pd.DataFrame, indicators: Dict) -> AnalysisResult:
        """Analyze mean reversion signals."""
        confidence = 0.0
        action = "hold"
        reasoning_parts = []
        
        current_price = indicators['current_price']
        
        # Bollinger Bands mean reversion
        bb_position = indicators['bb_position']
        if bb_position < 0.2:  # Near lower band
            confidence += 0.4
            reasoning_parts.append("Price near lower Bollinger Band")
        elif bb_position > 0.8:  # Near upper band
            confidence -= 0.4
            reasoning_parts.append("Price near upper Bollinger Band")
        
        # RSI mean reversion
        rsi = indicators['rsi']
        if rsi < 30:
            confidence += 0.3
            reasoning_parts.append("RSI oversold - potential reversal")
        elif rsi > 70:
            confidence -= 0.3
            reasoning_parts.append("RSI overbought - potential reversal")
        
        # Distance from moving average
        sma_20 = indicators['sma_20']
        price_deviation = (current_price - sma_20) / sma_20
        
        if price_deviation < -0.05:  # 5% below SMA
            confidence += 0.2
            reasoning_parts.append("Price significantly below SMA-20")
        elif price_deviation > 0.05:  # 5% above SMA
            confidence -= 0.2
            reasoning_parts.append("Price significantly above SMA-20")
        
        # Volatility consideration
        volatility = indicators['volatility_20']
        avg_volatility = df['close'].rolling(50).std().mean()
        if volatility > avg_volatility * 1.5:
            confidence *= 0.8  # Reduce confidence in high volatility
            reasoning_parts.append("High volatility reduces confidence")
        
        # Determine action
        if confidence > 0.3:
            action = "buy"
        elif confidence < -0.3:
            action = "sell"
        
        return AnalysisResult(
            symbol=df.iloc[-1].get('symbol', 'UNKNOWN'),
            signal_type="mean_reversion",
            action=action,
            confidence=abs(confidence),
            strength=self._confidence_to_strength(abs(confidence)),
            reasoning=f"Mean Reversion: {', '.join(reasoning_parts)}",
            indicators=indicators
        )
    
    def _breakout_analysis(self, df: pd.DataFrame, indicators: Dict) -> AnalysisResult:
        """Analyze breakout signals."""
        confidence = 0.0
        action = "hold"
        reasoning_parts = []
        
        current_price = indicators['current_price']
        
        # Support/Resistance breakout
        resistance = indicators['resistance']
        support = indicators['support']
        
        if current_price > resistance * 1.01:  # 1% above resistance
            confidence += 0.5
            reasoning_parts.append("Resistance breakout")
        elif current_price < support * 0.99:  # 1% below support
            confidence -= 0.5
            reasoning_parts.append("Support breakdown")
        
        # Bollinger Band breakout
        bb_position = indicators['bb_position']
        if bb_position > 1.0:  # Above upper band
            confidence += 0.3
            reasoning_parts.append("Bollinger Band upside breakout")
        elif bb_position < 0.0:  # Below lower band
            confidence -= 0.3
            reasoning_parts.append("Bollinger Band downside breakout")
        
        # Volume confirmation for breakouts
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if abs(confidence) > 0.3 and volume_ratio > 2.0:
            confidence *= 1.3  # Boost confidence with volume
            reasoning_parts.append("Strong volume confirmation")
        elif abs(confidence) > 0.3 and volume_ratio < 0.8:
            confidence *= 0.7  # Reduce confidence without volume
            reasoning_parts.append("Weak volume - reduces confidence")
        
        # ATR-based volatility adjustment
        atr = indicators['atr']
        if atr > df['close'].rolling(50).std().mean():
            confidence *= 0.9  # Slight reduction in high volatility
            reasoning_parts.append("High ATR volatility")
        
        # Determine action
        if confidence > 0.4:
            action = "buy"
        elif confidence < -0.4:
            action = "sell"
        
        return AnalysisResult(
            symbol=df.iloc[-1].get('symbol', 'UNKNOWN'),
            signal_type="breakout",
            action=action,
            confidence=abs(confidence),
            strength=self._confidence_to_strength(abs(confidence)),
            reasoning=f"Breakout: {', '.join(reasoning_parts)}",
            indicators=indicators
        )
    
    def _ensemble_decision(self, signals: List[AnalysisResult]) -> AnalysisResult:
        """Combine multiple signals using ensemble approach."""
        if not signals:
            return AnalysisResult("UNKNOWN", "ensemble", "hold", 0.0, SignalStrength.VERY_WEAK)
        
        # Weight different signal types
        weights = {
            "momentum": 0.4,
            "mean_reversion": 0.3,
            "breakout": 0.3
        }
        
        buy_score = 0.0
        sell_score = 0.0
        total_confidence = 0.0
        reasoning_parts = []
        
        for signal in signals:
            weight = weights.get(signal.signal_type, 0.33)
            weighted_confidence = signal.confidence * weight
            
            if signal.action == "buy":
                buy_score += weighted_confidence
            elif signal.action == "sell":
                sell_score += weighted_confidence
            
            total_confidence += weighted_confidence
            if signal.confidence > 0.3:  # Only include significant signals
                reasoning_parts.append(f"{signal.signal_type}: {signal.reasoning}")
        
        # Determine final action
        if buy_score > sell_score and buy_score > 0.3:
            action = "buy"
            confidence = buy_score
        elif sell_score > buy_score and sell_score > 0.3:
            action = "sell"
            confidence = sell_score
        else:
            action = "hold"
            confidence = max(buy_score, sell_score)
        
        # Use indicators from the most confident signal
        best_signal = max(signals, key=lambda s: s.confidence)
        
        return AnalysisResult(
            symbol=best_signal.symbol,
            signal_type="ensemble",
            action=action,
            confidence=min(confidence, 1.0),
            strength=self._confidence_to_strength(confidence),
            price_target=self._calculate_price_target(best_signal, action),
            stop_loss=self._calculate_stop_loss(best_signal, action),
            reasoning=f"Ensemble decision: {'; '.join(reasoning_parts)}",
            indicators=best_signal.indicators
        )
    
    def _confidence_to_strength(self, confidence: float) -> SignalStrength:
        """Convert confidence score to signal strength enum."""
        if confidence >= 0.8:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.6:
            return SignalStrength.STRONG
        elif confidence >= 0.4:
            return SignalStrength.MODERATE
        elif confidence >= 0.2:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def _calculate_price_target(self, signal: AnalysisResult, action: str) -> Optional[float]:
        """Calculate price target based on technical levels."""
        if not signal.indicators:
            return None
        
        current_price = signal.indicators['current_price']
        atr = signal.indicators.get('atr', current_price * 0.02)
        
        if action == "buy":
            # Target based on resistance or ATR multiple
            resistance = signal.indicators.get('resistance', current_price)
            return min(resistance, current_price + (atr * 2))
        elif action == "sell":
            # Target based on support or ATR multiple
            support = signal.indicators.get('support', current_price)
            return max(support, current_price - (atr * 2))
        
        return None
    
    def _calculate_stop_loss(self, signal: AnalysisResult, action: str) -> Optional[float]:
        """Calculate stop loss based on technical levels and volatility."""
        if not signal.indicators:
            return None
        
        current_price = signal.indicators['current_price']
        atr = signal.indicators.get('atr', current_price * 0.02)
        
        if action == "buy":
            # Stop below recent support or ATR-based
            support = signal.indicators.get('support', current_price)
            return max(support * 0.98, current_price - (atr * 1.5))
        elif action == "sell":
            # Stop above recent resistance or ATR-based
            resistance = signal.indicators.get('resistance', current_price)
            return min(resistance * 1.02, current_price + (atr * 1.5))
        
        return None

class FundamentalAnalysisAgent:
    """Agent specializing in fundamental analysis and market sentiment."""
    
    def __init__(self):
        self.name = "Fundamental Analysis Agent"
    
    def analyze_market_regime(self, symbols: List[str]) -> MarketCondition:
        """Analyze overall market conditions and regime using Yahoo Finance data."""
        try:
            logger.info("Analyzing market regime with Yahoo Finance data")
            
            # Get comprehensive market indices data (used for market context)
            yahoo_data.get_market_indices()
            
            # Additional individual symbol analysis for context
            market_indices = ["SPY", "QQQ", "IWM", "VIX"]
            index_data = {}
            
            for index in market_indices:
                try:
                    df = yahoo_data.get_stock_data(index, period="3mo", interval="1d")
                    if not df.empty:
                        index_data[index] = df
                        
                        # Get real-time data for current context
                        real_time = yahoo_data.get_real_time_price(index)
                        if "error" not in real_time:
                            index_data[f"{index}_current"] = real_time
                            
                except Exception as e:
                    logger.warning(f"Could not get data for {index}: {e}")
            
            # Determine market trend
            trend = self._analyze_market_trend(index_data)
            
            # Assess volatility regime
            volatility = self._assess_volatility_regime(index_data)
            
            # Analyze sector rotation (simplified)
            sector_performance = self._analyze_sector_rotation(symbols)
            
            return MarketCondition(
                market_trend=trend,
                volatility_regime=volatility,
                sector_rotation=sector_performance,
                fear_greed_index=self._calculate_fear_greed_proxy(index_data)
            )
            
        except Exception as e:
            logger.error(f"Market regime analysis error: {e}")
            return MarketCondition(
                market_trend="sideways",
                volatility_regime="medium",
                sector_rotation={}
            )
    
    def _analyze_market_trend(self, index_data: Dict[str, pd.DataFrame]) -> str:
        """Analyze overall market trend."""
        if "SPY" not in index_data:
            return "sideways"
        
        spy_data = index_data["SPY"]
        if len(spy_data) < 20:
            return "sideways"
        
        # Calculate trend indicators
        sma_20 = spy_data['close'].rolling(20).mean().iloc[-1]
        sma_50 = spy_data['close'].rolling(50).mean().iloc[-1] if len(spy_data) >= 50 else sma_20
        current_price = spy_data.iloc[-1]['close']
        
        # Price momentum
        price_change_5d = (current_price - spy_data.iloc[-5]['close']) / spy_data.iloc[-5]['close']
        price_change_20d = (current_price - spy_data.iloc[-20]['close']) / spy_data.iloc[-20]['close']
        
        if current_price > sma_20 > sma_50 and price_change_5d > 0.01 and price_change_20d > 0.05:
            return "bullish"
        elif current_price < sma_20 < sma_50 and price_change_5d < -0.01 and price_change_20d < -0.05:
            return "bearish"
        else:
            return "sideways"
    
    def _assess_volatility_regime(self, index_data: Dict[str, pd.DataFrame]) -> str:
        """Assess current volatility regime."""
        volatility_scores = []
        
        # VIX-based assessment
        if "VIX" in index_data:
            vix_data = index_data["VIX"]
            current_vix = vix_data.iloc[-1]['close']
            
            if current_vix > 30:
                volatility_scores.append("high")
            elif current_vix > 20:
                volatility_scores.append("medium")
            else:
                volatility_scores.append("low")
        
        # SPY volatility assessment
        if "SPY" in index_data:
            spy_data = index_data["SPY"]
            returns = spy_data['close'].pct_change().dropna()
            
            if len(returns) >= 20:
                volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
                
                if volatility > 0.25:
                    volatility_scores.append("high")
                elif volatility > 0.15:
                    volatility_scores.append("medium")
                else:
                    volatility_scores.append("low")
        
        # Return most common assessment
        if volatility_scores:
            return max(set(volatility_scores), key=volatility_scores.count)
        else:
            return "medium"
    
    def _analyze_sector_rotation(self, symbols: List[str]) -> Dict[str, float]:
        """Analyze sector performance using Yahoo Finance data."""
        sector_performance = {}
        
        for symbol in symbols:
            try:
                # Get financial metrics including sector information
                metrics = yahoo_data.get_financial_metrics(symbol)
                if "error" not in metrics:
                    sector = metrics.get("sector", "Other")
                    
                    # Get historical data for performance calculation
                    df = yahoo_data.get_stock_data(symbol, period="1mo", interval="1d")
                    if len(df) >= 20:
                        performance = (df.iloc[-1]['close'] - df.iloc[-20]['close']) / df.iloc[-20]['close']
                        
                        if sector not in sector_performance:
                            sector_performance[sector] = []
                        sector_performance[sector].append(performance)
            except Exception as e:
                logger.warning(f"Failed to get sector data for {symbol}: {e}")
                continue
        
        # Average performance by sector
        sector_avg = {}
        for sector, performances in sector_performance.items():
            if performances:
                sector_avg[sector] = sum(performances) / len(performances)
        
        return sector_avg
    
    def _calculate_fear_greed_proxy(self, index_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate a proxy for fear/greed index."""
        fear_greed_score = 50.0  # Neutral
        
        # VIX component
        if "VIX" in index_data:
            vix = index_data["VIX"].iloc[-1]['close']
            vix_score = max(0, min(100, 100 - (vix - 10) * 2))  # Invert VIX
            fear_greed_score = (fear_greed_score + vix_score) / 2
        
        # Market momentum component
        if "SPY" in index_data:
            spy_data = index_data["SPY"]
            if len(spy_data) >= 20:
                momentum = (spy_data.iloc[-1]['close'] - spy_data.iloc[-20]['close']) / spy_data.iloc[-20]['close']
                momentum_score = 50 + (momentum * 1000)  # Scale momentum
                momentum_score = max(0, min(100, momentum_score))
                fear_greed_score = (fear_greed_score + momentum_score) / 2
        
        return fear_greed_score

class QuantitativeAnalysisAgent:
    """Agent implementing quantitative trading strategies and statistical models."""
    
    def __init__(self):
        self.name = "Quantitative Analysis Agent"
        self.min_data_points = 100
    
    def analyze_portfolio_optimization(self, symbols: List[str], 
                                     portfolio_value: float) -> Dict[str, float]:
        """Analyze optimal portfolio weights using quantitative methods."""
        try:
            logger.info(f"Performing portfolio optimization analysis for ${portfolio_value:,.2f} portfolio")
            
            # Get return data for all symbols using Yahoo Finance
            returns_data = {}
            for symbol in symbols:
                try:
                    df = yahoo_data.get_stock_data(symbol, period="1y", interval="1d")
                    if len(df) >= 50:
                        returns = df['close'].pct_change().dropna()
                        returns_data[symbol] = returns
                except Exception as e:
                    logger.warning(f"Could not get return data for {symbol}: {e}")
            
            if len(returns_data) < 2:
                logger.warning("Insufficient data for portfolio optimization")
                return {}
            
            # Calculate expected returns and covariance
            expected_returns = {}
            for symbol, returns in returns_data.items():
                expected_returns[symbol] = returns.mean() * 252  # Annualized
            
            # Simple equal-weight optimization (can be enhanced with modern portfolio theory)
            optimal_weights = self._calculate_risk_parity_weights(returns_data)
            
            # Apply position size limits
            max_weight = settings.max_position_size
            total_weight = sum(optimal_weights.values())
            
            if total_weight > 0:
                for symbol in optimal_weights:
                    optimal_weights[symbol] = min(optimal_weights[symbol], max_weight)
                    optimal_weights[symbol] = optimal_weights[symbol] / total_weight
            
            return optimal_weights
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return {}
    
    def _calculate_risk_parity_weights(self, returns_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Calculate risk parity portfolio weights."""
        if not returns_data:
            return {}
        
        # Calculate volatilities
        volatilities = {}
        for symbol, returns in returns_data.items():
            volatilities[symbol] = returns.std() * np.sqrt(252)  # Annualized
        
        # Risk parity: weight inversely proportional to volatility
        total_inv_vol = sum(1/vol for vol in volatilities.values())
        
        weights = {}
        for symbol, vol in volatilities.items():
            weights[symbol] = (1/vol) / total_inv_vol
        
        return weights
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for a return series."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate  # Annualized
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        return excess_returns / volatility if volatility > 0 else 0.0
    
    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def detect_regime_change(self, prices: pd.Series, window: int = 50) -> bool:
        """Detect potential regime changes using statistical methods."""
        if len(prices) < window * 2:
            return False
        
        # Calculate rolling volatility
        returns = prices.pct_change().dropna()
        rolling_vol = returns.rolling(window).std()
        
        # Compare recent volatility to historical average
        recent_vol = rolling_vol.iloc[-10:].mean()
        historical_vol = rolling_vol.iloc[:-10].mean()
        
        # Detect significant volatility changes
        vol_change = abs(recent_vol - historical_vol) / historical_vol
        
        return vol_change > 0.5  # 50% change in volatility

# Agent factory for creating specialized agents
class MarketAnalysisAgentFactory:
    """Factory for creating and managing market analysis agents."""
    
    def __init__(self):
        self.technical_agent = TechnicalAnalysisAgent()
        self.fundamental_agent = FundamentalAnalysisAgent()
        self.quant_agent = QuantitativeAnalysisAgent()
    
    def analyze_symbols(self, symbols: List[str]) -> Dict[str, AnalysisResult]:
        """Perform comprehensive analysis on multiple symbols."""
        results = {}
        
        # Analyze market regime first
        market_condition = self.fundamental_agent.analyze_market_regime(symbols)
        
        # Analyze each symbol
        for symbol in symbols:
            try:
                # Technical analysis
                technical_result = self.technical_agent.analyze_symbol(symbol)
                
                # Adjust confidence based on market conditions
                adjusted_result = self._adjust_for_market_conditions(
                    technical_result, market_condition
                )
                
                results[symbol] = adjusted_result
                
            except Exception as e:
                logger.error(f"Analysis error for {symbol}: {e}")
                results[symbol] = AnalysisResult(
                    symbol=symbol,
                    signal_type="error",
                    action="hold",
                    confidence=0.0,
                    strength=SignalStrength.VERY_WEAK,
                    reasoning=f"Analysis failed: {e}"
                )
        
        return results
    
    def _adjust_for_market_conditions(self, result: AnalysisResult, 
                                    market_condition: MarketCondition) -> AnalysisResult:
        """Adjust trading signals based on overall market conditions."""
        adjustment_factor = 1.0
        
        # Adjust based on market trend
        if market_condition.market_trend == "bearish" and result.action == "buy":
            adjustment_factor *= 0.7  # Reduce buy confidence in bear market
        elif market_condition.market_trend == "bullish" and result.action == "sell":
            adjustment_factor *= 0.7  # Reduce sell confidence in bull market
        
        # Adjust based on volatility
        if market_condition.volatility_regime == "high":
            adjustment_factor *= 0.8  # Reduce confidence in high volatility
        elif market_condition.volatility_regime == "low":
            adjustment_factor *= 1.1  # Increase confidence in low volatility
        
        # Apply adjustment
        adjusted_confidence = result.confidence * adjustment_factor
        adjusted_confidence = min(1.0, max(0.0, adjusted_confidence))
        
        # Update result
        result.confidence = adjusted_confidence
        result.strength = self.technical_agent._confidence_to_strength(adjusted_confidence)
        
        # Add market condition context to reasoning
        market_context = f"Market: {market_condition.market_trend}, Vol: {market_condition.volatility_regime}"
        result.reasoning = f"{result.reasoning} | {market_context}"
        
        return result
    
    def get_portfolio_recommendations(self, symbols: List[str], 
                                    portfolio_value: float) -> Dict[str, Any]:
        """Get comprehensive portfolio recommendations."""
        
        # Individual symbol analysis
        symbol_analysis = self.analyze_symbols(symbols)
        
        # Portfolio optimization
        optimal_weights = self.quant_agent.analyze_portfolio_optimization(
            symbols, portfolio_value
        )
        
        # Market regime analysis
        market_condition = self.fundamental_agent.analyze_market_regime(symbols)
        
        return {
            "symbol_analysis": symbol_analysis,
            "optimal_weights": optimal_weights,
            "market_condition": market_condition,
            "recommendations": self._generate_portfolio_recommendations(
                symbol_analysis, optimal_weights, market_condition
            )
        }
    
    def _generate_portfolio_recommendations(self, symbol_analysis: Dict[str, AnalysisResult],
                                          optimal_weights: Dict[str, float],
                                          market_condition: MarketCondition) -> List[str]:
        """Generate actionable portfolio recommendations."""
        recommendations = []
        
        # Market-level recommendations
        if market_condition.volatility_regime == "high":
            recommendations.append("Consider reducing position sizes due to high volatility")
        
        if market_condition.market_trend == "bearish":
            recommendations.append("Be cautious with new long positions in bearish market")
        
        # Symbol-specific recommendations
        strong_buy_signals = [
            symbol for symbol, result in symbol_analysis.items()
            if result.action == "buy" and result.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]
        ]
        
        strong_sell_signals = [
            symbol for symbol, result in symbol_analysis.items()
            if result.action == "sell" and result.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]
        ]
        
        if strong_buy_signals:
            recommendations.append(f"Strong buy signals: {', '.join(strong_buy_signals)}")
        
        if strong_sell_signals:
            recommendations.append(f"Strong sell signals: {', '.join(strong_sell_signals)}")
        
        # Portfolio optimization recommendations
        if optimal_weights:
            overweight_positions = [
                symbol for symbol, weight in optimal_weights.items()
                if weight > settings.max_position_size * 0.8
            ]
            if overweight_positions:
                recommendations.append(f"Consider reducing exposure: {', '.join(overweight_positions)}")
        
        return recommendations

# Global factory instance
market_analysis_factory = MarketAnalysisAgentFactory()