#!/usr/bin/env python3
"""
CRYPTO TRADING DISABLED - All code commented out for later implementation
Uncomment when crypto trading is re-enabled
"""

# CRYPTO TRADING DISABLED - Comment out entire file for later implementation
'''
"""
Crypto Sentiment Analyzer
Specialized sentiment analysis for cryptocurrency using CryptoCompare, 
social media, and crypto-specific news sources.
"""

import asyncio
import logging
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import re

from config.settings import settings
from core.crypto_data_collector import crypto_collector, CryptoSentiment, CryptoNews

logger = logging.getLogger(__name__)

@dataclass
class CryptoSentimentResult:
    """Comprehensive crypto sentiment analysis result."""
    symbol: str
    overall_score: float      # -1 to 1
    overall_sentiment: str    # very_negative, negative, neutral, positive, very_positive
    confidence: float         # 0 to 1
    
    # Component scores
    news_sentiment: float
    social_sentiment: float
    technical_sentiment: float
    
    # Metrics
    news_count: int
    social_mentions: int
    price_momentum: float
    volume_spike: bool
    
    # Sources
    news_sources: List[str]
    key_themes: List[str]
    risk_factors: List[str]
    
    timestamp: datetime

class CryptoSentimentAnalyzer:
    """Advanced cryptocurrency sentiment analysis."""
    
    def __init__(self):
        self.session = None
        
        # Crypto-specific sentiment weights
        self.weights = {
            'news': 0.4,        # News sentiment
            'social': 0.3,      # Social media sentiment  
            'technical': 0.2,   # Technical/price momentum
            'volume': 0.1       # Volume analysis
        }
        
        # Crypto sentiment keywords
        self.crypto_bullish_keywords = [
            'adoption', 'institutional', 'etf', 'halving', 'defi', 'nft',
            'blockchain', 'web3', 'metaverse', 'staking', 'yield', 'apy',
            'bullish', 'moon', 'hodl', 'diamond hands', 'to the moon',
            'breakout', 'all-time high', 'ath', 'uptick', 'surge'
        ]
        
        self.crypto_bearish_keywords = [
            'regulation', 'sec', 'ban', 'crackdown', 'hack', 'exploit',
            'rug pull', 'scam', 'ponzi', 'bubble', 'crash', 'dump',
            'bearish', 'capitulation', 'liquidation', 'margin call',
            'paper hands', 'sell-off', 'correction', 'decline'
        ]
    
    async def analyze_crypto_sentiment(self, symbol: str) -> CryptoSentimentResult:
        """Comprehensive sentiment analysis for a crypto symbol."""
        
        logger.info(f"🔍 Analyzing crypto sentiment for {symbol}")
        
        try:
            # Get comprehensive data from crypto collector
            crypto_data = await crypto_collector.get_comprehensive_crypto_data([symbol])
            symbol_data = crypto_data.get(symbol, {})
            
            # Extract components
            sentiment_data = symbol_data.get('sentiment', {})
            price_data = symbol_data.get('price_data', {})
            
            # Get news data
            news_items = await crypto_collector.get_crypto_news_feed([symbol])
            
            # Analyze each component
            news_sentiment = await self._analyze_news_sentiment(news_items, symbol)
            social_sentiment = self._analyze_social_sentiment(sentiment_data)
            technical_sentiment = self._analyze_technical_sentiment(price_data)
            
            # Calculate weighted overall sentiment
            overall_score = (
                news_sentiment['score'] * self.weights['news'] +
                social_sentiment['score'] * self.weights['social'] +
                technical_sentiment['score'] * self.weights['technical']
            )
            
            # Volume analysis
            volume_spike = self._detect_volume_spike(price_data)
            if volume_spike:
                overall_score *= 1.1  # Boost sentiment on volume spikes
            
            # Classify overall sentiment
            overall_sentiment = self._classify_sentiment_level(overall_score)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                news_sentiment, social_sentiment, technical_sentiment, len(news_items)
            )
            
            # Extract insights
            key_themes = self._extract_key_themes(news_items)
            risk_factors = self._identify_risk_factors(news_items, overall_score)
            
            result = CryptoSentimentResult(
                symbol=symbol,
                overall_score=overall_score,
                overall_sentiment=overall_sentiment,
                confidence=confidence,
                
                news_sentiment=news_sentiment['score'],
                social_sentiment=social_sentiment['score'],
                technical_sentiment=technical_sentiment['score'],
                
                news_count=len(news_items),
                social_mentions=sentiment_data.get('social_mentions', 0),
                price_momentum=technical_sentiment.get('momentum', 0.0),
                volume_spike=volume_spike,
                
                news_sources=list(set([item.source for item in news_items[:10]])),
                key_themes=key_themes,
                risk_factors=risk_factors,
                
                timestamp=datetime.now()
            )
            
            logger.info(f"✅ {symbol} sentiment: {overall_sentiment} ({overall_score:.3f}, confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing crypto sentiment for {symbol}: {e}")
            return self._create_neutral_result(symbol)
    
    async def _analyze_news_sentiment(self, news_items: List[CryptoNews], symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from crypto news articles."""
        
        if not news_items:
            return {'score': 0.0, 'articles_analyzed': 0}
        
        sentiment_scores = []
        
        for article in news_items:
            try:
                # Combine title and summary for analysis
                text = f"{article.title} {article.summary}".lower()
                
                # Check for crypto-specific keywords
                bullish_count = sum(1 for keyword in self.crypto_bullish_keywords if keyword in text)
                bearish_count = sum(1 for keyword in self.crypto_bearish_keywords if keyword in text)
                
                # Calculate article sentiment
                if bullish_count > bearish_count:
                    score = min(0.8, 0.2 + (bullish_count - bearish_count) * 0.1)
                elif bearish_count > bullish_count:
                    score = max(-0.8, -0.2 - (bearish_count - bullish_count) * 0.1)
                else:
                    score = 0.0
                
                # Weight by recency (more recent = more important)
                hours_old = (datetime.now() - article.published_at).total_seconds() / 3600
                recency_weight = max(0.1, 1.0 - (hours_old / 168))  # Decay over 1 week
                
                weighted_score = score * recency_weight
                sentiment_scores.append(weighted_score)
                
            except Exception as e:
                logger.debug(f"Error analyzing article sentiment: {e}")
                continue
        
        if sentiment_scores:
            # Weight recent articles more heavily
            avg_score = np.mean(sentiment_scores)
            return {'score': avg_score, 'articles_analyzed': len(sentiment_scores)}
        
        return {'score': 0.0, 'articles_analyzed': 0}
    
    def _analyze_social_sentiment(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze social media sentiment for crypto."""
        
        if not sentiment_data:
            return {'score': 0.0, 'mentions': 0}
        
        # Extract social metrics from CryptoCompare data
        social_mentions = sentiment_data.get('social_mentions', 0)
        base_score = sentiment_data.get('sentiment_score', 0.0)
        
        # Adjust score based on mention volume
        if social_mentions > 1000:
            volume_multiplier = 1.2
        elif social_mentions > 100:
            volume_multiplier = 1.1
        else:
            volume_multiplier = 0.9
        
        adjusted_score = base_score * volume_multiplier
        
        return {
            'score': np.clip(adjusted_score, -1.0, 1.0),
            'mentions': social_mentions
        }
    
    def _analyze_technical_sentiment(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze technical/price momentum sentiment."""
        
        if not price_data:
            return {'score': 0.0, 'momentum': 0.0}
        
        try:
            # Extract price metrics
            change_24h = price_data.get('change_24h', 0.0)
            volume_24h = price_data.get('volume_24h', 0.0)
            
            # Convert percentage change to sentiment score
            # Strong moves get higher sentiment magnitude
            if abs(change_24h) > 10:
                # Large moves: amplify sentiment
                momentum_score = np.sign(change_24h) * min(0.8, abs(change_24h) / 20)
            elif abs(change_24h) > 5:
                # Medium moves: moderate sentiment
                momentum_score = np.sign(change_24h) * min(0.5, abs(change_24h) / 15)
            else:
                # Small moves: weak sentiment
                momentum_score = np.sign(change_24h) * min(0.3, abs(change_24h) / 10)
            
            return {
                'score': momentum_score,
                'momentum': change_24h,
                'volume': volume_24h
            }
            
        except Exception as e:
            logger.debug(f"Error analyzing technical sentiment: {e}")
            return {'score': 0.0, 'momentum': 0.0}
    
    def _detect_volume_spike(self, price_data: Dict[str, Any]) -> bool:
        """Detect unusual volume activity."""
        
        if not price_data:
            return False
        
        try:
            volume_24h = price_data.get('volume_24h', 0)
            change_24h = abs(price_data.get('change_24h', 0))
            
            # Simple heuristic: high volume with significant price change
            return volume_24h > 1000000 and change_24h > 3.0  # $1M+ volume and 3%+ move
            
        except:
            return False
    
    def _classify_sentiment_level(self, score: float) -> str:
        """Classify sentiment score into categories."""
        
        if score >= 0.6:
            return 'very_positive'
        elif score >= 0.2:
            return 'positive'
        elif score <= -0.6:
            return 'very_negative'
        elif score <= -0.2:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_confidence(self, news_sentiment: Dict, social_sentiment: Dict, 
                            technical_sentiment: Dict, news_count: int) -> float:
        """Calculate confidence in sentiment analysis."""
        
        # Base confidence from data availability
        data_confidence = min(1.0, (
            (news_count / 20.0) * 0.4 +  # More news = higher confidence
            (min(social_sentiment.get('mentions', 0), 1000) / 1000.0) * 0.3 +  # More mentions
            (1.0 if technical_sentiment.get('score', 0) != 0 else 0.5) * 0.3  # Price movement
        ))
        
        # Consistency bonus: all components agreeing increases confidence
        scores = [
            news_sentiment.get('score', 0),
            social_sentiment.get('score', 0),
            technical_sentiment.get('score', 0)
        ]
        
        # Check if all scores have the same sign (agreement)
        if all(s >= 0 for s in scores) or all(s <= 0 for s in scores):
            consistency_bonus = 0.2
        else:
            consistency_bonus = 0.0
        
        return min(1.0, data_confidence + consistency_bonus)
    
    def _extract_key_themes(self, news_items: List[CryptoNews]) -> List[str]:
        """Extract key themes from news articles."""
        
        themes = []
        theme_patterns = {
            'regulation': ['regulation', 'regulatory', 'sec', 'government', 'policy'],
            'adoption': ['adoption', 'institutional', 'enterprise', 'mainstream'],
            'technology': ['upgrade', 'protocol', 'blockchain', 'network', 'development'],
            'market': ['market', 'trading', 'exchange', 'liquidity', 'volume'],
            'partnerships': ['partnership', 'collaboration', 'integration', 'alliance']
        }
        
        for theme, keywords in theme_patterns.items():
            theme_count = 0
            for article in news_items[:20]:  # Check recent articles
                text = f"{article.title} {article.summary}".lower()
                if any(keyword in text for keyword in keywords):
                    theme_count += 1
            
            if theme_count >= 2:  # Theme appears in multiple articles
                themes.append(theme)
        
        return themes[:5]  # Return top 5 themes
    
    def _identify_risk_factors(self, news_items: List[CryptoNews], overall_score: float) -> List[str]:
        """Identify potential risk factors."""
        
        risk_factors = []
        
        # Check for negative news patterns
        risk_patterns = {
            'regulatory_risk': ['ban', 'regulation', 'crackdown', 'investigation'],
            'security_risk': ['hack', 'exploit', 'vulnerability', 'breach'],
            'market_risk': ['volatility', 'manipulation', 'crash', 'correction'],
            'technical_risk': ['bug', 'fork', 'downtime', 'congestion']
        }
        
        for risk_type, keywords in risk_patterns.items():
            risk_mentions = 0
            for article in news_items[:10]:  # Check recent articles
                text = f"{article.title} {article.summary}".lower()
                if any(keyword in text for keyword in keywords):
                    risk_mentions += 1
            
            if risk_mentions >= 1:
                risk_factors.append(risk_type)
        
        # Add sentiment-based risk factors
        if overall_score < -0.5:
            risk_factors.append('extreme_negative_sentiment')
        elif overall_score > 0.5:
            risk_factors.append('potential_overextension')
        
        return risk_factors[:3]  # Return top 3 risks
    
    def _create_neutral_result(self, symbol: str) -> CryptoSentimentResult:
        """Create a neutral sentiment result for error cases."""
        
        return CryptoSentimentResult(
            symbol=symbol,
            overall_score=0.0,
            overall_sentiment='neutral',
            confidence=0.1,
            
            news_sentiment=0.0,
            social_sentiment=0.0,
            technical_sentiment=0.0,
            
            news_count=0,
            social_mentions=0,
            price_momentum=0.0,
            volume_spike=False,
            
            news_sources=[],
            key_themes=[],
            risk_factors=['insufficient_data'],
            
            timestamp=datetime.now()
        )
    
    async def analyze_multiple_cryptos(self, symbols: List[str]) -> Dict[str, CryptoSentimentResult]:
        """Analyze sentiment for multiple crypto symbols concurrently."""
        
        logger.info(f"🔍 Analyzing crypto sentiment for {len(symbols)} symbols")
        
        # Analyze in parallel
        tasks = [self.analyze_crypto_sentiment(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        sentiment_results = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing {symbol}: {result}")
                sentiment_results[symbol] = self._create_neutral_result(symbol)
            else:
                sentiment_results[symbol] = result
        
        return sentiment_results
    
    async def close(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None

# Global instance
crypto_sentiment_analyzer = CryptoSentimentAnalyzer()'''
