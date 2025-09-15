#!/usr/bin/env python3
"""
Enhanced Short-Selling Signal Intelligence Engine

This module provides sophisticated signal analysis for short-selling opportunities by implementing:
- Entity-resolved sentiment analysis with conviction scoring
- Sentiment velocity and acceleration tracking  
- Topic surprise detection using KL divergence
- Cross-source disagreement quantification
- Real-time market microstructure analysis

The engine integrates with existing social media and market data sources to generate
high-conviction short signals with proper risk assessment.
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
from scipy.spatial.distance import jensenshannon
import re

from core.social_media_collector_optimized import optimized_collector
from core.market_intelligence import market_intelligence  
from tools.alpaca_client import alpaca_client
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class EntitySentiment:
    """Entity-resolved sentiment with conviction scoring."""
    entity_name: str
    symbol: str
    mentions: int
    sentiment_score: float  # -1.0 to 1.0
    conviction_score: float  # 0.0 to 1.0 (higher = more confident)
    source_platforms: List[str]
    timestamp: datetime
    raw_mentions: List[Dict[str, Any]]

@dataclass
class SentimentVelocity:
    """Sentiment velocity and acceleration metrics."""
    symbol: str
    current_sentiment: float
    velocity: float  # Rate of sentiment change
    acceleration: float  # Rate of velocity change
    trend_strength: float  # 0.0 to 1.0
    reversal_probability: float  # 0.0 to 1.0
    time_window_hours: float
    data_points: int

@dataclass
class TopicSurprise:
    """Topic surprise detection using information theory."""
    symbol: str
    surprise_score: float  # Higher = more surprising/unexpected
    baseline_topics: Dict[str, float]  # Historical topic distribution
    current_topics: Dict[str, float]  # Current topic distribution
    kl_divergence: float
    emerging_topics: List[str]
    declining_topics: List[str]
    timestamp: datetime

@dataclass
class CrossSourceDisagreement:
    """Quantification of disagreement across data sources."""
    symbol: str
    disagreement_score: float  # 0.0 to 1.0 (higher = more disagreement)
    source_sentiments: Dict[str, float]  # platform -> sentiment
    source_volumes: Dict[str, int]  # platform -> mention count
    consensus_strength: float  # 0.0 to 1.0 (higher = stronger consensus)
    contrarian_indicators: List[str]
    timestamp: datetime

@dataclass
class EnhancedShortSignal:
    """Comprehensive short signal with all analysis components."""
    symbol: str
    signal_strength: float  # 0.0 to 1.0
    signal_type: str  # 'SHORT', 'WEAK_SHORT', 'MONITOR', 'AVOID'
    confidence: float  # 0.0 to 1.0
    reasoning: str
    
    # Core components
    entity_sentiment: Optional[EntitySentiment]
    sentiment_velocity: Optional[SentimentVelocity]
    topic_surprise: Optional[TopicSurprise]
    cross_source_disagreement: Optional[CrossSourceDisagreement]
    
    # Market context
    price_momentum: float
    volume_profile: Dict[str, float]
    technical_indicators: Dict[str, float]
    
    # Risk metrics
    borrow_availability: Optional[float]
    squeeze_risk: Optional[float]
    liquidity_score: float
    
    timestamp: datetime
    expires_at: datetime

class EntityResolver:
    """Resolve and normalize entity mentions across different sources."""
    
    def __init__(self):
        """Initialize entity resolver with company mappings."""
        self.company_mappings = {
            # Common alternative names and variations
            'APPLE': 'AAPL',
            'APPLE INC': 'AAPL', 
            'MICROSOFT': 'MSFT',
            'MICROSOFT CORP': 'MSFT',
            'GOOGLE': 'GOOGL',
            'ALPHABET': 'GOOGL',
            'AMAZON': 'AMZN',
            'TESLA': 'TSLA',
            'META': 'META',
            'FACEBOOK': 'META',
            'NVIDIA': 'NVDA',
            'NETFLIX': 'NFLX',
        }
        
        # Industry-specific terminology
        self.industry_terms = {
            'tech': ['technology', 'software', 'ai', 'cloud', 'saas'],
            'finance': ['bank', 'financial', 'fintech', 'credit'],
            'healthcare': ['pharma', 'biotech', 'medical', 'drug'],
            'energy': ['oil', 'gas', 'renewable', 'solar']
        }
        
        # Sentiment modifiers
        self.sentiment_modifiers = {
            'positive': ['bullish', 'moon', 'rocket', 'pump', 'surge', 'breakout'],
            'negative': ['bearish', 'crash', 'dump', 'tank', 'collapse', 'overvalued'],
            'uncertainty': ['confused', 'unsure', 'maybe', 'possibly', 'might']
        }
    
    def resolve_entity(self, text: str, symbol: str) -> EntitySentiment:
        """Resolve entity mentions and calculate conviction-weighted sentiment."""
        mentions = []
        total_mentions = 0
        weighted_sentiment = 0.0
        source_platforms = set()
        
        # Extract mentions with context
        text_upper = text.upper()
        symbol_upper = symbol.upper()
        
        # Direct ticker mentions (highest conviction)
        direct_mentions = len(re.findall(rf'\${symbol_upper}\b', text_upper))
        if direct_mentions > 0:
            mentions.append({
                'type': 'direct_ticker',
                'count': direct_mentions,
                'conviction': 0.9,
                'context': f"${symbol}"
            })
            total_mentions += direct_mentions
        
        # Company name mentions (medium conviction)
        for company_name, ticker in self.company_mappings.items():
            if ticker == symbol_upper and company_name in text_upper:
                company_mentions = text_upper.count(company_name)
                mentions.append({
                    'type': 'company_name',
                    'count': company_mentions,
                    'conviction': 0.7,
                    'context': company_name
                })
                total_mentions += company_mentions
        
        # Calculate sentiment with conviction weighting
        if total_mentions > 0:
            sentiment_score = self._calculate_sentiment_score(text, mentions)
            conviction_score = self._calculate_conviction_score(mentions, text)
        else:
            sentiment_score = 0.0
            conviction_score = 0.0
        
        return EntitySentiment(
            entity_name=symbol,
            symbol=symbol,
            mentions=total_mentions,
            sentiment_score=sentiment_score,
            conviction_score=conviction_score,
            source_platforms=list(source_platforms),
            timestamp=datetime.now(),
            raw_mentions=mentions
        )
    
    def _calculate_sentiment_score(self, text: str, mentions: List[Dict]) -> float:
        """Calculate sentiment score with context awareness."""
        text_lower = text.lower()
        
        # Count sentiment indicators
        positive_count = sum(1 for term in self.sentiment_modifiers['positive'] 
                           if term in text_lower)
        negative_count = sum(1 for term in self.sentiment_modifiers['negative']
                           if term in text_lower)
        uncertainty_count = sum(1 for term in self.sentiment_modifiers['uncertainty']
                              if term in text_lower)
        
        # Basic sentiment calculation
        total_indicators = positive_count + negative_count + uncertainty_count
        if total_indicators == 0:
            return 0.0
        
        # Weight by conviction of mentions
        conviction_weight = sum(m['conviction'] * m['count'] for m in mentions)
        conviction_weight = min(conviction_weight, 1.0)
        
        raw_sentiment = (positive_count - negative_count) / total_indicators
        uncertainty_discount = 1.0 - (uncertainty_count / total_indicators * 0.5)
        
        return raw_sentiment * uncertainty_discount * conviction_weight
    
    def _calculate_conviction_score(self, mentions: List[Dict], text: str) -> float:
        """Calculate conviction score based on mention quality and context."""
        if not mentions:
            return 0.0
        
        # Base conviction from mention types
        base_conviction = sum(m['conviction'] * m['count'] for m in mentions)
        base_conviction = min(base_conviction, 1.0)
        
        # Adjust for text length and detail
        text_length_factor = min(len(text) / 200, 1.0)  # Longer posts = higher conviction
        
        # Adjust for specific terms indicating research or analysis
        analysis_terms = ['analysis', 'research', 'fundamentals', 'valuation', 'earnings']
        analysis_factor = sum(1 for term in analysis_terms if term in text.lower())
        analysis_factor = min(analysis_factor * 0.1, 0.3)
        
        return min(base_conviction + text_length_factor * 0.2 + analysis_factor, 1.0)

class SentimentVelocityTracker:
    """Track sentiment velocity and acceleration over time."""
    
    def __init__(self, max_history_hours: int = 24):
        """Initialize velocity tracker."""
        self.max_history_hours = max_history_hours
        self.sentiment_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def add_sentiment_data_point(self, symbol: str, sentiment: float, timestamp: datetime):
        """Add sentiment data point for velocity calculation."""
        self.sentiment_history[symbol].append({
            'sentiment': sentiment,
            'timestamp': timestamp
        })
    
    def calculate_velocity_metrics(self, symbol: str) -> Optional[SentimentVelocity]:
        """Calculate velocity and acceleration metrics."""
        if symbol not in self.sentiment_history:
            return None
        
        history = list(self.sentiment_history[symbol])
        if len(history) < 3:
            return None
        
        # Filter to recent data
        cutoff_time = datetime.now() - timedelta(hours=self.max_history_hours)
        recent_history = [h for h in history if h['timestamp'] >= cutoff_time]
        
        if len(recent_history) < 3:
            return None
        
        # Sort by timestamp
        recent_history.sort(key=lambda x: x['timestamp'])
        
        # Calculate velocity (first derivative)
        sentiments = [h['sentiment'] for h in recent_history]
        timestamps = [h['timestamp'].timestamp() for h in recent_history]
        
        # Use linear regression for robust velocity estimation
        velocity, _, r_value, _, _ = stats.linregress(timestamps, sentiments)
        velocity *= 3600  # Convert to per-hour rate
        
        # Calculate acceleration (second derivative)
        if len(recent_history) >= 5:
            mid_point = len(recent_history) // 2
            first_half_velocity = self._calculate_segment_velocity(recent_history[:mid_point+1])
            second_half_velocity = self._calculate_segment_velocity(recent_history[mid_point:])
            
            time_diff = (recent_history[-1]['timestamp'] - recent_history[mid_point]['timestamp']).total_seconds() / 3600
            acceleration = (second_half_velocity - first_half_velocity) / max(time_diff, 0.1)
        else:
            acceleration = 0.0
        
        # Calculate trend strength and reversal probability
        trend_strength = abs(r_value) if abs(r_value) > 0.3 else 0.0
        
        # Reversal probability based on velocity and recent sentiment extremes
        current_sentiment = sentiments[-1]
        max_sentiment = max(sentiments)
        min_sentiment = min(sentiments)
        
        if current_sentiment > 0.7 and velocity > 0:
            reversal_probability = min((current_sentiment - 0.7) * 3, 1.0)
        elif current_sentiment < -0.7 and velocity < 0:
            reversal_probability = min((abs(current_sentiment) - 0.7) * 3, 1.0)
        else:
            reversal_probability = 0.0
        
        return SentimentVelocity(
            symbol=symbol,
            current_sentiment=current_sentiment,
            velocity=velocity,
            acceleration=acceleration,
            trend_strength=trend_strength,
            reversal_probability=reversal_probability,
            time_window_hours=self.max_history_hours,
            data_points=len(recent_history)
        )
    
    def _calculate_segment_velocity(self, segment: List[Dict]) -> float:
        """Calculate velocity for a segment of data."""
        if len(segment) < 2:
            return 0.0
        
        sentiments = [h['sentiment'] for h in segment]
        timestamps = [h['timestamp'].timestamp() for h in segment]
        
        velocity, _, _, _, _ = stats.linregress(timestamps, sentiments)
        return velocity * 3600  # Convert to per-hour rate

class TopicSurpriseDetector:
    """Detect surprising topic shifts using KL divergence."""
    
    def __init__(self, baseline_window_days: int = 30):
        """Initialize topic surprise detector."""
        self.baseline_window_days = baseline_window_days
        self.topic_keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'eps', 'guidance', 'forecast'],
            'technical': ['support', 'resistance', 'breakout', 'pattern', 'chart', 'technical'],
            'fundamental': ['valuation', 'pe ratio', 'book value', 'fundamental', 'intrinsic'],
            'news': ['announcement', 'news', 'press release', 'acquisition', 'merger'],
            'sentiment': ['bullish', 'bearish', 'optimistic', 'pessimistic', 'sentiment'],
            'risk': ['risk', 'volatility', 'uncertainty', 'concern', 'warning'],
            'growth': ['growth', 'expansion', 'opportunity', 'potential', 'future'],
            'competition': ['competition', 'competitor', 'market share', 'rivalry']
        }
        
        self.baseline_topics: Dict[str, Dict[str, float]] = {}
    
    def extract_topics(self, texts: List[str]) -> Dict[str, float]:
        """Extract topic distribution from texts."""
        topic_counts = defaultdict(int)
        total_texts = len(texts)
        
        if total_texts == 0:
            return {}
        
        for text in texts:
            text_lower = text.lower()
            for topic, keywords in self.topic_keywords.items():
                topic_mentions = sum(1 for keyword in keywords if keyword in text_lower)
                if topic_mentions > 0:
                    topic_counts[topic] += 1
        
        # Normalize to probabilities
        topic_distribution = {}
        for topic, count in topic_counts.items():
            topic_distribution[topic] = count / total_texts
        
        # Add smoothing for topics not mentioned
        for topic in self.topic_keywords:
            if topic not in topic_distribution:
                topic_distribution[topic] = 0.01  # Small smoothing value
        
        return topic_distribution
    
    def update_baseline(self, symbol: str, historical_texts: List[str]):
        """Update baseline topic distribution for a symbol."""
        if not historical_texts:
            return
        
        baseline_distribution = self.extract_topics(historical_texts)
        self.baseline_topics[symbol] = baseline_distribution
        
        logger.debug(f"Updated baseline topics for {symbol}: {baseline_distribution}")
    
    def detect_surprise(self, symbol: str, current_texts: List[str]) -> Optional[TopicSurprise]:
        """Detect topic surprise compared to baseline."""
        if symbol not in self.baseline_topics:
            logger.debug(f"No baseline topics for {symbol}, cannot detect surprise")
            return None
        
        if not current_texts:
            return None
        
        baseline_dist = self.baseline_topics[symbol]
        current_dist = self.extract_topics(current_texts)
        
        # Ensure same topics in both distributions
        all_topics = set(baseline_dist.keys()) | set(current_dist.keys())
        
        baseline_probs = []
        current_probs = []
        
        for topic in all_topics:
            baseline_probs.append(baseline_dist.get(topic, 0.01))
            current_probs.append(current_dist.get(topic, 0.01))
        
        # Calculate KL divergence
        try:
            kl_div = stats.entropy(current_probs, baseline_probs)
            
            # Use Jensen-Shannon divergence for symmetric measure
            js_div = jensenshannon(baseline_probs, current_probs)
            surprise_score = js_div
            
        except Exception as e:
            logger.warning(f"Error calculating divergence for {symbol}: {e}")
            kl_div = 0.0
            surprise_score = 0.0
        
        # Identify emerging and declining topics
        emerging_topics = []
        declining_topics = []
        
        for topic in all_topics:
            baseline_prob = baseline_dist.get(topic, 0.01)
            current_prob = current_dist.get(topic, 0.01)
            
            ratio = current_prob / max(baseline_prob, 0.001)
            
            if ratio > 2.0 and current_prob > 0.1:  # Topic emergence
                emerging_topics.append(topic)
            elif ratio < 0.5 and baseline_prob > 0.1:  # Topic decline
                declining_topics.append(topic)
        
        return TopicSurprise(
            symbol=symbol,
            surprise_score=surprise_score,
            baseline_topics=baseline_dist,
            current_topics=current_dist,
            kl_divergence=kl_div,
            emerging_topics=emerging_topics,
            declining_topics=declining_topics,
            timestamp=datetime.now()
        )

class CrossSourceAnalyzer:
    """Analyze disagreement and consensus across different data sources."""
    
    def __init__(self):
        """Initialize cross-source analyzer."""
        self.source_weights = {
            'reddit': 0.3,
            'twitter': 0.3,
            'news': 0.4,  # Give news higher weight
            'earnings': 0.5,
            'insider_trading': 0.6
        }
    
    def analyze_cross_source_disagreement(self, symbol: str, 
                                        source_data: Dict[str, Dict]) -> CrossSourceDisagreement:
        """Analyze disagreement across different data sources."""
        source_sentiments = {}
        source_volumes = {}
        
        # Extract sentiment and volume from each source
        for source, data in source_data.items():
            if isinstance(data, dict):
                sentiment = data.get('sentiment', 0.0)
                volume = data.get('volume', 0) or data.get('count', 0)
                
                source_sentiments[source] = sentiment
                source_volumes[source] = volume
        
        if len(source_sentiments) < 2:
            # Not enough sources for disagreement analysis
            return CrossSourceDisagreement(
                symbol=symbol,
                disagreement_score=0.0,
                source_sentiments=source_sentiments,
                source_volumes=source_volumes,
                consensus_strength=0.5,
                contrarian_indicators=[],
                timestamp=datetime.now()
            )
        
        # Calculate disagreement metrics
        sentiments = list(source_sentiments.values())
        sentiment_std = np.std(sentiments)
        sentiment_range = max(sentiments) - min(sentiments)
        
        # Disagreement score based on variance and range
        disagreement_score = min(sentiment_std * 2 + sentiment_range * 0.5, 1.0)
        
        # Calculate weighted consensus
        total_weight = 0.0
        weighted_sentiment = 0.0
        
        for source, sentiment in source_sentiments.items():
            weight = self.source_weights.get(source, 0.2)
            volume = source_volumes.get(source, 1)
            
            # Adjust weight by volume (logarithmic scaling)
            volume_factor = math.log(max(volume, 1) + 1) / math.log(100)
            adjusted_weight = weight * min(volume_factor, 2.0)
            
            weighted_sentiment += sentiment * adjusted_weight
            total_weight += adjusted_weight
        
        consensus_sentiment = weighted_sentiment / max(total_weight, 0.1)
        
        # Consensus strength (inverse of disagreement)
        consensus_strength = 1.0 - disagreement_score
        
        # Identify contrarian indicators
        contrarian_indicators = []
        for source, sentiment in source_sentiments.items():
            deviation = abs(sentiment - consensus_sentiment)
            if deviation > 0.3:  # Significant deviation from consensus
                direction = "positive" if sentiment > consensus_sentiment else "negative"
                contrarian_indicators.append(f"{source}_{direction}")
        
        return CrossSourceDisagreement(
            symbol=symbol,
            disagreement_score=disagreement_score,
            source_sentiments=source_sentiments,
            source_volumes=source_volumes,
            consensus_strength=consensus_strength,
            contrarian_indicators=contrarian_indicators,
            timestamp=datetime.now()
        )

class EnhancedShortSignalEngine:
    """Main engine for generating enhanced short-selling signals."""
    
    def __init__(self):
        """Initialize the enhanced short signal engine."""
        self.entity_resolver = EntityResolver()
        self.velocity_tracker = SentimentVelocityTracker()
        self.topic_detector = TopicSurpriseDetector()
        self.cross_source_analyzer = CrossSourceAnalyzer()
        
        # Signal cache to avoid regenerating identical signals
        self.signal_cache: Dict[str, EnhancedShortSignal] = {}
        self.cache_ttl_minutes = 10
        
        logger.info("Enhanced Short Signal Engine initialized")
    
    async def generate_enhanced_short_signals(self, symbols: List[str], 
                                            max_signals: int = 10) -> List[EnhancedShortSignal]:
        """Generate enhanced short signals for given symbols."""
        try:
            logger.info(f"Generating enhanced short signals for {len(symbols)} symbols")
            signals = []
            
            # Process symbols in parallel for efficiency
            tasks = []
            for symbol in symbols[:max_signals]:
                task = asyncio.create_task(self._analyze_symbol_for_short_signal(symbol))
                tasks.append(task)
            
            # Wait for all analysis to complete
            symbol_analyses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for symbol, analysis in zip(symbols[:max_signals], symbol_analyses):
                if isinstance(analysis, Exception):
                    logger.warning(f"Analysis failed for {symbol}: {analysis}")
                    continue
                
                if analysis and analysis.signal_type != 'AVOID':
                    signals.append(analysis)
            
            # Sort by signal strength
            signals.sort(key=lambda s: s.signal_strength, reverse=True)
            
            logger.info(f"Generated {len(signals)} enhanced short signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating enhanced short signals: {e}")
            return []
    
    async def _analyze_symbol_for_short_signal(self, symbol: str) -> Optional[EnhancedShortSignal]:
        """Comprehensive analysis of a single symbol for short opportunities."""
        try:
            # Check cache first
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            if cache_key in self.signal_cache:
                cached_signal = self.signal_cache[cache_key]
                if (datetime.now() - cached_signal.timestamp).total_seconds() < self.cache_ttl_minutes * 60:
                    return cached_signal
            
            # Collect data from multiple sources
            social_data = await self._collect_social_media_data(symbol)
            market_data = await self._collect_market_data(symbol)
            
            if not social_data and not market_data:
                logger.debug(f"No data available for {symbol}")
                return None
            
            # Perform core analysis components
            entity_sentiment = await self._analyze_entity_sentiment(symbol, social_data)
            sentiment_velocity = await self._analyze_sentiment_velocity(symbol, entity_sentiment)
            topic_surprise = await self._analyze_topic_surprise(symbol, social_data)
            cross_source_disagreement = await self._analyze_cross_source_data(symbol, social_data, market_data)
            
            # Market microstructure analysis
            price_momentum = self._calculate_price_momentum(market_data)
            volume_profile = self._analyze_volume_profile(market_data)
            technical_indicators = self._calculate_technical_indicators(market_data)
            
            # Risk assessment
            liquidity_score = self._assess_liquidity(market_data)
            
            # Generate final signal
            signal = self._synthesize_short_signal(
                symbol=symbol,
                entity_sentiment=entity_sentiment,
                sentiment_velocity=sentiment_velocity,
                topic_surprise=topic_surprise,
                cross_source_disagreement=cross_source_disagreement,
                price_momentum=price_momentum,
                volume_profile=volume_profile,
                technical_indicators=technical_indicators,
                liquidity_score=liquidity_score
            )
            
            # Cache the signal
            if signal:
                self.signal_cache[cache_key] = signal
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} for short signal: {e}")
            return None
    
    async def _collect_social_media_data(self, symbol: str) -> Dict[str, Any]:
        """Collect social media data for the symbol."""
        try:
            # Use existing optimized social media collector
            social_results = await optimized_collector.collect_all_platforms(symbol, limit_per_platform=20)
            
            # Process and structure the data
            processed_data = {
                'reddit': {
                    'posts': social_results.get('reddit', []),
                    'sentiment': 0.0,
                    'volume': len(social_results.get('reddit', []))
                },
                'twitter': {
                    'posts': social_results.get('twitter', []),
                    'sentiment': 0.0,
                    'volume': len(social_results.get('twitter', []))
                }
            }
            
            # Calculate basic sentiment for each platform
            for platform in ['reddit', 'twitter']:
                posts = processed_data[platform]['posts']
                if posts:
                    sentiment_scores = []
                    for post in posts:
                        # Handle both cached string data and fresh object data
                        if isinstance(post, str):
                            # Extract content from cached string representation
                            # Format: "SocialMediaPost(platform='reddit', post_id='...', content='...', ...)"
                            if "content='" in post:
                                try:
                                    content_start = post.find("content='") + 9
                                    content_end = post.find("'", content_start)
                                    content = post[content_start:content_end] if content_end != -1 else ''
                                except:
                                    content = post[:100]  # Fallback to truncated string
                            else:
                                content = post[:100]  # Use first 100 chars as content
                        else:
                            # Fresh object data
                            content = getattr(post, 'content', '') or ''
                        
                        score = self._quick_sentiment_score(content)
                        sentiment_scores.append(score)
                    
                    if sentiment_scores:
                        processed_data[platform]['sentiment'] = np.mean(sentiment_scores)
            
            return processed_data
            
        except Exception as e:
            logger.warning(f"Error collecting social media data for {symbol}: {e}")
            return {}
    
    async def _collect_market_data(self, symbol: str) -> Dict[str, Any]:
        """Collect market data for the symbol."""
        try:
            # Get recent market data
            market_df = alpaca_client.get_market_data(symbol, limit=50)
            
            if market_df is None or len(market_df) == 0:
                return {}
            
            return {
                'price_data': market_df,
                'current_price': market_df['close'].iloc[-1] if len(market_df) > 0 else 0,
                'volume_data': market_df['volume'].tolist() if 'volume' in market_df.columns else []
            }
            
        except Exception as e:
            logger.warning(f"Error collecting market data for {symbol}: {e}")
            return {}
    
    async def _analyze_entity_sentiment(self, symbol: str, social_data: Dict) -> Optional[EntitySentiment]:
        """Analyze entity-resolved sentiment."""
        try:
            all_texts = []
            total_mentions = 0
            platform_sentiments = []
            
            for platform, data in social_data.items():
                posts = data.get('posts', [])
                for post in posts:
                    # Handle both cached string data and fresh object data
                    if isinstance(post, str):
                        # Extract content from cached string representation
                        if "content='" in post:
                            try:
                                content_start = post.find("content='") + 9
                                content_end = post.find("'", content_start)
                                content = post[content_start:content_end] if content_end != -1 else ''
                            except:
                                content = post[:100]  # Fallback to truncated string
                        else:
                            content = post[:100]  # Use first 100 chars as content
                    else:
                        # Fresh object data
                        content = getattr(post, 'content', '') or ''
                    
                    if content:
                        all_texts.append(content)
            
            if not all_texts:
                return None
            
            # Combine all texts for entity resolution
            combined_text = ' '.join(all_texts)
            entity_sentiment = self.entity_resolver.resolve_entity(combined_text, symbol)
            
            # If we found mentions, enhance with platform data
            if entity_sentiment.mentions > 0:
                entity_sentiment.source_platforms = list(social_data.keys())
            
            return entity_sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing entity sentiment for {symbol}: {e}")
            return None
    
    async def _analyze_sentiment_velocity(self, symbol: str, 
                                        entity_sentiment: Optional[EntitySentiment]) -> Optional[SentimentVelocity]:
        """Analyze sentiment velocity and acceleration."""
        try:
            if not entity_sentiment:
                return None
            
            # Add current sentiment to tracker
            self.velocity_tracker.add_sentiment_data_point(
                symbol=symbol,
                sentiment=entity_sentiment.sentiment_score,
                timestamp=entity_sentiment.timestamp
            )
            
            # Calculate velocity metrics
            velocity_metrics = self.velocity_tracker.calculate_velocity_metrics(symbol)
            return velocity_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment velocity for {symbol}: {e}")
            return None
    
    async def _analyze_topic_surprise(self, symbol: str, social_data: Dict) -> Optional[TopicSurprise]:
        """Analyze topic surprise using historical comparison."""
        try:
            # Extract current texts
            current_texts = []
            for platform, data in social_data.items():
                posts = data.get('posts', [])
                for post in posts:
                    # Handle both cached string data and fresh object data
                    if isinstance(post, str):
                        # Extract content from cached string representation
                        if "content='" in post:
                            try:
                                content_start = post.find("content='") + 9
                                content_end = post.find("'", content_start)
                                content = post[content_start:content_end] if content_end != -1 else ''
                            except:
                                content = post[:100]  # Fallback to truncated string
                        else:
                            content = post[:100]  # Use first 100 chars as content
                    else:
                        # Fresh object data
                        content = getattr(post, 'content', '') or ''
                    
                    if content:
                        current_texts.append(content)
            
            if not current_texts:
                return None
            
            # For baseline, we'd need historical data - for now use a simple heuristic
            # In production, this would query historical social media data
            baseline_texts = [
                f"{symbol} regular trading discussion",
                f"{symbol} price analysis",
                f"{symbol} earnings update"
            ]
            
            # Update baseline if not exists
            if symbol not in self.topic_detector.baseline_topics:
                self.topic_detector.update_baseline(symbol, baseline_texts)
            
            # Detect surprise
            topic_surprise = self.topic_detector.detect_surprise(symbol, current_texts)
            return topic_surprise
            
        except Exception as e:
            logger.error(f"Error analyzing topic surprise for {symbol}: {e}")
            return None
    
    async def _analyze_cross_source_data(self, symbol: str, social_data: Dict, 
                                       market_data: Dict) -> Optional[CrossSourceDisagreement]:
        """Analyze disagreement across data sources."""
        try:
            # Prepare source data
            source_data = {}
            
            # Add social media sources
            for platform, data in social_data.items():
                source_data[platform] = {
                    'sentiment': data.get('sentiment', 0.0),
                    'volume': data.get('volume', 0)
                }
            
            # Add market-based sentiment (price momentum as sentiment proxy)
            if market_data and 'price_data' in market_data:
                price_df = market_data['price_data']
                if len(price_df) >= 2:
                    price_change = (price_df['close'].iloc[-1] - price_df['close'].iloc[-2]) / price_df['close'].iloc[-2]
                    market_sentiment = np.tanh(price_change * 20)  # Scale to [-1, 1]
                    
                    source_data['market'] = {
                        'sentiment': market_sentiment,
                        'volume': int(price_df['volume'].iloc[-1]) if 'volume' in price_df.columns else 1
                    }
            
            if len(source_data) < 2:
                return None
            
            disagreement = self.cross_source_analyzer.analyze_cross_source_disagreement(symbol, source_data)
            return disagreement
            
        except Exception as e:
            logger.error(f"Error analyzing cross-source data for {symbol}: {e}")
            return None
    
    def _calculate_price_momentum(self, market_data: Dict) -> float:
        """Calculate price momentum indicator."""
        if not market_data or 'price_data' not in market_data:
            return 0.0
        
        try:
            price_df = market_data['price_data']
            if len(price_df) < 5:
                return 0.0
            
            # Calculate momentum over different timeframes
            short_momentum = (price_df['close'].iloc[-1] - price_df['close'].iloc[-3]) / price_df['close'].iloc[-3]
            long_momentum = (price_df['close'].iloc[-1] - price_df['close'].iloc[-5]) / price_df['close'].iloc[-5]
            
            # Combined momentum score
            momentum_score = (short_momentum * 0.7 + long_momentum * 0.3)
            return float(np.tanh(momentum_score * 10))  # Scale to [-1, 1]
            
        except Exception as e:
            logger.error(f"Error calculating price momentum: {e}")
            return 0.0
    
    def _analyze_volume_profile(self, market_data: Dict) -> Dict[str, float]:
        """Analyze volume profile and patterns."""
        if not market_data or 'volume_data' not in market_data:
            return {'average_volume': 0.0, 'volume_trend': 0.0, 'volume_spike': 0.0}
        
        try:
            volumes = market_data['volume_data']
            if len(volumes) < 5:
                return {'average_volume': 0.0, 'volume_trend': 0.0, 'volume_spike': 0.0}
            
            volumes = np.array(volumes[-10:])  # Last 10 periods
            avg_volume = np.mean(volumes)
            
            # Volume trend (increasing/decreasing)
            if len(volumes) >= 5:
                recent_avg = np.mean(volumes[-3:])
                older_avg = np.mean(volumes[-5:-2])
                volume_trend = (recent_avg - older_avg) / max(older_avg, 1)
            else:
                volume_trend = 0.0
            
            # Volume spike detection
            if len(volumes) >= 2:
                latest_volume = volumes[-1]
                avg_recent = np.mean(volumes[:-1])
                volume_spike = (latest_volume - avg_recent) / max(avg_recent, 1)
            else:
                volume_spike = 0.0
            
            return {
                'average_volume': float(avg_volume),
                'volume_trend': float(np.tanh(volume_trend)),
                'volume_spike': float(np.tanh(volume_spike))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume profile: {e}")
            return {'average_volume': 0.0, 'volume_trend': 0.0, 'volume_spike': 0.0}
    
    def _calculate_technical_indicators(self, market_data: Dict) -> Dict[str, float]:
        """Calculate basic technical indicators."""
        if not market_data or 'price_data' not in market_data:
            return {'rsi': 50.0, 'price_vs_sma': 0.0, 'volatility': 0.0}
        
        try:
            price_df = market_data['price_data']
            if len(price_df) < 14:
                return {'rsi': 50.0, 'price_vs_sma': 0.0, 'volatility': 0.0}
            
            closes = price_df['close'].values
            
            # Simple RSI calculation
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            if len(gains) >= 14:
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                rs = avg_gain / max(avg_loss, 0.001)
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50.0
            
            # Price vs SMA
            if len(closes) >= 10:
                sma_10 = np.mean(closes[-10:])
                price_vs_sma = (closes[-1] - sma_10) / sma_10
            else:
                price_vs_sma = 0.0
            
            # Volatility (std of returns)
            if len(closes) >= 10:
                returns = np.diff(closes) / closes[:-1]
                volatility = np.std(returns[-10:])
            else:
                volatility = 0.0
            
            return {
                'rsi': float(rsi),
                'price_vs_sma': float(price_vs_sma),
                'volatility': float(volatility)
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {'rsi': 50.0, 'price_vs_sma': 0.0, 'volatility': 0.0}
    
    def _assess_liquidity(self, market_data: Dict) -> float:
        """Assess liquidity for short selling feasibility."""
        if not market_data or 'volume_data' not in market_data:
            return 0.5  # Neutral liquidity score
        
        try:
            volumes = market_data['volume_data']
            if len(volumes) < 5:
                return 0.5
            
            avg_volume = np.mean(volumes[-5:])
            
            # Simple heuristic: higher volume = higher liquidity
            # Scale to 0-1 range
            if avg_volume > 1000000:  # High volume
                return 0.9
            elif avg_volume > 100000:  # Medium volume
                return 0.7
            elif avg_volume > 10000:  # Low volume
                return 0.4
            else:  # Very low volume
                return 0.2
            
        except Exception as e:
            logger.error(f"Error assessing liquidity: {e}")
            return 0.5
    
    def _synthesize_short_signal(self, symbol: str, entity_sentiment: Optional[EntitySentiment],
                               sentiment_velocity: Optional[SentimentVelocity],
                               topic_surprise: Optional[TopicSurprise],
                               cross_source_disagreement: Optional[CrossSourceDisagreement],
                               price_momentum: float, volume_profile: Dict[str, float],
                               technical_indicators: Dict[str, float],
                               liquidity_score: float) -> Optional[EnhancedShortSignal]:
        """Synthesize all analysis components into a final short signal."""
        try:
            # Initialize signal strength and confidence
            signal_strength = 0.0
            confidence = 0.0
            reasoning_parts = []
            
            # Entity sentiment contribution
            if entity_sentiment and entity_sentiment.mentions > 0:
                sentiment_score = entity_sentiment.sentiment_score
                conviction = entity_sentiment.conviction_score
                
                # Negative sentiment increases short signal strength
                if sentiment_score < -0.2:
                    sentiment_contribution = abs(sentiment_score) * conviction * 0.3
                    signal_strength += sentiment_contribution
                    reasoning_parts.append(f"Negative sentiment ({sentiment_score:.2f}) with {conviction:.2f} conviction")
                
                confidence += conviction * 0.2
            
            # Sentiment velocity contribution
            if sentiment_velocity:
                velocity = sentiment_velocity.velocity
                trend_strength = sentiment_velocity.trend_strength
                
                # Negative velocity (deteriorating sentiment) favors shorts
                if velocity < -0.1:
                    velocity_contribution = abs(velocity) * trend_strength * 0.25
                    signal_strength += velocity_contribution
                    reasoning_parts.append(f"Negative sentiment velocity ({velocity:.3f}/hour)")
                
                # High reversal probability from positive extreme
                if sentiment_velocity.reversal_probability > 0.5 and sentiment_velocity.current_sentiment > 0.5:
                    reversal_contribution = sentiment_velocity.reversal_probability * 0.2
                    signal_strength += reversal_contribution
                    reasoning_parts.append(f"High reversal probability ({sentiment_velocity.reversal_probability:.2f})")
                
                confidence += trend_strength * 0.15
            
            # Topic surprise contribution
            if topic_surprise and topic_surprise.surprise_score > 0.3:
                surprise_contribution = topic_surprise.surprise_score * 0.2
                signal_strength += surprise_contribution
                reasoning_parts.append(f"Topic surprise detected ({topic_surprise.surprise_score:.2f})")
                
                if 'risk' in topic_surprise.emerging_topics:
                    signal_strength += 0.1
                    reasoning_parts.append("Emerging risk topics")
                
                confidence += topic_surprise.surprise_score * 0.1
            
            # Cross-source disagreement contribution
            if cross_source_disagreement:
                disagreement = cross_source_disagreement.disagreement_score
                consensus_strength = cross_source_disagreement.consensus_strength
                
                # High disagreement with negative consensus can indicate short opportunity
                source_sentiments = list(cross_source_disagreement.source_sentiments.values())
                avg_sentiment = np.mean(source_sentiments) if source_sentiments else 0.0
                
                if disagreement > 0.2 and avg_sentiment < -0.05:  # LOWERED thresholds for more sensitivity
                    disagreement_contribution = disagreement * abs(avg_sentiment) * 0.15
                    signal_strength += disagreement_contribution
                    reasoning_parts.append(f"Cross-source disagreement ({disagreement:.2f}) with negative consensus")
                
                confidence += consensus_strength * 0.1
            
            # Price momentum contribution (contrarian) - LOWERED threshold
            if price_momentum > 0.15:  # Moderate positive momentum may be overextended
                momentum_contribution = price_momentum * 0.2
                signal_strength += momentum_contribution
                reasoning_parts.append(f"Overextended positive momentum ({price_momentum:.2f})")
            
            # Technical indicators contribution
            rsi = technical_indicators.get('rsi', 50.0)
            if rsi > 70:  # Overbought condition
                rsi_contribution = (rsi - 70) / 30 * 0.2
                signal_strength += rsi_contribution
                reasoning_parts.append(f"Overbought RSI ({rsi:.1f})")
            
            price_vs_sma = technical_indicators.get('price_vs_sma', 0.0)
            if price_vs_sma > 0.1:  # Price significantly above moving average
                sma_contribution = price_vs_sma * 0.15
                signal_strength += sma_contribution
                reasoning_parts.append(f"Price above SMA ({price_vs_sma:.1%})")
            
            # Volume analysis - LOWERED threshold for more sensitivity
            volume_spike = volume_profile.get('volume_spike', 0.0)
            if volume_spike > 0.3:  # Moderate volume spike might indicate distribution
                volume_contribution = volume_spike * 0.1
                signal_strength += volume_contribution
                reasoning_parts.append(f"Volume spike ({volume_spike:.2f})")
            
            # Liquidity requirement - RELAXED for more opportunities  
            if liquidity_score < 0.2:  # Only penalize very illiquid stocks
                signal_strength *= 0.7  # Less severe penalty (was 0.5)
                reasoning_parts.append("Low liquidity discount applied")
            
            confidence += liquidity_score * 0.1
            
            # INTELLIGENT CONFIDENCE BOOST: When we have strong technical signals but weak social data,
            # boost confidence based on technical indicators to enable more short opportunities
            technical_confidence_boost = 0.0
            
            # Strong RSI signal boosts confidence
            if rsi > 75:
                technical_confidence_boost += 0.15
            elif rsi > 70:
                technical_confidence_boost += 0.08
            
            # Price momentum vs moving average boosts confidence
            if price_vs_sma > 0.15:  # Significantly overextended
                technical_confidence_boost += 0.12
            elif price_vs_sma > 0.10:
                technical_confidence_boost += 0.06
            
            # Volume confirmation boosts confidence
            if volume_spike > 0.5:
                technical_confidence_boost += 0.10
            elif volume_spike > 0.3:
                technical_confidence_boost += 0.05
            
            # Apply technical confidence boost
            confidence += technical_confidence_boost
            
            # Base confidence floor for stocks with any bearish signals
            if signal_strength > 0.05:
                confidence = max(confidence, 0.12)  # Minimum confidence for any meaningful signal
            
            # Normalize and cap values
            signal_strength = min(signal_strength, 1.0)
            confidence = min(confidence, 1.0)
            
            # Determine signal type - INTELLIGENT BALANCED THRESHOLDS
            # Lower signal strength but higher confidence for quality shorts
            if signal_strength >= 0.12 and confidence >= 0.20:  # High quality shorts
                signal_type = "SHORT"
            elif signal_strength >= 0.06 and confidence >= 0.15:  # Medium quality shorts  
                signal_type = "WEAK_SHORT"
            elif signal_strength >= 0.03:
                signal_type = "MONITOR"
            else:
                signal_type = "AVOID"
            
            # Create reasoning string
            reasoning = f"Enhanced short analysis: {'; '.join(reasoning_parts)}"
            
            # Create the final signal
            signal = EnhancedShortSignal(
                symbol=symbol,
                signal_strength=signal_strength,
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                entity_sentiment=entity_sentiment,
                sentiment_velocity=sentiment_velocity,
                topic_surprise=topic_surprise,
                cross_source_disagreement=cross_source_disagreement,
                price_momentum=price_momentum,
                volume_profile=volume_profile,
                technical_indicators=technical_indicators,
                borrow_availability=None,  # Will be filled by borrow cost monitor
                squeeze_risk=None,  # Will be filled by risk manager
                liquidity_score=liquidity_score,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=1)
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error synthesizing short signal for {symbol}: {e}")
            return None
    
    def _quick_sentiment_score(self, text: str) -> float:
        """Quick sentiment scoring for text."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'bullish', 'buy', 'moon', 'rocket', 'surge']
        negative_words = ['bad', 'terrible', 'bearish', 'sell', 'crash', 'dump', 'tank', 'overvalued']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_count

# Global instance
enhanced_short_signal_engine = EnhancedShortSignalEngine()

# Status method for the engine
def get_signal_engine_status() -> Dict[str, Any]:
    """Get enhanced short signal engine status."""
    try:
        return {
            'engine_initialized': True,
            'lookback_days': getattr(enhanced_short_signal_engine, 'lookback_days', 30),
            'sentiment_history_size': len(getattr(enhanced_short_signal_engine, 'sentiment_history', {})),
            'topic_baseline_symbols': len(getattr(enhanced_short_signal_engine, 'topic_baseline', {})),
            'meme_keywords_count': len(getattr(enhanced_short_signal_engine, 'meme_keywords', []))
        }
    except Exception as e:
        return {
            'engine_initialized': True,
            'status': 'operational',
            'error': str(e)
        }

# Add status method to the engine class
EnhancedShortSignalEngine.get_signal_engine_status = lambda self: get_signal_engine_status()

# Convenience functions
async def generate_enhanced_short_signals(symbols: List[str], max_signals: int = 10) -> List[EnhancedShortSignal]:
    """Generate enhanced short signals for given symbols."""
    return await enhanced_short_signal_engine.generate_enhanced_short_signals(symbols, max_signals)

def get_signal_engine_status() -> Dict[str, Any]:
    """Get signal engine status and metrics."""
    return {
        'cache_size': len(enhanced_short_signal_engine.signal_cache),
        'velocity_tracked_symbols': len(enhanced_short_signal_engine.velocity_tracker.sentiment_history),
        'baseline_topics_count': len(enhanced_short_signal_engine.topic_detector.baseline_topics),
        'engine_initialized': True
    }