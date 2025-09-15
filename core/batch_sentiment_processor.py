"""
Batch Sentiment Processor for Trading System

Efficiently processes sentiment analysis for multiple stocks by:
1. Collecting news and social media data for all stocks concurrently
2. Batching text content for LLM analysis 
3. Processing in chunks to optimize API usage and costs
4. Maintaining individual stock context while maximizing efficiency
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import json
import time
from collections import defaultdict

from agents.sentiment_agent import ComprehensiveSentiment, SentimentAgent
from core.llm_sentiment_analyzer import LLMSentimentAnalyzer, SentimentAnalysis
from core.gpt_batch_sentiment_analyzer import GPTBatchSentimentAnalyzer
from core.social_media_collector_optimized import SocialMediaCollector
from core.market_intelligence import UnifiedMarketIntelligence

logger = logging.getLogger(__name__)

@dataclass
class BatchedContent:
    """Content collected for a stock for batch processing."""
    symbol: str
    news_content: List[str] = None
    social_content: Dict[str, List[str]] = None  # platform -> content list
    sec_content: List[str] = None
    earnings_content: List[str] = None
    
    def __post_init__(self):
        if self.news_content is None:
            self.news_content = []
        if self.social_content is None:
            self.social_content = {'reddit': [], 'twitter': []}
        if self.sec_content is None:
            self.sec_content = []
        if self.earnings_content is None:
            self.earnings_content = []
    
    def has_content(self) -> bool:
        """Check if there's any content to analyze."""
        # Accept any content source, even minimal data
        has_news = len(self.news_content) > 0
        has_social = any(len(posts) > 0 for posts in self.social_content.values())
        has_sec = len(self.sec_content) > 0
        has_earnings = len(self.earnings_content) > 0
        
        # Return True only if we have actual content
        return has_news or has_social or has_sec or has_earnings
    
    def get_combined_content(self, content_type: str = "all") -> str:
        """Get combined content for analysis."""
        content_parts = []
        
        if content_type in ["all", "news"] and self.news_content:
            content_parts.extend(self.news_content[:3])  # Limit to top 3 news
        
        if content_type in ["all", "social"] and self.social_content:
            for platform, posts in self.social_content.items():
                content_parts.extend(posts[:2])  # Limit to top 2 per platform
        
        if content_type in ["all", "sec"] and self.sec_content:
            content_parts.extend(self.sec_content[:2])  # Limit to top 2 SEC filings
        
        if content_type in ["all", "earnings"] and self.earnings_content:
            content_parts.extend(self.earnings_content[:1])  # Limit to 1 earnings call
        
        return "\n".join(content_parts)


class BatchSentimentProcessor:
    """
    Efficient batch processor for multiple stock sentiment analysis.
    
    Key optimizations:
    1. Concurrent data collection for all stocks
    2. Intelligent batching of LLM requests
    3. Fallback for failed individual requests
    4. Smart caching and rate limiting
    """
    
    def __init__(self):
        self.sentiment_agent = SentimentAgent()
        self.llm_analyzer = LLMSentimentAnalyzer()
        self.social_collector = SocialMediaCollector()
        self.market_intel = UnifiedMarketIntelligence()
        
        # Batch processing settings
        self.max_batch_size = 15  # Optimal for GPT-4o-mini
        self.max_concurrent_data_collection = 10  # Prevent overwhelming APIs
        self.max_content_length_per_stock = 2000  # Prevent token limit issues
        
        logger.info("✅ Batch Sentiment Processor initialized")
    
    async def analyze_stocks_batch(self, 
                                 symbols: List[str],
                                 timeout_seconds: float = 300.0) -> Dict[str, ComprehensiveSentiment]:
        """
        Analyze sentiment for multiple stocks efficiently using batching.
        
        Args:
            symbols: List of stock symbols to analyze
            timeout_seconds: Maximum time to wait for all analysis
            
        Returns:
            Dictionary mapping symbols to their sentiment analysis
        """
        if not symbols:
            return {}
        
        logger.info(f"🔄 Starting batch sentiment analysis for {len(symbols)} stocks")
        start_time = time.time()
        
        try:
            # Step 1: Collect data for all stocks concurrently
            logger.info("📊 Step 1: Collecting data for all stocks...")
            stock_data = await asyncio.wait_for(
                self._collect_data_for_stocks(symbols),
                timeout=timeout_seconds * 0.6  # 60% of total time for data collection
            )
            
            # Step 2: Process sentiment in optimized batches
            logger.info("🤖 Step 2: Processing sentiment in batches...")
            remaining_time = timeout_seconds - (time.time() - start_time)
            sentiment_results = await asyncio.wait_for(
                self._process_sentiment_batches(stock_data),
                timeout=max(30.0, remaining_time * 0.8)  # Reserve 20% buffer
            )
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Batch sentiment analysis completed in {processing_time:.2f}s")
            logger.info(f"   📈 Success rate: {len(sentiment_results)}/{len(symbols)} stocks")
            logger.info(f"   ⚡ Average: {processing_time/len(symbols):.2f}s per stock (vs ~4s individual)")
            
            return sentiment_results
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ Batch sentiment analysis timed out after {timeout_seconds}s")
            # Return partial results if any
            return {}
        except Exception as e:
            logger.error(f"❌ Batch sentiment analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    async def _collect_data_for_stocks(self, symbols: List[str]) -> Dict[str, BatchedContent]:
        """Collect all data for stocks concurrently."""
        
        # Create semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(self.max_concurrent_data_collection)
        
        async def collect_for_symbol(symbol: str) -> Tuple[str, BatchedContent]:
            async with semaphore:
                return symbol, await self._collect_single_stock_data(symbol)
        
        # Collect data for all stocks concurrently
        tasks = [collect_for_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        stock_data = {}
        successful_collections = 0
        
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Data collection failed for a symbol: {result}")
                continue
                
            symbol, batched_content = result
            if batched_content.has_content():
                stock_data[symbol] = batched_content
                successful_collections += 1
            else:
                logger.warning(f"No content collected for {symbol}")
        
        logger.info(f"📊 Data collection: {successful_collections}/{len(symbols)} stocks have content")
        return stock_data
    
    async def _collect_single_stock_data(self, symbol: str) -> BatchedContent:
        """Collect data for a single stock with optimized timeouts."""
        
        batched_content = BatchedContent(symbol=symbol)
        
        # Collect data from different sources concurrently with reasonable timeouts
        tasks = [
            self._collect_news_data(symbol, timeout=15.0),
            self._collect_social_data(symbol, timeout=20.0),
            self._collect_sec_data(symbol, timeout=25.0),
            self._collect_earnings_data(symbol, timeout=10.0)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        news_data, social_data, sec_data, earnings_data = results
        
        if not isinstance(news_data, Exception) and news_data:
            batched_content.news_content = news_data
        
        if not isinstance(social_data, Exception) and social_data:
            batched_content.social_content = social_data
        
        if not isinstance(sec_data, Exception) and sec_data:
            batched_content.sec_content = sec_data
        
        if not isinstance(earnings_data, Exception) and earnings_data:
            batched_content.earnings_content = earnings_data
        
        return batched_content
    
    async def _collect_news_data(self, symbol: str, timeout: float) -> List[str]:
        """Collect news data for a symbol."""
        try:
            # Use existing news collection from market intelligence
            news_content = await asyncio.wait_for(
                self._get_news_articles_text(symbol),
                timeout=timeout
            )
            return news_content[:3]  # Top 3 news articles
        except Exception as e:
            logger.debug(f"News collection failed for {symbol}: {e}")
            return []
    
    async def _collect_social_data(self, symbol: str, timeout: float) -> Dict[str, List[str]]:
        """Collect social media data for a symbol."""
        try:
            social_results = await asyncio.wait_for(
                self.social_collector.collect_all_platforms(symbol, limit_per_platform=3),
                timeout=timeout
            )
            
            social_content = {'reddit': [], 'twitter': []}
            
            for platform, posts in social_results.items():
                if posts:
                    social_content[platform] = [post.content for post in posts[:3]]
            
            return social_content
        except Exception as e:
            logger.debug(f"Social media collection failed for {symbol}: {e}")
            return {'reddit': [], 'twitter': []}
    
    async def _collect_sec_data(self, symbol: str, timeout: float) -> List[str]:
        """Collect SEC filing data for a symbol."""
        try:
            # Use sentiment agent's SEC collection method
            sec_results = await asyncio.wait_for(
                self.sentiment_agent._analyze_sec_filings_sentiment(symbol),
                timeout=timeout
            )
            
            if sec_results:
                return [f"SEC Filing analysis: {result.reasoning}" for result in sec_results.values()][:2]
            return []
        except Exception as e:
            logger.debug(f"SEC data collection failed for {symbol}: {e}")
            return []
    
    async def _collect_earnings_data(self, symbol: str, timeout: float) -> List[str]:
        """Collect earnings call data for a symbol."""
        try:
            earnings_results = await asyncio.wait_for(
                self.sentiment_agent._get_recent_earnings(symbol),
                timeout=timeout
            )
            
            if earnings_results:
                return [f"Earnings call summary: {earnings_results}"]
            return []
        except Exception as e:
            logger.debug(f"Earnings collection failed for {symbol}: {e}")
            return []
    
    async def _get_news_articles_text(self, symbol: str) -> List[str]:
        """Get news articles text for a symbol."""
        try:
            articles = await self.sentiment_agent._get_news_articles(symbol, self.market_intel)
            news_texts = []
            
            for article in articles[:3]:  # Top 3 articles
                title = article.get('title', '')
                description = article.get('description', '')
                if title or description:
                    news_texts.append(f"{title}. {description}")
            
            return news_texts
        except Exception as e:
            logger.debug(f"News text collection failed for {symbol}: {e}")
            return []
    
    async def _process_sentiment_batches(self, stock_data: Dict[str, BatchedContent]) -> Dict[str, ComprehensiveSentiment]:
        """Process sentiment analysis in optimized batches."""
        
        if not stock_data:
            return {}
        
        # Group stocks by content types for optimal batching
        content_groups = self._group_stocks_by_content(stock_data)
        
        # Process each group
        all_results = {}
        
        for group_name, stocks_in_group in content_groups.items():
            logger.info(f"🔄 Processing {group_name} group: {len(stocks_in_group)} stocks")
            
            group_results = await self._process_content_group(stocks_in_group, stock_data)
            all_results.update(group_results)
        
        return all_results
    
    def _group_stocks_by_content(self, stock_data: Dict[str, BatchedContent]) -> Dict[str, List[str]]:
        """Group stocks by their content types for optimal batching."""
        
        groups = {
            'news_heavy': [],      # Stocks with primarily news content
            'social_heavy': [],    # Stocks with primarily social content  
            'mixed_content': [],   # Stocks with mixed content types
            'minimal_content': []  # Stocks with very little content
        }
        
        for symbol, content in stock_data.items():
            news_count = len(content.news_content)
            social_count = sum(len(posts) for posts in content.social_content.values())
            sec_count = len(content.sec_content)
            earnings_count = len(content.earnings_content)
            
            total_sources = sum(1 for count in [news_count, social_count, sec_count, earnings_count] if count > 0)
            
            if total_sources <= 1:
                groups['minimal_content'].append(symbol)
            elif news_count >= 2 and social_count <= 1:
                groups['news_heavy'].append(symbol)
            elif social_count >= 3 and news_count <= 1:
                groups['social_heavy'].append(symbol)
            else:
                groups['mixed_content'].append(symbol)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    async def _process_content_group(self, symbols: List[str], stock_data: Dict[str, BatchedContent]) -> Dict[str, ComprehensiveSentiment]:
        """Process a group of stocks with similar content patterns."""
        
        # Split into batches for LLM processing
        results = {}
        
        for i in range(0, len(symbols), self.max_batch_size):
            batch_symbols = symbols[i:i + self.max_batch_size]
            
            batch_results = await self._process_single_batch(batch_symbols, stock_data)
            results.update(batch_results)
            
            # Brief pause between batches to be respectful to APIs
            if i + self.max_batch_size < len(symbols):
                await asyncio.sleep(0.5)
        
        return results
    
    async def _process_single_batch(self, symbols: List[str], stock_data: Dict[str, BatchedContent]) -> Dict[str, ComprehensiveSentiment]:
        """Process a single batch of stocks for sentiment analysis."""
        
        # Prepare batch content for LLM
        batch_texts = []
        symbol_mapping = []  # Track which text belongs to which symbol
        
        for symbol in symbols:
            content = stock_data.get(symbol)
            if not content:
                continue
            
            # Combine content for this symbol
            combined_content = content.get_combined_content()
            if not combined_content:
                continue
            
            # Truncate if too long
            if len(combined_content) > self.max_content_length_per_stock:
                combined_content = combined_content[:self.max_content_length_per_stock] + "... [truncated]"
            
            # Create batch text entry
            batch_text = f"Stock: {symbol}\nContent: {combined_content}"
            batch_texts.append(batch_text)
            symbol_mapping.append(symbol)
        
        if not batch_texts:
            return {}
        
        # Process with LLM analyzer in batch
        try:
            # Use the GPT batch analyzer for efficiency  
            batch_analyzer = GPTBatchSentimentAnalyzer()
            
            # For now, process concurrently (could be enhanced with true batching)
            sentiment_tasks = [
                self.llm_analyzer.analyze_text(text, "financial_news")
                for text in batch_texts
            ]
            
            sentiment_results = await asyncio.gather(*sentiment_tasks, return_exceptions=True)
            
            # Convert to ComprehensiveSentiment format
            batch_results = {}
            
            for symbol, sentiment_result in zip(symbol_mapping, sentiment_results):
                if isinstance(sentiment_result, Exception):
                    logger.warning(f"Sentiment analysis failed for {symbol}: {sentiment_result}")
                    continue
                
                # Convert to ComprehensiveSentiment
                comprehensive = await self._create_comprehensive_sentiment(
                    symbol, sentiment_result, stock_data.get(symbol)
                )
                batch_results[symbol] = comprehensive
            
            logger.info(f"✅ Batch processed: {len(batch_results)}/{len(symbols)} successful")
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fallback to individual processing
            return await self._fallback_individual_processing(symbols, stock_data)
    
    async def _create_comprehensive_sentiment(self, 
                                            symbol: str, 
                                            base_sentiment: SentimentAnalysis,
                                            content: BatchedContent) -> ComprehensiveSentiment:
        """Create a ComprehensiveSentiment from base sentiment and content data."""
        
        # Count data sources
        news_count = len(content.news_content) if content.news_content else 0
        social_count = sum(len(posts) for posts in content.social_content.values()) if content.social_content else 0
        sec_count = len(content.sec_content) if content.sec_content else 0
        earnings_count = len(content.earnings_content) if content.earnings_content else 0
        
        data_sources_count = sum(1 for count in [news_count, social_count, sec_count, earnings_count] if count > 0)
        
        # Create social sentiment breakdown
        social_sentiment = {}
        if content.social_content:
            for platform, posts in content.social_content.items():
                if posts:
                    # Create a simplified sentiment for each platform
                    platform_sentiment = SentimentAnalysis(
                        sentiment=base_sentiment.sentiment,
                        confidence=base_sentiment.confidence * 0.8,  # Slightly lower confidence
                        score=base_sentiment.score,
                        reasoning=f"Social media sentiment from {platform}",
                        key_phrases=base_sentiment.key_phrases[:2],
                        financial_impact=base_sentiment.financial_impact
                    )
                    social_sentiment[platform] = platform_sentiment
        
        # Calculate overall confidence adjustment
        confidence_adjustment = min(1.0, data_sources_count / 4.0)  # Max confidence with 4+ sources
        adjusted_confidence = base_sentiment.confidence * confidence_adjustment
        
        return ComprehensiveSentiment(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_sentiment=base_sentiment.sentiment.value,
            overall_score=base_sentiment.score,
            confidence=adjusted_confidence,
            
            # Source breakdown  
            news_sentiment=base_sentiment if news_count > 0 else None,
            social_sentiment=social_sentiment,
            earnings_sentiment={} if earnings_count > 0 else None,
            sec_filings_sentiment={} if sec_count > 0 else None,
            market_sentiment=None,
            
            # Metadata
            data_sources_count=data_sources_count,
            news_articles_count=news_count,
            social_posts_count=social_count,
            sec_filings_count=sec_count,
            has_recent_earnings=earnings_count > 0,
            has_recent_sec_filings=sec_count > 0,
            
            # Insights
            key_themes=base_sentiment.key_phrases,
            risk_factors=base_sentiment.risk_factors,
            opportunities=base_sentiment.opportunities
        )
    
    async def _fallback_individual_processing(self, symbols: List[str], stock_data: Dict[str, BatchedContent]) -> Dict[str, ComprehensiveSentiment]:
        """Fallback to individual processing if batch fails."""
        logger.warning("🔄 Falling back to individual sentiment processing")
        
        results = {}
        for symbol in symbols:
            try:
                # Use the original comprehensive sentiment analysis
                result = await asyncio.wait_for(
                    self.sentiment_agent.analyze_comprehensive_sentiment(symbol),
                    timeout=30.0
                )
                if result:
                    results[symbol] = result
            except Exception as e:
                logger.warning(f"Individual fallback failed for {symbol}: {e}")
        
        return results


# Global instance for efficiency
batch_sentiment_processor = BatchSentimentProcessor()