"""
Test script for the refactored LLM-based sentiment analysis system.

Tests all components:
- LLM sentiment analyzer
- Social media collectors (Reddit, Twitter, TikTok)  
- Earnings call scraper
- Comprehensive sentiment agent
- Integration with market intelligence
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add the trading_system directory to the Python path
sys.path.append('/Users/zac/Desktop/ML4T/trading_system')

from core.llm_sentiment_analyzer import LLMSentimentAnalyzer
from core.social_media_collector import SocialMediaCollector
from core.earnings_scraper import EarningsCallScraper
from agents.sentiment_agent import sentiment_agent
from core.market_intelligence import UnifiedMarketIntelligence
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_llm_sentiment_analyzer():
    """Test the LLM sentiment analyzer with sample financial text."""
    logger.info("=== Testing LLM Sentiment Analyzer ===")
    
    analyzer = LLMSentimentAnalyzer(
        openai_api_key=settings.openai_api_key,
        ollama_base_url=settings.ollama_base_url
    )
    
    # Test with sample financial news text
    sample_news = """
    Apple Inc. reported record quarterly revenue of $123.9 billion, beating analyst expectations 
    by $2.1 billion. The company's iPhone sales exceeded forecasts, and management expressed 
    confidence in continued growth despite economic headwinds. CEO Tim Cook highlighted strong 
    performance in emerging markets and robust demand for their latest products.
    """
    
    try:
        result = await analyzer.analyze_text(sample_news, "financial_news")
        logger.info(f"Sentiment: {result.sentiment}")
        logger.info(f"Score: {result.score:.3f}")
        logger.info(f"Confidence: {result.confidence:.3f}")
        logger.info(f"Reasoning: {result.reasoning}")
        logger.info(f"Key phrases: {result.key_phrases}")
        logger.info(f"Financial impact: {result.financial_impact}")
        return True
    except Exception as e:
        logger.error(f"LLM sentiment analyzer test failed: {e}")
        return False
    finally:
        await analyzer.close()


async def test_social_media_collector():
    """Test social media data collection."""
    logger.info("=== Testing Social Media Collector ===")
    
    collector = SocialMediaCollector()
    test_symbol = "AAPL"
    
    try:
        # Test collecting from all platforms
        results = await collector.collect_all_platforms(test_symbol, limit_per_platform=5)
        
        for platform, posts in results.items():
            logger.info(f"{platform.capitalize()}: {len(posts)} posts collected")
            if posts:
                sample_post = posts[0]
                logger.info(f"  Sample post: {sample_post.content[:100]}...")
        
        # Test platform statistics
        stats = collector.get_platform_stats(results)
        logger.info(f"Platform stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Social media collector test failed: {e}")
        return False


async def test_earnings_scraper():
    """Test earnings call transcript scraping."""
    logger.info("=== Testing Earnings Scraper ===")
    
    scraper = EarningsCallScraper()
    test_symbol = "AAPL"
    
    try:
        # Test searching for transcripts
        transcripts = await scraper.search_transcripts(test_symbol, days_back=120)
        logger.info(f"Found {len(transcripts)} earnings transcripts for {test_symbol}")
        
        if transcripts:
            logger.info(f"Latest transcript: {transcripts[0]['title']}")
            logger.info(f"Date: {transcripts[0]['date']}")
            
            # Test scraping the latest transcript
            latest = await scraper.scrape_transcript(transcripts[0]['url'], test_symbol)
            if latest:
                logger.info(f"Scraped transcript: {latest.company_name} {latest.quarter} {latest.year}")
                logger.info(f"Content length: {len(latest.full_text)} characters")
                logger.info(f"Key metrics: {latest.key_metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"Earnings scraper test failed: {e}")
        return False
    finally:
        await scraper.close()


async def test_comprehensive_sentiment_agent():
    """Test the comprehensive sentiment agent."""
    logger.info("=== Testing Comprehensive Sentiment Agent ===")
    
    test_symbol = "AAPL"
    
    try:
        # Test comprehensive sentiment analysis
        result = await sentiment_agent.analyze_comprehensive_sentiment(test_symbol)
        
        logger.info(f"Symbol: {result.symbol}")
        logger.info(f"Overall sentiment: {result.overall_sentiment}")
        logger.info(f"Overall score: {result.overall_score:.3f}")
        logger.info(f"Confidence: {result.confidence:.3f}")
        logger.info(f"Data sources: {result.data_sources_count}")
        logger.info(f"News articles: {result.news_articles_count}")
        logger.info(f"Social posts: {result.social_posts_count}")
        logger.info(f"Has recent earnings: {result.has_recent_earnings}")
        
        if result.key_themes:
            logger.info(f"Key themes: {result.key_themes[:3]}")
        if result.risk_factors:
            logger.info(f"Risk factors: {result.risk_factors[:2]}")
        if result.opportunities:
            logger.info(f"Opportunities: {result.opportunities[:2]}")
        
        # Test sentiment by source
        if result.news_sentiment:
            logger.info(f"News sentiment: {result.news_sentiment.sentiment} ({result.news_sentiment.score:.3f})")
        
        for platform, sentiment in result.social_sentiment.items():
            logger.info(f"{platform.capitalize()} sentiment: {sentiment.sentiment} ({sentiment.score:.3f})")
        
        for section, sentiment in result.earnings_sentiment.items():
            logger.info(f"Earnings {section} sentiment: {sentiment.sentiment} ({sentiment.score:.3f})")
        
        if result.market_sentiment:
            logger.info(f"Market sentiment: {result.market_sentiment.sentiment} ({result.market_sentiment.score:.3f})")
        
        return True
        
    except Exception as e:
        logger.error(f"Comprehensive sentiment agent test failed: {e}")
        return False


async def test_market_intelligence_integration():
    """Test integration with market intelligence system."""
    logger.info("=== Testing Market Intelligence Integration ===")
    
    market_intel = UnifiedMarketIntelligence()
    test_symbol = "AAPL"
    
    try:
        # Test the refactored sentiment analysis
        sentiment_score = await market_intel._analyze_sentiment(test_symbol)
        
        if sentiment_score is not None:
            logger.info(f"Market intelligence sentiment score for {test_symbol}: {sentiment_score:.3f}")
            
            # Test full market analysis
            analysis = await market_intel.analyze_stock(test_symbol)
            if analysis:
                logger.info(f"Overall analysis score: {analysis.score:.3f}")
                logger.info(f"Signal: {analysis.signal}")
                logger.info(f"Confidence: {analysis.confidence}")
                
                if hasattr(analysis, 'components') and analysis.components:
                    sentiment_component = analysis.components.get('sentiment')
                    if sentiment_component:
                        logger.info(f"Sentiment component: {sentiment_component:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Market intelligence integration test failed: {e}")
        return False


async def test_configuration():
    """Test system configuration."""
    logger.info("=== Testing Configuration ===")
    
    try:
        # Check API keys
        config_status = {
            'OpenAI API Key': bool(settings.openai_api_key),
            'News API Key': bool(settings.news_api_key),
            'Reddit Client ID': bool(settings.reddit_client_id),
            'Twitter Bearer Token': bool(settings.twitter_bearer_token),
            'Ollama Base URL': bool(settings.ollama_base_url),
        }
        
        for service, available in config_status.items():
            status = "✓ Available" if available else "✗ Missing"
            logger.info(f"{service}: {status}")
        
        # Check if we have at least one LLM option
        has_llm = settings.openai_api_key or settings.use_llama_fallback
        logger.info(f"LLM Available: {'✓ Yes' if has_llm else '✗ No'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("Starting Sentiment Analysis System Tests")
    logger.info("=" * 50)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Configuration", test_configuration),
        ("LLM Sentiment Analyzer", test_llm_sentiment_analyzer),
        ("Social Media Collector", test_social_media_collector),
        ("Earnings Scraper", test_earnings_scraper),
        ("Comprehensive Sentiment Agent", test_comprehensive_sentiment_agent),
        ("Market Intelligence Integration", test_market_intelligence_integration),
    ]
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\nStarting {test_name} test...")
            result = await test_func()
            test_results[test_name] = result
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test_name} test {status}")
        except Exception as e:
            logger.error(f"{test_name} test ERROR: {e}")
            test_results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Sentiment analysis system is ready.")
    else:
        logger.warning("⚠️  Some tests failed. Check configuration and dependencies.")
    
    # Cleanup
    await sentiment_agent.close()


if __name__ == "__main__":
    asyncio.run(main())