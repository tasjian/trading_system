"""
Basic sentiment analysis system test - focuses on core LLM functionality.
"""

import asyncio
import logging
import os
import sys

# Add the trading_system directory to the Python path
sys.path.append('/Users/zac/Desktop/ML4T/trading_system')

from core.llm_sentiment_analyzer import LLMSentimentAnalyzer
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
        logger.info("Analyzing sample Apple earnings news...")
        result = await analyzer.analyze_text(sample_news, "financial_news")
        
        logger.info(f"✅ Sentiment: {result.sentiment}")
        logger.info(f"✅ Score: {result.score:.3f}")
        logger.info(f"✅ Confidence: {result.confidence:.3f}")
        logger.info(f"✅ Reasoning: {result.reasoning}")
        logger.info(f"✅ Key phrases: {result.key_phrases}")
        logger.info(f"✅ Financial impact: {result.financial_impact}")
        
        # Test with negative news
        negative_news = """
        Tesla faces major recall of 2 million vehicles due to critical safety issues with 
        Autopilot system. Federal regulators cited multiple accidents and fatalities. 
        Stock price plummeted 15% in after-hours trading as investors panic over potential 
        lawsuits and regulatory action. Management remains silent on the crisis.
        """
        
        logger.info("\nAnalyzing negative Tesla news...")
        result2 = await analyzer.analyze_text(negative_news, "financial_news")
        
        logger.info(f"✅ Sentiment: {result2.sentiment}")
        logger.info(f"✅ Score: {result2.score:.3f}")
        logger.info(f"✅ Risk factors: {result2.risk_factors}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ LLM sentiment analyzer test failed: {e}")
        return False
    finally:
        await analyzer.close()


async def test_social_media_collector():
    """Test social media data collection (basic functionality)."""
    logger.info("\n=== Testing Social Media Collector (Basic) ===")
    
    try:
        from core.social_media_collector import RedditCollector
        
        reddit = RedditCollector()
        
        # Check if Reddit is initialized
        if reddit.reddit:
            logger.info("✅ Reddit API connection initialized successfully")
            
            # Test basic functionality without actually making API calls
            test_text = "Great earnings beat for $AAPL! Strong iPhone sales and guidance raised."
            mentions_symbol = reddit._mentions_symbol(test_text, "AAPL")
            logger.info(f"✅ Symbol mention detection: {mentions_symbol}")
            
            hashtags = reddit._extract_hashtags("#stocks #investing $AAPL to the moon!")
            logger.info(f"✅ Hashtag extraction: {hashtags}")
            
            stock_mentions = reddit._extract_mentions("Bullish on $AAPL $MSFT $GOOGL")
            logger.info(f"✅ Stock mention extraction: {stock_mentions}")
            
        else:
            logger.warning("⚠️  Reddit API not configured - skipping live tests")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Social media collector test failed: {e}")
        return False


async def test_earnings_scraper():
    """Test earnings call transcript scraping (basic functionality)."""
    logger.info("\n=== Testing Earnings Scraper (Basic) ===")
    
    try:
        from core.earnings_scraper import EarningsCallScraper
        
        scraper = EarningsCallScraper()
        
        # Test text parsing functions
        sample_transcript = """
        Good morning and welcome to Apple's Q4 2024 earnings call.
        
        Prepared Remarks:
        We're pleased to report record revenue of $95 billion, up 8% year-over-year.
        iPhone revenue was particularly strong at $43 billion.
        
        Questions and Answers:
        Analyst: What are your expectations for Q1?
        CEO: We remain optimistic about holiday season demand.
        """
        
        management, qa = scraper._parse_transcript_sections(sample_transcript)
        logger.info("✅ Transcript section parsing:")
        logger.info(f"   Management section: {len(management)} chars")
        logger.info(f"   Q&A section: {len(qa)} chars")
        
        # Test key metrics extraction
        metrics = scraper._extract_key_metrics(sample_transcript)
        logger.info(f"✅ Key metrics extracted: {metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Earnings scraper test failed: {e}")
        return False


async def test_configuration():
    """Test system configuration."""
    logger.info("\n=== Testing Configuration ===")
    
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
            status = "✅ Available" if available else "❌ Missing"
            logger.info(f"{service}: {status}")
        
        # Check if we have at least one LLM option
        has_llm = settings.openai_api_key or settings.use_llama_fallback
        logger.info(f"LLM Available: {'✅ Yes' if has_llm else '❌ No'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


async def test_market_data_sentiment():
    """Test basic market data sentiment analysis."""
    logger.info("\n=== Testing Market Data Sentiment ===")
    
    try:
        import yfinance as yf
        
        # Test with Apple stock
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="1mo")
        
        if not hist.empty:
            recent_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[-5]) - 1
            logger.info(f"✅ AAPL 5-day return: {recent_return*100:.2f}%")
            
            sentiment_direction = "positive" if recent_return > 0 else "negative"
            logger.info(f"✅ Market sentiment: {sentiment_direction}")
            
            return True
        else:
            logger.warning("⚠️  No market data available")
            return False
        
    except Exception as e:
        logger.error(f"❌ Market data sentiment test failed: {e}")
        return False


async def main():
    """Run basic tests."""
    logger.info("Starting Basic Sentiment Analysis System Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Configuration", test_configuration),
        ("LLM Sentiment Analyzer", test_llm_sentiment_analyzer),
        ("Social Media Collector (Basic)", test_social_media_collector),
        ("Earnings Scraper (Basic)", test_earnings_scraper),
        ("Market Data Sentiment", test_market_data_sentiment),
    ]
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n🧪 Running {test_name} test...")
            result = await test_func()
            test_results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name} test {status}")
        except Exception as e:
            logger.error(f"{test_name} test ❌ ERROR: {e}")
            test_results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All basic tests passed! Core sentiment system is functional.")
    elif passed >= total * 0.7:
        logger.info("✅ Most tests passed. System is largely functional.")
    else:
        logger.warning("⚠️  Several tests failed. Check configuration and dependencies.")


if __name__ == "__main__":
    asyncio.run(main())