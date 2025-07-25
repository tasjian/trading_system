"""
Test the complete trading system integration with new sentiment analysis.
"""

import asyncio
import logging
import sys

sys.path.append('/Users/zac/Desktop/ML4T/trading_system')

from core.market_intelligence import UnifiedMarketIntelligence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_market_intelligence_sentiment():
    """Test the refactored market intelligence with new sentiment analysis."""
    logger.info("=== Testing Market Intelligence Sentiment Integration ===")
    
    try:
        market_intel = UnifiedMarketIntelligence()
        
        # Test sentiment analysis for AAPL
        logger.info("Testing sentiment analysis for AAPL...")
        sentiment_score = await market_intel._analyze_sentiment("AAPL")
        
        if sentiment_score is not None:
            logger.info(f"✅ AAPL sentiment score: {sentiment_score:.3f}")
            
            sentiment_label = "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral"
            logger.info(f"✅ AAPL sentiment: {sentiment_label}")
            
        else:
            logger.warning("⚠️  No sentiment score returned")
        
        # Test full stock analysis
        logger.info("\nTesting full stock analysis...")
        analysis = await market_intel.analyze_stock("AAPL")
        
        if analysis:
            logger.info(f"✅ Overall analysis:")
            logger.info(f"   Score: {analysis.score:.3f}")
            logger.info(f"   Signal: {analysis.signal}")
            logger.info(f"   Confidence: {analysis.confidence}")
            
            if hasattr(analysis, 'components') and analysis.components:
                logger.info(f"✅ Component scores:")
                for component, score in analysis.components.items():
                    logger.info(f"   {component}: {score:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Market intelligence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_news_article_fetching():
    """Test news article fetching functionality."""
    logger.info("\n=== Testing News Article Fetching ===")
    
    try:
        market_intel = UnifiedMarketIntelligence()
        
        # Test news article fetching
        articles = await market_intel._get_news_articles("AAPL")
        
        logger.info(f"✅ Fetched {len(articles)} news articles for AAPL")
        
        if articles:
            sample_article = articles[0]
            logger.info(f"✅ Sample article:")
            logger.info(f"   Title: {sample_article.get('title', 'N/A')[:100]}...")
            logger.info(f"   Source: {sample_article.get('source', {}).get('name', 'N/A')}")
            logger.info(f"   Published: {sample_article.get('publishedAt', 'N/A')}")
        
        return len(articles) > 0
        
    except Exception as e:
        logger.error(f"❌ News article fetching failed: {e}")
        return False


async def main():
    """Run trading system integration tests."""
    logger.info("Testing Trading System Integration with New Sentiment Analysis")
    logger.info("=" * 70)
    
    tests = [
        ("Market Intelligence Sentiment", test_market_intelligence_sentiment),
        ("News Article Fetching", test_news_article_fetching),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name}...")
        try:
            result = await test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ ERROR - {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n📊 Overall: {passed}/{total} integration tests passed")
    
    if passed == total:
        logger.info("🎉 All integration tests passed! The sentiment system is fully integrated.")
    else:
        logger.info("⚠️  Some integration tests failed. Check logs for details.")


if __name__ == "__main__":
    asyncio.run(main())