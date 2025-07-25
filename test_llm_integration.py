"""
Test LLM integration specifically.
"""

import asyncio
import logging
import sys

sys.path.append('/Users/zac/Desktop/ML4T/trading_system')

from core.llm_sentiment_analyzer import LLMSentimentAnalyzer
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_openai_integration():
    """Test OpenAI API integration with real call."""
    logger.info("Testing OpenAI integration...")
    
    analyzer = LLMSentimentAnalyzer(
        openai_api_key=settings.openai_api_key,
        ollama_base_url=settings.ollama_base_url
    )
    
    simple_text = "Apple stock surged 5% after beating earnings expectations."
    
    try:
        result = await analyzer.analyze_text(simple_text, "financial_news")
        
        logger.info(f"✅ OpenAI Analysis Results:")
        logger.info(f"   Sentiment: {result.sentiment}")
        logger.info(f"   Score: {result.score:.3f}")
        logger.info(f"   Confidence: {result.confidence:.3f}")
        logger.info(f"   Reasoning: {result.reasoning[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenAI test failed: {e}")
        return False
    finally:
        await analyzer.close()


if __name__ == "__main__":
    asyncio.run(test_openai_integration())