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


async def test_anthropic_integration():
    """Test Anthropic Claude API integration with real call."""
    logger.info("Testing Anthropic Claude integration...")
    
    analyzer = LLMSentimentAnalyzer(
        anthropic_api_key=settings.anthropic_api_key,
        openai_api_key=settings.openai_api_key,
        ollama_base_url=settings.ollama_base_url
    )
    
    simple_text = "Apple stock surged 5% after beating earnings expectations with strong iPhone sales and AI initiatives showing promise."
    
    try:
        result = await analyzer.analyze_text(simple_text, "financial_news")
        
        logger.info(f"✅ Anthropic Analysis Results:")
        logger.info(f"   Sentiment: {result.sentiment}")
        logger.info(f"   Score: {result.score:.3f}")
        logger.info(f"   Confidence: {result.confidence:.3f}")
        logger.info(f"   Reasoning: {result.reasoning[:150]}...")
        logger.info(f"   Key phrases: {result.key_phrases[:3]}")
        logger.info(f"   Financial impact: {result.financial_impact[:100] if result.financial_impact else 'None'}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Anthropic test failed: {e}")
        return False
    finally:
        await analyzer.close()


async def test_llm_fallback():
    """Test LLM fallback from Anthropic to Ollama."""
    logger.info("Testing LLM fallback system...")
    
    from tools.llm_client import llm_client
    
    try:
        # Test basic connection
        connections = await llm_client.test_connection()
        logger.info(f"Provider connections: {connections}")
        
        # Test simple analysis
        response = await llm_client.generate_response(
            system_prompt="You are a financial analyst.",
            user_message="Analyze this: Tesla stock gained 3% on strong delivery numbers.",
            temperature=0.7,
            max_tokens=100
        )
        
        logger.info(f"✅ LLM Fallback Test Results:")
        logger.info(f"   Model used: {response.model}")
        logger.info(f"   Response time: {response.response_time:.2f}s")
        logger.info(f"   Content: {response.content[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ LLM fallback test failed: {e}")
        return False


if __name__ == "__main__":
    async def run_all_tests():
        print("🤖 TESTING LLM INTEGRATION")
        print("=" * 50)
        
        # Test 1: Anthropic integration
        success1 = await test_anthropic_integration()
        
        # Test 2: LLM fallback system
        success2 = await test_llm_fallback()
        
        if success1 and success2:
            print("✅ All LLM tests passed!")
            return True
        else:
            print("❌ Some LLM tests failed")
            return False
    
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)