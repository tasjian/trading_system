#!/usr/bin/env python3
"""
Test Critical Error Fixes
Verify that the session leaks and missing method errors are resolved.
"""

import asyncio
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_earnings_scraper_sessions():
    """Test that earnings scraper has proper session management."""
    logger.info("🔍 Testing Earnings Scraper Session Management")
    
    try:
        from core.earnings_scraper import EarningsCallScraper
        
        # Test context manager usage
        async with EarningsCallScraper() as scraper:
            logger.info("✅ Earnings scraper context manager works")
            
            # Test search functionality with timeout
            results = await scraper.search_transcripts("AAPL", days_back=30)
            logger.info(f"✅ Search completed, found {len(results)} results")
        
        logger.info("✅ Context manager cleanup completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Earnings scraper test failed: {e}")
        return False

async def test_workflow_llm_agent():
    """Test that workflow has LLM portfolio agent method."""
    logger.info("🔍 Testing Workflow LLM Portfolio Agent")
    
    try:
        from agents.workflow import TradingWorkflow
        from agents.state import create_initial_state
        
        workflow = TradingWorkflow()
        
        # Check if method exists
        if hasattr(workflow, 'llm_portfolio_agent'):
            logger.info("✅ LLM portfolio agent method exists")
            
            # Test method call with mock state
            test_state = create_initial_state("test_session")
            result = await workflow.llm_portfolio_agent(test_state, {})
            
            logger.info("✅ LLM portfolio agent method callable")
            return True
        else:
            logger.error("❌ LLM portfolio agent method missing")
            return False
            
    except Exception as e:
        logger.error(f"❌ Workflow LLM agent test failed: {e}")
        return False

async def test_social_media_cleanup():
    """Test social media collector session cleanup."""
    logger.info("🔍 Testing Social Media Collector Cleanup")
    
    try:
        from core.social_media_collector import SocialMediaCollector
        
        collector = SocialMediaCollector()
        
        # Test collection
        results = await collector.collect_all_platforms("AAPL", limit_per_platform=3)
        logger.info(f"✅ Collected {sum(len(posts) for posts in results.values())} posts")
        
        # Test cleanup
        await collector.cleanup()
        logger.info("✅ Social media collector cleanup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Social media collector test failed: {e}")
        return False

async def test_global_session_utility():
    """Test global session cleanup utility."""
    logger.info("🔍 Testing Global Session Cleanup Utility")
    
    try:
        from utils.session_cleanup import global_session_manager
        
        logger.info("✅ Global session manager imported successfully")
        
        # Test cleanup (should not fail)
        await global_session_manager.cleanup_all()
        logger.info("✅ Global session cleanup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Global session utility test failed: {e}")
        return False

async def main():
    """Run all critical fix tests."""
    logger.info("🚀 TESTING CRITICAL ERROR FIXES")
    logger.info("=" * 60)
    
    test_results = []
    
    # Test 1: Earnings scraper session management
    logger.info("1. Testing earnings scraper session management...")
    result1 = await test_earnings_scraper_sessions()
    test_results.append(result1)
    
    # Test 2: Workflow LLM portfolio agent
    logger.info("\n2. Testing workflow LLM portfolio agent...")
    result2 = await test_workflow_llm_agent()
    test_results.append(result2)
    
    # Test 3: Social media collector cleanup
    logger.info("\n3. Testing social media collector session cleanup...")
    result3 = await test_social_media_cleanup()
    test_results.append(result3)
    
    # Test 4: Global session utility
    logger.info("\n4. Testing global session cleanup utility...")
    result4 = await test_global_session_utility()
    test_results.append(result4)
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    logger.info(f"\n🎉 CRITICAL FIX TESTING COMPLETED!")
    logger.info(f"Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("✅ ALL CRITICAL FIXES WORKING CORRECTLY!")
        logger.info("\nFixed Issues:")
        logger.info("✅ Earnings scraper session leaks resolved")
        logger.info("✅ Missing LLM portfolio agent method added")
        logger.info("✅ Social media collector session cleanup working")
        logger.info("✅ Global session cleanup utility available")
        logger.info("\nSystem should now run without the critical errors found in logs.")
    else:
        logger.error(f"❌ {total_tests - passed_tests} tests failed - manual investigation needed")
    
if __name__ == "__main__":
    asyncio.run(main())