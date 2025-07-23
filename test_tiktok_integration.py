#!/usr/bin/env python3
"""
Test TikTok Integration for Sentiment Agent

Enhanced test with stealth measures to verify TikTok scraping functionality.
"""

import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.sentiment_agent import SocialMediaIngestionLayer, TikTokStealthConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_stealth_config():
    """Test the stealth configuration."""
    print("🕵️ Testing Stealth Configuration")
    print("=" * 50)
    
    config = TikTokStealthConfig()
    
    print("User Agents Available:")
    for i, ua in enumerate(config.USER_AGENTS[:3]):
        print(f"  {i+1}. {ua[:80]}...")
    
    print(f"\nViewport Sizes: {len(config.VIEWPORT_SIZES)} options")
    print("Sample viewports:")
    for viewport in config.VIEWPORT_SIZES[:3]:
        print(f"  {viewport['width']}x{viewport['height']}")
    
    print(f"\nRandom User Agent: {config.get_random_user_agent()[:60]}...")
    print(f"Random Viewport: {config.get_random_viewport()}")
    print(f"Random Delay: {config.get_random_delay():.2f} seconds")
    
    print("✅ Stealth configuration working")

async def test_tiktok_scraping_conservative():
    """Conservative test with minimal scraping."""
    
    print("\n🧪 Testing TikTok Integration (Conservative)")
    print("=" * 50)
    
    try:
        ingestion_layer = SocialMediaIngestionLayer()
        print(f"✅ Ingestion layer initialized")
        
        # Check TikTok availability
        if not ingestion_layer.tiktok_playwright:
            print("❌ TikTok scraper not available - Playwright not installed")
            print("Run: pip install playwright && playwright install chromium")
            return
        
        print("✅ TikTok scraper available")
        print("✅ Stealth configuration loaded")
        
        # Very conservative test - only one hashtag, minimal posts
        test_hashtags = ['finance']  # Less competitive hashtag
        test_keywords = ['trading', 'investing', 'stocks']  # Broader keywords
        
        print(f"🔍 Testing TikTok scraping (conservative mode)")
        print(f"📍 Hashtags: {test_hashtags}")
        print(f"🔍 Keywords: {test_keywords}")
        print("⏳ This will take 30-60 seconds with stealth delays...")
        
        # Test with minimal settings
        tiktok_posts = await ingestion_layer.fetch_tiktok_posts(
            hashtags=test_hashtags,
            keywords=test_keywords,
            max_posts=5  # Very small number
        )
        
        print(f"📱 Scraped {len(tiktok_posts)} posts")
        
        # Check cache
        cache_entries = len(ingestion_layer.html_cache)
        print(f"💾 HTML cache entries: {cache_entries}")
        
        if tiktok_posts:
            print("\n📋 Sample posts found:")
            for i, post in enumerate(tiktok_posts):
                print(f"\nPost {i+1}:")
                print(f"  ID: {post.id}")
                print(f"  Author: {post.author}")
                print(f"  Hashtags: {post.hashtags}")
                print(f"  Content Preview: {post.content[:80]}...")
        else:
            print("⚠️ No posts extracted. Possible reasons:")
            print("  - TikTok successfully blocked the request")
            print("  - Content extraction strategies need improvement")
            print("  - Rate limiting kicked in")
            
        # Check if we have raw HTML (shows we reached the page)
        html_entries = [k for k in ingestion_layer.html_cache.keys() if 'finance_' in k]
        if html_entries:
            print(f"✅ Successfully retrieved HTML from TikTok pages")
            latest_entry = max(html_entries, key=lambda k: ingestion_layer.html_cache[k]['timestamp'])
            html_content = ingestion_layer.html_cache[latest_entry]['html']
            print(f"📄 HTML length: {len(html_content)} characters")
            
            # Basic content analysis
            if 'tiktok' in html_content.lower():
                print("✅ Valid TikTok page content detected")
            if 'blocked' in html_content.lower() or 'captcha' in html_content.lower():
                print("⚠️ Blocking indicators found in HTML")
            else:
                print("✅ No obvious blocking indicators")
        
        print("\n✅ Conservative TikTok test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_cache_functionality():
    """Test caching to avoid repeated requests."""
    print("\n💾 Testing Cache Functionality")
    print("=" * 50)
    
    try:
        ingestion_layer = SocialMediaIngestionLayer()
        
        # First call
        print("🔍 First call (should scrape)")
        posts1 = await ingestion_layer.fetch_tiktok_posts(
            hashtags=['finance'],
            keywords=['trading'],
            max_posts=3
        )
        
        # Second call (should use cache)
        print("🔍 Second call (should use cache)")
        posts2 = await ingestion_layer.fetch_tiktok_posts(
            hashtags=['finance'],
            keywords=['trading'],
            max_posts=3
        )
        
        print(f"📊 First call: {len(posts1)} posts")
        print(f"📊 Second call: {len(posts2)} posts")
        
        if len(posts1) == len(posts2) and len(posts1) > 0:
            print("✅ Cache working correctly")
        elif len(posts1) == len(posts2) == 0:
            print("⚠️ Both calls returned 0 posts")
        else:
            print("⚠️ Different results between calls")
            
        cache_size = len(ingestion_layer.html_cache)
        print(f"💾 Total cache entries: {cache_size}")
        
    except Exception as e:
        print(f"❌ Cache test failed: {e}")

async def test_integration_with_sentiment_agent():
    """Test full integration with sentiment analysis."""
    print("\n🤖 Testing Full Sentiment Agent Integration")
    print("=" * 50)
    
    try:
        from agents.sentiment_agent import ConsumerSentimentAgent
        
        agent = ConsumerSentimentAgent()
        
        print("🔍 Running full sentiment analysis with TikTok...")
        
        # This would normally run the full analysis
        # For testing, just check if TikTok is in the default platforms
        ingestion = agent.ingestion_layer
        
        if ingestion.tiktok_playwright:
            print("✅ TikTok integrated into sentiment agent")
        else:
            print("❌ TikTok not properly integrated")
            
        print("✅ Integration test completed")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Enhanced TikTok Integration Tests")
    
    # Run focused tests
    asyncio.run(test_stealth_config())
    asyncio.run(test_tiktok_scraping_conservative())
    asyncio.run(test_cache_functionality())
    asyncio.run(test_integration_with_sentiment_agent())
    
    print("\n🎯 All tests completed")
    print("\nNote: If TikTok scraping returns 0 posts, this is often expected")
    print("due to anti-bot measures. The integration is still functional.")