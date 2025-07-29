#!/usr/bin/env python3
"""
Test Social Media APIs

Tests Reddit, Twitter/X, and TikTok APIs to ensure they're working properly.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_social_media_apis():
    """Test all social media APIs."""
    
    print("🧪 TESTING SOCIAL MEDIA APIs")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        from core.social_media_collector import SocialMediaCollector
        from config.settings import settings
        
        collector = SocialMediaCollector()
        test_symbol = "AAPL"
        
        print(f"🎯 Testing APIs for symbol: {test_symbol}")
        print("-" * 40)
        
        # Test Reddit API
        print("\n📱 Testing Reddit API...")
        try:
            reddit_posts = await collector.reddit_collector.collect_posts(test_symbol, limit=5)
            print(f"✅ Reddit: Collected {len(reddit_posts)} posts")
            if reddit_posts:
                print(f"   Latest post: {reddit_posts[0].content[:100]}...")
            else:
                print("   No posts found")
        except Exception as e:
            print(f"❌ Reddit failed: {e}")
        
        # Test Twitter API
        print("\n🐦 Testing Twitter/X API...")
        try:
            twitter_posts = await collector.twitter_collector.collect_posts(test_symbol, limit=5)
            print(f"✅ Twitter: Collected {len(twitter_posts)} posts")
            if twitter_posts:
                print(f"   Latest tweet: {twitter_posts[0].content[:100]}...")
            else:
                print("   No tweets found")
        except Exception as e:
            print(f"❌ Twitter failed: {e}")
        
        # Test TikTok API
        print("\n🎵 Testing TikTok API...")
        try:
            if settings.tiktok_enabled:
                tiktok_posts = await collector.tiktok_collector.collect_posts(test_symbol, limit=3)
                print(f"✅ TikTok: Collected {len(tiktok_posts)} posts")
                if tiktok_posts:
                    print(f"   Latest post: {tiktok_posts[0].content[:100]}...")
                else:
                    print("   No posts found")
            else:
                print("⚠️  TikTok disabled in settings")
        except Exception as e:
            print(f"❌ TikTok failed: {e}")
        
        # Test combined collection
        print("\n🌐 Testing Combined Collection...")
        try:
            platform_results = await collector.collect_all_platforms(test_symbol)
            total_posts = sum(len(posts) for posts in platform_results.values())
            print(f"✅ Combined: Collected {total_posts} total posts")
            
            for platform, posts in platform_results.items():
                print(f"   {platform}: {len(posts)} posts")
                
        except Exception as e:
            print(f"❌ Combined collection failed: {e}")
        
        # Test API credentials
        print(f"\n🔑 API Credentials Status:")
        print("-" * 30)
        print(f"Reddit Client ID: {'✅ Set' if settings.reddit_client_id else '❌ Missing'}")
        print(f"Reddit Client Secret: {'✅ Set' if settings.reddit_client_secret else '❌ Missing'}")
        print(f"Twitter Bearer Token: {'✅ Set' if settings.twitter_bearer_token else '❌ Missing'}")
        print(f"TikTok Enabled: {'✅ Yes' if settings.tiktok_enabled else '❌ No'}")
        
        print(f"\n🎉 SOCIAL MEDIA API TESTING COMPLETE!")
        
    except Exception as e:
        logger.error(f"API testing failed: {e}")
        print(f"\n❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_social_media_apis())
    sys.exit(0 if success else 1)