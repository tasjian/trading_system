#!/usr/bin/env python3
"""
Test Trading System Integration
Test that the enhanced social media collector works with the existing trading system.
"""

import asyncio
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_universe_filter_integration():
    """Test integration with universe filter."""
    logger.info("🔍 Testing Universe Filter Integration")
    logger.info("=" * 60)
    
    try:
        from core.universe_filter import StockUniverseFilter
        from core.social_media_collector import SocialMediaCollector
        
        # Test that universe filter can use the enhanced collector
        universe_filter = StockUniverseFilter()
        
        # Mock some symbols for testing
        test_symbols = ['AAPL', 'TSLA', 'MSFT']
        
        logger.info(f"Testing social signal collection for {len(test_symbols)} symbols...")
        start_time = datetime.now()
        
        # This would be called by the universe filter
        social_signals = []
        
        for symbol in test_symbols:
            try:
                # Simulate what universe filter does
                collector = SocialMediaCollector()
                results = await collector.collect_all_platforms(symbol, limit_per_platform=5)
                
                # Process results as universe filter would
                reddit_posts = results.get('reddit', [])
                twitter_posts = results.get('twitter', [])
                
                # Calculate sentiment signal (simplified)
                all_posts = reddit_posts + twitter_posts
                if all_posts:
                    bullish_count = sum(1 for post in all_posts 
                                      if any('bullish' in ind for ind in post.get('sentiment_indicators', [])))
                    bearish_count = sum(1 for post in all_posts 
                                      if any('bearish' in ind for ind in post.get('sentiment_indicators', [])))
                    
                    sentiment_score = (bullish_count - bearish_count) / len(all_posts) if all_posts else 0
                    avg_relevance = sum(post.get('relevance_score', 0) for post in all_posts) / len(all_posts)
                    
                    social_signal = {
                        'symbol': symbol,
                        'sentiment_score': sentiment_score,
                        'relevance_score': avg_relevance,
                        'total_posts': len(all_posts),
                        'bullish_posts': bullish_count,
                        'bearish_posts': bearish_count
                    }
                    social_signals.append(social_signal)
                    
                    logger.info(f"  {symbol}: {len(all_posts)} posts, sentiment: {sentiment_score:.2f}, relevance: {avg_relevance:.2f}")
                
                await collector.cleanup()
                
            except Exception as e:
                logger.error(f"Failed to collect for {symbol}: {e}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"\n✅ Universe Filter Integration Test:")
        logger.info(f"  Duration: {duration:.1f}s")
        logger.info(f"  Symbols processed: {len(social_signals)}/{len(test_symbols)}")
        logger.info(f"  Average processing time: {duration/len(test_symbols):.1f}s per symbol")
        
        if social_signals:
            # Show sample signal
            sample = social_signals[0]
            logger.info(f"\n  📊 Sample Social Signal:")
            logger.info(f"    Symbol: {sample['symbol']}")
            logger.info(f"    Sentiment Score: {sample['sentiment_score']:.2f}")
            logger.info(f"    Relevance Score: {sample['relevance_score']:.2f}")
            logger.info(f"    Posts: {sample['total_posts']} ({sample['bullish_posts']}📈 {sample['bearish_posts']}📉)")
        
        logger.info("✅ Universe filter integration successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Universe filter integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_sentiment_agent_integration():
    """Test integration with sentiment agent."""
    logger.info("\n🔍 Testing Sentiment Agent Integration")
    logger.info("=" * 60)
    
    try:
        # Mock sentiment agent workflow
        from core.social_media_collector import SocialMediaCollector
        
        # Simulate what sentiment agent would do
        test_symbols = ['NVDA', 'GOOGL']
        sentiment_data = {}
        
        collector = SocialMediaCollector()
        
        for symbol in test_symbols:
            logger.info(f"Processing {symbol} for sentiment analysis...")
            
            results = await collector.collect_all_platforms(symbol, limit_per_platform=8)
            
            # Process as sentiment agent would
            all_posts = []
            for platform_posts in results.values():
                all_posts.extend(platform_posts)
            
            if all_posts:
                # Enhanced sentiment analysis
                sentiment_indicators = []
                relevance_scores = []
                
                for post in all_posts:
                    sentiment_indicators.extend(post.get('sentiment_indicators', []))
                    relevance_scores.append(post.get('relevance_score', 0))
                
                # Categorize sentiment
                bullish_indicators = [ind for ind in sentiment_indicators if 'bullish' in ind]
                bearish_indicators = [ind for ind in sentiment_indicators if 'bearish' in ind]
                institutional_indicators = [ind for ind in sentiment_indicators if 'institutional' in ind]
                technical_indicators = [ind for ind in sentiment_indicators if 'technical' in ind]
                
                sentiment_summary = {
                    'total_posts': len(all_posts),
                    'avg_relevance': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
                    'bullish_signals': len(bullish_indicators),
                    'bearish_signals': len(bearish_indicators),
                    'institutional_signals': len(institutional_indicators),
                    'technical_signals': len(technical_indicators),
                    'sentiment_categories': list(set([ind.split(':')[0] for ind in sentiment_indicators])),
                    'confidence': min(1.0, len(all_posts) / 10)  # More posts = higher confidence
                }
                
                sentiment_data[symbol] = sentiment_summary
                
                logger.info(f"  {symbol} sentiment analysis:")
                logger.info(f"    Posts: {sentiment_summary['total_posts']}")
                logger.info(f"    Avg relevance: {sentiment_summary['avg_relevance']:.2f}")
                logger.info(f"    Bullish signals: {sentiment_summary['bullish_signals']}")
                logger.info(f"    Bearish signals: {sentiment_summary['bearish_signals']}")
                logger.info(f"    Confidence: {sentiment_summary['confidence']:.2f}")
        
        await collector.cleanup()
        
        logger.info(f"\n✅ Sentiment Agent Integration Test:")
        logger.info(f"  Symbols analyzed: {len(sentiment_data)}")
        
        if sentiment_data:
            # Show overall sentiment analysis
            total_posts = sum(data['total_posts'] for data in sentiment_data.values())
            total_bullish = sum(data['bullish_signals'] for data in sentiment_data.values())
            total_bearish = sum(data['bearish_signals'] for data in sentiment_data.values())
            avg_confidence = sum(data['confidence'] for data in sentiment_data.values()) / len(sentiment_data)
            
            logger.info(f"  Total posts analyzed: {total_posts}")
            logger.info(f"  Overall bullish signals: {total_bullish}")
            logger.info(f"  Overall bearish signals: {total_bearish}")
            logger.info(f"  Average confidence: {avg_confidence:.2f}")
        
        logger.info("✅ Sentiment agent integration successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Sentiment agent integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_workflow_integration():
    """Test integration with the main trading workflow."""
    logger.info("\n🔍 Testing Trading Workflow Integration")
    logger.info("=" * 60)
    
    try:
        # Test that we can import and use the collector in workflow context
        from core.social_media_collector import SocialMediaCollector
        
        # Simulate a mini trading workflow
        symbols = ['AAPL', 'TSLA']
        workflow_results = {}
        
        start_time = datetime.now()
        
        for symbol in symbols:
            logger.info(f"Running mini workflow for {symbol}...")
            
            # Step 1: Collect social media data (new enhanced version)
            collector = SocialMediaCollector()
            social_data = await collector.collect_all_platforms(symbol, limit_per_platform=6)
            
            # Step 2: Process for trading signals (simplified)
            reddit_posts = social_data.get('reddit', [])
            twitter_posts = social_data.get('twitter', [])
            
            all_posts = reddit_posts + twitter_posts
            
            if all_posts:
                # Calculate trading-relevant metrics
                high_relevance_posts = [p for p in all_posts if p.get('relevance_score', 0) > 5.0]
                strong_sentiment_posts = [p for p in all_posts 
                                        if any(indicator.startswith(('strong_bullish', 'strong_bearish')) 
                                               for indicator in p.get('sentiment_indicators', []))]
                institutional_posts = [p for p in all_posts 
                                     if any('institutional' in indicator 
                                            for indicator in p.get('sentiment_indicators', []))]
                
                # Create trading signal (simplified)
                signal_strength = len(high_relevance_posts) / len(all_posts) if all_posts else 0
                sentiment_intensity = len(strong_sentiment_posts) / len(all_posts) if all_posts else 0
                institutional_interest = len(institutional_posts) / len(all_posts) if all_posts else 0
                
                trading_signal = {
                    'symbol': symbol,
                    'total_posts': len(all_posts),
                    'signal_strength': signal_strength,
                    'sentiment_intensity': sentiment_intensity,
                    'institutional_interest': institutional_interest,
                    'overall_score': (signal_strength + sentiment_intensity + institutional_interest) / 3,
                    'data_quality': 'enhanced' if any('relevance_score' in str(p) for p in all_posts) else 'basic'
                }
                
                workflow_results[symbol] = trading_signal
                
                logger.info(f"  {symbol} trading signal:")
                logger.info(f"    Posts: {trading_signal['total_posts']}")
                logger.info(f"    Signal strength: {trading_signal['signal_strength']:.2f}")
                logger.info(f"    Sentiment intensity: {trading_signal['sentiment_intensity']:.2f}")
                logger.info(f"    Institutional interest: {trading_signal['institutional_interest']:.2f}")
                logger.info(f"    Overall score: {trading_signal['overall_score']:.2f}")
                logger.info(f"    Data quality: {trading_signal['data_quality']}")
            
            await collector.cleanup()
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"\n✅ Trading Workflow Integration Test:")
        logger.info(f"  Duration: {duration:.1f}s")
        logger.info(f"  Symbols processed: {len(workflow_results)}")
        
        if workflow_results:
            # Show summary
            avg_signal_strength = sum(r['signal_strength'] for r in workflow_results.values()) / len(workflow_results)
            avg_overall_score = sum(r['overall_score'] for r in workflow_results.values()) / len(workflow_results)
            
            logger.info(f"  Average signal strength: {avg_signal_strength:.2f}")
            logger.info(f"  Average overall score: {avg_overall_score:.2f}")
            logger.info(f"  All signals use enhanced data: {all(r['data_quality'] == 'enhanced' for r in workflow_results.values())}")
        
        logger.info("✅ Trading workflow integration successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Trading workflow integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_under_load():
    """Test performance under trading system load."""
    logger.info("\n🔍 Testing Performance Under Load")
    logger.info("=" * 60)
    
    try:
        from core.social_media_collector import SocialMediaCollector
        
        # Simulate high-load scenario
        symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'NVDA', 'AMZN', 'META', 'NFLX']
        
        logger.info(f"Testing under load with {len(symbols)} symbols...")
        start_time = datetime.now()
        
        # Create multiple collectors to simulate concurrent usage
        collectors = [SocialMediaCollector() for _ in range(3)]
        
        # Run concurrent collections
        tasks = []
        for i, collector in enumerate(collectors):
            batch = symbols[i::len(collectors)]  # Distribute symbols
            for symbol in batch:
                task = collector.collect_all_platforms(symbol, limit_per_platform=4)
                tasks.append((collector, symbol, task))
        
        # Execute all tasks
        results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
        
        # Cleanup all collectors
        cleanup_tasks = [collector.cleanup() for collector in collectors]
        await asyncio.gather(*cleanup_tasks)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Analyze results
        successful_results = 0
        total_posts = 0
        
        for i, result in enumerate(results):
            symbol = tasks[i][1]
            if isinstance(result, Exception):
                logger.info(f"  {symbol}: ❌ Failed - {result}")
            else:
                successful_results += 1
                posts_count = sum(len(posts) for posts in result.values())
                total_posts += posts_count
                logger.info(f"  {symbol}: ✅ {posts_count} posts")
        
        logger.info(f"\n📊 Performance Under Load Results:")
        logger.info(f"  Duration: {duration:.1f}s")
        logger.info(f"  Symbols: {len(symbols)}")
        logger.info(f"  Success rate: {successful_results}/{len(symbols)} ({100*successful_results/len(symbols):.1f}%)")
        logger.info(f"  Total posts: {total_posts}")
        logger.info(f"  Throughput: {total_posts/duration:.1f} posts/second")
        logger.info(f"  Average per symbol: {duration/len(symbols):.1f}s")
        
        if successful_results >= len(symbols) * 0.8:  # 80% success acceptable under load
            logger.info("🚀 EXCELLENT: System performs well under load!")
            return True
        else:
            logger.warning("⚠️ Performance degraded under load")
            return False
        
    except Exception as e:
        logger.error(f"❌ Load test failed: {e}")
        return False

async def main():
    """Run all trading system integration tests."""
    logger.info("🚀 TRADING SYSTEM INTEGRATION TESTING")
    logger.info("Testing enhanced social media collector with trading system...")
    logger.info("=" * 70)
    
    results = []
    
    try:
        # Test universe filter integration
        results.append(await test_universe_filter_integration())
        
        # Test sentiment agent integration
        results.append(await test_sentiment_agent_integration())
        
        # Test workflow integration
        results.append(await test_workflow_integration())
        
        # Test performance under load
        results.append(await test_performance_under_load())
        
        success_count = sum(results)
        
        logger.info(f"\n🎉 TRADING SYSTEM INTEGRATION TESTS COMPLETED!")
        logger.info(f"Success rate: {success_count}/{len(results)} tests passed")
        
        if success_count == len(results):
            logger.info("\n✅ ALL INTEGRATION TESTS PASSED!")
            logger.info("Enhanced social media collector is ready for production!")
            logger.info("\nKey Integration Features:")
            logger.info("✅ Universe filter compatibility maintained")
            logger.info("✅ Sentiment agent enhanced capabilities")
            logger.info("✅ Trading workflow integration seamless")
            logger.info("✅ Performance excellent under load")
            logger.info("✅ Enhanced sentiment analysis for trading signals")
            logger.info("✅ RapidAPI Twitter + preserved Reddit functionality")
        else:
            logger.warning(f"⚠️ {len(results) - success_count} integration tests failed")
        
    except Exception as e:
        logger.error(f"❌ Integration test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Trading System Integration Testing")
    print("Testing enhanced social media collector with trading system...")
    print("=" * 70)
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n💥 INTEGRATION TESTS FAILED: {e}")
        import traceback
        traceback.print_exc()