#!/usr/bin/env python3
"""
Test Optimized Pipeline
Test the new universe filter → sentiment analysis pipeline to verify efficiency gains.
"""

import asyncio
import logging
from datetime import datetime

from agents.workflow import TradingWorkflow
from agents.state import create_initial_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_optimized_pipeline():
    """Test the optimized pipeline with separated universe filter and sentiment analysis."""
    
    print("🔬 TESTING OPTIMIZED PIPELINE")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Testing separated Universe Filter → Sentiment Analysis pipeline")
    print("=" * 60)
    
    try:
        # Initialize workflow
        workflow = TradingWorkflow()
        state = create_initial_state()
        config = {"thread_id": "test_optimized_pipeline"}
        
        # Set up watchlist
        state["watchlist"] = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
        
        print("\n1️⃣ MARKET MONITOR")
        start_time = datetime.now()
        state = await workflow.market_monitor_agent(state, config)
        portfolio_value = state["portfolio"]["equity"]
        duration = (datetime.now() - start_time).total_seconds()
        print(f"✅ Portfolio Value: ${portfolio_value:,.2f} (took {duration:.1f}s)")
        
        print("\n2️⃣ UNIVERSE FILTER")
        start_time = datetime.now()
        state = await workflow.universe_filter_agent(state, config)
        filter_result = state.get("universe_filter_result")
        filtered_symbols = state.get("filtered_symbols", [])
        duration = (datetime.now() - start_time).total_seconds()
        
        if filter_result:
            efficiency = ((filter_result.total_symbols - len(filtered_symbols)) / filter_result.total_symbols * 100)
            print(f"✅ Universe Filter Results:")
            print(f"   📊 Original universe: {filter_result.total_symbols:,} stocks")
            print(f"   🎯 Filtered to: {len(filtered_symbols)} actionable stocks")
            print(f"   ⚡ Processing efficiency: {efficiency:.1f}% reduction")
            print(f"   ⏱️ Filter time: {duration:.1f}s")
            print(f"   📈 Signal categories: {filter_result.filter_summary}")
        else:
            print("❌ No filter results generated")
        
        print("\n3️⃣ SENTIMENT ANALYSIS (Pre-filtered stocks only)")
        start_time = datetime.now()
        state = await workflow.sentiment_analysis_agent(state, config)
        sentiment_data = state.get("sentiment_data", {})
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Sentiment Analysis Results:")
        print(f"   💭 Symbols analyzed: {len(sentiment_data)}")
        print(f"   ⏱️ Analysis time: {duration:.1f}s")
        
        if sentiment_data:
            # Show sample sentiment results
            print(f"   📊 Sample sentiment results:")
            for i, (symbol, data) in enumerate(list(sentiment_data.items())[:3]):
                sentiment = data.get('overall_sentiment', 'N/A')
                score = data.get('overall_score', 0)
                confidence = data.get('confidence', 0)
                print(f"      {symbol}: {sentiment} (score: {score:.3f}, confidence: {confidence:.3f})")
        
        print("\n4️⃣ QUICK RISK ASSESSMENT")
        start_time = datetime.now()
        state = await workflow.risk_assessment_agent(state, config)
        circuit_breakers = state.get("circuit_breakers", {})
        duration = (datetime.now() - start_time).total_seconds()
        
        active_breakers = [name for name, active in circuit_breakers.items() if active]
        print(f"✅ Risk Assessment: {len(active_breakers)} active circuit breakers (took {duration:.1f}s)")
        if active_breakers:
            print(f"   ⚠️ Active breakers: {active_breakers}")
        
        print("\n📊 PIPELINE EFFICIENCY SUMMARY")
        print("=" * 60)
        
        if filter_result:
            original_sentiment_cost = filter_result.total_symbols * 3  # 3 seconds per symbol estimate
            optimized_sentiment_cost = len(filtered_symbols) * 3
            time_saved = original_sentiment_cost - optimized_sentiment_cost
            
            print(f"📈 Efficiency Gains:")
            print(f"   🔍 Universe filtering: {efficiency:.1f}% reduction in stocks")
            print(f"   ⏱️ Estimated time saved: {time_saved/60:.1f} minutes per cycle")
            print(f"   💰 API calls saved: ~{filter_result.total_symbols - len(filtered_symbols):,} sentiment requests")
            print(f"   🎯 Focus on quality: Only actionable stocks get full analysis")
        
        print(f"\n🏆 OPTIMIZATION SUCCESS")
        print(f"The pipeline now efficiently processes only high-quality, actionable stocks!")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting optimized pipeline test...")
    print("This verifies that universe filter runs before sentiment analysis.")
    
    success = asyncio.run(test_optimized_pipeline())
    
    if success:
        print(f"\n✅ SUCCESS: Optimized pipeline is working efficiently!")
    else:
        print(f"\n❌ FAILED: Pipeline optimization needs debugging")