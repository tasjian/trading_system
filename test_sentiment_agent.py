#!/usr/bin/env python3
"""
Test script for the Consumer Sentiment Agent

Demonstrates the three-layer architecture:
1. Mock social media data ingestion
2. LLM sentiment processing  
3. Structured output for trading signals
"""

import asyncio
import sys
sys.path.append('.')

from agents.sentiment_agent import (
    ConsumerSentimentAgent, 
    SocialPost, 
    SentimentScore,
    TrendSignal
)
from datetime import datetime

async def test_sentiment_agent():
    """Test the sentiment agent with mock data."""
    print("🧪 Testing Consumer Sentiment Agent")
    print("=" * 50)
    
    # Create sentiment agent
    agent = ConsumerSentimentAgent()
    
    # Create mock social media posts
    mock_posts = [
        SocialPost(
            id="post1",
            platform="reddit", 
            content="$AAPL is going to the moon! 🚀 Just bought calls, this stock is unstoppable with their new AI features",
            author="bullish_trader",
            timestamp=datetime.now(),
            upvotes=150,
            comments=45,
            subreddit="wallstreetbets"
        ),
        SocialPost(
            id="post2",
            platform="reddit",
            content="TSLA is overvalued, time to short. The competition is catching up and Elon's drama is hurting the brand",
            author="bear_market_pro", 
            timestamp=datetime.now(),
            upvotes=85,
            comments=120,
            subreddit="stocks"
        ),
        SocialPost(
            id="post3", 
            platform="reddit",
            content="Diamond hands on NVDA! AI revolution just getting started, this is the next Apple 💎🙌",
            author="diamond_hands_holder",
            timestamp=datetime.now(),
            upvotes=200,
            comments=67,
            subreddit="investing"
        )
    ]
    
    print(f"📱 Created {len(mock_posts)} mock social media posts")
    
    # Test Layer 2: LLM Processing
    print("\n🧠 Testing LLM Sentiment Processing...")
    
    processor = agent.processing_layer
    analyses = await processor.analyze_sentiment(mock_posts)
    
    print(f"✅ Analyzed {len(analyses)} posts")
    
    for analysis in analyses:
        print(f"\nPost {analysis.post_id}:")
        print(f"  Sentiment: {analysis.sentiment.value}")
        print(f"  Confidence: {analysis.confidence:.2f}")
        print(f"  Entities: {[e.ticker for e in analysis.entities]}")
        print(f"  Trend Signals: {[t.value for t in analysis.trend_signals]}")
        print(f"  Reasoning: {analysis.reasoning[:100]}...")
    
    # Test Layer 3: Structured Output
    print("\n📊 Testing Structured Output Generation...")
    
    structured_outputs = await agent.output_layer.aggregate_sentiment_by_ticker(analyses)
    
    print(f"✅ Generated {len(structured_outputs)} ticker summaries")
    
    for output in structured_outputs:
        print(f"\n🎯 {output.ticker}:")
        print(f"  Overall Sentiment: {output.overall_sentiment.value}")
        print(f"  Confidence: {output.confidence:.2f}")
        print(f"  Volume: {output.volume} mentions")
        print(f"  Trend Signal: {output.trend_signal.value}")
        print(f"  Key Insights: {output.key_insights}")
    
    # Test inter-agent messaging
    print("\n📡 Testing Inter-Agent Communication...")
    await agent.output_layer.emit_structured_output(structured_outputs)
    print("✅ Messages sent to inter-agent queue")
    
    # Test retrieval
    print("\n🔍 Testing Sentiment Retrieval...")
    if structured_outputs:
        ticker = structured_outputs[0].ticker
        recent_sentiment = await agent.get_recent_sentiment(ticker)
        
        if recent_sentiment:
            print(f"✅ Retrieved recent sentiment for {ticker}: {recent_sentiment.overall_sentiment.value}")
        else:
            print(f"⚠️ No recent sentiment found for {ticker}")
    
    print("\n🎉 Sentiment Agent Test Complete!")
    print("\n📋 Summary:")
    print(f"  • Processed {len(mock_posts)} social media posts")  
    print(f"  • Generated {len(analyses)} sentiment analyses")
    print(f"  • Created {len(structured_outputs)} ticker summaries")
    print(f"  • Integrated with trading workflow via structured signals")
    
    return structured_outputs

def test_mock_workflow_integration():
    """Test integration with trading workflow."""
    print("\n🔄 Testing Trading Workflow Integration...")
    
    # Mock trading state
    mock_state = {
        "positions": {"AAPL": {"quantity": 100}, "TSLA": {"quantity": -50}},
        "candidate_symbols": ["NVDA", "MSFT", "GOOGL"],
        "trading_signals": []
    }
    
    print("✅ Mock trading state created")
    print(f"  Current positions: {list(mock_state['positions'].keys())}")
    print(f"  Candidate symbols: {mock_state['candidate_symbols']}")
    
    # The sentiment agent would be called from the workflow
    # and would add sentiment_data and sentiment_signals to the state
    
    print("✅ Sentiment agent ready for workflow integration")
    print("   (Use 'cycle' command in main.py to test full integration)")

if __name__ == "__main__":
    async def main():
        await test_sentiment_agent()
        test_mock_workflow_integration()
    
    asyncio.run(main())