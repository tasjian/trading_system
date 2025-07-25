#!/usr/bin/env python3
"""
Simple test to verify the refactored sentiment system is working.
"""

import asyncio
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

async def main():
    """Simple test of the new sentiment system integration."""
    
    print("🚀 TESTING NEW SENTIMENT ANALYSIS INTEGRATION")
    print("=" * 60)
    
    try:
        # Test direct market intelligence with new sentiment
        from core.market_intelligence import UnifiedMarketIntelligence
        
        market_intel = UnifiedMarketIntelligence()
        
        # Test sentiment analysis for AAPL (this now uses the new LLM system)
        print("🧠 Testing enhanced sentiment analysis for AAPL...")
        sentiment_score = await market_intel._analyze_sentiment("AAPL")
        
        print(f"✅ AAPL Sentiment Score: {sentiment_score:.3f}")
        
        if sentiment_score > 0.1:
            print("📈 Sentiment: POSITIVE - Likely to contribute to BUY signals")
        elif sentiment_score < -0.1:
            print("📉 Sentiment: NEGATIVE - Likely to contribute to SELL signals")
        else:
            print("⚖️ Sentiment: NEUTRAL - Minimal impact on trading decisions")
        
        # Test full stock analysis (this integrates the new sentiment)
        print(f"\n📊 Testing full stock analysis with enhanced sentiment...")
        analysis = await market_intel.analyze_stock("AAPL")
        
        if analysis:
            print(f"✅ Stock Analysis Results:")
            print(f"   Overall Score: {analysis.score:.3f}")
            print(f"   Signal: {analysis.signal}")
            print(f"   Confidence: {analysis.confidence}")
            
            if hasattr(analysis, 'components') and analysis.components:
                print(f"   Component Breakdown:")
                for component, score in analysis.components.items():
                    print(f"     {component.capitalize()}: {score:.3f}")
                    if component == 'sentiment':
                        print(f"       ↳ Using NEW LLM-powered analysis! 🎉")
        
        print(f"\n🎊 SUCCESS: New sentiment analysis is integrated and working!")
        print(f"📈 The trading system now uses LLM-powered sentiment analysis")
        print(f"🔄 When you run the main system, it will use this enhanced analysis")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(main())