#!/usr/bin/env python3
"""
Test script to cycle the trading system and test new sentiment analysis.
"""

import asyncio
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from agents.workflow import trading_workflow
from tools.alpaca_client import alpaca_client
from config.settings import settings

async def test_trading_cycle():
    """Test a trading cycle with new sentiment analysis."""
    
    print("🚀 TESTING TRADING CYCLE WITH NEW SENTIMENT ANALYSIS")
    print("=" * 60)
    
    try:
        # Get current account info
        account_info = alpaca_client.get_account_info()
        print(f"📊 Portfolio Value: ${account_info['equity']:,.2f}")
        print(f"💰 Cash Available: ${account_info['cash']:,.2f}")
        
        # Get current positions
        positions = alpaca_client.get_positions()
        print(f"📈 Current Positions: {len(positions)}")
        
        if positions:
            print("\nCurrent Holdings:")
            for pos in positions:
                symbol = pos.get('symbol', '')
                qty = float(pos.get('qty', 0))
                market_value = float(pos.get('market_value', 0))
                unrealized_pl = float(pos.get('unrealized_pl', 0))
                side = 'LONG' if qty > 0 else 'SHORT'
                print(f"  {symbol}: {abs(qty)} shares ({side}), "
                      f"Value: ${market_value:,.2f}, P&L: ${unrealized_pl:,.2f}")
        
        print(f"\n🔄 RUNNING TRADING CYCLE")
        print("-" * 40)
        
        # Run trading cycle with workflow
        result = await trading_workflow.run_cycle()
        
        print(f"✅ Trading cycle completed")
        print(f"Result: {result}")
        
        # Check if any trades were made
        new_positions = alpaca_client.get_positions()
        if len(new_positions) != len(positions):
            print(f"📊 Position changes detected:")
            print(f"   Before: {len(positions)} positions")
            print(f"   After: {len(new_positions)} positions")
        else:
            print("📊 No position changes this cycle")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during trading cycle: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_sentiment_analysis():
    """Test the new sentiment analysis directly."""
    
    print(f"\n🧠 TESTING NEW SENTIMENT ANALYSIS")
    print("-" * 40)
    
    try:
        from agents.sentiment_agent import sentiment_agent
        
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        for symbol in test_symbols:
            print(f"\nAnalyzing {symbol}...")
            result = await sentiment_agent.analyze_comprehensive_sentiment(symbol)
            
            print(f"  Sentiment: {result.overall_sentiment}")
            print(f"  Score: {result.overall_score:.3f}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Data sources: {result.data_sources_count}")
            
            if result.key_themes:
                print(f"  Key themes: {', '.join(result.key_themes[:3])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing sentiment analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    
    print("🧪 TRADING SYSTEM TEST WITH NEW SENTIMENT ANALYSIS")
    print("=" * 70)
    
    # Test sentiment analysis first
    sentiment_test = await test_sentiment_analysis()
    
    if sentiment_test:
        print(f"\n✅ Sentiment analysis test passed")
    else:
        print(f"\n❌ Sentiment analysis test failed")
    
    # Test trading cycle
    cycle_result = await test_trading_cycle()
    
    if cycle_result:
        print(f"\n✅ Trading cycle test completed")
    else:
        print(f"\n❌ Trading cycle test failed")
    
    print(f"\n" + "=" * 70)
    print("🏁 TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())