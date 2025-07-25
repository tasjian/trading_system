#!/usr/bin/env python3
"""
Quick trade execution with basic sentiment analysis (no API delays).
"""

import asyncio
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from core.market_intelligence import UnifiedMarketIntelligence
from tools.alpaca_client import alpaca_client

async def quick_trade():
    """Execute a quick trade with sentiment analysis."""
    
    print("⚡ QUICK TRADE WITH SENTIMENT ANALYSIS")
    print("=" * 50)
    
    try:
        # Get account info
        account_info = alpaca_client.get_account_info()
        buying_power = float(account_info['buying_power'])
        
        print(f"💰 Buying Power: ${buying_power:,.2f}")
        
        if buying_power < 100:
            print("❌ Insufficient buying power for trading")
            return
        
        # Test one symbol with simplified sentiment (avoids API timeouts)
        symbol = "AAPL"
        print(f"\n🎯 Analyzing {symbol}...")
        
        market_intel = UnifiedMarketIntelligence()
        
        # Get basic sentiment (should use fallback if APIs timeout)
        sentiment_score = await market_intel._analyze_sentiment(symbol)
        print(f"🧠 Sentiment Score: {sentiment_score:.3f}")
        
        if sentiment_score is None:
            sentiment_score = 0.0
            print("⚠️ Using neutral sentiment (API timeout)")
        
        # Simple trading logic based on sentiment
        if sentiment_score > 0.3:  # Positive sentiment
            print(f"📈 POSITIVE SENTIMENT - Placing BUY order")
            
            # Calculate small position size
            investment = min(buying_power * 0.05, 200)  # 5% or $200
            
            # Get current price
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            current_price = ticker.info.get('currentPrice', 230)  # Default price for AAPL
            
            shares = max(1, int(investment / current_price))
            
            print(f"   Shares: {shares}")
            print(f"   Est. Cost: ${shares * current_price:.2f}")
            
            # Place order
            result = alpaca_client.place_order(
                symbol=symbol,
                qty=shares,
                side='buy',
                order_type='market',
                time_in_force='day'
            )
            
            if result:
                print(f"✅ BUY ORDER PLACED!")
                print(f"   Order ID: {result}")
                
                # Check new positions
                await asyncio.sleep(2)  # Wait for order to process
                positions = alpaca_client.get_positions()
                print(f"📊 New portfolio has {len(positions)} positions")
                
            else:
                print(f"❌ Order failed")
                
        elif sentiment_score < -0.3:  # Negative sentiment
            print(f"📉 NEGATIVE SENTIMENT - Would place SELL order")
            print("   (Skipping sell for safety in demo)")
            
        else:
            print(f"⚖️ NEUTRAL SENTIMENT - No trade executed")
            print(f"   Sentiment score {sentiment_score:.3f} is between -0.3 and 0.3")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(quick_trade())