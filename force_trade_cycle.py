#!/usr/bin/env python3
"""
Force a trading cycle with sentiment analysis - bypassing some risk restrictions for testing.
"""

import asyncio
import sys
import os
import logging

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from core.market_intelligence import UnifiedMarketIntelligence
from tools.alpaca_client import alpaca_client
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def force_trading_cycle():
    """Force a trading cycle with sentiment analysis."""
    
    print("🚀 FORCE TRADING CYCLE WITH NEW SENTIMENT ANALYSIS")
    print("=" * 60)
    
    try:
        # Get current account info
        account_info = alpaca_client.get_account_info()
        portfolio_value = account_info['equity']
        buying_power = account_info['buying_power']
        
        print(f"📊 Portfolio Value: ${portfolio_value:,.2f}")
        print(f"💰 Buying Power: ${buying_power:,.2f}")
        
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
        
        # Initialize market intelligence with new sentiment analysis
        market_intel = UnifiedMarketIntelligence()
        
        # Test stocks to analyze
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        print(f"\n🧠 ANALYZING STOCKS WITH NEW SENTIMENT SYSTEM")
        print("-" * 50)
        
        best_analysis = None
        best_score = -1
        
        for symbol in test_symbols:
            try:
                print(f"\nAnalyzing {symbol}...")
                
                # This now uses our new LLM-powered sentiment analysis!
                analysis = await market_intel.analyze_stock(symbol)
                
                if analysis:
                    print(f"  Overall Score: {analysis.score:.3f}")
                    print(f"  Signal: {analysis.signal}")
                    print(f"  Confidence: {analysis.confidence}")
                    
                    if hasattr(analysis, 'components') and analysis.components:
                        sentiment_score = analysis.components.get('sentiment', 0)
                        print(f"  🧠 NEW Sentiment Score: {sentiment_score:.3f}")
                        
                        if sentiment_score > 0.5:
                            print(f"    ↳ 📈 Very Positive Sentiment!")
                        elif sentiment_score > 0.2:
                            print(f"    ↳ 📊 Positive Sentiment")
                        elif sentiment_score < -0.2:
                            print(f"    ↳ 📉 Negative Sentiment")
                        else:
                            print(f"    ↳ ⚖️ Neutral Sentiment")
                    
                    # Track best opportunity
                    if analysis.score > best_score and analysis.signal.value in ['buy', 'strong_buy']:
                        best_score = analysis.score
                        best_analysis = (symbol, analysis)
                
            except Exception as e:
                print(f"  ❌ Error analyzing {symbol}: {e}")
        
        # Execute trade on best opportunity if we have buying power
        if best_analysis and buying_power > 100:
            symbol, analysis = best_analysis
            
            print(f"\n🎯 BEST OPPORTUNITY: {symbol}")
            print(f"   Score: {analysis.score:.3f}")
            print(f"   Signal: {analysis.signal}")
            print(f"   Sentiment: {analysis.components.get('sentiment', 0):.3f}")
            
            # Calculate position size (conservative)
            max_investment = min(buying_power * 0.1, 500)  # 10% of buying power or $500, whichever is smaller
            
            # Get current price to calculate shares
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                current_price = ticker.info.get('currentPrice', ticker.info.get('regularMarketPrice', 100))
                
                shares_to_buy = int(max_investment / current_price)
                
                if shares_to_buy > 0:
                    print(f"\n💰 EXECUTING TRADE:")
                    print(f"   Symbol: {symbol}")
                    print(f"   Shares: {shares_to_buy}")
                    print(f"   Est. Cost: ${shares_to_buy * current_price:.2f}")
                    print(f"   Reason: High sentiment + technical analysis")
                    
                    # Place the order
                    result = alpaca_client.place_order(
                        symbol=symbol,
                        qty=shares_to_buy,
                        side='buy',
                        order_type='market',
                        time_in_force='day'
                    )
                    
                    if result:
                        print(f"   ✅ ORDER PLACED SUCCESSFULLY!")
                        print(f"   Order ID: {result}")
                    else:
                        print(f"   ❌ Order failed to place")
                        
                else:
                    print(f"\n⚠️ Position size too small (${max_investment:.2f} / ${current_price:.2f} = {shares_to_buy} shares)")
                    
            except Exception as e:
                print(f"   ❌ Error executing trade: {e}")
        
        elif not best_analysis:
            print(f"\n📊 No strong buy signals found in current market conditions")
        
        elif buying_power <= 100:
            print(f"\n💰 Insufficient buying power (${buying_power:.2f}) for new positions")
        
        # Show updated positions
        print(f"\n📈 FINAL PORTFOLIO STATUS")
        print("-" * 30)
        
        new_positions = alpaca_client.get_positions()
        new_account_info = alpaca_client.get_account_info()
        
        print(f"Portfolio Value: ${new_account_info['equity']:,.2f}")
        print(f"Positions: {len(new_positions)}")
        
        if len(new_positions) != len(positions):
            print(f"🎉 PORTFOLIO CHANGED!")
            print(f"   Before: {len(positions)} positions")
            print(f"   After: {len(new_positions)} positions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during forced trading cycle: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(force_trading_cycle())