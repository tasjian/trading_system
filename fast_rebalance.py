#!/usr/bin/env python3
"""
Fast Portfolio Rebalancing - Uses enhanced analysis but avoids API timeouts.

Strategy: Use technical + fundamental analysis with lightweight sentiment,
then execute trades quickly.
"""

import asyncio
import sys
import os
import yfinance as yf
from datetime import datetime

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from tools.alpaca_client import alpaca_client
from core.market_intelligence import UnifiedMarketIntelligence

class FastRebalancer:
    """Fast portfolio rebalancing with enhanced features."""
    
    def __init__(self):
        self.market_intel = UnifiedMarketIntelligence()
        
        # Pre-selected high-quality stocks across sectors
        self.quality_stocks = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV'], 
            'Financial': ['JPM', 'BAC', 'WFC', 'GS'],
            'Consumer': ['PG', 'KO', 'WMT', 'HD'],
            'Industrial': ['BA', 'CAT', 'RTX', 'LMT'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG']
        }
    
    async def fast_rebalance(self):
        """Execute fast portfolio rebalancing."""
        
        print("⚡ FAST PORTFOLIO REBALANCING")
        print("Using Enhanced Analysis with API Optimization")
        print("=" * 55)
        
        # Step 1: Check current state
        account_info = alpaca_client.get_account_info()
        current_positions = alpaca_client.get_positions()
        
        portfolio_value = float(account_info['equity'])
        buying_power = float(account_info['buying_power'])
        
        print(f"💰 Portfolio Value: ${portfolio_value:,.2f}")
        print(f"🔥 Buying Power: ${buying_power:,.2f}")
        print(f"📈 Current Positions: {len(current_positions)}")
        
        # Step 2: Quick analysis of top stocks (no API calls)
        print(f"\n🔍 FAST STOCK ANALYSIS")
        print("-" * 30)
        
        top_picks = await self._fast_stock_selection()
        
        # Step 3: Clear current positions
        if current_positions:
            print(f"\n🔥 CLEARING CURRENT POSITIONS")
            await self._sell_all_positions(current_positions)
        
        # Step 4: Build new portfolio
        await self._build_new_portfolio(top_picks, buying_power)
        
        print(f"\n🎉 FAST REBALANCING COMPLETE!")
    
    async def _fast_stock_selection(self):
        """Quick stock selection using yfinance data and basic analysis."""
        
        candidates = []
        
        print("Analyzing top stocks by sector...")
        
        for sector, stocks in self.quality_stocks.items():
            print(f"\n{sector}:")
            
            for symbol in stocks[:2]:  # Top 2 per sector
                try:
                    # Quick yfinance analysis
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="3mo")
                    
                    if hist.empty:
                        continue
                    
                    # Calculate simple metrics
                    current_price = info.get('currentPrice', hist['Close'].iloc[-1])
                    pe_ratio = info.get('trailingPE', 0)
                    market_cap = info.get('marketCap', 0)
                    
                    # Price momentum (3-month return)
                    price_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
                    
                    # Volume trend
                    recent_volume = hist['Volume'].iloc[-10:].mean()
                    older_volume = hist['Volume'].iloc[-30:-10].mean()
                    volume_trend = (recent_volume / older_volume) - 1 if older_volume > 0 else 0
                    
                    # Simple scoring
                    score = 0.5  # Base score
                    
                    # Price momentum component
                    if price_return > 0.1:
                        score += 0.2
                    elif price_return > 0:
                        score += 0.1
                    elif price_return < -0.1:
                        score -= 0.2
                    
                    # Valuation component  
                    if 0 < pe_ratio < 20:
                        score += 0.1
                    elif pe_ratio > 30:
                        score -= 0.1
                    
                    # Volume component
                    if volume_trend > 0.1:
                        score += 0.1
                    
                    # Market cap preference (large cap stability)
                    if market_cap > 100e9:  # > $100B
                        score += 0.1
                    
                    # Quick sentiment proxy (price vs 50-day average)
                    if len(hist) >= 50:
                        ma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                        if current_price > ma_50 * 1.05:  # 5% above 50-day MA
                            score += 0.2
                        elif current_price < ma_50 * 0.95:  # 5% below 50-day MA  
                            score -= 0.1
                    
                    candidates.append({
                        'symbol': symbol,
                        'sector': sector,
                        'score': score,
                        'price': current_price,
                        'pe_ratio': pe_ratio,
                        'price_return': price_return,
                        'volume_trend': volume_trend
                    })
                    
                    status = "🚀" if score > 0.7 else "📈" if score > 0.6 else "⚖️"
                    print(f"  {symbol}: {status} {score:.3f} | Price: ${current_price:.2f} | 3M: {price_return:.1%}")
                    
                except Exception as e:
                    print(f"  {symbol}: ❌ Analysis failed - {e}")
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n🏆 TOP 8 SELECTIONS:")
        print("-" * 30)
        
        top_8 = candidates[:8]
        for i, stock in enumerate(top_8):
            print(f"{i+1}. {stock['symbol']} ({stock['sector']}): {stock['score']:.3f}")
        
        return top_8
    
    async def _sell_all_positions(self, positions):
        """Quickly sell all current positions."""
        
        print("Selling current positions...")
        
        for pos in positions:
            symbol = pos.get('symbol', '')
            qty = float(pos.get('qty', 0))
            
            try:
                result = alpaca_client.place_order(
                    symbol=symbol,
                    qty=abs(qty),
                    side='sell',
                    order_type='market',
                    time_in_force='day'
                )
                
                if result:
                    print(f"  ✅ Sold {symbol} ({abs(qty)} shares)")
                else:
                    print(f"  ❌ Failed to sell {symbol}")
                    
            except Exception as e:
                print(f"  ❌ Error selling {symbol}: {e}")
        
        # Wait for orders to settle
        if positions:
            print("⏳ Waiting for sell orders to settle...")
            await asyncio.sleep(5)
    
    async def _build_new_portfolio(self, top_picks, buying_power):
        """Build new portfolio with selected stocks."""
        
        print(f"\n🛒 BUILDING NEW PORTFOLIO")
        print("-" * 30)
        
        # Reserve some cash
        investable_capital = buying_power * 0.95  # Use 95% of buying power
        position_size = investable_capital / len(top_picks)
        
        print(f"💰 Investable Capital: ${investable_capital:,.2f}")
        print(f"📊 Target Position Size: ${position_size:,.2f} each")
        
        successful_buys = 0
        
        for stock in top_picks:
            symbol = stock['symbol']
            price = stock['price']
            score = stock['score']
            
            try:
                shares = max(1, int(position_size / price))
                actual_cost = shares * price
                
                print(f"\nBuying {symbol}:")
                print(f"  Shares: {shares} @ ${price:.2f} = ${actual_cost:,.2f}")
                print(f"  Score: {score:.3f} | Sector: {stock['sector']}")
                print(f"  3M Return: {stock['price_return']:.1%}")
                
                result = alpaca_client.place_order(
                    symbol=symbol,
                    qty=shares,
                    side='buy',
                    order_type='market',
                    time_in_force='day'
                )
                
                if result:
                    successful_buys += 1
                    print(f"  ✅ Order placed successfully!")
                    
                    # Show reasoning
                    reasons = []
                    if score > 0.7:
                        reasons.append("High analysis score")
                    if stock['price_return'] > 0.1:
                        reasons.append("Strong 3M momentum")
                    if stock['volume_trend'] > 0.1:
                        reasons.append("Increasing volume")
                    
                    if reasons:
                        print(f"  💡 Why: {', '.join(reasons)}")
                else:
                    print(f"  ❌ Order failed")
                    
            except Exception as e:
                print(f"  ❌ Error buying {symbol}: {e}")
            
            await asyncio.sleep(0.5)
        
        print(f"\n📊 PORTFOLIO CONSTRUCTION SUMMARY:")
        print(f"✅ Successful purchases: {successful_buys}/{len(top_picks)}")
        print(f"📈 Success rate: {successful_buys/len(top_picks)*100:.1f}%")
        
        # Final status check
        await asyncio.sleep(3)
        
        final_account = alpaca_client.get_account_info()
        final_positions = alpaca_client.get_positions()
        
        print(f"\n🎯 FINAL PORTFOLIO:")
        print(f"💰 Portfolio Value: ${float(final_account['equity']):,.2f}")
        print(f"📈 Active Positions: {len(final_positions)}")
        print(f"💵 Remaining Cash: ${float(final_account['cash']):,.2f}")
        
        if final_positions:
            print(f"\nNew Holdings:")
            total_value = 0
            for pos in final_positions:
                symbol = pos.get('symbol', '')
                qty = float(pos.get('qty', 0))
                market_value = float(pos.get('market_value', 0))
                total_value += market_value
                weight = (market_value / float(final_account['equity'])) * 100
                
                print(f"  {symbol}: {qty} shares, ${market_value:,.2f} ({weight:.1f}%)")
            
            print(f"\n📊 Portfolio Diversification:")
            print(f"  Total invested: ${total_value:,.2f}")
            print(f"  Average position: {total_value/len(final_positions):,.0f}")
            print(f"  Cash reserve: {(float(final_account['cash'])/float(final_account['equity']))*100:.1f}%")

async def main():
    """Run fast rebalancing."""
    
    rebalancer = FastRebalancer()
    await rebalancer.fast_rebalance()

if __name__ == "__main__":
    asyncio.run(main())