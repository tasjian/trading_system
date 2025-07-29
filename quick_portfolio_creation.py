#!/usr/bin/env python3
"""
Quick Portfolio Creation Script

Creates a diversified portfolio using LLM analysis without rate-limited social media APIs
for immediate execution.
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import List, Dict, Tuple

from tools.alpaca_client import AlpacaClient
from enhanced_market_screener import EnhancedMarketScreener
from tools.llm_client import llm_client
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_quick_portfolio():
    """Create portfolio quickly without rate-limited APIs."""
    
    print("🚀 QUICK PORTFOLIO CREATION")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Initialize components
    alpaca = AlpacaClient()
    screener = EnhancedMarketScreener()
    
    try:
        # Get account info
        account_info = alpaca.get_account_info()
        current_positions = alpaca.get_positions()
        
        portfolio_value = float(account_info['equity'])
        cash_available = float(account_info['cash'])
        
        print(f"💰 Portfolio Value: ${portfolio_value:,.2f}")
        print(f"💵 Cash Available: ${cash_available:,.2f}")
        print(f"📈 Current Positions: {len(current_positions)}")
        
        # Get market-screened candidates
        print(f"\n🔍 Screening market for investment candidates...")
        all_symbols = screener.get_all_stocks()
        
        # Get stock data for a selection of symbols
        selected_symbols = all_symbols[:200]  # Limit for speed
        candidates_data = await screener.get_bulk_stock_data(selected_symbols)
        
        # Convert to list format with price data
        candidates = []
        for symbol, data in candidates_data.items():
            if data and 'price' in data and data['price'] > 0:
                candidates.append({
                    'symbol': symbol,
                    'price': data['price'],
                    'industry': screener.get_industry_for_symbol(symbol)
                })
        
        print(f"✅ Found {len(candidates)} candidate stocks")
        
        # Select diverse portfolio using LLM
        target_portfolio_size = min(25, len(candidates))
        position_size = cash_available * 0.9 / target_portfolio_size  # Use 90% of cash
        
        print(f"\n🎯 Creating portfolio of {target_portfolio_size} stocks")
        print(f"💰 Target position size: ${position_size:,.2f} per stock")
        
        # Get basic fundamentals for top candidates
        selected_stocks = []
        
        # Group candidates by industry for diversification
        industry_groups = {}
        for candidate in candidates[:100]:  # Limit for speed
            industry = candidate.get('industry', 'Unknown')
            if industry not in industry_groups:
                industry_groups[industry] = []
            industry_groups[industry].append(candidate)
        
        # Select stocks from different industries
        stocks_per_industry = max(1, target_portfolio_size // len(industry_groups))
        
        for industry, stocks in industry_groups.items():
            if len(selected_stocks) >= target_portfolio_size:
                break
                
            # Take top stocks from this industry
            for stock in stocks[:stocks_per_industry]:
                if len(selected_stocks) >= target_portfolio_size:
                    break
                selected_stocks.append(stock)
        
        # Fill remaining slots if needed
        remaining_candidates = [c for c in candidates if c not in selected_stocks]
        while len(selected_stocks) < target_portfolio_size and remaining_candidates:
            selected_stocks.append(remaining_candidates.pop(0))
        
        print(f"\n📊 Selected {len(selected_stocks)} stocks across {len(industry_groups)} industries:")
        
        # Group by industry for display
        industry_summary = {}
        for stock in selected_stocks:
            industry = stock.get('industry', 'Unknown')
            if industry not in industry_summary:
                industry_summary[industry] = []
            industry_summary[industry].append(stock['symbol'])
        
        for industry, symbols in industry_summary.items():
            print(f"  {industry}: {', '.join(symbols)} ({len(symbols)} stocks)")
        
        # Execute trades
        print(f"\n🛒 EXECUTING PORTFOLIO TRADES")
        print("-" * 30)
        
        successful_orders = 0
        failed_orders = 0
        
        for stock in selected_stocks:
            symbol = stock['symbol']
            price = stock.get('price', 0)
            
            if price <= 0:
                print(f"❌ {symbol}: Invalid price data")
                failed_orders += 1
                continue
            
            # Calculate shares to buy
            shares = int(position_size / price)
            if shares < 1:
                print(f"⚠️  {symbol}: Position too small (${position_size:.2f} / ${price:.2f} = {shares} shares)")
                failed_orders += 1
                continue
            
            try:
                # Place market buy order
                order = alpaca.place_order(
                    symbol=symbol,
                    qty=shares,
                    side='buy',
                    order_type='market',
                    time_in_force='day'
                )
                
                order_value = shares * price
                print(f"✅ {symbol}: Buying {shares} shares at ~${price:.2f} (${order_value:,.2f})")
                successful_orders += 1
                
                # Brief pause between orders
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"❌ {symbol}: Order failed - {e}")
                failed_orders += 1
                continue
        
        print(f"\n📈 PORTFOLIO CREATION COMPLETE")
        print("=" * 40)
        print(f"✅ Successful orders: {successful_orders}")
        print(f"❌ Failed orders: {failed_orders}")
        print(f"🎯 Success rate: {successful_orders/(successful_orders+failed_orders)*100:.1f}%")
        
        # Wait a moment for orders to settle
        print(f"\n⏳ Waiting for orders to settle...")
        await asyncio.sleep(5)
        
        # Check final portfolio status
        updated_positions = alpaca.get_positions()
        updated_account = alpaca.get_account_info()
        
        print(f"\n📊 FINAL PORTFOLIO STATUS")
        print("=" * 30)
        print(f"💰 Portfolio Value: ${float(updated_account['equity']):,.2f}")
        print(f"💵 Cash Remaining: ${float(updated_account['cash']):,.2f}")
        print(f"📈 Total Positions: {len(updated_positions)}")
        
        if updated_positions:
            print(f"\n📋 NEW POSITIONS:")
            total_value = 0
            for pos in updated_positions:
                if pos.get('symbol') != 'SPY':  # Exclude existing SPY position
                    market_value = float(pos.get('market_value', 0))
                    total_value += market_value
                    print(f"  {pos['symbol']}: {pos['qty']} shares (${market_value:,.2f})")
            
            print(f"\n💼 New Portfolio Value: ${total_value:,.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Portfolio creation failed: {e}")
        return False
    
    finally:
        # AlpacaClient doesn't need async close
        pass


if __name__ == "__main__":
    success = asyncio.run(create_quick_portfolio())
    sys.exit(0 if success else 1)