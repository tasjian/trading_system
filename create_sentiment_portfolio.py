#!/usr/bin/env python3
"""
Create Portfolio Using Sentiment Analysis

Uses the enhanced sentiment analyzer and LLM portfolio management
to create a diversified portfolio based on sentiment analysis.
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

async def create_sentiment_based_portfolio():
    """Create portfolio using sentiment analysis."""
    
    print("🎯 SENTIMENT-DRIVEN PORTFOLIO CREATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Import required modules
        from agents.llm_portfolio_management import construct_llm_portfolio
        from enhanced_market_screener import enhanced_screener
        from tools.alpaca_client import alpaca_client
        from config.settings import settings
        
        print("📊 Getting account information...")
        account_info = alpaca_client.get_account_info()
        portfolio_value = float(account_info['equity'])
        cash_available = float(account_info['cash'])
        
        print(f"💰 Portfolio Value: ${portfolio_value:,.2f}")
        print(f"💵 Cash Available: ${cash_available:,.2f}")
        
        # Get diversified candidate universe
        print("\n🔍 Building candidate universe...")
        all_candidates = enhanced_screener.get_all_stocks()
        print(f"📈 Total candidate stocks: {len(all_candidates)}")
        
        # Create LLM-enhanced portfolio with sentiment analysis
        print("\n🧠 Analyzing sentiment and creating portfolio...")
        print("This may take a few minutes as we analyze sentiment across multiple sources...")
        
        portfolio_recommendation = await construct_llm_portfolio(
            candidate_symbols=all_candidates,
            portfolio_value=portfolio_value,
            risk_profile="moderate",
            max_positions=25  # Target 25 positions for good diversification
        )
        
        print("\n✅ PORTFOLIO CONSTRUCTION COMPLETE!")
        print("=" * 60)
        
        # Display results
        print(f"🎯 Market Regime: {portfolio_recommendation.market_regime.value}")
        print(f"📈 Expected Return: {portfolio_recommendation.expected_return:.1%}")
        print(f"🎪 Confidence: {portfolio_recommendation.confidence:.1%}")
        print(f"💵 Cash Allocation: {portfolio_recommendation.cash_allocation:.1%}")
        print(f"📊 Diversification Score: {portfolio_recommendation.diversification_score:.2f}")
        print(f"⚠️  Total Risk Score: {portfolio_recommendation.total_risk_score:.2f}")
        
        print(f"\n🏢 PORTFOLIO ALLOCATIONS ({len(portfolio_recommendation.allocations)} positions):")
        print("-" * 80)
        
        total_allocation = 0.0
        by_industry = {}
        
        for i, allocation in enumerate(portfolio_recommendation.allocations, 1):
            weight_pct = allocation.target_weight * 100
            value = portfolio_value * allocation.target_weight
            
            print(f"{i:2d}. {allocation.symbol:<6} {weight_pct:5.1f}% (${value:8,.0f}) "
                  f"Sentiment: {allocation.sentiment_score:+.2f} Risk: {allocation.risk_level}")
            
            total_allocation += allocation.target_weight
            
            # Group by industry
            industry = allocation.industry
            if industry not in by_industry:
                by_industry[industry] = []
            by_industry[industry].append(allocation)
        
        print("-" * 80)
        print(f"Total Allocated: {total_allocation:.1%}")
        print(f"Cash Reserve: {1-total_allocation:.1%}")
        
        # Show industry diversification
        print(f"\n🏭 INDUSTRY DIVERSIFICATION:")
        print("-" * 50)
        
        for industry, allocations in sorted(by_industry.items(), key=lambda x: len(x[1]), reverse=True):
            industry_weight = sum(a.target_weight for a in allocations)
            industry_display = enhanced_screener.get_industry_description(industry)
            print(f"{industry_display:<30} {len(allocations):2d} stocks ({industry_weight:.1%})")
        
        print(f"\n💡 PORTFOLIO RATIONALE:")
        print("-" * 50)
        print(portfolio_recommendation.rationale)
        
        # Ask if user wants to execute trades
        print(f"\n🚀 READY TO EXECUTE TRADES")
        print("-" * 30)
        response = input("Execute these trades? (y/N): ").strip().lower()
        
        if response == 'y':
            print("\n⚡ Executing portfolio trades...")
            
            # Get current positions
            current_positions = alpaca_client.get_positions()
            current_symbols = {pos['symbol'] for pos in current_positions}
            
            # Calculate orders needed
            orders_to_place = []
            
            for allocation in portfolio_recommendation.allocations:
                target_value = portfolio_value * allocation.target_weight
                
                # This is a simplified order calculation
                # In practice, you'd want more sophisticated order management
                if allocation.symbol not in current_symbols:
                    orders_to_place.append({
                        'symbol': allocation.symbol,
                        'action': 'buy',
                        'target_value': target_value,
                        'reasoning': allocation.reasoning
                    })
            
            print(f"📋 Orders to place: {len(orders_to_place)}")
            
            # For safety, we'll just display what would be ordered
            # Actual execution would require more careful order management
            for order in orders_to_place[:10]:  # Show first 10
                print(f"  BUY {order['symbol']}: ${order['target_value']:,.0f}")
            
            if len(orders_to_place) > 10:
                print(f"  ... and {len(orders_to_place)-10} more orders")
            
            print("\n⚠️  For safety, automatic execution is disabled.")
            print("Review the portfolio and use the trading system to execute trades.")
        
        else:
            print("Portfolio creation complete. No trades executed.")
        
        print(f"\n🎉 SENTIMENT-BASED PORTFOLIO READY!")
        
    except Exception as e:
        logger.error(f"Portfolio creation failed: {e}")
        print(f"\n❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(create_sentiment_based_portfolio())
    sys.exit(0 if success else 1)