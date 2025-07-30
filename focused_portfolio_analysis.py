#!/usr/bin/env python3
"""
Focused Portfolio Analysis
Analyze current holdings performance and identify underperformers for replacement.
"""

import asyncio
import logging
from datetime import datetime
from tools.alpaca_client import alpaca_client
from agents.workflow import TradingWorkflow
from agents.state import create_initial_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def analyze_current_portfolio():
    """Analyze current portfolio holdings for underperformers."""
    
    print("🔄 FOCUSED PORTFOLIO ANALYSIS")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Analyzing current holdings and identifying underperformers")
    print("=" * 60)
    
    try:
        # Get current portfolio data
        print("\n📊 Getting current portfolio...")
        account_info = alpaca_client.get_account_info()
        positions_list = alpaca_client.get_positions()
        
        portfolio_value = account_info["equity"]
        available_cash = account_info["cash"]
        
        print(f"   Portfolio Value: ${portfolio_value:,.2f}")
        print(f"   Available Cash: ${available_cash:,.2f}")
        print(f"   Current Holdings: {len(positions_list)}")
        
        if not positions_list:
            print("   ⚠️ No current positions to analyze")
            return False
        
        # Convert to dictionary for easier processing
        current_positions = {}
        for pos in positions_list:
            symbol = pos.get('symbol')
            if symbol:
                current_positions[symbol] = pos
        
        # Get top holdings by value for focused analysis
        sorted_positions = sorted(current_positions.items(), 
                                key=lambda x: float(x[1].get('market_value', 0)), 
                                reverse=True)
        
        # Focus on top 15 holdings for comprehensive analysis
        top_holdings = sorted_positions[:15]
        focus_symbols = [symbol for symbol, _ in top_holdings]
        
        print(f"\n🎯 FOCUSING ON TOP {len(focus_symbols)} HOLDINGS:")
        total_analyzed_value = 0
        for symbol, pos in top_holdings:
            qty = float(pos.get('qty', 0))
            value = float(pos.get('market_value', 0))
            weight = (value / portfolio_value * 100) if portfolio_value > 0 else 0
            total_analyzed_value += value
            print(f"   {symbol}: {qty:.1f} shares = ${value:,.2f} ({weight:.1f}%)")
        
        analysis_coverage = (total_analyzed_value / portfolio_value * 100) if portfolio_value > 0 else 0
        print(f"\n   Analysis Coverage: ${total_analyzed_value:,.2f} ({analysis_coverage:.1f}% of portfolio)")
        
        # Initialize workflow for sentiment analysis
        print(f"\n🧠 RUNNING SENTIMENT ANALYSIS...")
        workflow = TradingWorkflow()
        state = create_initial_state()
        state["watchlist"] = focus_symbols
        config = {"thread_id": "focused_portfolio_analysis"}
        
        # Run sentiment analysis on focused holdings
        print("   📈 Analyzing comprehensive sentiment (news, earnings, social, market)")
        state = await workflow.sentiment_analysis_agent(state, config)
        
        sentiment_data = state.get("sentiment_data", {})
        print(f"   ✅ Sentiment analysis complete for {len(sentiment_data)} symbols")
        
        # Analyze each holding
        print(f"\n📊 HOLDINGS PERFORMANCE ANALYSIS:")
        
        underperformers = []
        strong_performers = []
        earnings_integrated = []
        
        for symbol in focus_symbols:
            if symbol in sentiment_data:
                sentiment = sentiment_data[symbol]
                # Handle both object and dictionary formats
                if isinstance(sentiment, dict):
                    score = sentiment.get('overall_score', 0.0)
                    confidence = sentiment.get('confidence', 0.5)
                    has_earnings = sentiment.get('has_recent_earnings', False)
                else:
                    score = sentiment.overall_score
                    confidence = sentiment.confidence
                    has_earnings = getattr(sentiment, 'has_recent_earnings', False)
                
                # Get position data
                position = current_positions[symbol]
                value = float(position.get('market_value', 0))
                weight = (value / portfolio_value * 100) if portfolio_value > 0 else 0
                
                if has_earnings:
                    earnings_integrated.append(symbol)
                
                # Classify performance with weighted consideration of portfolio impact
                if score < -0.15:  # Lowered threshold for focused analysis
                    underperformers.append((symbol, score, confidence, value, weight))
                    status = "🔴 UNDERPERFORMING"
                elif score > 0.25:  # Lowered threshold for focused analysis
                    strong_performers.append((symbol, score, confidence, value, weight))
                    status = "🟢 STRONG"
                else:
                    status = "🟡 NEUTRAL"
                
                earnings_marker = "📈" if has_earnings else "📊"
                print(f"   {earnings_marker} {symbol}: {status}")
                print(f"      Score: {score:.3f}, Confidence: {confidence:.3f}")
                print(f"      Position: ${value:,.2f} ({weight:.1f}% of portfolio)")
            else:
                print(f"   ❓ {symbol}: NO SENTIMENT DATA")
        
        # Show Motley Fool earnings integration
        if earnings_integrated:
            print(f"\n📈 MOTLEY FOOL EARNINGS DATA INTEGRATED:")
            print(f"   {', '.join(earnings_integrated)}")
        
        # Report findings
        print(f"\n🎯 ANALYSIS RESULTS:")
        
        if underperformers:
            total_underperformer_value = sum(value for _, _, _, value, _ in underperformers)
            underperformer_weight = (total_underperformer_value / portfolio_value * 100) if portfolio_value > 0 else 0
            
            print(f"   🔴 UNDERPERFORMERS FOUND: {len(underperformers)}")
            print(f"      Total Value: ${total_underperformer_value:,.2f} ({underperformer_weight:.1f}% of portfolio)")
            
            for symbol, score, confidence, value, weight in underperformers:
                earnings_note = " (with earnings data)" if symbol in earnings_integrated else ""
                print(f"      • {symbol}: score {score:.3f}, ${value:,.2f} ({weight:.1f}%){earnings_note}")
        else:
            print(f"   ✅ NO UNDERPERFORMERS DETECTED")
            print(f"      All analyzed holdings showing neutral or positive sentiment")
        
        if strong_performers:
            total_strong_value = sum(value for _, _, _, value, _ in strong_performers)
            strong_weight = (total_strong_value / portfolio_value * 100) if portfolio_value > 0 else 0
            
            print(f"   🟢 STRONG PERFORMERS: {len(strong_performers)}")
            print(f"      Total Value: ${total_strong_value:,.2f} ({strong_weight:.1f}% of portfolio)")
            
            for symbol, score, confidence, value, weight in strong_performers:
                earnings_note = " (with earnings data)" if symbol in earnings_integrated else ""
                print(f"      • {symbol}: score {score:.3f}, ${value:,.2f} ({weight:.1f}%){earnings_note}")
        
        # Generate recommendations
        print(f"\n💡 REBALANCING RECOMMENDATIONS:")
        
        if underperformers:
            print("   🔄 RECOMMENDED ACTIONS:")
            print("      1. Consider reducing positions in underperforming holdings")
            print("      2. Reallocate capital to stronger sentiment opportunities")
            print("      3. Maintain risk management and diversification")
            
            # Suggest specific actions for largest underperformers
            largest_underperformers = sorted(underperformers, key=lambda x: x[3], reverse=True)[:3]
            if largest_underperformers:
                print("   🎯 PRIORITY ACTIONS (largest positions):")
                for symbol, score, confidence, value, weight in largest_underperformers:
                    print(f"      • Reduce {symbol}: ${value:,.2f} position (score: {score:.3f})")
        else:
            print("   🎉 PORTFOLIO PERFORMING WELL:")
            print("      Current holdings show balanced sentiment")
            print("      Continue monitoring for emerging opportunities")
        
        # Final assessment
        success = len(sentiment_data) > 0
        
        print(f"\n🏆 ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"📊 COVERAGE: {len(sentiment_data)}/{len(focus_symbols)} symbols analyzed ({len(sentiment_data)/len(focus_symbols)*100:.0f}%)")
        print(f"📈 EARNINGS DATA: {len(earnings_integrated)} symbols with Motley Fool integration")
        print(f"🔴 UNDERPERFORMERS: {len(underperformers)} positions requiring attention")
        print(f"🟢 STRONG PERFORMERS: {len(strong_performers)} positions showing strength")
        
        data_integration = "✅ SUCCESS" if len(sentiment_data) > 0 else "❌ FAILED"
        print(f"🧠 DATA INTEGRATION: {data_integration}")
        
        if len(earnings_integrated) > 0:
            print(f"📈 EARNINGS INTEGRATION: ✅ SUCCESS ({len(earnings_integrated)} symbols)")
        
        recommendation_quality = "✅ HIGH QUALITY" if len(sentiment_data) >= len(focus_symbols) * 0.8 else "⚠️ PARTIAL"
        print(f"💡 RECOMMENDATION QUALITY: {recommendation_quality}")
        
        return success
        
    except Exception as e:
        logger.error(f"Portfolio analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting focused portfolio analysis...")
    print("This will analyze current holdings for underperformers.")
    
    success = asyncio.run(analyze_current_portfolio())
    
    if success:
        print(f"\n🎉 ANALYSIS COMPLETE: Portfolio analyzed with comprehensive sentiment data!")
    else:
        print(f"\n❌ ANALYSIS FAILED: Check logs for issues")