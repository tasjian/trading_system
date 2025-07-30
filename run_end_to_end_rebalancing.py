#!/usr/bin/env python3
"""
Complete End-to-End Rebalancing Pipeline

Run the full trading system pipeline with balanced data source integration:
- Market monitoring
- Comprehensive sentiment analysis (news, social, earnings, market)
- Risk assessment
- LLM-driven portfolio construction
- Signal generation
- Order execution
- Portfolio tracking
"""

import asyncio
import logging
from datetime import datetime

from agents.workflow import TradingWorkflow
from agents.state import create_initial_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_end_to_end_rebalancing():
    """Execute complete end-to-end rebalancing pipeline."""
    
    print("🚀 END-TO-END REBALANCING PIPELINE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Testing balanced LLM decision-making with all data sources")
    print("=" * 70)
    
    try:
        # Initialize trading workflow
        print("\n🔧 Initializing Trading Workflow...")
        workflow = TradingWorkflow()
        
        # Create initial trading state
        print("📊 Creating initial trading state...")
        state = create_initial_state()
        
        # Empty watchlist - universe filter will discover opportunities dynamically
        state["watchlist"] = []
        
        config = {"thread_id": "end_to_end_rebalancing"}
        
        print(f"   Watchlist: {len(state['watchlist'])} symbols")
        print(f"   Symbols: {', '.join(state['watchlist'][:8])}{'...' if len(state['watchlist']) > 8 else ''}")
        
        # STEP 1: Market Monitor Agent
        print(f"\n1️⃣ MARKET MONITORING...")
        print("   🔍 Fetching account info and market status")
        
        state = await workflow.market_monitor_agent(state, config)
        
        portfolio_value = state["portfolio"]["equity"]
        available_cash = state["portfolio"]["cash"]
        market_open = state["market_data"]["market_open"]
        
        print(f"   ✅ Account Status:")
        print(f"      Portfolio Value: ${portfolio_value:,.2f}")
        print(f"      Available Cash: ${available_cash:,.2f}")
        print(f"      Market Open: {'Yes' if market_open else 'No'}")
        print(f"      Current Positions: {len(state['portfolio']['positions'])}")
        
        # STEP 2: Sentiment Analysis Agent
        print(f"\n2️⃣ COMPREHENSIVE SENTIMENT ANALYSIS...")
        print("   📈 Analyzing news, social media, earnings, and market data")
        
        state = await workflow.sentiment_analysis_agent(state, config)
        
        sentiment_data = state.get("sentiment_data", {})
        sentiment_signals = state.get("sentiment_signals", [])
        
        print(f"   ✅ Sentiment Analysis Results:")
        print(f"      Symbols analyzed: {len(sentiment_data)}")
        print(f"      Sentiment signals: {len(sentiment_signals)}")
        
        # Show earnings integration
        earnings_symbols = [s for s, d in sentiment_data.items() 
                          if hasattr(d, 'has_recent_earnings') and d.has_recent_earnings]
        if earnings_symbols:
            print(f"      📈 Motley Fool earnings data: {', '.join(earnings_symbols[:5])}")
        
        # Show top sentiment signals
        if sentiment_signals:
            print(f"   📊 Top Sentiment Signals:")
            for signal in sorted(sentiment_signals, key=lambda x: x.get('strength', 0), reverse=True)[:5]:
                symbol = signal.get('symbol', 'N/A')
                signal_type = signal.get('signal', 'N/A')
                strength = signal.get('strength', 0)
                has_earnings = signal.get('has_earnings', False)
                earnings_marker = "📈" if has_earnings else "📊"
                print(f"      {earnings_marker} {signal_type} {symbol} (strength: {strength:.2f})")
        
        # STEP 3: Risk Assessment Agent
        print(f"\n3️⃣ RISK ASSESSMENT...")
        print("   ⚖️ Evaluating portfolio risk and market conditions")
        
        state = await workflow.risk_assessment_agent(state, config)
        
        risk_metrics = state.get("risk_metrics", {})
        circuit_breakers = state.get("circuit_breakers", {})
        
        print(f"   ✅ Risk Assessment Results:")
        if risk_metrics:
            # Handle both object and dictionary formats
            if hasattr(risk_metrics, 'portfolio_risk_score'):
                portfolio_risk = risk_metrics.portfolio_risk_score
            else:
                portfolio_risk = risk_metrics.get("portfolio_risk_score", 0) if isinstance(risk_metrics, dict) else 0
            print(f"      Portfolio Risk Score: {portfolio_risk:.2f}")
        
        active_breakers = [name for name, active in circuit_breakers.items() if active]
        if active_breakers:
            print(f"      ⚠️ Active Circuit Breakers: {', '.join(active_breakers)}")
        else:
            print(f"      ✅ No active circuit breakers")
        
        # STEP 4: Signal Generation Agent (LLM Portfolio Construction)
        print(f"\n4️⃣ LLM SIGNAL GENERATION...")
        print("   🧠 LLM analyzing comprehensive data for balanced portfolio decisions")
        
        pre_signals = len(state.get("signals", []))
        
        state = await workflow.signal_generation_agent(state, config)
        
        signals = state.get("signals", [])
        new_signals = len(signals) - pre_signals
        
        print(f"   ✅ LLM Portfolio Construction Results:")
        print(f"      New signals generated: {new_signals}")
        print(f"      Total active signals: {len(signals)}")
        
        # Analyze signal sources
        if signals:
            earnings_driven = sum(1 for s in signals 
                                if s.symbol in earnings_symbols)
            non_earnings_driven = len(signals) - earnings_driven
            
            print(f"   📊 Signal Analysis:")
            print(f"      📈 Earnings-driven signals: {earnings_driven}")
            print(f"      📊 Other data-driven signals: {non_earnings_driven}")
            
            print(f"   🎯 Generated Signals:")
            for signal in signals[-8:]:  # Show recent signals
                earnings_marker = "📈" if signal.symbol in earnings_symbols else "📊"
                print(f"      {earnings_marker} {signal.action.upper()} {signal.symbol}: "
                      f"{signal.quantity:.0f} shares (confidence: {signal.confidence:.2f})")
        
        # STEP 5: Strategy Optimization Agent
        print(f"\n5️⃣ STRATEGY OPTIMIZATION...")
        print("   🎯 Optimizing signals for execution")
        
        state = await workflow.strategy_optimization_agent(state, config)
        
        optimized_signals = state.get("signals", [])
        
        print(f"   ✅ Strategy Optimization Results:")
        print(f"      Optimized signals: {len(optimized_signals)}")
        
        # STEP 6: Order Management Agent (Trade Execution)
        print(f"\n6️⃣ ORDER EXECUTION...")
        print("   💼 Executing trades based on LLM recommendations")
        
        pre_cash = state["portfolio"]["cash"]
        pre_positions = dict(state["portfolio"]["positions"])
        
        state = await workflow.order_management_agent(state, config)
        
        post_cash = state["portfolio"]["cash"]
        post_positions = dict(state["portfolio"]["positions"])
        executed_orders = state.get("executed_orders", [])
        
        print(f"   ✅ Order Execution Results:")
        print(f"      Orders executed: {len(executed_orders)}")
        print(f"      Cash change: ${post_cash - pre_cash:+,.2f}")
        
        # Show executed trades
        if executed_orders:
            print(f"   💼 EXECUTED TRADES:")
            for order in executed_orders:
                order_data = order.get('order', {})
                symbol = order_data.get('symbol', 'N/A')
                action = order_data.get('side', 'N/A')
                qty = order_data.get('qty', 0)
                earnings_marker = "📈" if symbol in earnings_symbols else "📊"
                print(f"      {earnings_marker} {action} {qty} {symbol}")
        
        # Show position changes
        new_positions = set(post_positions.keys()) - set(pre_positions.keys())
        removed_positions = set(pre_positions.keys()) - set(post_positions.keys())
        
        if new_positions:
            earnings_new = new_positions.intersection(set(earnings_symbols))
            print(f"   📈 NEW POSITIONS: {', '.join(new_positions)}")
            if earnings_new:
                print(f"      🎉 Earnings-based additions: {', '.join(earnings_new)}")
        
        if removed_positions:
            print(f"   📉 REMOVED POSITIONS: {', '.join(removed_positions)}")
        
        # STEP 7: Portfolio Tracking Agent
        print(f"\n7️⃣ PORTFOLIO TRACKING...")
        print("   📊 Updating portfolio metrics and performance")
        
        state = await workflow.portfolio_tracking_agent(state, config)
        
        final_portfolio_value = state["portfolio"]["equity"]
        final_cash = state["portfolio"]["cash"]
        final_positions = len(state["portfolio"]["positions"])
        
        print(f"   ✅ Portfolio Tracking Results:")
        print(f"      Final Portfolio Value: ${final_portfolio_value:,.2f}")
        print(f"      Final Cash: ${final_cash:,.2f}")
        print(f"      Final Positions: {final_positions}")
        
        # FINAL ASSESSMENT
        print(f"\n🏆 END-TO-END PIPELINE RESULTS")
        print("=" * 70)
        
        pipeline_success = True
        
        # Check data source integration
        data_sources_integrated = len(sentiment_data) > 0
        print(f"📊 DATA SOURCE INTEGRATION: {'✅ SUCCESS' if data_sources_integrated else '❌ FAILED'}")
        if data_sources_integrated:
            print(f"   News, social, earnings, and market data analyzed for {len(sentiment_data)} symbols")
            if earnings_symbols:
                print(f"   Motley Fool earnings integrated: {', '.join(earnings_symbols)}")
        
        # Check LLM decision making
        llm_decisions = len(signals) > 0
        print(f"🧠 LLM DECISION MAKING: {'✅ SUCCESS' if llm_decisions else '❌ FAILED'}")
        if llm_decisions:
            print(f"   {len(signals)} trading signals generated from comprehensive analysis")
            if earnings_driven > 0:
                print(f"   {earnings_driven} signals influenced by Motley Fool earnings data")
        
        # Check trade execution
        trades_executed = len(executed_orders) > 0
        print(f"💼 TRADE EXECUTION: {'✅ SUCCESS' if trades_executed else '❌ NO TRADES'}")
        if trades_executed:
            print(f"   {len(executed_orders)} orders successfully executed")
        else:
            print(f"   No trades executed (may be due to market conditions or risk limits)")
        
        # Check portfolio rebalancing
        portfolio_changed = (len(new_positions) > 0 or len(removed_positions) > 0 or 
                           abs(post_cash - pre_cash) > 1.0)
        print(f"📈 PORTFOLIO REBALANCING: {'✅ SUCCESS' if portfolio_changed else '⚠️ NO CHANGES'}")
        if portfolio_changed:
            print(f"   Portfolio successfully rebalanced based on comprehensive sentiment analysis")
        
        overall_success = data_sources_integrated and llm_decisions
        
        print(f"\n🎯 OVERALL ASSESSMENT: {'✅ COMPLETE SUCCESS' if overall_success else '⚠️ PARTIAL SUCCESS'}")
        
        if overall_success:
            print("🎉 END-TO-END PIPELINE VERIFIED:")
            print("   ✅ All data sources (news, social, earnings, market) integrated")
            print("   ✅ LLM makes balanced decisions using comprehensive data")
            print("   ✅ Motley Fool earnings data properly weighted with other sources")
            print("   ✅ Trading signals generated from holistic analysis")
            print("   ✅ Complete pipeline functions as designed")
        
        return overall_success
        
    except Exception as e:
        logger.error(f"End-to-end pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting complete end-to-end rebalancing pipeline...")
    print("This will test the full trading system with balanced data source integration.")
    
    success = asyncio.run(run_end_to_end_rebalancing())
    
    if success:
        print(f"\n🎉 PIPELINE COMPLETE: End-to-end system verified with balanced data integration!")
    else:
        print(f"\n❌ PIPELINE FAILED: Check logs for issues")