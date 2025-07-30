#!/usr/bin/env python3
"""
Test Signal to Trade Execution
Simplified test focusing on LLM signal generation and actual trade execution.
"""

import asyncio
import logging
from datetime import datetime

from agents.workflow import TradingWorkflow
from agents.state import create_initial_state, TradingSignal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_signal_to_trade_execution():
    """Test the signal generation to trade execution path."""
    
    print("🎯 SIGNAL TO TRADE EXECUTION TEST")
    print("=" * 50)
    print(f"Testing LLM signal generation → actual trade execution")
    print("=" * 50)
    
    try:
        # Initialize workflow
        workflow = TradingWorkflow()
        state = create_initial_state()
        config = {"thread_id": "test_signal_to_trade"}
        
        # Set up watchlist
        state["watchlist"] = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
        
        print("\n1️⃣ MARKET MONITOR")
        start_time = datetime.now()
        state = await workflow.market_monitor_agent(state, config)
        portfolio_value = state["portfolio"]["equity"]
        buying_power = state["portfolio"]["buying_power"]
        duration = (datetime.now() - start_time).total_seconds()
        print(f"✅ Portfolio: ${portfolio_value:,.2f}, Buying Power: ${buying_power:,.2f} ({duration:.1f}s)")
        
        print("\n2️⃣ MOCK SENTIMENT DATA")
        # Create mock sentiment data for faster testing
        mock_sentiment_data = {
            "AAPL": {
                'overall_sentiment': 'positive',
                'overall_score': 0.7,
                'confidence': 0.8,
                'data_sources_count': 3,
                'has_recent_earnings': False,
                'key_themes': ['AI growth', 'Strong iPhone sales', 'Services expansion'],
                'opportunities': ['AI integration', 'Market expansion']
            },
            "MSFT": {
                'overall_sentiment': 'positive',
                'overall_score': 0.6,
                'confidence': 0.75,
                'data_sources_count': 3,
                'has_recent_earnings': False,
                'key_themes': ['Cloud growth', 'AI leadership', 'Enterprise strength'],
                'opportunities': ['Azure expansion', 'AI monetization']
            },
            "GOOGL": {
                'overall_sentiment': 'neutral',
                'overall_score': 0.2,
                'confidence': 0.7,
                'data_sources_count': 2,
                'has_recent_earnings': False,
                'key_themes': ['Search dominance', 'Cloud competition', 'Regulatory pressure'],
                'opportunities': ['Gemini AI', 'YouTube growth']
            }
        }
        
        state["sentiment_data"] = mock_sentiment_data
        state["filtered_symbols"] = list(mock_sentiment_data.keys())
        print(f"✅ Mock sentiment data created for {len(mock_sentiment_data)} symbols")
        
        print("\n3️⃣ RISK ASSESSMENT")
        start_time = datetime.now()
        state = await workflow.risk_assessment_agent(state, config)
        circuit_breakers = state.get("circuit_breakers", {})
        active_breakers = [name for name, active in circuit_breakers.items() if active]
        duration = (datetime.now() - start_time).total_seconds()
        print(f"✅ Risk check complete, {len(active_breakers)} active breakers ({duration:.1f}s)")
        
        print("\n4️⃣ LLM SIGNAL GENERATION")
        start_time = datetime.now()
        pre_signals = len(state.get("signals", []))
        state = await workflow.signal_generation_agent(state, config)
        post_signals = len(state.get("signals", []))
        signals_generated = post_signals - pre_signals
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ LLM Signal Generation:")
        print(f"   🧠 Signals generated: {signals_generated}")
        print(f"   ⏱️ Generation time: {duration:.1f}s")
        
        if signals_generated > 0:
            print(f"   📊 Generated signals:")
            for i, signal in enumerate(state.get("signals", [])[-signals_generated:]):
                print(f"      {i+1}. {signal.symbol}: {signal.action} (confidence: {signal.confidence:.2f})")
        
        print("\n5️⃣ STRATEGY OPTIMIZATION")
        start_time = datetime.now()
        state = await workflow.strategy_optimization_agent(state, config)
        duration = (datetime.now() - start_time).total_seconds()
        print(f"✅ Strategy optimization complete ({duration:.1f}s)")
        
        print("\n6️⃣ ORDER MANAGEMENT - ACTUAL TRADE EXECUTION")
        if signals_generated > 0:
            start_time = datetime.now()
            pre_orders = len(state.get("executed_orders", []))
            state = await workflow.order_management_agent(state, config)
            post_orders = len(state.get("executed_orders", []))
            orders_executed = post_orders - pre_orders
            duration = (datetime.now() - start_time).total_seconds()
            
            print(f"✅ Order Management Results:")
            print(f"   💼 Orders executed: {orders_executed}")
            print(f"   ⏱️ Execution time: {duration:.1f}s")
            
            if orders_executed > 0:
                executed_orders = state.get("executed_orders", [])
                print(f"   📋 Executed Orders:")
                for order_info in executed_orders[-orders_executed:]:
                    order = order_info.get('order', {})
                    print(f"      🎯 {order.get('symbol')}: {order.get('side')} {order.get('qty')} - ID: {order.get('id')}")
                    print(f"         Status: {order.get('status')}")
            else:
                print(f"   ❌ No orders were executed")
        else:
            print(f"❌ No signals generated - skipping order management")
            orders_executed = 0
        
        print("\n7️⃣ PORTFOLIO TRACKING")
        start_time = datetime.now()
        state = await workflow.portfolio_tracking_agent(state, config)
        final_portfolio_value = state["portfolio"]["equity"]
        portfolio_change = final_portfolio_value - portfolio_value
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Portfolio tracking complete ({duration:.1f}s)")
        print(f"   💰 Final value: ${final_portfolio_value:,.2f}")
        print(f"   📈 Change: ${portfolio_change:+,.2f}")
        
        # Final assessment
        success = orders_executed > 0
        
        print(f"\n🏆 FINAL RESULTS:")
        print(f"Signals Generated: {signals_generated}")
        print(f"Orders Executed: {orders_executed}")
        print(f"Portfolio Change: ${portfolio_change:+,.2f}")
        print(f"Trade Execution: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        if success:
            print(f"\n🎉 BREAKTHROUGH: Signals successfully converted to actual trades!")
            print(f"The end-to-end pipeline is working!")
        else:
            print(f"\n❌ ISSUE: Generated {signals_generated} signals but executed {orders_executed} orders")
            print(f"Trade execution pipeline needs debugging")
        
        return success
        
    except Exception as e:
        logger.error(f"Signal to trade test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing signal generation to trade execution...")
    
    success = asyncio.run(test_signal_to_trade_execution())
    
    if success:
        print(f"\n🚀 SUCCESS: End-to-end signal to trade execution is working!")
    else:
        print(f"\n🔧 NEEDS FIX: Trade execution still not working properly")