#!/usr/bin/env python3
"""
Test Trade Execution
Focused test to verify that trading signals actually result in executed orders.
"""

import asyncio
import logging
from datetime import datetime

from agents.workflow import TradingWorkflow
from agents.state import create_initial_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_trade_execution():
    """Test that signals are properly converted to executed trades."""
    
    print("💼 TESTING TRADE EXECUTION")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Focused test to verify order execution works")
    print("=" * 50)
    
    try:
        # Initialize workflow
        workflow = TradingWorkflow()
        state = create_initial_state()
        config = {"thread_id": "test_trade_execution"}
        
        # Set up minimal test data
        state["watchlist"] = ["AAPL", "MSFT", "GOOGL"]
        
        print(f"\n1️⃣ MARKET MONITOR")
        print("Getting account info and market data...")
        state = await workflow.market_monitor_agent(state, config)
        portfolio_value = state["portfolio"]["equity"]
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        
        print(f"\n2️⃣ CREATING TEST SIGNALS")
        print("Adding manual trading signals for testing...")
        
        # Import TradingSignal
        from agents.state import TradingSignal
        
        # Create test signals manually
        test_signals = [
            TradingSignal(
                symbol="AAPL",
                action="buy", 
                confidence=0.8,
                quantity=1.0,  # Small quantity for testing
                price_target=220.0,
                stop_loss=200.0,
                reasoning="Test signal for trade execution verification"
            ),
            TradingSignal(
                symbol="MSFT", 
                action="buy",
                confidence=0.7,
                quantity=1.0,  # Small quantity for testing  
                price_target=530.0,
                stop_loss=500.0,
                reasoning="Test signal for trade execution verification"
            )
        ]
        
        # Add signals to state
        state["signals"] = test_signals
        print(f"Created {len(test_signals)} test signals")
        
        for signal in test_signals:
            print(f"  - {signal.symbol}: {signal.action} {signal.quantity} shares (confidence: {signal.confidence})")
        
        print(f"\n3️⃣ ORDER MANAGEMENT")
        print("Executing trade orders...")
        
        pre_orders = len(state.get("executed_orders", []))
        state = await workflow.order_management_agent(state, config)
        post_orders = len(state.get("executed_orders", []))
        
        executed_orders = state.get("executed_orders", [])
        orders_executed = post_orders - pre_orders
        
        print(f"\n✅ TRADE EXECUTION RESULTS:")
        print(f"Orders Executed: {orders_executed}")
        print(f"Total Executed Orders: {len(executed_orders)}")
        
        if executed_orders:
            print(f"\n📋 EXECUTED ORDERS:")
            for i, order_info in enumerate(executed_orders):
                order = order_info.get('order', {})
                print(f"  {i+1}. Order ID: {order.get('id', 'N/A')}")
                print(f"     Symbol: {order.get('symbol', 'N/A')}")
                print(f"     Side: {order.get('side', 'N/A')}")
                print(f"     Quantity: {order.get('qty', 'N/A')}")
                print(f"     Status: {order.get('status', 'N/A')}")
        else:
            print("\n❌ NO ORDERS EXECUTED")
            print("Checking debug information...")
        
        print(f"\n4️⃣ PORTFOLIO TRACKING")
        print("Updating portfolio after trades...")
        
        state = await workflow.portfolio_tracking_agent(state, config)
        final_portfolio_value = state["portfolio"]["equity"]
        portfolio_change = final_portfolio_value - portfolio_value
        
        print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
        print(f"Portfolio Change: ${portfolio_change:+,.2f}")
        
        # Check for new positions
        current_positions = state["portfolio"].get("positions", {})
        print(f"Current Positions: {len(current_positions)}")
        
        if current_positions:
            print("📊 CURRENT POSITIONS:")
            for symbol, position in current_positions.items():
                qty = position.get('qty', 0)
                market_value = position.get('market_value', 0)
                print(f"  {symbol}: {qty} shares, ${market_value:,.2f}")
        
        # Determine success
        success = orders_executed > 0
        
        print(f"\n🏆 TEST RESULTS:")
        print(f"Signals Created: {len(test_signals)}")
        print(f"Orders Executed: {orders_executed}")
        print(f"Trade Execution: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        if not success:
            print(f"\n🔍 DEBUGGING INFO:")
            print("Check the order management logs above for specific error messages")
            print("Verify that Alpaca API credentials are working")
            print("Ensure paper trading account has sufficient buying power")
        
        return success
        
    except Exception as e:
        logger.error(f"Trade execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting trade execution test...")
    print("This will test if trading signals actually result in executed orders.")
    
    success = asyncio.run(test_trade_execution())
    
    if success:
        print(f"\n🎉 SUCCESS: Trade execution is working!")
    else:
        print(f"\n❌ FAILED: Trade execution needs debugging")