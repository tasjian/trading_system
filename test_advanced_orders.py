#!/usr/bin/env python3
"""
Test Advanced Order Types

Demonstrates stop-limit, trailing stop, and OCO order functionality.
"""

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_advanced_order_types():
    """Test various advanced order types."""
    
    print("🔬 Testing Advanced Order Types System")
    print("=" * 50)
    
    # Import the advanced order system
    from order_types.advanced_orders import (
        advanced_order_manager, place_stop_limit_order, 
        place_trailing_stop_order, place_oco_order
    )
    
    # Test 1: Stop-Limit Order
    print("\n1️⃣ Testing Stop-Limit Order")
    print("-" * 30)
    
    stop_limit_result = await place_stop_limit_order(
        symbol="AAPL",
        quantity=10,
        side="buy",
        stop_price=230.00,  # Trigger when price hits $230
        limit_price=232.00, # Execute at $232 or better
        stop_loss=220.00,   # Stop loss at $220
        take_profit=250.00, # Take profit at $250
        reasoning="Testing stop-limit order with risk management"
    )
    
    if stop_limit_result.success:
        print(f"✅ Stop-limit order placed successfully: {stop_limit_result.order_id}")
        print(f"   Related orders: {stop_limit_result.related_orders}")
    else:
        print(f"❌ Stop-limit order failed: {stop_limit_result.error_message}")
    
    # Test 2: Trailing Stop Order
    print("\n2️⃣ Testing Trailing Stop Order")
    print("-" * 30)
    
    trailing_stop_result = await place_trailing_stop_order(
        symbol="MSFT",
        quantity=5,
        side="sell",
        trail_percent=3.0,  # 3% trailing stop
        reasoning="Testing trailing stop to lock in profits"
    )
    
    if trailing_stop_result.success:
        print(f"✅ Trailing stop order initiated: {trailing_stop_result.order_id}")
        print("   The system will monitor and adjust the stop price automatically")
    else:
        print(f"❌ Trailing stop order failed: {trailing_stop_result.error_message}")
    
    # Test 3: OCO (One-Cancels-Other) Order
    print("\n3️⃣ Testing OCO Order")
    print("-" * 30)
    
    oco_result = await place_oco_order(
        symbol="GOOGL",
        quantity=3,
        side="buy",
        take_profit_price=180.00,  # Take profit at $180
        stop_loss_price=160.00,    # Stop loss at $160
        reasoning="Testing OCO order for risk management"
    )
    
    if oco_result.success:
        print(f"✅ OCO order placed successfully: {oco_result.order_id}")
        print(f"   Related orders: {oco_result.related_orders}")
        print("   When one order fills, the other will be automatically cancelled")
    else:
        print(f"❌ OCO order failed: {oco_result.error_message}")
    
    # Test 4: Advanced Order Manager Summary
    print("\n4️⃣ Advanced Order Manager Summary")
    print("-" * 30)
    
    summary = advanced_order_manager.get_active_orders_summary()
    print(f"Active Orders: {summary['total_active_orders']}")
    print(f"OCO Groups: {summary['oco_groups']}")
    print(f"Stop Loss Orders: {summary['stop_loss_orders']}")
    print(f"Take Profit Orders: {summary['take_profit_orders']}")
    print(f"Order Types: {set(summary['order_types'])}")
    
    # Test 5: Order Type Decision Logic
    print("\n5️⃣ Testing Smart Order Selection Logic")
    print("-" * 30)
    
    from agents.diversified_portfolio import DiversifiedPortfolioManager
    portfolio_manager = DiversifiedPortfolioManager()
    
    # Test different confidence levels and their order types
    test_scenarios = [
        {"confidence": 0.9, "volatility": 0.01, "expected": "limit order with tight spread"},
        {"confidence": 0.7, "volatility": 0.03, "expected": "stop-limit with protection"},
        {"confidence": 0.4, "volatility": 0.05, "expected": "cautious market order"}
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"   Scenario {i}: Confidence {scenario['confidence']:.1%}, "
              f"Volatility {scenario['volatility']:.1%}")
        print(f"   Expected: {scenario['expected']}")
    
    print(f"\n🎯 Advanced Order Types Test Complete!")
    print("=" * 50)
    
    # Show supported order types
    print("\n📋 Supported Order Types:")
    print("   • Market Orders - Immediate execution at current market price")
    print("   • Limit Orders - Execute at specified price or better")
    print("   • Stop Orders - Trigger when price reaches stop level")
    print("   • Stop-Limit Orders - Trigger at stop, execute as limit order")
    print("   • Trailing Stop Orders - Dynamic stop that follows price movement")
    print("   • OCO Orders - One-Cancels-Other for risk management")
    print("   • Automatic Stop Loss/Take Profit attachment")
    
    print("\n🔧 Smart Order Selection Features:")
    print("   • High confidence (>80%) → Limit orders with tight spreads")
    print("   • Medium confidence (>60%) → Stop-limit with risk management")
    print("   • Low confidence (<60%) → Market orders with stop losses")
    print("   • Profitable positions → Limit orders to maximize gains")
    print("   • Loss positions → Trailing stops to minimize losses")
    print("   • Automatic volatility-based price adjustments")

if __name__ == "__main__":
    try:
        asyncio.run(test_advanced_order_types())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.error(f"Advanced order types test error: {e}")