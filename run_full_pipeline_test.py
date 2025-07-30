#!/usr/bin/env python3
"""
Run Full Pipeline Test
Test the complete pipeline after position liquidation to verify trade execution.
"""

import asyncio
import logging
from datetime import datetime

from continuous_rebalancer import ContinuousRebalancer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_full_pipeline_test():
    """Run a complete pipeline test after liquidation."""
    
    print("🚀 FULL PIPELINE TEST AFTER LIQUIDATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Testing complete pipeline with fresh portfolio state")
    print("=" * 60)
    
    try:
        # Create continuous rebalancer instance
        rebalancer = ContinuousRebalancer()
        
        print("\n📊 STARTING FULL PIPELINE TEST")
        print("This will run all 8 steps of the trading pipeline:")
        print("1. Market Monitor")
        print("2. Universe Filter") 
        print("3. Sentiment Analysis")
        print("4. Risk Assessment")
        print("5. Signal Generation")
        print("6. Strategy Optimization")
        print("7. Order Management")
        print("8. Portfolio Tracking")
        
        # Run single pipeline iteration
        result = await rebalancer._run_rebalancing_pipeline()
        
        print(f"\n🏆 PIPELINE TEST RESULTS:")
        print(f"Success: {result.success}")
        print(f"Pipeline Stage: {result.stage}")
        print(f"Duration: {result.duration:.1f}s")
        print(f"Signals Generated: {result.signals_generated}")
        print(f"Orders Executed: {result.orders_executed}")
        print(f"Portfolio Value: ${result.final_portfolio_value:,.2f}")
        print(f"Portfolio Change: ${result.portfolio_change:+,.2f}")
        
        if result.error_message:
            print(f"Error: {result.error_message}")
        
        # Determine success
        pipeline_success = result.success and result.orders_executed > 0
        
        print(f"\n🎯 TRADE EXECUTION ASSESSMENT:")
        print(f"Pipeline Completed: {'✅' if result.success else '❌'}")
        print(f"Signals Generated: {'✅' if result.signals_generated > 0 else '❌'} ({result.signals_generated})")
        print(f"Orders Executed: {'✅' if result.orders_executed > 0 else '❌'} ({result.orders_executed})")
        print(f"Overall Success: {'✅ TRADES EXECUTED' if pipeline_success else '❌ NO TRADES EXECUTED'}")
        
        return pipeline_success
        
    except Exception as e:
        logger.error(f"Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting full pipeline test after position liquidation...")
    
    success = asyncio.run(run_full_pipeline_test())
    
    if success:
        print(f"\n🎉 SUCCESS: Full pipeline executed trades successfully!")
        print("The trading system is working end-to-end!")
    else:
        print(f"\n❌ FAILED: Full pipeline did not execute trades")
        print("Trade execution issue still needs resolution.")