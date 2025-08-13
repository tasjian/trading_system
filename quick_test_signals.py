#!/usr/bin/env python3
"""
Quick Signal System Test
Fast validation of the new multi-source signal generation system.
"""

import asyncio
import logging
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def quick_test():
    """Run a quick test of the signal system."""
    logger.info("🚀 QUICK SIGNAL SYSTEM TEST")
    
    try:
        # Test the resilient orchestrator
        from tools.resilient_signal_orchestrator import get_resilient_price_signals_sync
        
        # Use minimal symbols to avoid rate limits
        test_symbols = ['AAPL', 'SPY']
        
        logger.info(f"Testing with symbols: {test_symbols}")
        
        # Test signal generation  
        from tools.resilient_signal_orchestrator import signal_orchestrator
        
        result = await signal_orchestrator.orchestrate_signals(
            symbols=test_symbols,
            threshold=0.01,  # Lower threshold
            min_signals=2
        )
        
        signals = result.signals
        
        logger.info(f"✅ SUCCESS: Generated {len(signals)} signals")
        
        # Display signals
        for i, signal in enumerate(signals):
            logger.info(f"  Signal {i+1}: {signal['symbol']} {signal['direction']} "
                       f"strength={signal['strength']:.2f} "
                       f"source={signal.get('data_source', 'unknown')}")
        
        # Test universe filter integration
        logger.info("Testing universe filter integration...")
        
        from core.universe_filter import filter_stock_universe
        
        result = await filter_stock_universe(
            base_symbols=test_symbols,
            max_symbols=5,
            include_watchlist=False
        )
        
        logger.info(f"✅ Universe filter: {result.total_symbols} → {len(result.filtered_symbols)} symbols")
        logger.info(f"Generated {len(result.signals)} total signals")
        logger.info(f"Filter summary: {result.filter_summary}")
        
        # Validate minimum requirements
        if len(result.signals) >= 2:
            logger.info("🎉 SUCCESS: Signal system meets minimum requirements!")
            return True
        else:
            logger.error(f"❌ FAILED: Only {len(result.signals)} signals generated (minimum 2 required)")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(quick_test())
        if success:
            print("\n✅ SIGNAL SYSTEM VALIDATION PASSED")
            print("The new multi-source signal generation system is working correctly!")
        else:
            print("\n❌ SIGNAL SYSTEM VALIDATION FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)