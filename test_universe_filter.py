#!/usr/bin/env python3
"""
Test Universe Filter
Quick test of the stock universe filtering functionality.
"""

import asyncio
import logging
from datetime import datetime

from core.universe_filter import filter_stock_universe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_universe_filter():
    """Test the universe filter functionality."""
    
    print("🔍 TESTING STOCK UNIVERSE FILTER")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Testing lightweight filtering to reduce processing overhead")
    print("=" * 50)
    
    try:
        # Test the universe filter
        print("\n🚀 Running universe filter...")
        
        result = await filter_stock_universe(
            base_symbols=None,      # Use full universe
            max_symbols=50,         # Small test set
            include_watchlist=True
        )
        
        print(f"\n✅ FILTER RESULTS:")
        print(f"   Original universe: {result.total_symbols} symbols")
        print(f"   Filtered universe: {len(result.filtered_symbols)} symbols")
        print(f"   Processing time: {result.processing_time:.1f}s")
        print(f"   Reduction ratio: {len(result.filtered_symbols)/result.total_symbols:.1%}")
        
        print(f"\n📊 SIGNAL BREAKDOWN:")
        for signal_type, count in result.filter_summary.items():
            print(f"   {signal_type}: {count} signals")
        
        print(f"\n🎯 TOP FILTERED SYMBOLS:")
        for i, symbol in enumerate(result.filtered_symbols[:20]):
            print(f"   {i+1:2d}. {symbol}")
        
        print(f"\n🔍 SIGNAL DETAILS:")
        # Group signals by symbol for analysis
        symbol_signals = {}
        for signal in result.signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)
        
        # Show top 10 symbols with their signals
        top_symbols = result.filtered_symbols[:10]
        for symbol in top_symbols:
            if symbol in symbol_signals:
                signals = symbol_signals[symbol]
                signal_types = [s.signal_type for s in signals]
                total_strength = sum(s.strength for s in signals)
                print(f"   {symbol}: {len(signals)} signals ({', '.join(set(signal_types))}) - strength: {total_strength:.2f}")
        
        print(f"\n🏆 PERFORMANCE ANALYSIS:")
        efficiency_gain = (result.total_symbols - len(result.filtered_symbols)) / result.total_symbols
        print(f"   Processing reduction: {efficiency_gain:.1%}")
        print(f"   Estimated time savings: {efficiency_gain * 100:.0f}% faster processing")
        print(f"   Focus on actionable stocks: {len([s for s in result.signals if s.strength > 0.5])} high-strength signals")
        
        success = len(result.filtered_symbols) > 0 and result.processing_time < 60
        return success
        
    except Exception as e:
        logger.error(f"Universe filter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting universe filter test...")
    print("This will test the lightweight stock filtering system.")
    
    success = asyncio.run(test_universe_filter())
    
    if success:
        print(f"\n🎉 TEST COMPLETE: Universe filter working efficiently!")
    else:
        print(f"\n❌ TEST FAILED: Check logs for issues")