#!/usr/bin/env python3
"""
Start Trading System
Launch the market-aware trading system that automatically manages timing.
"""

import asyncio
import sys
from pathlib import Path
from market_aware_scheduler import start_market_aware_trading

def main():
    """Start the market-aware trading system."""
    print("🚀 MARKET-AWARE TRADING SYSTEM")
    print("=" * 50)
    print("This system will automatically:")
    print("🌅 Start 30 minutes before market open (9:00 AM ET)")
    print("📈 Run continuously during trading hours (9:30 AM - 4:00 PM ET)")
    print("🌙 Run once after market close for end-of-day review")
    print("⏰ Wait until next trading day when market is closed")
    print()
    print("Current schedule:")
    print("- PRE-MARKET: Single portfolio preparation run")
    print("- TRADING: Continuous rebalancing every 15 minutes") 
    print("- AFTER-HOURS: Single end-of-day review")
    print()
    print("Press Ctrl+C to stop gracefully")
    print("=" * 50)
    
    # Create required directories
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    try:
        # Start the market-aware trading system
        asyncio.run(start_market_aware_trading())
    except KeyboardInterrupt:
        print("\n👋 Trading system stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()