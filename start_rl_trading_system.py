#!/usr/bin/env python3
"""
RL Trading System Starter
Starts the enhanced continuous rebalancing system with RL integration and conducts test trades.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add trading_system to path
sys.path.insert(0, str(Path(__file__).parent / 'trading_system'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rl_trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """Main function to start RL trading system."""
    
    print("🚀 RL TRADING SYSTEM LAUNCHER")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("online_learning_checkpoints").mkdir(exist_ok=True)
    
    try:
        # Import after path setup
        from continuous_rebalancer_with_rl import conduct_test_trades, start_enhanced_continuous_rebalancing
        
        print("🧪 Phase 1: Conducting Test Trades")
        print("This will validate RL system integration...")
        
        test_success = await conduct_test_trades()
        
        if test_success:
            print("✅ Test trades completed successfully!")
            print("✅ RL system integration validated!")
            
            print("\n🚀 Phase 2: Starting Continuous Rebalancing")
            print("Starting enhanced continuous rebalancing with RL...")
            print("Press Ctrl+C to stop the system gracefully.\n")
            
            # Start the continuous system
            await start_enhanced_continuous_rebalancing()
            
        else:
            print("❌ Test trades failed!")
            print("Please check the logs and system configuration.")
            return False
    
    except KeyboardInterrupt:
        print("\n👋 RL Trading System stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.error(f"Fatal error in RL trading system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("✅ RL Trading System completed successfully")
    else:
        print("❌ RL Trading System encountered errors")
        sys.exit(1)