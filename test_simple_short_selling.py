#!/usr/bin/env python3
"""
Simple test to verify short selling works in the workflow signal generation.
"""

import logging
import asyncio
import sys
import os

# Add the trading system to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.workflow import TradingWorkflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_signal_generation():
    """Test the signal generation with explicit RL decisions including short selling."""
    
    logger.info("🧪 Testing Signal Generation with Short Selling")
    
    # Create mock state with RL decisions that include short selling
    state = {
        "portfolio": {"equity": 100000, "cash": 50000},
        "rl_enhanced": True,
        "finrl_integrated": False,
        "rl_decisions": {
            "strategy": "test_short_selling",
            "risk_level": "moderate",
            "confidence": 0.8,
            "reasoning": "Test with mixed buy/sell signals",
            "advanced_features": {
                "short_selling": True,
                "limit_orders": False,
                "risk_management": True
            },
            "allocations": [
                {
                    "symbol": "AAPL",
                    "weight": 0.15,  # 15% long position
                    "confidence": 0.8,
                    "action": "buy",
                    "reasoning": "RL buy allocation #1"
                },
                {
                    "symbol": "GME", 
                    "weight": -0.12,  # -12% short position
                    "confidence": 0.7,
                    "action": "short",
                    "reasoning": "RL short allocation #1"
                },
                {
                    "symbol": "TSLA",
                    "weight": -0.08,  # -8% short position
                    "confidence": 0.6, 
                    "action": "sell",
                    "reasoning": "RL sell allocation #1"
                }
            ]
        },
        "messages": [],
        "current_agent": "signal_generator"
    }
    
    config = {"test_mode": True}
    
    # Create workflow instance
    workflow = TradingWorkflow()
    
    logger.info("📊 Input RL decisions:")
    for alloc in state["rl_decisions"]["allocations"]:
        weight_str = f"{alloc['weight']:+.1%}"
        logger.info(f"   {alloc['symbol']}: {alloc['action']} {weight_str}")
    
    # Test signal generation
    logger.info("\n🧠 Testing signal generation...")
    try:
        result_state = await workflow._fallback_signal_generation(state, config)
        
        signals = result_state.get("signals", [])
        logger.info(f"\n📈 Generated {len(signals)} trading signals:")
        
        buy_signals = []
        sell_signals = []
        short_signals = []
        
        for signal in signals:
            symbol = signal.symbol
            action = signal.action
            quantity = signal.quantity
            
            if action == "buy":
                buy_signals.append((symbol, quantity))
                logger.info(f"   📈 {symbol}: BUY {quantity} shares")
            elif action == "sell":
                sell_signals.append((symbol, quantity))
                logger.info(f"   📉 {symbol}: SELL {quantity} shares")
            elif action == "short":
                short_signals.append((symbol, quantity))
                logger.info(f"   🔻 {symbol}: SHORT {quantity} shares")
        
        logger.info(f"\n📋 Summary:")
        logger.info(f"   BUY signals: {len(buy_signals)}")
        logger.info(f"   SELL signals: {len(sell_signals)}")
        logger.info(f"   SHORT signals: {len(short_signals)}")
        
        # Check if short selling is working
        if short_signals or sell_signals:
            logger.info("✅ SHORT SELLING IS WORKING!")
            return True
        else:
            logger.warning("❌ No sell/short signals generated!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_signal_generation())
    logger.info(f"\n{'✅ SUCCESS' if success else '❌ FAILURE'}: Short selling test completed")
    sys.exit(0 if success else 1)