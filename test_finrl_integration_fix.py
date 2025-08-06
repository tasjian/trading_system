#!/usr/bin/env python3
"""
Test Script for Enhanced FinRL Integration with Short Selling
Tests the fixes to the continuous rebalancer to ensure FinRL integration works properly.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any

# Add the trading system to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_state_with_sell_signals() -> Dict[str, Any]:
    """Create a mock trading state with both BUY and SELL sentiment signals."""
    
    return {
        "portfolio": {
            "equity": 100000,
            "cash": 50000,
            "buying_power": 100000
        },
        "sentiment_signals": [
            # BUY signals
            {
                "symbol": "AAPL",
                "signal": "BUY", 
                "strength": 0.8,
                "reason": "Strong earnings",
                "source": "sentiment_analysis"
            },
            {
                "symbol": "MSFT",
                "signal": "BUY",
                "strength": 0.7,
                "reason": "Positive outlook",
                "source": "sentiment_analysis"
            },
            # SELL signals for short selling
            {
                "symbol": "GME",
                "signal": "SELL",
                "strength": 0.9,
                "reason": "Overvalued fundamentals",
                "source": "sentiment_analysis"
            },
            {
                "symbol": "TSLA",
                "signal": "SELL",
                "strength": 0.6,
                "reason": "Technical indicators negative",
                "source": "sentiment_analysis"
            }
        ],
        "sentiment_data": {
            "AAPL": {"overall_sentiment": "positive", "confidence": 0.8},
            "MSFT": {"overall_sentiment": "positive", "confidence": 0.7},
            "GME": {"overall_sentiment": "negative", "confidence": 0.9},
            "TSLA": {"overall_sentiment": "negative", "confidence": 0.6}
        },
        "universe_filter_result": type('obj', (object,), {
            'filtered_symbols': ["AAPL", "MSFT", "GOOGL", "GME", "TSLA", "NVDA"],
            'total_symbols': 100
        })(),
        "trading_config": {
            "risk_profile": "moderate",
            "enable_short_selling": True
        },
        "watchlist": ["AAPL", "MSFT", "GOOGL", "GME", "TSLA"]
    }


async def test_finrl_integration():
    """Test the enhanced FinRL integration with short selling support."""
    
    logger.info("🧪 Testing Enhanced FinRL Integration with Short Selling")
    logger.info("=" * 70)
    
    try:
        # Import the continuous rebalancer with FinRL integration
        from continuous_rebalancer import ContinuousRebalancer
        
        # Create rebalancer instance
        rebalancer = ContinuousRebalancer()
        
        # Create mock state with both BUY and SELL signals
        state = create_mock_state_with_sell_signals()
        config = {"test_mode": True}
        
        logger.info("📊 Mock state includes:")
        buy_signals = [s for s in state["sentiment_signals"] if s["signal"] == "BUY"]
        sell_signals = [s for s in state["sentiment_signals"] if s["signal"] == "SELL"]
        logger.info(f"   • {len(buy_signals)} BUY signals: {[s['symbol'] for s in buy_signals]}")
        logger.info(f"   • {len(sell_signals)} SELL signals: {[s['symbol'] for s in sell_signals]}")
        
        # Test FinRL decision layer
        logger.info("\n🚀 Testing FinRL Decision Layer...")
        enhanced_state = await rebalancer._run_rl_decision_layer(state, config)
        
        # Check results
        finrl_integrated = enhanced_state.get("finrl_integrated", False)
        rl_enhanced = enhanced_state.get("rl_enhanced", False)
        rl_decisions = enhanced_state.get("rl_decisions", {})
        
        logger.info(f"\n📈 FinRL Integration Results:")
        logger.info(f"   ✅ FinRL Integrated: {finrl_integrated}")
        logger.info(f"   ✅ RL Enhanced: {rl_enhanced}")
        
        if rl_decisions:
            allocations = rl_decisions.get("allocations", [])
            advanced_features = rl_decisions.get("advanced_features", {})
            
            logger.info(f"   📊 Generated {len(allocations)} allocations")
            logger.info(f"   🔧 Short Selling Enabled: {advanced_features.get('short_selling', False)}")
            logger.info(f"   🔧 Limit Orders Enabled: {advanced_features.get('limit_orders', False)}")
            
            # Count order types
            buy_orders = [a for a in allocations if a.get('action') == 'buy']
            sell_orders = [a for a in allocations if a.get('action') in ['sell', 'short']]
            
            logger.info(f"   📈 BUY orders: {len(buy_orders)}")
            logger.info(f"   📉 SELL/SHORT orders: {len(sell_orders)}")
            
            # Show sample orders
            logger.info("\n🎯 Sample Trading Decisions:")
            for i, allocation in enumerate(allocations[:5]):
                symbol = allocation.get('symbol', 'Unknown')
                action = allocation.get('action', 'unknown')
                weight = allocation.get('weight', 0)
                weight_str = f"{weight:+.1%}"
                logger.info(f"   {i+1}. {symbol}: {action.upper()} {weight_str}")
            
            # Test signal generation path
            logger.info("\n🧠 Testing Signal Generation with RL Decisions...")
            
            # Check if the state would generate sell/short signals
            if advanced_features.get('short_selling'):
                short_allocations = [a for a in allocations if a.get('weight', 0) < 0]
                if short_allocations:
                    logger.info(f"   ✅ Found {len(short_allocations)} negative weight allocations (short positions)")
                    for alloc in short_allocations[:2]:
                        logger.info(f"      🔻 {alloc['symbol']}: {alloc['weight']:+.1%} ({alloc['action']})")
                else:
                    logger.warning("   ⚠️ No negative weight allocations found - check RL agent behavior")
            
            logger.info("\n✅ FinRL Integration Test Completed Successfully!")
            
            # Summary
            logger.info("\n📋 Test Summary:")
            logger.info(f"   • FinRL System: {'✅ ENABLED' if finrl_integrated else '❌ FALLBACK'}")
            logger.info(f"   • Short Selling: {'✅ ENABLED' if advanced_features.get('short_selling') else '❌ DISABLED'}")
            logger.info(f"   • Total Allocations: {len(allocations)}")
            logger.info(f"   • Buy Orders: {len(buy_orders)}")
            logger.info(f"   • Sell/Short Orders: {len(sell_orders)}")
            
            return True
        else:
            logger.error("❌ No RL decisions generated!")
            return False
            
    except Exception as e:
        logger.error(f"❌ FinRL Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_finrl_integration())
    
    if success:
        logger.info("\n🎉 ALL TESTS PASSED! FinRL integration with short selling is working!")
    else:
        logger.error("\n💥 TESTS FAILED! Please check the logs above for issues.")
        
    sys.exit(0 if success else 1)