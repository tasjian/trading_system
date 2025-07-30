#!/usr/bin/env python3
"""
Test LLM Signal Generation
Direct test of the LLM portfolio construction to debug signal generation issues.
"""

import asyncio
import logging
from datetime import datetime

from agents.workflow import TradingWorkflow
from agents.state import create_initial_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_llm_signal_generation():
    """Test LLM signal generation directly."""
    
    print("🧪 TESTING LLM SIGNAL GENERATION")
    print("=" * 50)
    
    try:
        # Initialize workflow
        workflow = TradingWorkflow()
        state = create_initial_state()
        
        # Set up test data with strong sentiment signals
        state["watchlist"] = ["NVDA", "MSFT", "BAC", "UNH", "GOOGL"]
        state["portfolio"]["equity"] = 26000.0
        state["portfolio"]["cash"] = -5000.0
        
        # Add mock market data
        state["market_data"]["symbols"] = {
            "NVDA": {"price": 178.0, "volume": 1000000},
            "MSFT": {"price": 512.0, "volume": 800000},
            "BAC": {"price": 48.0, "volume": 2000000},
            "UNH": {"price": 262.0, "volume": 500000},
            "GOOGL": {"price": 195.0, "volume": 700000}
        }
        
        # Add mock sentiment data with clear buy/sell signals
        state["sentiment_data"] = {
            "NVDA": {
                "overall_score": 0.407,
                "confidence": 0.736,
                "has_recent_earnings": False,
                "overall_sentiment": "positive"
            },
            "MSFT": {
                "overall_score": 0.364,
                "confidence": 0.736,
                "has_recent_earnings": False,
                "overall_sentiment": "positive"
            },
            "BAC": {
                "overall_score": -0.600,
                "confidence": 0.750,
                "has_recent_earnings": False,
                "overall_sentiment": "very_negative"
            },
            "UNH": {
                "overall_score": -0.521,
                "confidence": 0.764,
                "has_recent_earnings": False,
                "overall_sentiment": "very_negative"
            },
            "GOOGL": {
                "overall_score": 0.500,
                "confidence": 0.650,
                "has_recent_earnings": False,
                "overall_sentiment": "positive"
            }
        }
        
        config = {"thread_id": "test_llm_signals"}
        
        print(f"📊 Input State:")
        print(f"   Watchlist: {state['watchlist']}")
        print(f"   Portfolio Value: ${state['portfolio']['equity']:,.2f}")
        print(f"   Market Data: {len(state['market_data']['symbols'])} symbols")
        print(f"   Sentiment Data: {len(state['sentiment_data'])} symbols")
        
        # Test signal generation
        print(f"\n🧠 Running LLM Signal Generation...")
        initial_signals = len(state.get("signals", []))
        print(f"   Initial signals: {initial_signals}")
        
        result_state = await workflow.signal_generation_agent(state, config)
        
        final_signals = result_state.get("signals", [])
        new_signals = len(final_signals) - initial_signals
        
        print(f"\n✅ LLM Signal Generation Results:")
        print(f"   Final signals: {len(final_signals)}")
        print(f"   New signals generated: {new_signals}")
        
        if final_signals:
            print(f"\n📈 Generated Signals:")
            for i, signal in enumerate(final_signals):
                print(f"   {i+1}. {signal.symbol}: {signal.action}")
                print(f"      Confidence: {signal.confidence:.3f}")
                print(f"      Quantity: {signal.quantity:.1f}")
                print(f"      Reasoning: {signal.reasoning}")
        else:
            print(f"\n❌ No signals generated!")
            print(f"   This indicates an issue with the LLM signal generation process")
        
        return len(final_signals) > 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_llm_signal_generation())
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: LLM signal generation test")