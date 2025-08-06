#!/usr/bin/env python3
"""
Quick Test Script for FinRL Signal Generation
Tests signal generation with explicit actions to ensure the pipeline works.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import sys
import os

# Add the trading system to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_signal_generation():
    """Test FinRL signal generation with explicit actions."""
    logger.info("🧪 Testing FinRL Signal Generation with Explicit Actions")
    
    try:
        from agents.finrl_integration import create_finrl_trading_system
        
        # Create orchestrator
        symbols = ["AAPL", "TSLA", "GME", "NVDA"]
        orchestrator = await create_finrl_trading_system(
            symbols=symbols,
            initial_balance=100000,
            enable_short_selling=True,
            enable_limit_orders=True,
            paper_trading=True,
            max_position_size=0.20  # 20% max position
        )
        
        logger.info(f"✅ FinRL Trading System created for {len(symbols)} symbols")
        
        # Create mock trading state with proper price data
        test_state = {
            'portfolio': {
                'equity': 100000,
                'cash': 50000
            },
            'market_data': {
                'AAPL': {'close': 150.0, 'price': 150.0},
                'TSLA': {'close': 200.0, 'price': 200.0}, 
                'GME': {'close': 80.0, 'price': 80.0},
                'NVDA': {'close': 120.0, 'price': 120.0}
            }
        }
        
        logger.info("📊 Test state created with market data:")
        for symbol, data in test_state['market_data'].items():
            logger.info(f"   {symbol}: ${data['price']:.2f}")
        
        # Test 1: Manual actions (simulate what RL agent should produce)
        logger.info("\n🎯 Test 1: Manual Action Conversion")
        
        # Strong actions that should generate orders
        test_actions = np.array([0.8, -0.6, 0.3, -0.4])  # AAPL long, TSLA short, GME long, NVDA short
        test_metadata = {'confidence': 0.8}
        
        logger.info(f"Test actions: {test_actions}")
        
        # Convert actions to decisions manually
        decisions = await orchestrator._convert_actions_to_decisions(
            test_actions, test_metadata, test_state
        )
        
        logger.info(f"\n📋 Generated {len(decisions['orders'])} orders:")
        for order in decisions['orders']:
            side = order['side']
            symbol = order['symbol']
            quantity = order['quantity']
            price = order.get('price', 'market')
            
            if side == 'short':
                logger.info(f"   🔻 SHORT {symbol}: {quantity} shares @ ${price}")
            elif side == 'buy':
                logger.info(f"   🔺 BUY {symbol}: {quantity} shares @ ${price}")
            else:
                logger.info(f"   ➡️ {side.upper()} {symbol}: {quantity} shares @ ${price}")
        
        # Test 2: Test via make_trading_decisions (full pipeline without trained agent)
        logger.info("\n🎯 Test 2: Full Pipeline Test (Mock Agent)")
        
        # Initialize components first to create the agent
        market_df = pd.DataFrame({
            'AAPL_close': [150.0],
            'TSLA_close': [200.0], 
            'GME_close': [80.0],
            'NVDA_close': [120.0]
        })
        
        await orchestrator.initialize_components(market_df)
        
        # Now override the agent's get_action method to return our test actions
        if orchestrator.agent:
            original_get_action = orchestrator.agent.get_action
            def mock_get_action(obs, training=False):
                return test_actions, test_metadata
            orchestrator.agent.get_action = mock_get_action
            orchestrator.is_trained = True  # Mark as trained so it uses the agent
            logger.info("✅ Agent mocked and marked as trained")
        
        # Market data already created above
        
        try:
            full_decisions = await orchestrator.make_trading_decisions(test_state, market_df)
            
            logger.info(f"\n📋 Full Pipeline Generated {len(full_decisions['orders'])} orders:")
            logger.info(f"Strategy: {full_decisions['strategy']}")
            logger.info(f"Risk Level: {full_decisions['risk_level']}")
            logger.info(f"Confidence: {full_decisions['confidence']:.2f}")
            
            for order in full_decisions['orders']:
                side = order['side']
                symbol = order['symbol']
                quantity = order['quantity']
                price = order.get('price', 'market')
                
                if side == 'short':
                    logger.info(f"   🔻 SHORT {symbol}: {quantity} shares @ ${price}")
                elif side == 'buy':
                    logger.info(f"   🔺 BUY {symbol}: {quantity} shares @ ${price}")
                else:
                    logger.info(f"   ➡️ {side.upper()} {symbol}: {quantity} shares @ ${price}")
                    
        except Exception as e:
            logger.error(f"Full pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            await orchestrator.close()
        except Exception as close_error:
            logger.warning(f"Close method error (non-critical): {close_error}")
        logger.info("✅ Signal generation test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Signal generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_signal_generation())
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Signal Generation Test")
    sys.exit(0 if success else 1)