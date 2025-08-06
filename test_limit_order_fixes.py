#!/usr/bin/env python3
"""
Quick test to validate limit order fixes
"""

import numpy as np
import pandas as pd
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_limit_order_agent_fixes():
    """Test that the limit order agent NaN issues are fixed."""
    try:
        from agents.finrl_limit_order_env import create_limit_order_finrl_env
        from agents.finrl_limit_order_agent import create_limit_order_finrl_agent, RLConfig
        
        # Create small test environment
        symbols = ["AAPL", "TSLA", "GME"]
        env = create_limit_order_finrl_env(symbols=symbols, initial_balance=100000)
        
        # Generate simple test data
        test_data = {}
        for symbol in symbols:
            prices = np.random.uniform(100, 200, 20)
            volumes = np.random.uniform(1e6, 5e6, 20)
            
            test_data[f"{symbol}_close"] = prices
            test_data[f"{symbol}_volume"] = volumes
        
        df = pd.DataFrame(test_data)
        env.set_data(df)
        
        # Create agent with conservative config
        config = RLConfig(
            hidden_dim=128,  # Smaller network
            num_layers=2,
            learning_rate=1e-4,
            batch_size=32
        )
        
        agent = create_limit_order_finrl_agent(env, config)
        
        logger.info(f"✅ Agent created successfully")
        logger.info(f"   State dim: {agent.state_dim}")
        logger.info(f"   Action dim: {agent.action_dim}")
        
        # Test action generation
        obs = env.reset()
        logger.info(f"✅ Environment reset - obs shape: {obs.shape}")
        
        # Check for NaN in observation
        if np.isnan(obs).any():
            logger.error("❌ NaN found in observation")
            return False
        
        # Test action generation
        action, metadata = agent.get_action(obs, training=False)
        
        logger.info(f"✅ Action generated successfully")
        logger.info(f"   Action shape: {action.shape}")
        logger.info(f"   Confidence: {metadata['confidence']:.3f}")
        
        # Check for NaN in action
        if np.isnan(action).any():
            logger.error("❌ NaN found in action")
            return False
        
        # Test step execution
        next_obs, reward, done, info = env.step(action)
        
        logger.info(f"✅ Environment step completed")
        logger.info(f"   Reward: {reward:.6f}")
        logger.info(f"   Portfolio Value: ${info['portfolio_value']:,.2f}")
        
        # Check for NaN in results
        if np.isnan(next_obs).any():
            logger.error("❌ NaN found in next observation")
            return False
        
        if np.isnan(reward) or np.isinf(reward):
            logger.error("❌ Invalid reward value")
            return False
        
        try:
            env.close()
        except Exception as close_error:
            logger.info(f"Close method error (non-critical): {close_error}")
        logger.info("✅ All NaN issues resolved!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_limit_order_agent_fixes()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Limit Order Fixes Test")
    sys.exit(0 if success else 1)