#!/usr/bin/env python3
"""
Test Script for FinRL Integration with Short Selling and Limit Orders
Tests the complete FinRL-enhanced trading system with advanced features.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os

# Add the trading system to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_finrl_short_selling.log')
    ]
)
logger = logging.getLogger(__name__)


def generate_test_market_data(symbols: List[str], days: int = 90) -> pd.DataFrame:
    """Generate synthetic market data for testing."""
    logger.info(f"Generating test market data for {len(symbols)} symbols over {days} days")
    
    # Generate dates (trading days only)
    dates = pd.bdate_range(start=datetime.now() - timedelta(days=days), periods=days)
    
    data_dict = {}
    
    for symbol in symbols:
        # Generate realistic price data with trends
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        
        initial_price = np.random.uniform(50, 200)
        returns = np.random.normal(0.001, 0.02, days)  # 0.1% daily return, 2% volatility
        
        # Add some trending behavior
        if symbol == "AAPL":
            returns += np.linspace(0, 0.005, days)  # Upward trend
        elif symbol == "TSLA":
            returns += np.sin(np.linspace(0, 4*np.pi, days)) * 0.01  # Cyclical
        elif symbol == "GME":
            returns -= np.linspace(0, 0.003, days)  # Downward trend (good for shorting)
        
        # Calculate cumulative prices
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        opens = prices * np.random.uniform(0.995, 1.005, days)
        highs = np.maximum(opens, prices) * np.random.uniform(1.0, 1.02, days)
        lows = np.minimum(opens, prices) * np.random.uniform(0.98, 1.0, days)
        closes = prices
        volumes = np.random.uniform(1000000, 10000000, days)
        
        # Add columns to data dictionary
        data_dict[f"{symbol}_open"] = opens
        data_dict[f"{symbol}_high"] = highs
        data_dict[f"{symbol}_low"] = lows
        data_dict[f"{symbol}_close"] = closes
        data_dict[f"{symbol}_volume"] = volumes
        
        # Add some technical indicators
        data_dict[f"{symbol}_rsi"] = 50 + 30 * np.sin(np.linspace(0, 10*np.pi, days))
        data_dict[f"{symbol}_macd"] = np.random.normal(0, 2, days)
        data_dict[f"{symbol}_bb_upper"] = closes * 1.1
        data_dict[f"{symbol}_bb_lower"] = closes * 0.9
    
    # Create DataFrame
    market_data = pd.DataFrame(data_dict, index=dates)
    market_data = market_data.fillna(method='ffill').fillna(method='bfill')
    
    logger.info(f"Generated market data: {market_data.shape[0]} rows, {market_data.shape[1]} columns")
    return market_data


async def test_finrl_environment():
    """Test FinRL enhanced environment with short selling."""
    logger.info("=" * 60)
    logger.info("TESTING FINRL ENHANCED ENVIRONMENT")
    logger.info("=" * 60)
    
    try:
        from agents.finrl_enhanced_env import create_enhanced_finrl_env, OrderSide, OrderType
        
        # Test symbols with different expected behaviors
        symbols = ["AAPL", "TSLA", "GME", "NVDA"]
        logger.info(f"Testing with symbols: {symbols}")
        
        # Create environment with short selling enabled
        env = create_enhanced_finrl_env(
            symbols=symbols,
            initial_balance=100000,
            enable_short_selling=True,
            enable_limit_orders=True,
            commission_rate=0.001,
            margin_requirement=0.5,
            short_borrow_rate=0.03,
            max_position_size=0.2
        )
        
        logger.info(f"✅ Environment created")
        logger.info(f"   Action space: {env.action_space}")
        logger.info(f"   Short selling: {env.enable_short_selling}")
        logger.info(f"   Limit orders: {env.enable_limit_orders}")
        
        # Generate and set test data
        market_data = generate_test_market_data(symbols, days=60)
        env.set_data(market_data)
        
        logger.info(f"✅ Market data loaded")
        logger.info(f"   Observation space: {env.observation_space}")
        
        # Test environment reset
        obs = env.reset()
        logger.info(f"✅ Environment reset - Observation shape: {obs.shape}")
        
        # Test mixed actions (long, short, hold)
        test_actions = [
            np.array([0.8, -0.6, 0.1, -0.3]),   # AAPL long, TSLA short, NVDA small long, GME short
            np.array([-0.5, 0.7, -0.2, 0.4]),   # Mixed positions
            np.array([0.0, 0.0, 0.0, 0.0]),     # All hold
        ]
        
        for i, action in enumerate(test_actions):
            logger.info(f"\n--- Test Action {i+1}: {action} ---")
            
            obs, reward, done, info = env.step(action)
            
            logger.info(f"Reward: {reward:.6f}")
            logger.info(f"Portfolio Value: ${info['portfolio_value']:,.2f}")
            logger.info(f"Cash: ${info['cash']:,.2f}")
            logger.info(f"Trades Executed: {info['trades_executed']}")
            logger.info(f"Active Positions: {info['positions']}")
            
            # Get portfolio summary
            portfolio_summary = env.get_portfolio_summary()
            
            # Log positions
            for symbol, pos_info in portfolio_summary['positions'].items():
                side = pos_info['side']
                quantity = pos_info['quantity']
                pnl = pos_info['unrealized_pnl']
                
                if side == 'short':
                    logger.info(f"   🔻 SHORT {symbol}: {abs(quantity):.0f} shares, P&L: ${pnl:+.2f}")
                elif side == 'long':
                    logger.info(f"   🔺 LONG {symbol}: {quantity:.0f} shares, P&L: ${pnl:+.2f}")
            
            if done:
                break
        
        # Test limit orders
        logger.info(f"\n--- Testing Limit Orders ---")
        current_prices = env._get_current_prices()
        
        if "AAPL" in current_prices:
            current_price = current_prices["AAPL"]
            limit_price = current_price * 0.98  # 2% below current
            
            success = env.place_limit_order("AAPL", OrderSide.BUY, 10, limit_price)
            logger.info(f"Limit order placed: {success}")
            logger.info(f"   BUY 10 AAPL @ ${limit_price:.2f} (current: ${current_price:.2f})")
            
            # Short limit order
            short_limit_price = current_price * 1.02  # 2% above current
            success = env.place_limit_order("AAPL", OrderSide.SHORT, 5, short_limit_price)
            logger.info(f"Short limit order placed: {success}")
            logger.info(f"   SHORT 5 AAPL @ ${short_limit_price:.2f}")
        
        # Final portfolio summary
        final_summary = env.get_portfolio_summary()
        logger.info(f"\n--- Final Portfolio Summary ---")
        logger.info(f"Total Return: {final_summary['portfolio_return']:.2%}")
        logger.info(f"Cash: ${final_summary['cash']:,.2f}")
        logger.info(f"Equity: ${final_summary['equity']:,.2f}")
        logger.info(f"Short Interest: ${final_summary['short_interest']:,.2f}")
        logger.info(f"Pending Orders: {final_summary['pending_orders']}")
        
        env.close()
        logger.info("✅ FinRL Environment test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ FinRL Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_finrl_agent():
    """Test FinRL agent with short selling capabilities."""
    logger.info("=" * 60)
    logger.info("TESTING FINRL AGENT WITH SHORT SELLING")
    logger.info("=" * 60)
    
    try:
        from agents.finrl_enhanced_env import create_enhanced_finrl_env
        from agents.finrl_rl_agent import create_finrl_agent, RLConfig
        
        # Create environment and agent
        symbols = ["AAPL", "TSLA", "GME"]
        env = create_enhanced_finrl_env(
            symbols=symbols,
            enable_short_selling=True,
            enable_limit_orders=True,
            initial_balance=100000
        )
        
        # Set test data
        market_data = generate_test_market_data(symbols, days=50)
        env.set_data(market_data)
        
        # Create agent with test configuration
        config = RLConfig(
            hidden_dim=128,
            num_layers=2,
            batch_size=32,
            buffer_size=5000,
            min_buffer_size=50,
            ppo_epochs=5,
            update_frequency=2
        )
        
        agent = create_finrl_agent(env, config)
        
        logger.info(f"✅ Agent created")
        logger.info(f"   State dim: {agent.state_dim}")
        logger.info(f"   Action dim: {agent.action_dim}")
        logger.info(f"   Device: {agent.device}")
        
        # Test agent actions
        logger.info(f"\n--- Testing Agent Actions ---")
        
        obs = env.reset()
        
        for step in range(5):
            logger.info(f"\nStep {step + 1}:")
            
            # Get action from agent
            action, metadata = agent.get_action(obs, training=True)
            
            logger.info(f"Action: {action}")
            logger.info(f"Metadata: {metadata}")
            
            # Execute action
            next_obs, reward, done, info = env.step(action)
            
            # Add experience to agent
            agent.add_experience(obs, action, reward, next_obs, done, metadata)
            
            logger.info(f"Reward: {reward:.6f}")
            logger.info(f"Portfolio Value: ${info['portfolio_value']:,.2f}")
            logger.info(f"Buffer Size: {len(agent.buffer)}")
            
            obs = next_obs
            
            if done:
                break
        
        # Test training update
        if len(agent.buffer) >= agent.config.min_buffer_size:
            logger.info(f"\n--- Testing Training Update ---")
            training_metrics = agent.update_policy(batch_size=32)
            
            if training_metrics:
                logger.info(f"Training metrics:")
                for key, value in training_metrics.items():
                    logger.info(f"   {key}: {value:.6f}")
        
        # Test performance statistics
        perf_stats = agent.get_performance_stats()
        logger.info(f"\n--- Performance Statistics ---")
        for key, value in perf_stats.items():
            logger.info(f"   {key}: {value}")
        
        # Test save/load
        logger.info(f"\n--- Testing Save/Load ---")
        agent.save_model("test_short_selling_checkpoint")
        
        # Create new agent and load
        agent2 = create_finrl_agent(env, config)
        success = agent2.load_model(agent.model_dir / "test_short_selling_checkpoint.pt")
        logger.info(f"Model load success: {success}")
        
        env.close()
        logger.info("✅ FinRL Agent test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ FinRL Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_trading_system_integration():
    """Test integration with the main trading system."""
    logger.info("=" * 60)
    logger.info("TESTING TRADING SYSTEM INTEGRATION")
    logger.info("=" * 60)
    
    try:
        from agents.finrl_integration import create_finrl_trading_system, integrate_finrl_with_workflow
        
        # Create orchestrator
        symbols = ["AAPL", "MSFT", "GME", "NVDA"]
        orchestrator = await create_finrl_trading_system(
            symbols=symbols,
            initial_balance=100000,
            enable_short_selling=True,
            enable_limit_orders=True,
            paper_trading=True,
            max_position_size=0.15
        )
        
        logger.info(f"✅ FinRL Trading System created")
        
        # Generate test market data (use existing generator instead of yfinance)
        market_data = generate_test_market_data(symbols, days=40)
        
        # Initialize components
        await orchestrator.initialize_components(market_data)
        
        logger.info(f"✅ Components initialized")
        
        # Quick training (for testing)
        logger.info(f"\n--- Quick Training Session ---")
        training_results = await orchestrator.train_agent(
            training_data=market_data,
            num_episodes=20,  # Quick training for test
            eval_frequency=10
        )
        
        logger.info(f"Training completed:")
        perf_stats = training_results['performance_stats']
        logger.info(f"   Episodes: {perf_stats['total_episodes']}")
        logger.info(f"   Average reward: {perf_stats.get('average_reward', 0):.4f}")
        
        # Test trading decisions
        logger.info(f"\n--- Testing Trading Decisions ---")
        
        # Mock workflow state
        test_state = {
            'portfolio': {
                'equity': 100000,
                'cash': 50000
            },
            'market_data': {
                symbol: {'close': np.random.uniform(100, 200)} 
                for symbol in symbols
            },
            'sentiment_data': {
                symbol: {'score': np.random.uniform(-0.5, 0.5)}
                for symbol in symbols
            },
            'sentiment_signals': [
                {'symbol': symbol, 'signal': 'BUY', 'strength': np.random.uniform(0.3, 0.8)}
                for symbol in symbols[:3]
            ]
        }
        
        # Make trading decisions
        decisions = await orchestrator.make_trading_decisions(test_state, market_data)
        
        logger.info(f"Generated {len(decisions['orders'])} trading decisions:")
        logger.info(f"Strategy: {decisions['strategy']}")
        logger.info(f"Risk Level: {decisions['risk_level']}")
        logger.info(f"Confidence: {decisions['confidence']:.2f}")
        
        # Log orders with short selling details
        for order in decisions['orders']:
            side = order['side']
            symbol = order['symbol']
            quantity = order['quantity']
            price = order.get('price', 'market')
            
            if side == 'short':
                logger.info(f"   🔻 SHORT {symbol}: {quantity} shares @ ${price}")
            elif side == 'buy':
                logger.info(f"   🔺 BUY {symbol}: {quantity} shares @ ${price}")
            elif side == 'sell':
                logger.info(f"   ➡️ SELL {symbol}: {quantity} shares @ ${price}")
        
        # Test workflow integration
        logger.info(f"\n--- Testing Workflow Integration ---")
        enhanced_state = await integrate_finrl_with_workflow(orchestrator, test_state)
        
        finrl_decisions = enhanced_state.get('finrl_decisions', {})
        finrl_enhanced = enhanced_state.get('finrl_enhanced', False)
        
        logger.info(f"Workflow integration success: {finrl_enhanced}")
        logger.info(f"FinRL orders generated: {len(finrl_decisions.get('orders', []))}")
        
        # Status summary
        status = orchestrator.get_status_summary()
        logger.info(f"\n--- System Status ---")
        logger.info(f"Trained: {status['orchestrator_status']['is_trained']}")
        logger.info(f"Circuit Breakers: {status['orchestrator_status']['circuit_breakers']}")
        logger.info(f"Agent Episodes: {status['agent_status'].get('total_episodes', 0)}")
        
        await orchestrator.close()
        logger.info("✅ Trading System Integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Trading System Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_trading_signal_enhancements():
    """Test enhanced TradingSignal class with short selling support."""
    logger.info("=" * 60)
    logger.info("TESTING ENHANCED TRADING SIGNALS")
    logger.info("=" * 60)
    
    try:
        from agents.state import TradingSignal
        
        # Test long position signal
        long_signal = TradingSignal(
            symbol="AAPL",
            action="buy",
            confidence=0.8,
            price_target=150.0,
            stop_loss=140.0,
            quantity=100,
            reasoning="Bullish sentiment analysis",
            order_type="limit",
            limit_price=148.0
        )
        
        logger.info(f"✅ Long signal created:")
        logger.info(f"   Symbol: {long_signal.symbol}")
        logger.info(f"   Action: {long_signal.action}")
        logger.info(f"   Is Short: {long_signal.is_short_position}")
        logger.info(f"   Margin Req: {long_signal.margin_requirement}")
        
        # Test short position signal
        short_signal = TradingSignal(
            symbol="GME",
            action="short",
            confidence=0.7,
            price_target=80.0,  # Target price below current
            stop_loss=90.0,     # Stop loss above current
            quantity=50,
            reasoning="Bearish technical analysis",
            order_type="limit",
            limit_price=85.0
        )
        
        logger.info(f"✅ Short signal created:")
        logger.info(f"   Symbol: {short_signal.symbol}")
        logger.info(f"   Action: {short_signal.action}")
        logger.info(f"   Is Short: {short_signal.is_short_position}")
        logger.info(f"   Margin Req: ${short_signal.margin_requirement:,.2f}")
        
        # Test cover position signal
        cover_signal = TradingSignal(
            symbol="GME",
            action="cover",
            confidence=0.6,
            price_target=82.0,
            stop_loss=88.0,
            quantity=30,
            reasoning="Cover short position"
        )
        
        logger.info(f"✅ Cover signal created:")
        logger.info(f"   Symbol: {cover_signal.symbol}")
        logger.info(f"   Action: {cover_signal.action}")
        logger.info(f"   Is Short: {cover_signal.is_short_position}")
        
        # Test conversion to trading orders
        logger.info(f"\n--- Testing Order Conversion ---")
        
        for signal_name, signal in [("Long", long_signal), ("Short", short_signal), ("Cover", cover_signal)]:
            order = signal.to_trading_order()
            
            if order:
                logger.info(f"{signal_name} Order:")
                logger.info(f"   Side: {order['side']}")
                logger.info(f"   Quantity: {order['quantity']}")
                logger.info(f"   Order Type: {order['order_type']}")
                logger.info(f"   Limit Price: ${order['limit_price']}")
                logger.info(f"   Metadata: {order['metadata']['is_short_position']}")
        
        # Test invalid action
        try:
            invalid_signal = TradingSignal(
                symbol="TEST",
                action="invalid_action",
                confidence=0.5,
                quantity=10
            )
            logger.error("❌ Should have raised ValueError for invalid action")
            return False
        except ValueError as e:
            logger.info(f"✅ Correctly caught invalid action: {e}")
        
        logger.info("✅ Enhanced Trading Signals test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Trading Signals test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    logger.info("🚀 STARTING FINRL SHORT SELLING INTEGRATION TESTS")
    logger.info("=" * 80)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("FinRL Environment", test_finrl_environment),
        ("FinRL Agent", test_finrl_agent),
        ("Trading System Integration", test_trading_system_integration),
        ("Trading Signal Enhancements", test_trading_signal_enhancements),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            result = await test_func()
            test_results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: EXCEPTION - {e}")
            test_results.append((test_name, False))
    
    # Summary
    logger.info("=" * 80)
    logger.info("🏁 TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! FinRL short selling integration is working correctly.")
    else:
        logger.error(f"⚠️ {total-passed} tests failed. Please review the logs above.")
    
    return passed == total


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)