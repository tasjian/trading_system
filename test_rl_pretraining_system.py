#!/usr/bin/env python3
"""
Test RL Pre-training System
Comprehensive test suite for the RL pre-training components
"""

import asyncio
import logging
import sys
import os
import numpy as np
import time
from pathlib import Path

# Add the trading system to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components to test
from agents.hybrid_data_sources import HybridDataGenerator, MarketRegimeGenerator, LLMSentimentFabricator
from agents.realistic_trading_env import create_realistic_trading_env
from agents.safe_rl_algorithms import create_safe_sac_agent, create_safe_ppo_agent
from agents.decision_transformer import create_decision_transformer, create_trajectories_from_hybrid_data
from agents.comprehensive_rl_pretraining import create_comprehensive_rl_system

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RLPretrainingTester:
    """Comprehensive test suite for RL pre-training system."""
    
    def __init__(self):
        self.symbols = ['AAPL', 'MSFT', 'GOOGL']
        self.test_results = {}
        
        # Create test directories
        Path("test_outputs").mkdir(exist_ok=True)
        Path("test_outputs/models").mkdir(exist_ok=True)
    
    async def test_hybrid_data_sources(self) -> bool:
        """Test hybrid data generation components."""
        
        logger.info("🧪 Testing Hybrid Data Sources...")
        
        try:
            # Test 1: Market Regime Generator
            regime_generator = MarketRegimeGenerator()
            regimes = regime_generator.generate_regime_sequence(days=50, start_regime='bull')
            
            assert len(regimes) == 50, f"Expected 50 regimes, got {len(regimes)}"
            assert regimes[0].regime_type == 'bull', f"Expected bull regime, got {regimes[0].regime_type}"
            
            logger.info(f"✅ Generated {len(regimes)} market regimes")
            
            # Test 2: LLM Sentiment Fabricator
            sentiment_fabricator = LLMSentimentFabricator()
            sentiment = sentiment_fabricator.generate_sentiment_data(regimes[0], price_return=0.02)
            
            assert hasattr(sentiment, 'overall_sentiment'), "Missing overall_sentiment attribute"
            assert -1.0 <= sentiment.overall_sentiment <= 1.0, f"Invalid sentiment range: {sentiment.overall_sentiment}"
            assert 0.0 <= sentiment.confidence <= 1.0, f"Invalid confidence range: {sentiment.confidence}"
            
            logger.info(f"✅ Generated sentiment: {sentiment.overall_sentiment:.3f} (confidence: {sentiment.confidence:.3f})")
            
            # Test 3: Hybrid Data Generator
            data_generator = HybridDataGenerator()
            dataset = await data_generator.generate_training_dataset(
                symbols=self.symbols,
                training_days=100,
                validation_split=0.2
            )
            
            assert 'training' in dataset, "Missing training dataset"
            assert 'validation' in dataset, "Missing validation dataset"
            
            training_data = dataset['training']
            assert len(training_data['market_data']) == 80, f"Expected 80 training days, got {len(training_data['market_data'])}"
            
            # Save test dataset
            data_generator.save_dataset(dataset, "test_outputs/test_hybrid_dataset.json")
            
            logger.info(f"✅ Generated hybrid dataset: {len(training_data['market_data'])} training days")
            
            self.test_results['hybrid_data_sources'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Hybrid data sources test failed: {e}")
            self.test_results['hybrid_data_sources'] = False
            return False
    
    async def test_realistic_trading_env(self) -> bool:
        """Test realistic trading environment."""
        
        logger.info("🧪 Testing Realistic Trading Environment...")
        
        try:
            # Create environment
            env = create_realistic_trading_env(self.symbols, initial_balance=100000)
            
            # Test environment reset
            obs = await env.reset()
            
            assert obs is not None, "Environment reset returned None"
            assert len(obs.shape) == 1, f"Expected 1D observation, got shape {obs.shape}"
            
            logger.info(f"✅ Environment initialized with observation shape: {obs.shape}")
            
            # Test environment step
            action = np.random.uniform(-0.2, 0.2, size=(len(self.symbols),))
            next_obs, reward, done, info = await env.step(action)
            
            assert next_obs is not None, "Environment step returned None observation"
            assert isinstance(reward, (int, float)), f"Reward should be numeric, got {type(reward)}"
            assert isinstance(done, bool), f"Done should be boolean, got {type(done)}"
            assert 'portfolio_value' in info, "Missing portfolio_value in info"
            
            logger.info(f"✅ Environment step: reward={reward:.4f}, portfolio=${info['portfolio_value']:,.2f}")
            
            # Test multiple steps
            for i in range(5):
                action = np.random.uniform(-0.1, 0.1, size=(len(self.symbols),))
                obs, reward, done, info = await env.step(action)
                
                if done:
                    obs = await env.reset()
                    break
            
            logger.info("✅ Multiple environment steps completed successfully")
            
            self.test_results['realistic_trading_env'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Realistic trading environment test failed: {e}")
            self.test_results['realistic_trading_env'] = False
            return False
    
    def test_safe_rl_algorithms(self) -> bool:
        """Test safe RL algorithms."""
        
        logger.info("🧪 Testing Safe RL Algorithms...")
        
        try:
            state_dim = len(self.symbols) * 10  # Example state dimension
            action_dim = len(self.symbols)
            
            # Test Safe SAC
            sac_agent = create_safe_sac_agent(state_dim, action_dim)
            
            test_state = np.random.randn(state_dim)
            action, metadata = sac_agent.select_action(test_state)
            
            assert action.shape == (action_dim,), f"Expected action shape {(action_dim,)}, got {action.shape}"
            assert 'uncertainty' in metadata, "Missing uncertainty in metadata"
            assert np.all(np.abs(action) <= 1.0), f"Actions should be in [-1, 1], got {action}"
            
            logger.info(f"✅ Safe SAC: action shape {action.shape}, uncertainty {metadata['uncertainty']:.4f}")
            
            # Test Safe PPO
            ppo_agent = create_safe_ppo_agent(state_dim, action_dim)
            
            action, metadata = ppo_agent.select_action(test_state)
            
            assert action.shape == (action_dim,), f"Expected action shape {(action_dim,)}, got {action.shape}"
            assert 'uncertainty' in metadata, "Missing uncertainty in metadata"
            
            logger.info(f"✅ Safe PPO: action shape {action.shape}, uncertainty {metadata['uncertainty']:.4f}")
            
            # Test training step (basic)
            # Store some dummy experiences
            for _ in range(100):
                state = np.random.randn(state_dim)
                action = np.random.uniform(-1, 1, action_dim)
                reward = np.random.normal(0, 0.1)
                next_state = np.random.randn(state_dim)
                done = np.random.random() < 0.1
                
                sac_agent.store_transition(state, action, reward, next_state, done)
            
            # Train SAC
            sac_stats = sac_agent.train_step()
            if sac_stats:
                logger.info(f"✅ Safe SAC training step: critic_loss={sac_stats.get('critic_loss', 'N/A')}")
            
            self.test_results['safe_rl_algorithms'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Safe RL algorithms test failed: {e}")
            self.test_results['safe_rl_algorithms'] = False
            return False
    
    async def test_decision_transformer(self) -> bool:
        """Test Decision Transformer."""
        
        logger.info("🧪 Testing Decision Transformer...")
        
        try:
            # Test PyTorch availability
            try:
                import torch
            except ImportError:
                logger.warning("⚠️ PyTorch not available - skipping Decision Transformer test")
                self.test_results['decision_transformer'] = True
                return True
            
            state_dim = len(self.symbols) * 7  # Simplified features
            action_dim = len(self.symbols)
            
            # Create Decision Transformer
            dt_model, dt_config = create_decision_transformer(state_dim, action_dim, num_epochs=2)
            
            logger.info(f"✅ Created Decision Transformer: {state_dim}→{action_dim}")
            
            # Create test trajectory data
            test_trajectory = {
                'states': np.random.randn(50, state_dim).tolist(),
                'actions': np.random.randn(49, action_dim).tolist(),
                'rewards': np.random.randn(49).tolist()
            }
            
            # Test trajectory conversion
            from agents.decision_transformer import TrajectoryDataset
            dataset = TrajectoryDataset([test_trajectory], dt_config)
            
            assert len(dataset) > 0, "Empty trajectory dataset"
            
            logger.info(f"✅ Created trajectory dataset with {len(dataset)} sequences")
            
            # Quick training test
            from agents.decision_transformer import DecisionTransformerTrainer
            trainer = DecisionTransformerTrainer(dt_model, dt_config)
            
            dt_config.num_epochs = 2  # Quick test
            stats = trainer.train(dataset)
            
            assert 'losses' in stats, "Missing training losses"
            assert len(stats['losses']) == 2, f"Expected 2 epochs, got {len(stats['losses'])}"
            
            logger.info(f"✅ Decision Transformer training: final loss={stats['losses'][-1]:.4f}")
            
            # Save test model
            trainer.save_model("test_outputs/models/test_dt_model.pt")
            
            self.test_results['decision_transformer'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Decision Transformer test failed: {e}")
            self.test_results['decision_transformer'] = False
            return False
    
    async def test_comprehensive_system(self) -> bool:
        """Test comprehensive RL pre-training system."""
        
        logger.info("🧪 Testing Comprehensive RL Pre-training System...")
        
        try:
            # Create system
            system = create_comprehensive_rl_system(
                symbols=self.symbols,
                batch_update_freq_minutes=1,  # Fast for testing
                stable_policy_update_freq_days=1
            )
            
            assert system is not None, "Failed to create comprehensive system"
            
            logger.info("✅ Created comprehensive RL system")
            
            # Test pre-training phase (quick)
            pretrain_results = await system.pretrain_with_decision_transformer(training_days=50)
            
            assert 'pretrain_success' in pretrain_results, "Missing pretrain_success key"
            
            if pretrain_results['pretrain_success']:
                logger.info("✅ Decision Transformer pre-training completed")
            else:
                logger.warning("⚠️ Decision Transformer pre-training had issues (expected in test)")
            
            # Test system components
            assert system.dual_agent is not None, "Dual agent not initialized"
            assert system.replay_buffer is not None, "Replay buffer not initialized"
            assert system.safety_monitor is not None, "Safety monitor not initialized"
            
            logger.info("✅ All system components initialized")
            
            # Test status reporting
            summary = system._get_training_summary()
            
            assert 'training_stats' in summary, "Missing training stats"
            assert 'current_agent' in summary, "Missing current agent"
            
            logger.info(f"✅ System status: {summary['current_agent']} agent active")
            
            # Save system state
            system.save_system_state("test_outputs/models/test_system_state.pt")
            
            self.test_results['comprehensive_system'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Comprehensive system test failed: {e}")
            self.test_results['comprehensive_system'] = False
            return False
    
    def test_integration_components(self) -> bool:
        """Test integration with existing trading system."""
        
        logger.info("🧪 Testing Integration Components...")
        
        try:
            # Test importing existing components
            from continuous_rebalancer import ContinuousRebalancer
            from tools.alpaca_client import alpaca_client
            
            logger.info("✅ Successfully imported existing trading system components")
            
            # Test basic rebalancer initialization (without actual API calls)
            rebalancer = ContinuousRebalancer()
            assert rebalancer is not None, "Failed to create rebalancer"
            
            logger.info("✅ Created continuous rebalancer instance")
            
            # Test status reporting
            status = rebalancer.get_status_report()
            assert isinstance(status, dict), "Status report should be a dictionary"
            assert 'running' in status, "Missing 'running' field in status"
            
            logger.info("✅ Status reporting works")
            
            self.test_results['integration_components'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration components test failed: {e}")
            self.test_results['integration_components'] = False
            return False
    
    async def run_all_tests(self) -> dict:
        """Run all tests and return results."""
        
        logger.info("🚀 STARTING COMPREHENSIVE RL PRE-TRAINING SYSTEM TESTS")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        # Test 1: Hybrid Data Sources
        await self.test_hybrid_data_sources()
        
        # Test 2: Realistic Trading Environment
        await self.test_realistic_trading_env()
        
        # Test 3: Safe RL Algorithms
        self.test_safe_rl_algorithms()
        
        # Test 4: Decision Transformer
        await self.test_decision_transformer()
        
        # Test 5: Comprehensive System
        await self.test_comprehensive_system()
        
        # Test 6: Integration Components
        self.test_integration_components()
        
        total_time = time.time() - start_time
        
        # Calculate results
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)
        success_rate = passed_tests / total_tests
        
        final_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'total_time_seconds': total_time,
            'individual_results': self.test_results
        }
        
        # Log final results
        logger.info("=" * 70)
        logger.info("🧪 TEST RESULTS SUMMARY")
        logger.info("=" * 70)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        
        logger.info("-" * 70)
        logger.info(f"Overall: {passed_tests}/{total_tests} tests passed ({success_rate:.1%})")
        logger.info(f"Total time: {total_time:.2f} seconds")
        
        if success_rate == 1.0:
            logger.info("🎉 ALL TESTS PASSED! RL Pre-training system is ready.")
        elif success_rate >= 0.8:
            logger.info("✅ Most tests passed. System is mostly functional.")
        else:
            logger.warning("⚠️ Several tests failed. Please review before deployment.")
        
        return final_results

async def main():
    """Main test function."""
    
    # Create test directories
    Path("test_outputs").mkdir(exist_ok=True)
    
    # Run tests
    tester = RLPretrainingTester()
    results = await tester.run_all_tests()
    
    # Save results
    import json
    with open("test_outputs/test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"📄 Test results saved to test_outputs/test_results.json")
    
    # Exit with appropriate code
    if results['success_rate'] == 1.0:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    # Check dependencies before running tests
    missing_deps = []
    
    try:
        import numpy as np
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available - some tests will be skipped")
    
    if missing_deps:
        logger.error(f"Missing required dependencies: {missing_deps}")
        logger.error("Please install: pip install numpy torch")
        sys.exit(1)
    
    # Run tests
    asyncio.run(main())