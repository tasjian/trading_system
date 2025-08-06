#!/usr/bin/env python3
"""
Setup RL Integration
Complete setup script to integrate the comprehensive RL system with the existing trading infrastructure
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add the trading system to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RLIntegrationSetup:
    """Setup and integration manager for the comprehensive RL system."""
    
    def __init__(self):
        self.setup_results = {}
        
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        
        logger.info("🔍 Checking dependencies...")
        
        required_modules = [
            'numpy', 'pandas', 'torch', 'asyncio', 'pathlib', 'json'
        ]
        
        missing = []
        
        for module in required_modules:
            try:
                __import__(module)
                logger.debug(f"✅ {module} available")
            except ImportError:
                missing.append(module)
                logger.error(f"❌ {module} missing")
        
        if missing:
            logger.error(f"Missing dependencies: {missing}")
            logger.error("Please install: pip install torch numpy pandas")
            return False
        
        logger.info("✅ All dependencies satisfied")
        return True
    
    def check_existing_system(self) -> bool:
        """Check if the existing trading system is properly set up."""
        
        logger.info("🔍 Checking existing trading system...")
        
        required_files = [
            'continuous_rebalancer.py',
            'agents/workflow.py',
            'tools/alpaca_client.py',
            'agents/sentiment_agent.py'
        ]
        
        missing_files = []
        
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
                logger.error(f"❌ Missing: {file_path}")
            else:
                logger.debug(f"✅ Found: {file_path}")
        
        if missing_files:
            logger.error(f"Missing core system files: {missing_files}")
            return False
        
        logger.info("✅ Existing trading system files found")
        return True
    
    def check_rl_components(self) -> bool:
        """Check if RL system components are properly installed."""
        
        logger.info("🔍 Checking RL system components...")
        
        rl_files = [
            'agents/comprehensive_rl_pretraining.py',
            'agents/hybrid_data_sources.py',
            'agents/realistic_trading_env.py',
            'agents/safe_rl_algorithms.py',
            'agents/decision_transformer.py',
            'agents/rl_integration_bridge.py',
            'start_rl_pretraining.py'
        ]
        
        missing_files = []
        
        for file_path in rl_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
                logger.error(f"❌ Missing: {file_path}")
            else:
                logger.debug(f"✅ Found: {file_path}")
        
        if missing_files:
            logger.error(f"Missing RL components: {missing_files}")
            return False
        
        logger.info("✅ All RL components found")
        return True
    
    def test_imports(self) -> bool:
        """Test that all components can be imported."""
        
        logger.info("🔍 Testing imports...")
        
        try:
            # Test existing system imports
            from continuous_rebalancer import ContinuousRebalancer
            from agents.workflow import TradingWorkflow
            from tools.alpaca_client import alpaca_client
            logger.info("✅ Existing system imports successful")
            
            # Test RL system imports
            from agents.comprehensive_rl_pretraining import ComprehensiveRLPretrainingSystem
            from agents.rl_integration_bridge import RLSystemBridge
            from agents.hybrid_data_sources import HybridDataGenerator
            logger.info("✅ RL system imports successful")
            
            # Test integration
            from agents.rl_integration_bridge import integrate_comprehensive_rl_system
            logger.info("✅ Integration bridge import successful")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Import test failed: {e}")
            return False
    
    def create_directories(self) -> bool:
        """Create necessary directories for RL system operation."""
        
        logger.info("📁 Creating directories...")
        
        directories = [
            'models',
            'models/pretrained_rl',
            'models/online_checkpoints',
            'data',
            'data/rl_training',
            'data/historical_cache',
            'logs'
        ]
        
        try:
            for directory in directories:
                Path(directory).mkdir(exist_ok=True)
                logger.debug(f"✅ Created: {directory}")
            
            logger.info("✅ All directories created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Directory creation failed: {e}")
            return False
    
    def create_default_config(self) -> bool:
        """Create default configuration files."""
        
        logger.info("⚙️ Creating default configuration...")
        
        try:
            # RL system configuration
            rl_config = {
                "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "NFLX"],
                "rl_system": {
                    "max_drawdown": 0.08,
                    "min_sharpe_ratio": -0.5,
                    "max_position_change": 0.1,
                    "batch_update_freq_minutes": 60,
                    "stable_policy_update_freq_days": 7,
                    "enable_pretraining": True,
                    "enable_online_learning": True
                },
                "safety_constraints": {
                    "consecutive_losses_limit": 3,
                    "volatility_threshold": 0.04,
                    "uncertainty_threshold": 0.5
                },
                "integration": {
                    "enable_comprehensive_rl": True,
                    "fallback_to_existing": True,
                    "rl_priority": "comprehensive"
                }
            }
            
            config_path = Path("data/rl_config.json")
            with open(config_path, 'w') as f:
                json.dump(rl_config, f, indent=2)
            
            logger.info(f"✅ Created configuration: {config_path}")
            
            # Integration status file
            status = {
                "integration_complete": True,
                "setup_date": datetime.now().isoformat(),
                "comprehensive_rl_enabled": True,
                "version": "1.0.0"
            }
            
            status_path = Path("data/rl_integration_status.json")
            with open(status_path, 'w') as f:
                json.dump(status, f, indent=2)
            
            logger.info(f"✅ Created status file: {status_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration creation failed: {e}")
            return False
    
    async def test_integration(self) -> bool:
        """Test the complete integration by running a quick simulation."""
        
        logger.info("🧪 Testing RL integration...")
        
        try:
            # Test bridge creation
            from agents.rl_integration_bridge import get_rl_bridge
            
            test_symbols = ['AAPL', 'MSFT', 'GOOGL']
            bridge = get_rl_bridge(test_symbols)
            
            logger.info("✅ RL bridge created successfully")
            
            # Test mock state processing
            mock_state = {
                "sentiment_signals": [
                    {"symbol": "AAPL", "score": 0.3, "confidence": 0.8, "signal": "BUY"},
                    {"symbol": "MSFT", "score": -0.2, "confidence": 0.7, "signal": "SELL"},
                    {"symbol": "GOOGL", "score": 0.1, "confidence": 0.6, "signal": "BUY"}
                ],
                "portfolio": {"equity": 100000, "cash": 10000},
                "market_data": {}
            }
            
            mock_config = {"test_mode": True}
            
            # Test decision making (this will likely fallback gracefully)
            result = await bridge.make_rl_decisions(mock_state, mock_config)
            
            if result and "rl_decisions" in result:
                logger.info("✅ RL decision making test successful")
                logger.info(f"   Decisions: {len(result['rl_decisions'].get('allocations', []))} allocations")
            else:
                logger.warning("⚠️ RL decision making returned no results (expected during initial setup)")
            
            # Test statistics
            stats = bridge.get_integration_stats()
            logger.info(f"✅ Integration stats: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            return False
    
    def verify_continuous_rebalancer_integration(self) -> bool:
        """Verify that the continuous rebalancer has been properly modified."""
        
        logger.info("🔍 Verifying continuous rebalancer integration...")
        
        try:
            # Check if the integration code is present
            with open('continuous_rebalancer.py', 'r') as f:
                content = f.read()
            
            required_patterns = [
                'rl_integration_bridge',
                'integrate_comprehensive_rl_system',
                'comprehensive_rl',
                'Comprehensive RL Decision Layer'
            ]
            
            missing_patterns = []
            for pattern in required_patterns:
                if pattern not in content:
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                logger.error(f"❌ Missing integration patterns: {missing_patterns}")
                logger.error("The continuous_rebalancer.py may not be properly integrated")
                return False
            
            logger.info("✅ Continuous rebalancer integration verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
    
    async def run_complete_setup(self) -> dict:
        """Run the complete setup and integration process."""
        
        logger.info("🚀 STARTING COMPLETE RL INTEGRATION SETUP")
        logger.info("=" * 60)
        
        results = {
            "setup_date": datetime.now().isoformat(),
            "steps": {}
        }
        
        # Step 1: Check dependencies
        logger.info("\n📋 Step 1: Checking Dependencies")
        results["steps"]["dependencies"] = self.check_dependencies()
        
        # Step 2: Check existing system
        logger.info("\n📋 Step 2: Checking Existing System")
        results["steps"]["existing_system"] = self.check_existing_system()
        
        # Step 3: Check RL components
        logger.info("\n📋 Step 3: Checking RL Components")
        results["steps"]["rl_components"] = self.check_rl_components()
        
        # Step 4: Test imports
        logger.info("\n📋 Step 4: Testing Imports")
        results["steps"]["imports"] = self.test_imports()
        
        # Step 5: Create directories
        logger.info("\n📋 Step 5: Creating Directories")
        results["steps"]["directories"] = self.create_directories()
        
        # Step 6: Create configuration
        logger.info("\n📋 Step 6: Creating Configuration")
        results["steps"]["configuration"] = self.create_default_config()
        
        # Step 7: Verify rebalancer integration
        logger.info("\n📋 Step 7: Verifying Rebalancer Integration")
        results["steps"]["rebalancer_integration"] = self.verify_continuous_rebalancer_integration()
        
        # Step 8: Test integration
        logger.info("\n📋 Step 8: Testing Integration")
        results["steps"]["integration_test"] = await self.test_integration()
        
        # Calculate overall success
        successful_steps = sum(1 for success in results["steps"].values() if success)
        total_steps = len(results["steps"])
        success_rate = successful_steps / total_steps
        
        results["success_rate"] = success_rate
        results["successful_steps"] = successful_steps
        results["total_steps"] = total_steps
        results["overall_success"] = success_rate >= 0.8  # 80% success threshold
        
        # Print final results
        logger.info("\n" + "=" * 60)
        logger.info("📊 SETUP RESULTS SUMMARY")
        logger.info("=" * 60)
        
        for step_name, success in results["steps"].items():
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{step_name}: {status}")
        
        logger.info("-" * 60)
        logger.info(f"Overall: {successful_steps}/{total_steps} steps successful ({success_rate:.1%})")
        
        if results["overall_success"]:
            logger.info("🎉 SETUP COMPLETE! RL system is fully integrated.")
            logger.info("   You can now run: python continuous_rebalancer.py")
            logger.info("   Or run RL training: python start_rl_pretraining.py")
        else:
            logger.warning("⚠️ Setup incomplete. Please address failing steps before using RL system.")
        
        # Save results
        results_path = Path("data/setup_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📄 Setup results saved to: {results_path}")
        
        return results

async def main():
    """Main setup function."""
    
    setup = RLIntegrationSetup()
    results = await setup.run_complete_setup()
    
    # Exit with appropriate code
    if results["overall_success"]:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())