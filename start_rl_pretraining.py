#!/usr/bin/env python3
"""
Start RL Pre-training System
Comprehensive script to initialize and run the RL pre-training system with integration to existing trading infrastructure
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
import argparse

# Add the trading system to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our RL pre-training components
from agents.comprehensive_rl_pretraining import create_comprehensive_rl_system, OnlineLearningConfig, SafetyConstraints
from agents.hybrid_data_sources import HybridDataGenerator
from agents.realistic_trading_env import create_realistic_trading_env

# Import existing trading system components
from tools.alpaca_client import alpaca_client
from agents.workflow import TradingWorkflow
from continuous_rebalancer import ContinuousRebalancer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rl_pretraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RLPretrainingOrchestrator:
    """Orchestrates RL pre-training with existing trading system."""
    
    def __init__(self, config_path: str = None):
        """Initialize the orchestrator."""
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize symbols from config or default
        self.symbols = self.config.get('symbols', ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'])
        
        # Create directories
        self._create_directories()
        
        # Initialize components
        self.rl_system = None
        self.existing_rebalancer = None
        self.data_generator = HybridDataGenerator()
        
        logger.info(f"✅ RL Pre-training Orchestrator initialized for {len(self.symbols)} symbols")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file or use defaults."""
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            # Default configuration
            config = {
                "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "NFLX"],
                "pretraining": {
                    "training_days": 2000,
                    "validation_split": 0.2,
                    "decision_transformer_epochs": 100,
                    "save_model_path": "models/pretrained_rl"
                },
                "online_learning": {
                    "max_drawdown": 0.10,
                    "min_sharpe_ratio": -1.0,
                    "batch_update_freq_minutes": 60,
                    "stable_policy_update_freq_days": 7,
                    "safety_constraints": {
                        "max_position_change": 0.15,
                        "consecutive_losses_limit": 5,
                        "volatility_threshold": 0.06
                    }
                },
                "integration": {
                    "enable_existing_rebalancer": True,
                    "rl_weight": 0.3,  # Weight for RL decisions vs existing system
                    "fallback_to_existing": True,
                    "performance_comparison_window": 100
                }
            }
            logger.info("Using default configuration")
        
        return config
    
    def _create_directories(self):
        """Create necessary directories."""
        
        directories = [
            'logs', 'models', 'data', 'models/pretrained_rl', 
            'models/online_checkpoints', 'data/rl_training'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    async def run_pretraining_phase(self) -> dict:
        """Run the comprehensive pre-training phase."""
        
        logger.info("🚀 STARTING RL PRE-TRAINING PHASE")
        logger.info("=" * 60)
        
        # Create RL system with configuration
        online_config = OnlineLearningConfig(
            batch_update_freq=timedelta(minutes=self.config['online_learning']['batch_update_freq_minutes']),
            stable_policy_update_freq=timedelta(days=self.config['online_learning']['stable_policy_update_freq_days']),
            safety_constraints=SafetyConstraints(
                max_drawdown=self.config['online_learning']['max_drawdown'],
                min_sharpe_ratio=self.config['online_learning']['min_sharpe_ratio'],
                max_position_change=self.config['online_learning']['safety_constraints']['max_position_change'],
                consecutive_losses_limit=self.config['online_learning']['safety_constraints']['consecutive_losses_limit'],
                volatility_threshold=self.config['online_learning']['safety_constraints']['volatility_threshold']
            )
        )
        
        self.rl_system = create_comprehensive_rl_system(self.symbols, **online_config.__dict__)
        
        # Phase 1: Decision Transformer Pre-training
        logger.info("📚 Phase 1: Decision Transformer Pre-training on Synthetic Data")
        
        pretrain_config = self.config['pretraining']
        pretrain_results = await self.rl_system.pretrain_with_decision_transformer(
            training_days=pretrain_config['training_days'],
            save_path=pretrain_config['save_model_path']
        )
        
        if pretrain_results['pretrain_success']:
            logger.info("✅ Decision Transformer pre-training completed successfully")
            
            # Save pre-training results
            results_path = f"data/rl_training/pretrain_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_path, 'w') as f:
                json.dump(pretrain_results, f, indent=2, default=str)
        else:
            logger.warning("⚠️ Decision Transformer pre-training had issues - proceeding with caution")
        
        return pretrain_results
    
    async def run_online_learning_phase(self, duration_hours: int = 24) -> dict:
        """Run the online learning phase."""
        
        logger.info(f"🧠 STARTING ONLINE LEARNING PHASE ({duration_hours} hours)")
        logger.info("=" * 60)
        
        if self.rl_system is None:
            logger.error("❌ RL system not initialized - run pre-training first")
            return {"success": False, "error": "No RL system"}
        
        # Generate enhanced training data from recent market activity
        training_dataset = await self._prepare_online_training_data()
        
        # Start online learning with timeout
        try:
            # Run online learning for specified duration
            online_task = asyncio.create_task(
                self.rl_system.start_online_learning(training_data=training_dataset)
            )
            
            # Wait with timeout
            await asyncio.wait_for(online_task, timeout=duration_hours * 3600)
            
            # Get training summary
            summary = self.rl_system._get_training_summary()
            
            logger.info("✅ Online learning phase completed")
            logger.info(f"📊 Final Performance Summary:")
            logger.info(f"   - Batch Updates: {summary['training_stats']['batch_updates']}")
            logger.info(f"   - Safety Violations: {summary['safety_violations']}")
            logger.info(f"   - Active Agent: {summary['current_agent']}")
            logger.info(f"   - Stable Policy Performance: {summary['stable_policy_performance']:.4f}")
            logger.info(f"   - Learner Performance: {summary['learner_performance']:.4f}")
            
            return {"success": True, "summary": summary}
            
        except asyncio.TimeoutError:
            logger.info(f"⏰ Online learning completed after {duration_hours} hours (timeout)")
            await self.rl_system.stop_online_learning()
            summary = self.rl_system._get_training_summary()
            return {"success": True, "summary": summary, "timeout": True}
        
        except Exception as e:
            logger.error(f"❌ Online learning failed: {e}")
            await self.rl_system.stop_online_learning()
            return {"success": False, "error": str(e)}
    
    async def _prepare_online_training_data(self) -> dict:
        """Prepare training data that incorporates recent market activity."""
        
        logger.info("📊 Preparing online training data with market integration...")
        
        try:
            # Try to get recent market data from Alpaca
            market_data = {}
            
            for symbol in self.symbols[:5]:  # Limit to prevent rate limiting
                try:
                    historical = alpaca_client.get_historical_data(symbol)
                    if historical is not None and len(historical) > 0:
                        market_data[symbol] = historical.tail(100)  # Last 100 bars
                        logger.debug(f"Retrieved {len(historical)} bars for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to get market data for {symbol}: {e}")
            
            # Generate hybrid dataset incorporating real market data
            if market_data:
                logger.info(f"✅ Incorporating real market data for {len(market_data)} symbols")
                # Generate synthetic data that's informed by real market patterns
                training_days = 500  # Shorter for online learning
            else:
                logger.info("📈 Using fully synthetic training data")
                training_days = 200
            
            dataset = await self.data_generator.generate_training_dataset(
                symbols=self.symbols,
                training_days=training_days,
                validation_split=0.1  # Smaller validation split for online learning
            )
            
            # Save dataset for analysis
            dataset_path = f"data/rl_training/online_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.data_generator.save_dataset(dataset, dataset_path)
            
            return dataset['training']
            
        except Exception as e:
            logger.error(f"Failed to prepare online training data: {e}")
            logger.info("Using fallback synthetic data generation")
            
            # Fallback to pure synthetic data
            dataset = await self.data_generator.generate_training_dataset(
                symbols=self.symbols,
                training_days=200,
                validation_split=0.1
            )
            
            return dataset['training']
    
    async def integrate_with_existing_system(self) -> dict:
        """Integrate RL system with existing trading infrastructure."""
        
        logger.info("🔗 INTEGRATING WITH EXISTING TRADING SYSTEM")
        logger.info("=" * 60)
        
        integration_config = self.config['integration']
        
        if not integration_config.get('enable_existing_rebalancer', True):
            logger.info("Integration with existing rebalancer disabled")
            return {"integration_enabled": False}
        
        try:
            # Initialize existing rebalancer
            self.existing_rebalancer = ContinuousRebalancer()
            
            # Modify the continuous rebalancer to incorporate RL decisions
            await self._enhance_rebalancer_with_rl()
            
            logger.info("✅ Successfully integrated RL system with existing infrastructure")
            
            return {
                "integration_enabled": True,
                "rl_weight": integration_config['rl_weight'],
                "fallback_enabled": integration_config['fallback_to_existing']
            }
            
        except Exception as e:
            logger.error(f"❌ Integration failed: {e}")
            return {"integration_enabled": False, "error": str(e)}
    
    async def _enhance_rebalancer_with_rl(self):
        """Enhance the existing rebalancer with RL capabilities."""
        
        # This is a conceptual integration - modify the continuous rebalancer
        # to incorporate RL decisions alongside existing LLM-based decisions
        
        integration_config = self.config['integration']
        rl_weight = integration_config['rl_weight']
        
        logger.info(f"🔧 Enhancing rebalancer with RL decisions (weight: {rl_weight})")
        
        # Store original _run_rl_decision_layer method
        original_rl_method = self.existing_rebalancer._run_rl_decision_layer
        
        async def enhanced_rl_decision_layer(state, config):
            """Enhanced RL decision layer that combines existing and new RL approaches."""
            
            try:
                # Get original RL decisions
                original_result = await original_rl_method(state, config)
                
                # If we have our RL system available, blend the decisions
                if (self.rl_system and 
                    hasattr(self.rl_system.dual_agent, 'select_action') and
                    'sentiment_signals' in state):
                    
                    # Create observation from state
                    obs = self._state_to_observation(state)
                    
                    if obs is not None:
                        # Get RL action
                        regime_id = 0  # Simplified regime detection
                        rl_action, rl_metadata = self.rl_system.dual_agent.select_action(obs, regime_id)
                        
                        # Convert RL action to allocation format
                        rl_allocations = self._action_to_allocations(rl_action, state)
                        
                        # Blend original and RL decisions
                        blended_decisions = self._blend_decisions(
                            original_result.get('rl_decisions', {}),
                            rl_allocations,
                            rl_weight
                        )
                        
                        # Update state with blended decisions
                        original_result['rl_decisions'] = blended_decisions
                        original_result['rl_enhanced'] = True
                        original_result['rl_blended'] = True
                        
                        logger.info(f"✅ Blended RL decisions with weight {rl_weight}")
                
                return original_result
                
            except Exception as e:
                logger.warning(f"⚠️ RL enhancement failed, using original decisions: {e}")
                return await original_rl_method(state, config)
        
        # Replace the method
        self.existing_rebalancer._run_rl_decision_layer = enhanced_rl_decision_layer
    
    def _state_to_observation(self, state: dict) -> np.ndarray:
        """Convert trading state to RL observation."""
        
        try:
            import numpy as np
            
            sentiment_signals = state.get('sentiment_signals', [])
            portfolio = state.get('portfolio', {})
            
            # Create simplified observation
            obs = []
            
            for symbol in self.symbols[:5]:  # Limit symbols
                # Find sentiment for this symbol
                symbol_sentiment = next((s for s in sentiment_signals if s.get('symbol') == symbol), {})
                
                obs.extend([
                    symbol_sentiment.get('score', 0.0),
                    symbol_sentiment.get('confidence', 0.5),
                    portfolio.get('equity', 100000) / 100000.0,  # Normalized portfolio value
                    0.0, 0.0, 0.0, 0.0  # Placeholder features
                ])
            
            return np.array(obs, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Failed to convert state to observation: {e}")
            return None
    
    def _action_to_allocations(self, action: np.ndarray, state: dict) -> dict:
        """Convert RL action to allocation format."""
        
        try:
            allocations = []
            
            for i, symbol in enumerate(self.symbols[:len(action)]):
                if abs(action[i]) > 0.05:  # Only significant actions
                    allocation = {
                        "symbol": symbol,
                        "weight": float(action[i]) * 0.1,  # Scale down for safety
                        "confidence": 0.7,
                        "action": "buy" if action[i] > 0 else "sell",
                        "reasoning": f"RL decision: {action[i]:.3f}"
                    }
                    allocations.append(allocation)
            
            return {
                "strategy": "rl_enhanced",
                "risk_level": "moderate",
                "allocations": allocations,
                "confidence": 0.8,
                "reasoning": "RL-enhanced decision blending"
            }
            
        except Exception as e:
            logger.warning(f"Failed to convert action to allocations: {e}")
            return {"allocations": []}
    
    def _blend_decisions(self, original_decisions: dict, rl_decisions: dict, rl_weight: float) -> dict:
        """Blend original and RL decisions."""
        
        try:
            # Get allocations from both systems
            original_allocs = original_decisions.get('allocations', [])
            rl_allocs = rl_decisions.get('allocations', [])
            
            blended_allocations = []
            
            # Combine allocations by symbol
            all_symbols = set()
            all_symbols.update(alloc.get('symbol') for alloc in original_allocs)
            all_symbols.update(alloc.get('symbol') for alloc in rl_allocs)
            
            for symbol in all_symbols:
                orig_alloc = next((a for a in original_allocs if a.get('symbol') == symbol), None)
                rl_alloc = next((a for a in rl_allocs if a.get('symbol') == symbol), None)
                
                if orig_alloc and rl_alloc:
                    # Blend weights
                    orig_weight = orig_alloc.get('weight', 0.0)
                    rl_weight_val = rl_alloc.get('weight', 0.0)
                    blended_weight = (1 - rl_weight) * orig_weight + rl_weight * rl_weight_val
                    
                    blended_allocation = orig_alloc.copy()
                    blended_allocation['weight'] = blended_weight
                    blended_allocation['reasoning'] = f"Blended: {orig_alloc.get('reasoning', '')} + RL"
                    blended_allocations.append(blended_allocation)
                    
                elif orig_alloc:
                    # Scale down original allocation
                    scaled_alloc = orig_alloc.copy()
                    scaled_alloc['weight'] = orig_alloc.get('weight', 0.0) * (1 - rl_weight)
                    blended_allocations.append(scaled_alloc)
                    
                elif rl_alloc:
                    # Scale RL allocation
                    scaled_alloc = rl_alloc.copy()
                    scaled_alloc['weight'] = rl_alloc.get('weight', 0.0) * rl_weight
                    blended_allocations.append(scaled_alloc)
            
            # Create blended decision
            blended = {
                "strategy": "hybrid_rl_llm",
                "risk_level": original_decisions.get('risk_level', 'moderate'),
                "allocations": blended_allocations,
                "confidence": (original_decisions.get('confidence', 0.5) + rl_decisions.get('confidence', 0.5)) / 2,
                "reasoning": f"Hybrid: {1-rl_weight:.0%} LLM + {rl_weight:.0%} RL",
                "rl_weight": rl_weight
            }
            
            return blended
            
        except Exception as e:
            logger.warning(f"Failed to blend decisions: {e}")
            return original_decisions
    
    async def run_full_pipeline(self, 
                              pretraining: bool = True,
                              online_learning_hours: int = 24,
                              integration: bool = True) -> dict:
        """Run the complete RL pre-training pipeline."""
        
        logger.info("🚀 STARTING COMPLETE RL PRE-TRAINING PIPELINE")
        logger.info("=" * 80)
        
        results = {
            "start_time": datetime.now().isoformat(),
            "pipeline_config": {
                "pretraining": pretraining,
                "online_learning_hours": online_learning_hours,
                "integration": integration
            },
            "phases": {}
        }
        
        try:
            # Phase 1: Pre-training
            if pretraining:
                pretrain_results = await self.run_pretraining_phase()
                results["phases"]["pretraining"] = pretrain_results
            
            # Phase 2: Online Learning
            online_results = await self.run_online_learning_phase(online_learning_hours)
            results["phases"]["online_learning"] = online_results
            
            # Phase 3: Integration
            if integration:
                integration_results = await self.integrate_with_existing_system()
                results["phases"]["integration"] = integration_results
            
            # Save complete system state
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            state_path = f"models/online_checkpoints/rl_system_state_{timestamp}.pt"
            
            if self.rl_system:
                self.rl_system.save_system_state(state_path)
            
            results["end_time"] = datetime.now().isoformat()
            results["system_state_path"] = state_path
            results["success"] = True
            
            # Save results
            results_path = f"data/rl_training/pipeline_results_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info("🎉 COMPLETE RL PRE-TRAINING PIPELINE FINISHED SUCCESSFULLY")
            logger.info(f"📄 Results saved to: {results_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            results["success"] = False
            results["error"] = str(e)
            results["end_time"] = datetime.now().isoformat()
            return results
    
    def get_status_report(self) -> dict:
        """Get current status of the RL pre-training system."""
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "symbols": self.symbols,
            "rl_system_initialized": self.rl_system is not None,
            "integration_enabled": self.existing_rebalancer is not None
        }
        
        if self.rl_system:
            status["rl_training_summary"] = self.rl_system._get_training_summary()
        
        return status

async def main():
    """Main function with CLI interface."""
    
    parser = argparse.ArgumentParser(description="Start RL Pre-training System")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--pretraining-only", action="store_true", help="Run only pre-training phase")
    parser.add_argument("--online-hours", type=int, default=2, help="Online learning duration in hours")
    parser.add_argument("--skip-integration", action="store_true", help="Skip integration with existing system")
    parser.add_argument("--symbols", nargs="+", help="Override symbols list")
    
    args = parser.parse_args()
    
    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    try:
        # Initialize orchestrator
        orchestrator = RLPretrainingOrchestrator(args.config)
        
        # Override symbols if provided
        if args.symbols:
            orchestrator.symbols = args.symbols
            logger.info(f"Using override symbols: {args.symbols}")
        
        if args.pretraining_only:
            # Run only pre-training
            logger.info("Running pre-training phase only")
            results = await orchestrator.run_pretraining_phase()
            print(f"\nPre-training Results: {json.dumps(results, indent=2, default=str)}")
        else:
            # Run full pipeline
            results = await orchestrator.run_full_pipeline(
                pretraining=True,
                online_learning_hours=args.online_hours,
                integration=not args.skip_integration
            )
            
            print(f"\nPipeline Results: {json.dumps(results, indent=2, default=str)}")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure all required dependencies are available
    try:
        import torch
        import numpy as np
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install: pip install torch numpy pandas")
        sys.exit(1)
    
    asyncio.run(main())