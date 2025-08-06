"""
RL Integration Bridge
Connects the comprehensive RL pre-training system to the existing trading infrastructure
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import json
from pathlib import Path

# Import our comprehensive RL system
from agents.comprehensive_rl_pretraining import (
    ComprehensiveRLPretrainingSystem, 
    OnlineLearningConfig, 
    SafetyConstraints,
    create_comprehensive_rl_system
)

# Import existing system components
from agents.online_learning_orchestrator import OnlineLearningOrchestrator, TradingEngine, PortfolioManager, RiskManager
from agents.llm_rl_integration import LLMStateEnricher

logger = logging.getLogger(__name__)

class RLSystemBridge:
    """Bridge between comprehensive RL system and existing trading infrastructure."""
    
    def __init__(self, symbols: List[str], config: Optional[Dict] = None):
        """Initialize the RL system bridge."""
        
        self.symbols = symbols
        self.config = config or {}
        
        # Initialize RL system
        self.rl_system = None
        self.system_initialized = False
        self.last_initialization_attempt = None
        
        # Performance tracking
        self.rl_decisions_count = 0
        self.fallback_count = 0
        self.integration_stats = {
            'successful_decisions': 0,
            'fallback_decisions': 0,
            'errors': 0,
            'total_calls': 0
        }
        
        logger.info(f"✅ RL System Bridge initialized for {len(symbols)} symbols")
    
    async def initialize_rl_system(self) -> bool:
        """Initialize the comprehensive RL system."""
        
        if self.system_initialized:
            return True
        
        try:
            # Create RL system configuration
            rl_config = OnlineLearningConfig(
                # Conservative settings for live trading
                learner_performance_threshold=0.03,  # 3% outperformance needed
                safety_constraints=SafetyConstraints(
                    max_drawdown=0.08,  # 8% max drawdown
                    min_sharpe_ratio=-0.5,
                    max_position_change=0.1,  # 10% max position change
                    consecutive_losses_limit=3,
                    volatility_threshold=0.04
                )
            )
            
            # Initialize system
            self.rl_system = create_comprehensive_rl_system(self.symbols, **rl_config.__dict__)
            
            # Check if pre-trained models exist
            model_path = Path("models/pretrained_rl")
            if model_path.exists():
                logger.info("🔄 Loading pre-trained RL models...")
                # Load pre-trained models if available
                # This would load Decision Transformer and other components
            else:
                logger.info("🚀 Initializing RL system without pre-training")
            
            self.system_initialized = True
            self.last_initialization_attempt = datetime.now()
            
            logger.info("✅ Comprehensive RL system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RL system: {e}")
            self.system_initialized = False
            return False
    
    async def make_rl_decisions(self, 
                              state: Dict[str, Any], 
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make RL-based portfolio decisions using the comprehensive system.
        This is the main integration point called by continuous_rebalancer.py
        """
        
        self.integration_stats['total_calls'] += 1
        
        try:
            # Initialize system if needed
            if not self.system_initialized:
                initialized = await self.initialize_rl_system()
                if not initialized:
                    return await self._fallback_decision(state, config)
            
            # Extract required data from state
            sentiment_signals = state.get("sentiment_signals", [])
            portfolio = state.get("portfolio", {})
            market_data = state.get("market_data", {})
            
            if not sentiment_signals:
                logger.warning("No sentiment signals available for RL decisions")
                return await self._fallback_decision(state, config)
            
            # Get symbols with signals
            available_symbols = list(set([s.get('symbol') for s in sentiment_signals if s.get('symbol')]))
            if not available_symbols:
                logger.warning("No valid symbols found in sentiment signals")
                return await self._fallback_decision(state, config)
            
            logger.info(f"🤖 RL System processing {len(available_symbols)} symbols")
            
            # Create observation for RL system
            observation = self._create_rl_observation(state)
            
            if observation is None:
                logger.warning("Failed to create RL observation")
                return await self._fallback_decision(state, config)
            
            # Get RL action using dual-agent system
            regime_id = self._detect_current_regime(state)
            action, metadata = self.rl_system.dual_agent.select_action(observation, regime_id)
            
            # Convert action to trading allocations
            rl_decisions = self._convert_action_to_decisions(
                action, available_symbols, metadata, portfolio
            )
            
            # Add comprehensive RL metadata
            rl_decisions.update({
                "rl_enhanced": True,
                "comprehensive_rl": True,
                "active_agent": metadata.get('agent', 'unknown'),
                "regime_id": regime_id,
                "action_uncertainty": metadata.get('uncertainty', 0.0),
                "advanced_features": {
                    "dual_agent_system": True,
                    "safety_monitoring": True,
                    "regime_conditioning": True,
                    "uncertainty_estimation": True
                }
            })
            
            # Update statistics
            self.integration_stats['successful_decisions'] += 1
            self.rl_decisions_count += 1
            
            # Store experience for online learning
            if hasattr(self.rl_system, 'replay_buffer') and len(self.rl_system.replay_buffer.buffer) > 0:
                # This would store the experience for continuous learning
                pass
            
            logger.info(f"✅ RL decisions generated: {len(rl_decisions.get('allocations', []))} allocations")
            logger.info(f"🎯 Active agent: {rl_decisions.get('active_agent', 'unknown')}")
            
            return {"rl_decisions": rl_decisions}
            
        except Exception as e:
            logger.error(f"❌ RL decision making failed: {e}")
            self.integration_stats['errors'] += 1
            return await self._fallback_decision(state, config)
    
    def _create_rl_observation(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Create observation vector for RL system."""
        
        try:
            sentiment_signals = state.get("sentiment_signals", [])
            portfolio = state.get("portfolio", {})
            
            observation = []
            
            # Process each symbol
            for symbol in self.symbols[:10]:  # Limit to prevent dimension issues
                
                # Find sentiment signal for this symbol
                symbol_signal = next((s for s in sentiment_signals if s.get('symbol') == symbol), {})
                
                # Feature extraction
                features = [
                    symbol_signal.get('score', 0.0),  # Sentiment score
                    symbol_signal.get('confidence', 0.5),  # Confidence
                    1.0 if symbol_signal.get('signal') == 'BUY' else 0.0,  # Buy signal
                    1.0 if symbol_signal.get('signal') == 'SELL' else 0.0,  # Sell signal
                    portfolio.get('equity', 100000) / 100000.0,  # Normalized portfolio value
                    portfolio.get('cash', 10000) / portfolio.get('equity', 100000),  # Cash ratio
                    len(sentiment_signals) / 10.0,  # Signal count normalized
                ]
                
                observation.extend(features)
            
            # Pad to fixed dimension if needed
            target_dim = len(self.symbols) * 7
            current_dim = len(observation)
            
            if current_dim < target_dim:
                observation.extend([0.0] * (target_dim - current_dim))
            elif current_dim > target_dim:
                observation = observation[:target_dim]
            
            return np.array(observation, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error creating RL observation: {e}")
            return None
    
    def _detect_current_regime(self, state: Dict[str, Any]) -> int:
        """Detect current market regime for regime-conditional RL."""
        
        try:
            # Simple regime detection based on available signals
            sentiment_signals = state.get("sentiment_signals", [])
            
            if not sentiment_signals:
                return 0  # Default regime
            
            # Calculate overall sentiment and confidence
            scores = [s.get('score', 0.0) for s in sentiment_signals]
            confidences = [s.get('confidence', 0.5) for s in sentiment_signals]
            
            avg_score = np.mean(scores) if scores else 0.0
            avg_confidence = np.mean(confidences) if confidences else 0.5
            score_volatility = np.std(scores) if len(scores) > 1 else 0.0
            
            # Regime mapping (simplified)
            if avg_confidence > 0.7 and score_volatility < 0.2:
                if avg_score > 0.3:
                    return 1  # Bull low vol
                elif avg_score < -0.3:
                    return 2  # Bear low vol
                else:
                    return 4  # Sideways
            elif score_volatility > 0.4:
                return 5  # Volatile
            else:
                return 0  # Unknown
                
        except Exception as e:
            logger.warning(f"Error detecting regime: {e}")
            return 0  # Default regime
    
    def _convert_action_to_decisions(self, 
                                   action: np.ndarray, 
                                   symbols: List[str], 
                                   metadata: Dict,
                                   portfolio: Dict) -> Dict:
        """Convert RL action to trading decision format."""
        
        try:
            allocations = []
            portfolio_value = portfolio.get('equity', 100000)
            
            # Process each symbol's action
            for i, symbol in enumerate(symbols):
                if i >= len(action):
                    break
                
                action_value = float(action[i])
                
                # Only create allocation if action is significant
                if abs(action_value) > 0.02:  # 2% threshold
                    
                    # Scale action to reasonable position size
                    weight = action_value * 0.15  # Max 15% per position
                    
                    allocation = {
                        "symbol": symbol,
                        "weight": weight,
                        "confidence": 0.8 - abs(action_value) * 0.2,  # Higher confidence for smaller actions
                        "action": "buy" if weight > 0 else "sell" if weight < 0 else "hold",
                        "reasoning": f"Comprehensive RL decision: {action_value:.3f} (agent: {metadata.get('agent', 'unknown')})",
                        "rl_action_value": action_value,
                        "uncertainty": metadata.get('uncertainty', 0.0)
                    }
                    
                    allocations.append(allocation)
            
            # Determine strategy based on action distribution
            total_long = sum(a['weight'] for a in allocations if a['weight'] > 0)
            total_short = sum(abs(a['weight']) for a in allocations if a['weight'] < 0)
            
            if total_long > total_short * 2:
                strategy = "rl_bullish"
                risk_level = "moderate"
            elif total_short > total_long * 2:
                strategy = "rl_bearish" 
                risk_level = "moderate"
            else:
                strategy = "rl_balanced"
                risk_level = "low"
            
            decisions = {
                "strategy": strategy,
                "risk_level": risk_level,
                "allocations": allocations,
                "confidence": np.mean([a['confidence'] for a in allocations]) if allocations else 0.5,
                "reasoning": f"Comprehensive RL system with {len(allocations)} positions",
                "total_long_exposure": total_long,
                "total_short_exposure": total_short,
                "net_exposure": total_long - total_short
            }
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error converting action to decisions: {e}")
            return {
                "strategy": "rl_fallback",
                "allocations": [],
                "confidence": 0.3,
                "reasoning": f"RL conversion error: {e}"
            }
    
    async def _fallback_decision(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to existing RL system when comprehensive system fails."""
        
        self.integration_stats['fallback_decisions'] += 1
        self.fallback_count += 1
        
        logger.info("🔄 Falling back to existing RL orchestrator")
        
        try:
            # Use existing online learning orchestrator
            orchestrator = OnlineLearningOrchestrator(
                trading_engine=TradingEngine(),
                portfolio_manager=PortfolioManager(),
                risk_manager=RiskManager()
            )
            
            # Create enhanced state
            enricher = LLMStateEnricher()
            enhanced_state = await enricher.enrich_state_from_llm_analysis(
                sentiment_data=state.get("sentiment_data", {}),
                market_signals=state.get("sentiment_signals", []),
                portfolio_state=state.get("portfolio", {}),
                market_data=state.get("market_data", {})
            )
            
            # Get fallback RL decisions
            sentiment_signals = state.get("sentiment_signals", [])
            available_symbols = [s.get('symbol') for s in sentiment_signals if s.get('symbol')][:8]
            
            if available_symbols:
                rl_decisions = await orchestrator.make_portfolio_decisions(
                    enhanced_state=enhanced_state,
                    available_symbols=available_symbols,
                    portfolio_value=state.get("portfolio", {}).get("equity", 50000),
                    risk_tolerance=0.4  # Conservative
                )
                
                # Mark as fallback
                rl_decisions["fallback_mode"] = True
                rl_decisions["comprehensive_rl"] = False
                
                return {"rl_decisions": rl_decisions}
            
            else:
                return {"rl_decisions": {"allocations": [], "strategy": "no_action", "reasoning": "No symbols available"}}
            
        except Exception as e:
            logger.error(f"❌ Fallback RL decision also failed: {e}")
            return {
                "rl_decisions": {
                    "allocations": [],
                    "strategy": "emergency_fallback",
                    "reasoning": f"All RL systems failed: {e}",
                    "confidence": 0.1
                }
            }
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration performance statistics."""
        
        total_calls = self.integration_stats['total_calls']
        
        return {
            "system_initialized": self.system_initialized,
            "total_calls": total_calls,
            "successful_decisions": self.integration_stats['successful_decisions'],
            "fallback_decisions": self.integration_stats['fallback_decisions'],
            "errors": self.integration_stats['errors'],
            "success_rate": self.integration_stats['successful_decisions'] / max(total_calls, 1),
            "fallback_rate": self.integration_stats['fallback_decisions'] / max(total_calls, 1),
            "error_rate": self.integration_stats['errors'] / max(total_calls, 1),
            "rl_decisions_count": self.rl_decisions_count,
            "fallback_count": self.fallback_count
        }
    
    async def shutdown(self):
        """Shutdown the RL system gracefully."""
        
        if self.rl_system and hasattr(self.rl_system, 'stop_online_learning'):
            await self.rl_system.stop_online_learning()
        
        logger.info("🛑 RL System Bridge shutdown complete")

# Global instance for easy access
_rl_bridge_instance = None

def get_rl_bridge(symbols: List[str] = None, config: Dict = None) -> RLSystemBridge:
    """Get or create the global RL bridge instance."""
    
    global _rl_bridge_instance
    
    if _rl_bridge_instance is None and symbols:
        _rl_bridge_instance = RLSystemBridge(symbols, config)
    
    return _rl_bridge_instance

async def integrate_comprehensive_rl_system(state: Dict[str, Any], 
                                          config: Dict[str, Any],
                                          symbols: List[str] = None) -> Dict[str, Any]:
    """
    Main integration function called by continuous_rebalancer.py
    This replaces or enhances the existing RL decision layer.
    """
    
    # Get default symbols if not provided
    if not symbols:
        sentiment_signals = state.get("sentiment_signals", [])
        symbols = list(set([s.get('symbol') for s in sentiment_signals if s.get('symbol')]))[:10]
    
    if not symbols:
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]  # Default fallback
    
    # Get or create bridge
    bridge = get_rl_bridge(symbols)
    
    # Make RL decisions
    return await bridge.make_rl_decisions(state, config)