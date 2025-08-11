"""
RL Integration Bridge
Connects the comprehensive RL pre-training system to the existing trading infrastructure
"""

import asyncio
import logging
import numpy as np
import pandas as pd
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

# Note: Online learning orchestrator is integrated within the comprehensive RL system

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
        self.integration_stats = {
            'successful_decisions': 0,
            'errors': 0,
            'total_calls': 0
        }
        
        logger.info(f"✅ RL System Bridge initialized for {len(symbols)} symbols")
    
    async def initialize_rl_system(self) -> bool:
        """Initialize the comprehensive RL system."""
        
        if self.system_initialized:
            return True
        
        try:
            # Create comprehensive RL system
            rl_config = OnlineLearningConfig(
                buffer_size=self.config.get('buffer_size', 10000),
                batch_update_freq=pd.Timedelta(minutes=self.config.get('batch_update_minutes', 30)),
                stable_policy_update_freq=pd.Timedelta(hours=self.config.get('stable_update_hours', 6))
            )
            
            self.rl_system = create_comprehensive_rl_system(self.symbols, **self.config)
            self.system_initialized = True
            self.last_initialization_attempt = datetime.now()
            
            logger.info(f"✅ Comprehensive RL system initialized for {len(self.symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize comprehensive RL system: {e}")
            self.system_initialized = False
            self.last_initialization_attempt = datetime.now()
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
                    logger.error("❌ Comprehensive RL system failed to initialize")
                    return {"rl_decisions": {"allocations": [], "strategy": "no_action", "reasoning": "RL system initialization failed"}}
            
            # Extract required data from state
            sentiment_signals = state.get("sentiment_signals", [])
            portfolio = state.get("portfolio", {})
            market_data = state.get("market_data", {})
            
            if not sentiment_signals:
                logger.warning("No sentiment signals available for RL decisions")
                return {"rl_decisions": {"allocations": [], "strategy": "no_action", "reasoning": "No sentiment signals available"}}
            
            # Get symbols with signals
            available_symbols = list(set([s.get('symbol') for s in sentiment_signals if s.get('symbol')]))
            if not available_symbols:
                logger.warning("No valid symbols found in sentiment signals")
                return {"rl_decisions": {"allocations": [], "strategy": "no_action", "reasoning": "No valid symbols found"}}
            
            logger.info(f"🤖 RL System processing {len(available_symbols)} symbols")
            
            # Create observation for RL system
            observation = self._create_rl_observation(state)
            
            if observation is None:
                logger.warning("Failed to create RL observation")
                return {"rl_decisions": {"allocations": [], "strategy": "no_action", "reasoning": "Failed to create RL observation"}}
            
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
            return {"rl_decisions": {"allocations": [], "strategy": "error_fallback", "reasoning": f"RL system error: {str(e)}", "confidence": 0.1}}
    
    def _create_rl_observation(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Create observation vector for RL system."""
        
        try:
            sentiment_signals = state.get("sentiment_signals", [])
            portfolio = state.get("portfolio", {})
            
            observation = []
            
            # Use a fixed number of symbols to ensure consistent dimensions
            # This must match what the comprehensive RL system expects: len(symbols) * 10
            max_symbols = len(self.symbols) if len(self.symbols) <= 10 else 10  # Match RL system expectations
            symbols_to_process = self.symbols[:max_symbols]
            
            # Pad symbols list if needed to maintain consistent dimensions
            while len(symbols_to_process) < max_symbols:
                symbols_to_process.append(f"DUMMY_{len(symbols_to_process)}")
            
            # Process each symbol with exactly 10 features per symbol (to match RL system)
            for symbol in symbols_to_process:
                
                # Find sentiment signal for this symbol
                symbol_signal = next((s for s in sentiment_signals if s.get('symbol') == symbol), {})
                
                # Feature extraction - exactly 10 features per symbol to match RL system
                features = [
                    symbol_signal.get('score', 0.0),  # Sentiment score
                    symbol_signal.get('confidence', 0.5),  # Confidence
                    1.0 if symbol_signal.get('signal') == 'BUY' else 0.0,  # Buy signal
                    1.0 if symbol_signal.get('signal') == 'SELL' else 0.0,  # Sell signal
                    1.0 if symbol_signal.get('signal') == 'SHORT' else 0.0,  # Short signal
                    symbol_signal.get('strength', 0.5),  # Signal strength
                    portfolio.get('equity', 100000) / 100000.0,  # Normalized portfolio value
                    portfolio.get('cash', 10000) / portfolio.get('equity', 100000),  # Cash ratio
                    len(sentiment_signals) / 20.0,  # Signal count normalized
                    float(symbol_signal.get('has_earnings', False)),  # Earnings flag
                ]
                
                observation.extend(features)
            
            # Final dimension check - should be exactly max_symbols * 10 
            target_dim = max_symbols * 10
            current_dim = len(observation)
            
            if current_dim != target_dim:
                logger.warning(f"Dimension mismatch: expected {target_dim}, got {current_dim}. Fixing...")
                if current_dim < target_dim:
                    observation.extend([0.0] * (target_dim - current_dim))
                else:
                    observation = observation[:target_dim]
            
            final_observation = np.array(observation, dtype=np.float32)
            logger.debug(f"Created RL observation with shape: {final_observation.shape}")
            
            return final_observation
            
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
    
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration performance statistics."""
        
        total_calls = self.integration_stats['total_calls']
        
        return {
            "system_initialized": self.system_initialized,
            "total_calls": total_calls,
            "successful_decisions": self.integration_stats['successful_decisions'],
            "errors": self.integration_stats['errors'],
            "success_rate": self.integration_stats['successful_decisions'] / max(total_calls, 1),
            "error_rate": self.integration_stats['errors'] / max(total_calls, 1),
            "rl_decisions_count": self.rl_decisions_count
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
    
    # If no sentiment signals, use filtered symbols from universe filter
    if not symbols:
        filtered_symbols = state.get("filtered_symbols", [])
        if filtered_symbols:
            symbols = filtered_symbols[:10]  # Use top 10 filtered symbols
            logger.info(f"🔄 Using top {len(symbols)} filtered symbols for RL: {symbols}")
        else:
            # Last resort: use current portfolio positions if any
            portfolio_positions = state.get("portfolio", {}).get("positions", [])
            if portfolio_positions:
                symbols = [pos.get('symbol') for pos in portfolio_positions if pos.get('symbol')][:5]
                logger.info(f"🔄 Using portfolio positions for RL: {symbols}")
    
    # Only use hardcoded symbols if absolutely no other data available
    if not symbols:
        logger.warning("⚠️ No symbols available from sentiment, universe filter, or portfolio - skipping RL decisions")
        return []
    
    # Get or create bridge
    bridge = get_rl_bridge(symbols)
    
    # Make RL decisions
    return await bridge.make_rl_decisions(state, config)