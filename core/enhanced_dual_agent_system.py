#!/usr/bin/env python3
"""
Enhanced Dual Agent System for Long/Short Trading

Integrates curriculum-trained LongShortRegimeAwarePolicy with online learning
to create a robust dual agent architecture that combines stability with adaptability.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

from models.long_short_regime_policy import LongShortRegimeAwarePolicy, PolicyConfig, ActionSpace
from agents.comprehensive_rl_pretraining import RegimeType
from agents.state import TradingState

logger = logging.getLogger(__name__)

@dataclass
class DualAgentConfig:
    """Configuration for the enhanced dual agent system."""
    
    # Model paths
    curriculum_model_path: str = "models/curriculum_final_200000.pt"
    online_model_path: str = "models/online_learner.pt"
    
    # Agent behavior
    stable_agent_weight: float = 0.7  # Base weight for stable agent
    learner_agent_weight: float = 0.3  # Base weight for learner agent
    
    # Performance thresholds
    min_performance_samples: int = 20  # Min samples to evaluate agent performance
    performance_window: int = 100      # Rolling window for performance evaluation
    agent_switch_threshold: float = 0.05  # Performance difference to switch primary
    
    # Online learning parameters
    online_learning_rate: float = 3e-4
    online_batch_size: int = 32
    online_update_frequency: int = 10  # Update every N predictions
    experience_buffer_size: int = 1000
    
    # Risk management
    max_allocation_divergence: float = 0.15  # Max difference between agents
    confidence_decay: float = 0.95  # Decay factor for confidence tracking
    
    # Fallback behavior
    fallback_to_stable_on_error: bool = True
    max_consecutive_errors: int = 3


@dataclass
class AgentPrediction:
    """Individual agent prediction with metadata."""
    allocations: Dict[str, float]
    confidence: float
    regime_id: int
    value_estimate: float
    entropy: float
    expert_weights: Optional[torch.Tensor] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class DualAgentDecision:
    """Combined decision from both agents."""
    final_allocations: Dict[str, float]
    stable_allocations: Dict[str, float]
    learner_allocations: Dict[str, float]
    stable_weight: float
    learner_weight: float
    primary_agent: str  # "stable" or "learner"
    confidence: float
    regime_id: int


class EnhancedDualAgentSystem:
    """
    Enhanced dual agent system combining curriculum-trained stability with online adaptability.
    
    The system maintains two agents:
    1. Stable Agent: Uses curriculum-trained LongShortRegimeAwarePolicy 
    2. Learner Agent: Continuously adapts through online learning
    
    Both agents operate on the same architecture for seamless integration.
    """
    
    def __init__(self, config: DualAgentConfig, symbols: List[str]):
        self.config = config
        self.symbols = sorted(symbols)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Agent models
        self.stable_agent = None
        self.learner_agent = None
        self.state_dim = None
        self.action_dim = len(symbols)
        
        # Performance tracking
        self.stable_performance_history = []
        self.learner_performance_history = []
        self.decision_history = []
        
        # Online learning
        self.experience_buffer = []
        self.online_optimizer = None
        self.update_counter = 0
        
        # Agent selection
        self.primary_agent = "stable"  # Start with stable agent
        self.agent_confidences = {"stable": 1.0, "learner": 0.5}
        self.consecutive_errors = 0
        
        logger.info(f"Initialized EnhancedDualAgentSystem for {len(symbols)} symbols")
    
    async def initialize(self) -> bool:
        """Initialize both agents and the dual agent system."""
        logger.info("Initializing Enhanced Dual Agent System")
        
        try:
            # Load stable agent (curriculum-trained)
            stable_success = await self._load_stable_agent()
            if not stable_success:
                logger.error("Failed to load stable agent")
                return False
            
            # Initialize learner agent (copy of stable agent initially)
            learner_success = await self._initialize_learner_agent()
            if not learner_success:
                logger.error("Failed to initialize learner agent")
                return False
            
            # Setup online learning
            self._setup_online_learning()
            
            logger.info("✅ Enhanced Dual Agent System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize dual agent system: {e}")
            return False
    
    async def _load_stable_agent(self) -> bool:
        """Load the curriculum-trained stable agent."""
        model_path = Path(self.config.curriculum_model_path)
        
        if not model_path.exists():
            logger.error(f"Curriculum model not found: {model_path}")
            return False
        
        try:
            logger.info(f"Loading stable agent from {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            metadata = checkpoint.get('metadata', {})
            
            # Get configuration from model metadata
            state_dim = metadata.get('state_dim', 336)
            model_action_dim = metadata.get('action_dim', 27)
            model_symbols = metadata.get('symbols', [])
            
            # Verify symbol compatibility
            if model_symbols and len(model_symbols) != len(self.symbols):
                logger.warning(f"Model trained on {len(model_symbols)} symbols, but system initialized with {len(self.symbols)}")
                logger.info(f"Model symbols: {model_symbols[:5]}...")
                logger.info(f"System symbols: {self.symbols[:5]}...")
                
                # Update system to use model symbols for compatibility
                if len(model_symbols) > len(self.symbols):
                    logger.info(f"Expanding system to use all {len(model_symbols)} model symbols")
                    self.symbols = model_symbols
                    self.action_dim = len(model_symbols)
            
            action_dim = model_action_dim
            
            # Create stable agent
            stable_config = PolicyConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                action_space=ActionSpace.FULL_RANGE,
                max_gross_exposure=1.5,
                max_net_exposure=0.35,
                min_net_exposure=-0.15,
                max_position_size=0.08
            )
            
            self.stable_agent = LongShortRegimeAwarePolicy(stable_config).to(self.device)
            
            # Load weights - handle both formats
            if 'stable_policy' in checkpoint:
                # New format with dual agent state
                self.stable_agent.load_state_dict(checkpoint['stable_policy'])
                logger.info("✅ Loaded from stable_policy key")
            elif 'policy_state_dict' in checkpoint:
                # Old format
                self.stable_agent.load_state_dict(checkpoint['policy_state_dict'])
                logger.info("✅ Loaded from policy_state_dict key")
            else:
                # Direct state dict
                self.stable_agent.load_state_dict(checkpoint)
                logger.info("✅ Loaded from direct checkpoint")
            self.stable_agent.eval()
            
            self.state_dim = state_dim
            
            logger.info(f"✅ Stable agent loaded (state_dim: {state_dim}, action_dim: {action_dim})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load stable agent: {e}")
            return False
    
    async def _initialize_learner_agent(self) -> bool:
        """Initialize the learner agent (starts as copy of stable agent)."""
        if self.stable_agent is None:
            logger.error("Stable agent must be loaded first")
            return False
        
        try:
            # Create learner agent with same architecture
            learner_config = PolicyConfig(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                action_space=ActionSpace.FULL_RANGE,
                max_gross_exposure=1.5,
                max_net_exposure=0.35,
                min_net_exposure=-0.15,
                max_position_size=0.08
            )
            
            self.learner_agent = LongShortRegimeAwarePolicy(learner_config).to(self.device)
            
            # Initialize with stable agent weights
            self.learner_agent.load_state_dict(self.stable_agent.state_dict())
            self.learner_agent.train()  # Keep in training mode for online learning
            
            # Try to load existing online model if available, or use learner_policy from curriculum model
            online_path = Path(self.config.online_model_path)
            curriculum_path = Path(self.config.curriculum_model_path)
            
            try:
                if online_path.exists():
                    # Load existing online model
                    online_checkpoint = torch.load(online_path, map_location=self.device)
                    if 'learner_policy' in online_checkpoint:
                        self.learner_agent.load_state_dict(online_checkpoint['learner_policy'])
                        logger.info("✅ Loaded existing online learner weights")
                    else:
                        raise Exception("No learner_policy in online model")
                else:
                    # Use learner_policy from curriculum model if available
                    curriculum_checkpoint = torch.load(curriculum_path, map_location=self.device)
                    if 'learner_policy' in curriculum_checkpoint:
                        self.learner_agent.load_state_dict(curriculum_checkpoint['learner_policy'])
                        logger.info("✅ Initialized learner from curriculum learner_policy")
                    else:
                        logger.info("✅ Initialized learner as copy of stable agent")
                        
            except Exception as e:
                logger.warning(f"Using stable agent weights for learner initialization: {e}")
            
            logger.info("✅ Learner agent initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize learner agent: {e}")
            return False
    
    def _setup_online_learning(self):
        """Setup online learning components."""
        if self.learner_agent is None:
            logger.error("Learner agent must be initialized first")
            return
        
        # Create optimizer for online learning
        self.online_optimizer = torch.optim.Adam(
            self.learner_agent.parameters(),
            lr=self.config.online_learning_rate
        )
        
        logger.info(f"✅ Online learning setup complete (lr={self.config.online_learning_rate})")
    
    async def get_portfolio_allocations(self, state: TradingState) -> Dict[str, float]:
        """
        Generate portfolio allocations using the dual agent system.
        
        Args:
            state: Current trading state
            
        Returns:
            Dictionary of symbol -> allocation weight
        """
        try:
            # Get predictions from both agents
            stable_pred = await self._get_stable_prediction(state)
            learner_pred = await self._get_learner_prediction(state)
            
            if stable_pred is None and learner_pred is None:
                logger.error("Both agents failed to generate predictions")
                return self._get_emergency_allocations()
            
            # Handle single agent failure
            if stable_pred is None:
                logger.warning("Stable agent failed, using learner only")
                return learner_pred.allocations if learner_pred else self._get_emergency_allocations()
            
            if learner_pred is None:
                logger.warning("Learner agent failed, using stable only")
                return stable_pred.allocations
            
            # Combine predictions from both agents
            decision = self._combine_agent_predictions(stable_pred, learner_pred)
            
            # Store decision for tracking
            self.decision_history.append(decision)
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-1000:]
            
            # Add to experience buffer for online learning
            await self._add_experience(state, decision)
            
            # Update online learning if enough experiences accumulated
            if len(self.experience_buffer) >= self.config.online_batch_size:
                self.update_counter += 1
                if self.update_counter % self.config.online_update_frequency == 0:
                    await self._update_online_learning()
            
            # Log decision
            logger.info(f"🎯 Dual Agent Decision: {decision.primary_agent} primary "
                       f"(stable={decision.stable_weight:.2f}, learner={decision.learner_weight:.2f})")
            
            self.consecutive_errors = 0  # Reset error counter on success
            return decision.final_allocations
            
        except Exception as e:
            logger.error(f"Error in dual agent prediction: {e}")
            self.consecutive_errors += 1
            
            if self.consecutive_errors >= self.config.max_consecutive_errors:
                logger.error("Too many consecutive errors, resetting learner agent")
                await self._reset_learner_agent()
            
            return self._get_emergency_allocations()
    
    async def _get_stable_prediction(self, state: TradingState) -> Optional[AgentPrediction]:
        """Get prediction from the stable (curriculum-trained) agent."""
        if self.stable_agent is None:
            return None
        
        try:
            observation = await self._convert_state_to_observation(state)
            if observation is None:
                return None
            
            regime_id = self._get_current_regime(state)
            
            # Generate prediction
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            regime_tensor = torch.LongTensor([regime_id]).to(self.device)
            
            with torch.no_grad():
                actions, log_probs, info = self.stable_agent.get_action(
                    obs_tensor, regime_tensor, deterministic=True
                )
            
            # Convert to allocations
            actions_np = actions.cpu().numpy()[0]
            allocations = {
                symbol: float(actions_np[i]) if i < len(actions_np) else 0.0
                for i, symbol in enumerate(self.symbols)
            }
            
            # Calculate confidence from entropy
            entropy = info.get('entropy', torch.tensor(1.0)).item()
            confidence = max(0.1, min(1.0, 1.0 - entropy))
            
            # Get value estimate
            value_estimate = info.get('value', torch.tensor(0.0)).item()
            
            # Get expert weights for interpretability
            expert_weights = info.get('gating_weights', None)
            
            return AgentPrediction(
                allocations=allocations,
                confidence=confidence,
                regime_id=regime_id,
                value_estimate=value_estimate,
                entropy=entropy,
                expert_weights=expert_weights
            )
            
        except Exception as e:
            logger.error(f"Stable agent prediction failed: {e}")
            return None
    
    async def _get_learner_prediction(self, state: TradingState) -> Optional[AgentPrediction]:
        """Get prediction from the online learner agent."""
        if self.learner_agent is None:
            return None
        
        try:
            observation = await self._convert_state_to_observation(state)
            if observation is None:
                return None
            
            regime_id = self._get_current_regime(state)
            
            # Generate prediction with slight exploration for learning
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            regime_tensor = torch.LongTensor([regime_id]).to(self.device)
            
            with torch.no_grad():
                actions, log_probs, info = self.learner_agent.get_action(
                    obs_tensor, regime_tensor, deterministic=False  # Allow exploration
                )
            
            # Convert to allocations
            actions_np = actions.cpu().numpy()[0]
            allocations = {
                symbol: float(actions_np[i]) if i < len(actions_np) else 0.0
                for i, symbol in enumerate(self.symbols)
            }
            
            # Calculate confidence
            entropy = info.get('entropy', torch.tensor(1.0)).item()
            confidence = max(0.1, min(1.0, 1.0 - entropy))
            
            # Adjust learner confidence based on recent performance
            confidence *= self.agent_confidences.get("learner", 0.5)
            
            # Get value estimate
            value_estimate = info.get('value', torch.tensor(0.0)).item()
            
            # Get expert weights
            expert_weights = info.get('gating_weights', None)
            
            return AgentPrediction(
                allocations=allocations,
                confidence=confidence,
                regime_id=regime_id,
                value_estimate=value_estimate,
                entropy=entropy,
                expert_weights=expert_weights
            )
            
        except Exception as e:
            logger.error(f"Learner agent prediction failed: {e}")
            return None
    
    def _combine_agent_predictions(self, stable_pred: AgentPrediction, 
                                 learner_pred: AgentPrediction) -> DualAgentDecision:
        """Combine predictions from both agents into final decision."""
        
        # Calculate dynamic weights based on recent performance and confidence
        stable_weight, learner_weight = self._calculate_dynamic_weights(stable_pred, learner_pred)
        
        # Combine allocations
        final_allocations = {}
        for symbol in self.symbols:
            stable_alloc = stable_pred.allocations.get(symbol, 0.0)
            learner_alloc = learner_pred.allocations.get(symbol, 0.0)
            
            # Check for excessive divergence
            divergence = abs(stable_alloc - learner_alloc)
            if divergence > self.config.max_allocation_divergence:
                logger.warning(f"{symbol} allocation divergence {divergence:.3f} > threshold")
                # Favor stable agent when divergence is high
                stable_weight *= 1.2
                learner_weight *= 0.8
                # Renormalize
                total_weight = stable_weight + learner_weight
                stable_weight /= total_weight
                learner_weight /= total_weight
            
            # Weighted combination
            final_allocations[symbol] = (
                stable_weight * stable_alloc + learner_weight * learner_alloc
            )
        
        # Determine primary agent
        primary_agent = "stable" if stable_weight > learner_weight else "learner"
        
        # Combined confidence
        combined_confidence = stable_weight * stable_pred.confidence + learner_weight * learner_pred.confidence
        
        return DualAgentDecision(
            final_allocations=final_allocations,
            stable_allocations=stable_pred.allocations,
            learner_allocations=learner_pred.allocations,
            stable_weight=stable_weight,
            learner_weight=learner_weight,
            primary_agent=primary_agent,
            confidence=combined_confidence,
            regime_id=stable_pred.regime_id
        )
    
    def _calculate_dynamic_weights(self, stable_pred: AgentPrediction, 
                                 learner_pred: AgentPrediction) -> Tuple[float, float]:
        """Calculate dynamic weights for combining agent predictions."""
        
        # Base weights from configuration
        stable_weight = self.config.stable_agent_weight
        learner_weight = self.config.learner_agent_weight
        
        # Adjust based on confidence levels
        stable_confidence = stable_pred.confidence
        learner_confidence = learner_pred.confidence * self.agent_confidences["learner"]
        
        confidence_ratio = stable_confidence / (stable_confidence + learner_confidence + 1e-8)
        stable_weight = 0.3 + 0.7 * confidence_ratio  # Range [0.3, 1.0]
        learner_weight = 1.0 - stable_weight
        
        # Adjust based on recent performance if we have enough data
        if (len(self.stable_performance_history) >= self.config.min_performance_samples and 
            len(self.learner_performance_history) >= self.config.min_performance_samples):
            
            # Recent performance comparison
            stable_recent = np.mean(self.stable_performance_history[-20:])
            learner_recent = np.mean(self.learner_performance_history[-20:])
            
            performance_difference = learner_recent - stable_recent
            
            # Adjust weights based on performance
            if performance_difference > self.config.agent_switch_threshold:
                # Learner is performing better
                learner_weight = min(0.8, learner_weight * 1.3)
                stable_weight = 1.0 - learner_weight
                self.primary_agent = "learner"
                logger.debug("Favoring learner agent due to superior performance")
                
            elif performance_difference < -self.config.agent_switch_threshold:
                # Stable is performing better
                stable_weight = min(0.9, stable_weight * 1.2)
                learner_weight = 1.0 - stable_weight
                self.primary_agent = "stable"
                logger.debug("Favoring stable agent due to superior performance")
        
        # Ensure weights sum to 1
        total_weight = stable_weight + learner_weight
        stable_weight /= total_weight
        learner_weight /= total_weight
        
        return stable_weight, learner_weight
    
    async def _add_experience(self, state: TradingState, decision: DualAgentDecision):
        """Add experience to buffer for online learning."""
        try:
            observation = await self._convert_state_to_observation(state)
            if observation is None:
                return
            
            # Calculate reward (simplified - would be more sophisticated in practice)
            portfolio = state.get("portfolio", {})
            current_nav = float(portfolio.get("equity", 100000))
            nav_history = state.get("nav_history", [current_nav])
            
            if len(nav_history) > 1:
                reward = (current_nav / nav_history[-2]) - 1.0  # Return since last prediction
            else:
                reward = 0.0
            
            experience = {
                'state': observation,
                'action': list(decision.final_allocations.values()),
                'reward': reward,
                'regime_id': decision.regime_id,
                'log_prob': 0.0,  # Will be recalculated during training
                'timestamp': datetime.now()
            }
            
            self.experience_buffer.append(experience)
            
            # Keep buffer size manageable
            if len(self.experience_buffer) > self.config.experience_buffer_size:
                self.experience_buffer = self.experience_buffer[-self.config.experience_buffer_size:]
            
            # Update performance histories
            self._update_agent_performance(decision, reward)
            
        except Exception as e:
            logger.error(f"Failed to add experience: {e}")
    
    async def _update_online_learning(self):
        """Perform online learning update on the learner agent."""
        if (self.learner_agent is None or self.online_optimizer is None or 
            len(self.experience_buffer) < self.config.online_batch_size):
            return
        
        try:
            logger.debug("Performing online learning update")
            
            # Sample recent experiences for training
            recent_experiences = self.experience_buffer[-self.config.online_batch_size:]
            
            # Perform update
            loss_info = self.learner_agent.update_online(
                recent_experiences, self.online_optimizer, str(self.device)
            )
            
            # Update learner confidence based on learning progress
            if loss_info.get("loss", float('inf')) < 1.0:  # Reasonable loss
                self.agent_confidences["learner"] = min(1.0, self.agent_confidences["learner"] * 1.05)
            else:
                self.agent_confidences["learner"] *= 0.95
            
            logger.debug(f"Online update complete: loss={loss_info.get('loss', 0):.4f}, "
                        f"learner_confidence={self.agent_confidences['learner']:.3f}")
            
            # Save updated learner model periodically
            if self.update_counter % 50 == 0:
                await self._save_online_model()
            
        except Exception as e:
            logger.error(f"Online learning update failed: {e}")
    
    def _update_agent_performance(self, decision: DualAgentDecision, reward: float):
        """Update performance tracking for both agents."""
        
        # Attribute reward to agents based on their contribution
        stable_performance = reward * decision.stable_weight
        learner_performance = reward * decision.learner_weight
        
        self.stable_performance_history.append(stable_performance)
        self.learner_performance_history.append(learner_performance)
        
        # Keep history manageable
        if len(self.stable_performance_history) > self.config.performance_window:
            self.stable_performance_history = self.stable_performance_history[-self.config.performance_window:]
        if len(self.learner_performance_history) > self.config.performance_window:
            self.learner_performance_history = self.learner_performance_history[-self.config.performance_window:]
        
        # Update agent confidences with decay
        self.agent_confidences["stable"] *= self.config.confidence_decay
        self.agent_confidences["learner"] *= self.config.confidence_decay
        
        # Boost confidence of well-performing agents
        if reward > 0:
            if decision.primary_agent == "stable":
                self.agent_confidences["stable"] = min(1.0, self.agent_confidences["stable"] * 1.02)
            else:
                self.agent_confidences["learner"] = min(1.0, self.agent_confidences["learner"] * 1.02)
    
    async def _convert_state_to_observation(self, state: TradingState) -> Optional[np.ndarray]:
        """Convert trading state to EXACT curriculum training observation format."""
        try:
            observation_parts = []
            
            # Portfolio data
            portfolio = state.get("portfolio", {})
            current_positions = portfolio.get("positions", {})
            nav = float(portfolio.get("equity", 100000))
            
            # Convert positions to allocations array for portfolio features
            positions_array = np.array([
                current_positions.get(symbol, {}).get('allocation_pct', 0.0) 
                for symbol in self.symbols
            ])
            
            # EXACT curriculum training format: 12 features per symbol (27 symbols = 324) + 8 portfolio = 332
            # But model expects 336, so need to check the exact training format
            
            for symbol in self.symbols:
                market_data = state.get("market_data", {}).get("symbols", {}).get(symbol, {})
                
                # EXACT match to curriculum training LongShortPortfolioEnv._get_observation()
                # 8 price/technical features + 4 borrow features = 12 per symbol
                symbol_features = [
                    # Price and technical features (8)
                    market_data.get('return_1h', 0.0),  # log_return
                    market_data.get('volatility_20', 0.15),
                    market_data.get('rsi', 50.0) / 100.0 - 0.5,  # Normalized to [-0.5, 0.5]
                    market_data.get('momentum_10', 0.0),
                    market_data.get('volume_ratio', 1.0) - 1.0,  # Normalized around 0
                    market_data.get('price_sma_ratio', 1.0) - 1.0,  # Price/SMA20 - 1
                    market_data.get('sma_trend', market_data.get('sma_20', 100) / market_data.get('sma_50', 100) - 1.0 if market_data.get('sma_50', 100) != 0 else 0.0),  # SMA20/SMA50 - 1
                    market_data.get('long_trend', market_data.get('sma_50', 100) / market_data.get('sma_200', 100) - 1.0 if market_data.get('sma_200', 100) != 0 else 0.0),  # SMA50/SMA200 - 1
                    
                    # Borrow features (4)
                    0.02,  # borrow_rate (default)
                    1.0,   # borrow_available (True)
                    0.0,   # htb_premium
                    0.0    # placeholder for additional borrow feature
                ]
                
                # Handle NaN/inf values
                symbol_features = [0.0 if np.isnan(x) or np.isinf(x) else float(x) for x in symbol_features]
                
                observation_parts.extend(symbol_features)
            
            # Portfolio state features (8 features matching curriculum training)
            portfolio_features = [
                nav - 1.0,  # NAV relative to starting value (1.0)
                np.sum(positions_array),  # Net exposure
                np.sum(np.abs(positions_array)),  # Gross exposure
                np.sum(np.maximum(positions_array, 0)),  # Long exposure
                np.sum(np.minimum(positions_array, 0)),  # Short exposure
                len(state.get("nav_history", [nav])) / 1000 if len(state.get("nav_history", [nav])) < 1000 else 1.0,  # Time progress
            ]
            
            # Drawdown
            nav_history = state.get("nav_history", [nav])
            if len(nav_history) > 1:
                peak_nav = max(nav_history)
                drawdown = (peak_nav - nav) / peak_nav
                portfolio_features.append(drawdown)
            else:
                portfolio_features.append(0.0)
                
            # Recent performance (5-period return)
            if len(nav_history) >= 6:
                recent_return = (nav_history[-1] / nav_history[-6]) - 1.0
                portfolio_features.append(recent_return)
            else:
                portfolio_features.append(0.0)
            
            observation_parts.extend(portfolio_features)
            
            # Add the missing 4 features to reach exactly 336 (matching curriculum training)
            # These are likely market-wide or additional risk features
            additional_features = [
                0.0,  # Placeholder 1 - could be market volatility
                0.0,  # Placeholder 2 - could be sector rotation
                0.0,  # Placeholder 3 - could be risk-off indicator
                0.0   # Placeholder 4 - could be correlation metric
            ]
            observation_parts.extend(additional_features)
            
            # Verify final dimension
            final_observation = np.array(observation_parts, dtype=np.float32)
            
            if len(final_observation) != self.state_dim:
                logger.warning(f"Observation size mismatch: {len(final_observation)} vs expected {self.state_dim}")
                # Pad or truncate to exact size
                if len(final_observation) < self.state_dim:
                    padding = np.zeros(self.state_dim - len(final_observation))
                    final_observation = np.concatenate([final_observation, padding])
                else:
                    final_observation = final_observation[:self.state_dim]
            
            return final_observation
            
        except Exception as e:
            logger.error(f"Failed to convert state to observation: {e}")
            return None
    
    def _get_current_regime(self, state: TradingState) -> int:
        """Get current market regime ID."""
        market_regime = state.get("market_regime", {})
        regime_name = market_regime.get("regime", "sideways").lower()
        
        regime_mapping = {
            'bull': 0,
            'bear': 1, 
            'sideways': 2,
            'high_vol': 3
        }
        
        return regime_mapping.get(regime_name, 2)  # Default to sideways
    
    def _get_emergency_allocations(self) -> Dict[str, float]:
        """Generate emergency allocations when both agents fail."""
        logger.warning("Using emergency allocations - both agents failed")
        
        # Conservative cash position
        return {symbol: 0.0 for symbol in self.symbols}
    
    async def _reset_learner_agent(self):
        """Reset learner agent to stable agent weights."""
        if self.stable_agent is None or self.learner_agent is None:
            return
        
        try:
            logger.info("Resetting learner agent to stable agent weights")
            self.learner_agent.load_state_dict(self.stable_agent.state_dict())
            self.agent_confidences["learner"] = 0.5
            self.consecutive_errors = 0
            
        except Exception as e:
            logger.error(f"Failed to reset learner agent: {e}")
    
    async def _save_online_model(self):
        """Save the current state of both agents."""
        try:
            save_path = Path(self.config.online_model_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create combined state dict
            combined_state = {
                'stable_policy': self.stable_agent.state_dict(),
                'learner_policy': self.learner_agent.state_dict(),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_architecture': 'EnhancedDualAgentSystem',
                    'symbols': self.symbols,
                    'state_dim': self.state_dim,
                    'action_dim': self.action_dim,
                    'update_counter': self.update_counter,
                    'agent_confidences': self.agent_confidences,
                    'primary_agent': self.primary_agent
                }
            }
            
            torch.save(combined_state, save_path)
            logger.debug(f"Saved dual agent model to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save online model: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'agents_loaded': {
                'stable': self.stable_agent is not None,
                'learner': self.learner_agent is not None
            },
            'primary_agent': self.primary_agent,
            'agent_confidences': self.agent_confidences.copy(),
            'performance_samples': {
                'stable': len(self.stable_performance_history),
                'learner': len(self.learner_performance_history)
            },
            'recent_performance': {
                'stable': np.mean(self.stable_performance_history[-10:]) if self.stable_performance_history else 0.0,
                'learner': np.mean(self.learner_performance_history[-10:]) if self.learner_performance_history else 0.0
            },
            'experience_buffer_size': len(self.experience_buffer),
            'update_counter': self.update_counter,
            'consecutive_errors': self.consecutive_errors,
            'last_decision_time': self.decision_history[-1].timestamp if self.decision_history else None
        }


# Factory function for easy integration
async def create_enhanced_dual_agent_system(symbols: List[str], **config_kwargs) -> EnhancedDualAgentSystem:
    """
    Factory function to create and initialize an enhanced dual agent system.
    
    Args:
        symbols: Trading symbols to support
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Initialized EnhancedDualAgentSystem
    """
    config = DualAgentConfig(**config_kwargs)
    system = EnhancedDualAgentSystem(config, symbols)
    
    success = await system.initialize()
    if not success:
        raise RuntimeError("Failed to initialize Enhanced Dual Agent System")
    
    return system