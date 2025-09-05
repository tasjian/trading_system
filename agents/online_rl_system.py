#!/usr/bin/env python3
"""
Online RL Trading System with Dual-Agent Architecture
Implements safe online learning for live/paper trading with continuous adaptation.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, namedtuple
import json
import threading
import time
from enum import Enum
from pathlib import Path
import copy

# Handle optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal
    import torch.distributions.kl as kl
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False

# Import model version manager
try:
    from agents.model_version_manager import get_version_manager
    VERSION_MANAGER_AVAILABLE = True
except ImportError:
    VERSION_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import curriculum policy for trained models
try:
    from models.long_short_regime_policy import RegimeAwarePolicy as CurriculumPolicy
    CURRICULUM_POLICY_AVAILABLE = True
    logger.info("✅ Curriculum policy architecture available")
except ImportError:
    CURRICULUM_POLICY_AVAILABLE = False
    logger.warning("⚠️ Curriculum policy architecture not available - using local policy")

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'info'])

class MarketRegime(Enum):
    """Market regime types for contextual learning."""
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRISIS = "crisis"
    UNKNOWN = "unknown"

@dataclass
class SafetyConstraints:
    """Safety constraints for online learning."""
    max_drawdown: float = 0.10           # 10% max drawdown trigger (relaxed for paper trading)
    min_sharpe_ratio: float = -1.0       # Minimum Sharpe before rollback (relaxed)
    max_kl_divergence: float = 0.05      # Trust region constraint (relaxed)
    max_position_change: float = 0.2     # Max 20% position change per update (relaxed)
    consecutive_losses_limit: int = 10    # Max consecutive losing trades (relaxed for paper trading)
    volatility_threshold: float = 0.08   # Max vol increase before safety mode (relaxed)
    daily_loss_limit: float = 0.08       # 8% daily loss limit (relaxed for learning)

@dataclass
class OnlineLearningConfig:
    """Configuration for the online learning system."""
    
    # Dual-agent parameters
    stable_update_frequency: timedelta = timedelta(days=7)    # Weekly stable policy updates
    learner_performance_threshold: float = 0.03              # 3% outperformance needed to switch
    performance_evaluation_window: int = 100                 # Number of trades to evaluate
    
    # Experience replay
    buffer_size: int = 50000
    priority_alpha: float = 0.6          # Prioritization strength
    priority_beta: float = 0.4           # Importance sampling correction
    min_replay_size: int = 1000          # Minimum experiences before learning
    
    # Batch update settings
    batch_update_frequency: timedelta = timedelta(hours=6)   # 4x daily updates
    min_batch_size: int = 64
    max_batch_size: int = 256
    gradient_steps_per_update: int = 32
    
    # Early stopping and regularization
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    l2_regularization: float = 0.0001
    
    # Exploration parameters
    epsilon_decay: float = 0.995         # Decay exploration over time
    min_epsilon: float = 0.01            # Minimum exploration
    uncertainty_exploration_weight: float = 0.1
    
    # Safety constraints
    safety_constraints: Optional[SafetyConstraints] = None
    
    def __post_init__(self):
        if self.safety_constraints is None:
            self.safety_constraints = SafetyConstraints()

class PrioritizedReplayBuffer:
    """Enhanced prioritized replay buffer with uncertainty and regime awareness."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta) / 1000000  # Anneal beta over time
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.uncertainties = deque(maxlen=capacity)
        self.regimes = deque(maxlen=capacity)
        self.timestamps = deque(maxlen=capacity)
        
        self.max_priority = 1.0
        self.position = 0
        
    def push(self, experience: Experience, uncertainty: float = 0.0, 
             regime: MarketRegime = MarketRegime.UNKNOWN, timestamp: datetime = None):
        """Add experience with priority calculation."""
        
        if timestamp is None:
            timestamp = datetime.now()
            
        # Calculate priority based on TD error (simplified)
        td_error = abs(experience.reward) + uncertainty * 0.1
        priority = (td_error + 1e-6) ** self.alpha
        
        # Boost priority for regime transitions
        if len(self.regimes) > 0 and self.regimes[-1] != regime:
            priority *= 1.5  # Boost regime transition experiences
            
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.uncertainties.append(uncertainty)
        self.regimes.append(regime)
        self.timestamps.append(timestamp)
        
        self.max_priority = max(self.max_priority, priority)
        
    def sample(self, batch_size: int) -> Tuple[List[Experience], List[int], np.ndarray]:
        """Sample batch with prioritized sampling."""
        
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        # Convert to numpy for efficient sampling
        priorities = np.array(self.priorities)
        priorities_sum = priorities.sum()
        if priorities_sum > 0:
            probabilities = priorities / priorities_sum
        else:
            # Equal probabilities if all priorities are 0
            probabilities = np.ones(len(priorities)) / len(priorities)
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        max_weight = weights.max()
        if max_weight > 0:
            weights = weights / max_weight  # Normalize
        else:
            weights = np.ones_like(weights)  # Equal weights if max is 0
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
        
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

class RegimeAwarePolicy(nn.Module):
    """Policy network with regime conditioning."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            return
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Regime embedding
        self.regime_embedding = nn.Embedding(len(MarketRegime), 16)
        
        # Main policy network
        self.network = nn.Sequential(
            nn.Linear(state_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.zero_()
            
    def forward(self, state, regime_id):
        """Forward pass with regime conditioning."""
        if not TORCH_AVAILABLE:
            return None, None
            
        # Get regime embedding
        regime_emb = self.regime_embedding(regime_id)
        
        # Concatenate state and regime
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(regime_emb.shape) == 1:
            regime_emb = regime_emb.unsqueeze(0)
            
        state_regime = torch.cat([state, regime_emb], dim=-1)
        
        # Get policy output
        output = self.network(state_regime)
        mean, log_std = output.chunk(2, dim=-1)
        
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        return mean, std

class SafetyMonitor:
    """Monitors trading safety metrics and triggers safeguards."""
    
    def __init__(self, constraints: SafetyConstraints):
        self.constraints = constraints
        self.violation_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        self.drawdown_tracker = deque(maxlen=100)
        self.consecutive_losses = 0
        self.daily_loss_tracker = {}
        
    def check_safety_violations(self, 
                              current_performance: Dict[str, float],
                              current_portfolio_value: float,
                              policy_kl: float = 0.0) -> Dict[str, bool]:
        """Check for safety violations."""
        
        violations = {}
        today = datetime.now().date()
        
        # Track performance
        self.performance_history.append(current_performance)
        
        # 1. Drawdown check
        if len(self.drawdown_tracker) > 0:
            peak_value = max(self.drawdown_tracker)
            if peak_value > 0:
                current_drawdown = (peak_value - current_portfolio_value) / peak_value
                violations['max_drawdown'] = current_drawdown > self.constraints.max_drawdown
            else:
                violations['max_drawdown'] = False  # Cannot calculate drawdown if peak is 0
        else:
            violations['max_drawdown'] = False
            
        self.drawdown_tracker.append(current_portfolio_value)
        
        # 2. Sharpe ratio check
        if len(self.performance_history) >= 30:
            returns = [p.get('return', 0.0) for p in list(self.performance_history)[-30:]]
            if returns and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                violations['min_sharpe'] = sharpe < self.constraints.min_sharpe_ratio
            else:
                violations['min_sharpe'] = False
        else:
            violations['min_sharpe'] = False
            
        # 3. KL divergence check (policy drift)
        violations['kl_divergence'] = policy_kl > self.constraints.max_kl_divergence
        
        # 4. Consecutive losses check
        if current_performance.get('return', 0) < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        violations['consecutive_losses'] = self.consecutive_losses > self.constraints.consecutive_losses_limit
        
        # 5. Daily loss limit
        if today not in self.daily_loss_tracker:
            self.daily_loss_tracker[today] = 0.0
        self.daily_loss_tracker[today] += current_performance.get('return', 0)
        violations['daily_loss'] = self.daily_loss_tracker[today] < -self.constraints.daily_loss_limit
        
        # Clean up old daily data
        cutoff_date = datetime.now().date() - timedelta(days=7)
        self.daily_loss_tracker = {k: v for k, v in self.daily_loss_tracker.items() if k > cutoff_date}
        
        # Log violations
        active_violations = [k for k, v in violations.items() if v]
        if active_violations:
            logger.warning(f"🚨 Safety violations detected: {active_violations}")
            self.violation_history.append({
                'timestamp': datetime.now(),
                'violations': active_violations,
                'performance': current_performance
            })
        
        return violations

class DualAgentSystem:
    """Dual-agent system with stable policy and learning agent."""
    
    def __init__(self, state_dim: int, action_dim: int, config: OnlineLearningConfig, device: str = 'cpu', symbols: List[str] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        self.symbols = symbols or []
        
        # Dynamic dimension tracking for compatibility
        self.expected_state_dim = state_dim
        self.actual_state_dim = None
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - using dummy agent")
            return
        
        logger.info(f"🤖 Initializing DualAgentSystem: {state_dim} state features → {action_dim} actions")
        
        # Initialize policies with correct dimensions - use curriculum policy if available
        if CURRICULUM_POLICY_AVAILABLE:
            logger.info("🎓 Using trained curriculum policy architecture")
            self.stable_policy = CurriculumPolicy(state_dim, action_dim).to(device)
            self.learner_policy = CurriculumPolicy(state_dim, action_dim).to(device)
        else:
            logger.info("🔄 Using local RegimeAwarePolicy architecture")
            self.stable_policy = RegimeAwarePolicy(state_dim, action_dim).to(device)
            self.learner_policy = RegimeAwarePolicy(state_dim, action_dim).to(device)
            
        self.stable_optimizer = optim.Adam(self.stable_policy.parameters(), lr=0.0001)
        self.last_stable_update = datetime.now()
        self.learner_optimizer = optim.Adam(self.learner_policy.parameters(), lr=0.0003)
        
        # Try to load compatible models
        self._load_compatible_models()
        
        # Performance tracking
        self.stable_performance = deque(maxlen=config.performance_evaluation_window)
        self.learner_performance = deque(maxlen=config.performance_evaluation_window)
        self.active_agent = "stable"  # Start with stable policy
        
        # Exploration parameters
        self.epsilon = 0.1
        self.epsilon_decay = config.epsilon_decay
        self.min_epsilon = config.min_epsilon
        
        # Copy stable policy to learner initially
        self.sync_learner_to_stable()
        
        logger.info(f"✅ DualAgentSystem initialized with {len(self.symbols)} symbols")
        
    def _adapt_state_tensor(self, state_tensor: torch.Tensor, actual_size: int) -> torch.Tensor:
        """Adapt state tensor to expected dimensions."""
        try:
            if actual_size > self.expected_state_dim:
                # Truncate excess features (keep most important ones)
                adapted_tensor = state_tensor[:self.expected_state_dim]
                logger.info(f"🔧 Truncated state tensor from {actual_size} to {self.expected_state_dim}")
            elif actual_size < self.expected_state_dim:
                # Pad with zeros or repeat pattern
                padding_size = self.expected_state_dim - actual_size
                if actual_size > 0:
                    # Repeat pattern to fill gaps
                    repeat_pattern = state_tensor[:(padding_size % actual_size)] if padding_size % actual_size > 0 else torch.zeros(1)
                    padding = torch.cat([state_tensor[-padding_size//actual_size:].repeat(padding_size//actual_size), repeat_pattern])
                else:
                    padding = torch.zeros(padding_size)
                    
                adapted_tensor = torch.cat([state_tensor, padding[:padding_size]])
                logger.info(f"🔧 Padded state tensor from {actual_size} to {self.expected_state_dim}")
            else:
                adapted_tensor = state_tensor
                
            return adapted_tensor
            
        except Exception as e:
            logger.error(f"❌ State tensor adaptation failed: {e}")
            # Return truncated or zero-padded tensor as fallback
            if actual_size > self.expected_state_dim:
                return state_tensor[:self.expected_state_dim]
            else:
                padded = torch.zeros(self.expected_state_dim)
                padded[:actual_size] = state_tensor
                return padded
    
    def _attempt_model_adaptation(self, actual_state_dim: int) -> bool:
        """Attempt to adapt model architecture to current state dimensions."""
        try:
            logger.info(f"🔄 Adapting model architecture from {self.expected_state_dim} to {actual_state_dim} state dimensions")
            
            # Create new models with correct dimensions
            new_stable_policy = RegimeAwarePolicy(actual_state_dim, self.action_dim).to(self.device)
            new_learner_policy = RegimeAwarePolicy(actual_state_dim, self.action_dim).to(self.device)
            
            # Try to transfer weights if possible using model version manager
            if VERSION_MANAGER_AVAILABLE:
                try:
                    version_manager = get_version_manager()
                    
                    # Save current model state for migration
                    old_model_dict = {
                        'stable_policy': self.stable_policy.state_dict(),
                        'learner_policy': self.learner_policy.state_dict()
                    }
                    
                    # Create metadata for migration
                    old_metadata = {
                        'state_dim': self.expected_state_dim,
                        'action_dim': self.action_dim,
                        'symbols': self.symbols
                    }
                    
                    # Attempt migration to new dimensions
                    migrated_model = version_manager._migrate_model(
                        old_model_dict,
                        old_metadata,
                        actual_state_dim,
                        self.action_dim
                    )
                    
                    if migrated_model and 'stable_policy' in migrated_model:
                        new_stable_policy.load_state_dict(migrated_model['stable_policy'])
                        new_learner_policy.load_state_dict(migrated_model['learner_policy'])
                        logger.info("✅ Successfully migrated model weights to new dimensions")
                    else:
                        logger.info("🆕 Migration not possible, using fresh model weights")
                        
                except Exception as migration_error:
                    logger.warning(f"⚠️ Migration attempt failed: {migration_error}, using fresh weights")
            
            # Replace old models with new ones
            self.stable_policy = new_stable_policy
            self.learner_policy = new_learner_policy
            
            # Update optimizers
            self.stable_optimizer = optim.Adam(self.stable_policy.parameters(), lr=0.0001)
            self.learner_optimizer = optim.Adam(self.learner_policy.parameters(), lr=0.0003)
            
            # Update expected dimensions
            self.expected_state_dim = actual_state_dim
            self.state_dim = actual_state_dim
            
            logger.info(f"✅ Model architecture successfully adapted to {actual_state_dim} state dimensions")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model adaptation failed: {e}")
            return False
    
    def _load_curriculum_with_adaptation(self, checkpoint: Dict, orig_state_dim: int, orig_action_dim: int) -> bool:
        """Load curriculum model with dimension adaptation for current symbol universe."""
        try:
            stable_state_dict = checkpoint['stable_policy']
            learner_state_dict = checkpoint['learner_policy']
            
            # Adapt state dimensions (input layers)
            adapted_stable = self._adapt_state_dict_dimensions(
                stable_state_dict, orig_state_dim, self.state_dim, orig_action_dim, self.action_dim
            )
            adapted_learner = self._adapt_state_dict_dimensions(
                learner_state_dict, orig_state_dim, self.state_dim, orig_action_dim, self.action_dim
            )
            
            # Load adapted weights
            self.stable_policy.load_state_dict(adapted_stable, strict=False)
            self.learner_policy.load_state_dict(adapted_learner, strict=False)
            
            logger.info(f"✅ Successfully adapted curriculum model: {orig_state_dim}→{self.state_dim} state, {orig_action_dim}→{self.action_dim} actions")
            return True
            
        except Exception as e:
            logger.error(f"❌ Curriculum adaptation failed: {e}")
            return False
    
    def _adapt_state_dict_dimensions(self, state_dict: Dict, orig_state: int, new_state: int, orig_action: int, new_action: int) -> Dict:
        """Adapt state dict dimensions for current symbol universe."""
        adapted = {}
        
        for key, tensor in state_dict.items():
            if 'actor_backbone.0.weight' in key and tensor.shape[1] == orig_state:
                # Input layer adaptation - map actor_backbone to actor
                if new_state <= orig_state:
                    adapted[key.replace('actor_backbone.0', 'actor.0')] = tensor[:, :new_state]
                else:
                    padding = torch.zeros(tensor.shape[0], new_state - orig_state)
                    adapted[key.replace('actor_backbone.0', 'actor.0')] = torch.cat([tensor, padding], dim=1)
                    
            elif 'actor_backbone.0.bias' in key:
                adapted[key.replace('actor_backbone.0', 'actor.0')] = tensor
                
            elif 'actor_backbone.2.weight' in key:
                adapted[key.replace('actor_backbone.2', 'actor.2')] = tensor
                
            elif 'actor_backbone.2.bias' in key:
                adapted[key.replace('actor_backbone.2', 'actor.2')] = tensor
                    
            elif 'mu_head.weight' in key and tensor.shape[0] == orig_action:
                # Output layer adaptation
                if new_action <= orig_action:
                    adapted[key] = tensor[:new_action, :]
                else:
                    padding = torch.randn(new_action - orig_action, tensor.shape[1]) * 0.01
                    adapted[key] = torch.cat([tensor, padding], dim=0)
                    
            elif 'mu_head.bias' in key and tensor.shape[0] == orig_action:
                if new_action <= orig_action:
                    adapted[key] = tensor[:new_action]
                else:
                    padding = torch.zeros(new_action - orig_action)
                    adapted[key] = torch.cat([tensor, padding], dim=0)
                    
            elif 'log_std' in key and tensor.shape[0] == orig_action:
                # Adapt log_std parameter
                if new_action <= orig_action:
                    adapted[key] = tensor[:new_action]
                else:
                    padding = torch.zeros(new_action - orig_action)
                    adapted[key] = torch.cat([tensor, padding], dim=0)
                    
            elif 'critic.0.weight' in key:
                # Critic input layer adaptation 
                # Saved model has regime embedding (352 = 336 + 16), current model doesn't (just state_dim)
                critic_input_dim = tensor.shape[1]  # 352 = 336 state + 16 regime
                
                if new_state <= orig_state:
                    # Truncate to new state dimension only (ignore regime embedding part)
                    adapted[key] = tensor[:, :new_state]
                else:
                    # Use original state features + pad for additional features
                    state_part = tensor[:, :orig_state]  # Take first 336 features (ignore regime embedding)
                    padding = torch.zeros(tensor.shape[0], new_state - orig_state)
                    adapted[key] = torch.cat([state_part, padding], dim=1)
                    
            else:
                # Keep other layers as-is (critic hidden layers, biases, etc.)
                adapted[key] = tensor
        
        return adapted
        
    def _load_compatible_models(self):
        """Load compatible models using version manager or curriculum model."""
        
        # Check for curriculum model first (highest priority)
        curriculum_marker = Path(".curriculum_model_enabled")
        if curriculum_marker.exists():
            curriculum_model_path = Path("models/latest_curriculum_model.pt")
            if curriculum_model_path.exists():
                try:
                    logger.info("🎓 Loading curriculum-trained model...")
                    curriculum_checkpoint = torch.load(curriculum_model_path, map_location=self.device)
                    
                    if 'stable_policy' in curriculum_checkpoint and 'learner_policy' in curriculum_checkpoint:
                        # Check if dimensions match current system
                        curriculum_metadata = curriculum_checkpoint.get('metadata', {})
                        curriculum_state_dim = curriculum_metadata.get('state_dim', self.state_dim)
                        curriculum_action_dim = curriculum_metadata.get('action_dim', self.action_dim)
                        
                        if curriculum_state_dim != self.state_dim or curriculum_action_dim != self.action_dim:
                            logger.warning(f"🔄 Curriculum model dimension mismatch: trained({curriculum_state_dim}→{curriculum_action_dim}) vs current({self.state_dim}→{self.action_dim})")
                            logger.info("🎯 Adapting curriculum model to current symbol universe...")
                            
                            # Load curriculum model with dimension adaptation
                            success = self._load_curriculum_with_adaptation(
                                curriculum_checkpoint, curriculum_state_dim, curriculum_action_dim
                            )
                            if not success:
                                raise Exception("Curriculum model adaptation failed")
                        else:
                            # Direct loading when dimensions match
                            self.stable_policy.load_state_dict(curriculum_checkpoint['stable_policy'])
                            self.learner_policy.load_state_dict(curriculum_checkpoint['learner_policy'])
                        
                        # Load optimizers if available
                        if 'stable_optimizer' in curriculum_checkpoint:
                            self.stable_optimizer.load_state_dict(curriculum_checkpoint['stable_optimizer'])
                        if 'learner_optimizer' in curriculum_checkpoint:
                            self.learner_optimizer.load_state_dict(curriculum_checkpoint['learner_optimizer'])
                        
                        reward_weights = curriculum_checkpoint.get('reward_weights', {})
                        logger.info("🎉 Successfully loaded curriculum-trained model!")
                        logger.info(f"📊 Reward weights: α={reward_weights.get('alpha', 1.0):.2f}, β={reward_weights.get('beta', 0.5):.2f}, γ={reward_weights.get('gamma', 0.3):.2f}, δ={reward_weights.get('delta', 0.2):.2f}")
                        
                        return
                    else:
                        logger.warning("⚠️ Curriculum model missing required keys, falling back to version manager")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load curriculum model: {e}, falling back to version manager")
        
        if not VERSION_MANAGER_AVAILABLE or not self.symbols:
            logger.info("📝 Using fresh models (no version manager or symbols)")
            return
        
        try:
            version_manager = get_version_manager()
            compatible_model = version_manager.load_compatible_model(
                self.symbols, self.state_dim, self.action_dim
            )
            
            if compatible_model and 'stable_policy' in compatible_model:
                # Load stable policy with detailed error handling
                try:
                    self.stable_policy.load_state_dict(compatible_model['stable_policy'])
                    logger.info("✅ Loaded compatible stable policy")
                except RuntimeError as e:
                    if "size mismatch" in str(e):
                        logger.warning(f"⚠️ Model dimension mismatch, using fresh models: {e}")
                        logger.info(f"Expected dimensions: state={self.state_dim}, action={self.action_dim}")
                        return
                    else:
                        raise
                
                # Load learner policy if available
                if 'learner_policy' in compatible_model:
                    self.learner_policy.load_state_dict(compatible_model['learner_policy'])
                    logger.info("✅ Loaded compatible learner policy")
                else:
                    # Sync learner to stable
                    self.sync_learner_to_stable()
                    logger.info("🔄 Synced learner to stable policy")
                
                # Load optimizers if available and compatible
                if 'stable_optimizer' in compatible_model:
                    try:
                        self.stable_optimizer.load_state_dict(compatible_model['stable_optimizer'])
                        logger.info("✅ Loaded stable optimizer")
                    except Exception as e:
                        logger.warning(f"Could not load stable optimizer: {e}")
                
                if 'learner_optimizer' in compatible_model:
                    try:
                        self.learner_optimizer.load_state_dict(compatible_model['learner_optimizer'])
                        logger.info("✅ Loaded learner optimizer")
                    except Exception as e:
                        logger.warning(f"Could not load learner optimizer: {e}")
                
                # Log metadata (compatible_model contains the metadata)
                if compatible_model and 'metadata' in compatible_model:
                    model_metadata = compatible_model['metadata']
                    old_arch = model_metadata.get('architecture', {})
                    logger.info(f"📊 Migrated from: {len(old_arch.get('symbols', []))} symbols, "
                               f"{old_arch.get('state_dim', 0)}→{old_arch.get('action_dim', 0)}")
                elif compatible_model:
                    logger.info("📊 Loaded model without detailed metadata")
            else:
                logger.info("📝 No compatible models found - using fresh models")
                
        except Exception as e:
            logger.warning(f"⚠️ Model loading failed, using fresh models: {e}")
    
    def sync_learner_to_stable(self):
        """Sync learner policy to stable policy."""
        if TORCH_AVAILABLE:
            self.learner_policy.load_state_dict(self.stable_policy.state_dict())
    
    def save_models(self, training_stats: Dict = None):
        """Save models using version manager."""
        if not TORCH_AVAILABLE or not VERSION_MANAGER_AVAILABLE or not self.symbols:
            return
        
        try:
            version_manager = get_version_manager()
            
            # Validate inputs
            if not isinstance(self.symbols, list) or len(self.symbols) == 0:
                raise ValueError("Symbols list is empty or invalid")
            if not isinstance(self.state_dim, int) or self.state_dim <= 0:
                raise ValueError(f"Invalid state_dim: {self.state_dim}")
            if not isinstance(self.action_dim, int) or self.action_dim <= 0:
                raise ValueError(f"Invalid action_dim: {self.action_dim}")
            
            model_state_dict = {
                'stable_policy': self.stable_policy.state_dict(),
                'learner_policy': self.learner_policy.state_dict(),
                'stable_optimizer': self.stable_optimizer.state_dict(),
                'learner_optimizer': self.learner_optimizer.state_dict()
            }
            
            # Call with correct signature (5 parameters total)
            saved_path = version_manager.save_model_with_metadata(
                model_state_dict,
                self.symbols,
                self.state_dim,
                self.action_dim,
                training_stats
            )
            
            if saved_path:
                logger.info(f"💾 Models saved with version control: {saved_path}")
            else:
                logger.warning("Model saving returned None path")
                
        except TypeError as e:
            if "positional arguments" in str(e):
                logger.error(f"❌ Model saving signature mismatch: {e}")
                logger.error("Expected: save_model_with_metadata(model_state_dict, symbols, state_dim, action_dim, training_stats=None)")
            else:
                logger.error(f"❌ Model saving type error: {e}")
        except Exception as e:
            logger.error(f"❌ Model saving failed: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
        
    def select_action(self, state: np.ndarray, regime: MarketRegime, 
                     deterministic: bool = False, uncertainty_estimate: float = 0.0) -> Tuple[np.ndarray, Dict]:
        """Select action using appropriate agent."""
        
        if not TORCH_AVAILABLE:
            # Return more realistic random actions instead of zeros
            logger.info("🎲 Using fallback action generation (PyTorch not available)")
            # Generate small random actions between -0.2 and 0.2
            random_actions = np.random.uniform(-0.2, 0.2, self.action_dim).astype(np.float32)
            return random_actions, {"agent": "fallback", "uncertainty": 0.3}
        
        # Choose active policy
        policy = self.stable_policy if self.active_agent == "stable" else self.learner_policy
        
        try:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                
                # Dynamic dimension adaptation
                actual_state_size = state_tensor.shape[0] if state_tensor.dim() == 1 else state_tensor.shape[1]
                if self.actual_state_dim is None:
                    self.actual_state_dim = actual_state_size
                    if actual_state_size != self.expected_state_dim:
                        logger.warning(f"⚠️ State dimension mismatch detected: expected {self.expected_state_dim}, got {actual_state_size}")
                
                # Handle dimension mismatch with adaptive reshaping
                if actual_state_size != self.expected_state_dim:
                    state_tensor = self._adapt_state_tensor(state_tensor, actual_state_size)
                    
                logger.debug(f"State tensor shape: {state_tensor.shape}, expected: ({self.expected_state_dim},)")
                
                # Convert integer regime to enum if needed
                if isinstance(regime, int):
                    regime_list = list(MarketRegime)
                    if 0 <= regime < len(regime_list):
                        regime_enum = regime_list[regime]
                    else:
                        regime_enum = MarketRegime.UNKNOWN
                    regime_tensor = torch.LongTensor([regime]).to(self.device)
                else:
                    regime_tensor = torch.LongTensor([list(MarketRegime).index(regime)]).to(self.device)
                
                logger.debug(f"Regime tensor shape: {regime_tensor.shape}")
                
                try:
                    mean, std = policy(state_tensor, regime_tensor)
                    logger.debug(f"Policy output shapes - mean: {mean.shape}, std: {std.shape}, expected action_dim: {self.action_dim}")
                except RuntimeError as model_error:
                    if "mat1 and mat2 shapes cannot be multiplied" in str(model_error):
                        logger.warning(f"🔄 Model dimension mismatch detected: {model_error}")
                        logger.warning(f"🔧 Attempting model architecture adaptation...")
                        
                        # Try to create compatible models with current dimensions
                        if self._attempt_model_adaptation(actual_state_size):
                            # Retry with adapted models
                            policy = self.stable_policy if self.active_agent == "stable" else self.learner_policy
                            mean, std = policy(state_tensor, regime_tensor)
                            logger.info(f"✅ Successfully adapted model architecture")
                        else:
                            raise model_error
                    else:
                        raise model_error
                
                # Check if the policy returned valid outputs
                if mean is None or std is None:
                    logger.warning("Policy returned None outputs, using fallback")
                    raise RuntimeError("Policy returned None")
            
                if deterministic:
                    action = mean
                else:
                    # Add uncertainty-based exploration
                    exploration_noise = uncertainty_estimate * self.config.uncertainty_exploration_weight
                    noise = torch.randn_like(mean) * (std + exploration_noise)
                    action = mean + noise
                    
                    # Epsilon-greedy component
                    if np.random.random() < self.epsilon:
                        random_action = torch.randn_like(action) * 0.1
                        action = action + random_action
                
                # Decay epsilon
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                
                # Clip action to reasonable bounds
                action = torch.clamp(action, -1.0, 1.0)
                
                # Ensure action has correct shape (handle batch dimensions carefully)
                logger.debug(f"Action tensor shape before processing: {action.shape}, expected action_dim: {self.action_dim}")
                
                if action.dim() > 1:
                    # Check if we have a proper batch dimension to remove
                    if action.shape[0] == 1 and action.shape[1] == self.action_dim:
                        action = action.squeeze(0)  # Remove batch dimension
                        logger.debug(f"Removed batch dimension: {action.shape}")
                    elif action.shape[0] == self.action_dim and action.shape[1] == 1:
                        action = action.squeeze(1)  # Remove trailing dimension
                        logger.debug(f"Removed trailing dimension: {action.shape}")
                    else:
                        logger.warning(f"Unexpected action tensor shape: {action.shape}, flattening")
                        action = action.flatten()[:self.action_dim]
                        
                # Enhanced dimension validation with detailed logging
                action_numpy = action.cpu().numpy()
                expected_shape = (self.action_dim,)
                actual_shape = action_numpy.shape
                
                if actual_shape != expected_shape:
                    logger.warning(f"Action shape mismatch in select_action: got {actual_shape}, expected {expected_shape}")
                    logger.debug(f"Raw action tensor shape: {action.shape}, symbols: {len(self.symbols)}, action_dim: {self.action_dim}")
                    
                    if action_numpy.size == 1 and self.action_dim > 1:
                        # Broadcast single action to all dimensions
                        single_value = action_numpy.item() if action_numpy.ndim > 0 else float(action_numpy)
                        action_numpy = np.full(self.action_dim, single_value, dtype=np.float32)
                        logger.info(f"Broadcasted single action {single_value:.4f} to {self.action_dim} dimensions")
                    elif action_numpy.size > self.action_dim:
                        action_numpy = action_numpy[:self.action_dim]
                        logger.info(f"Truncated action from {action_numpy.size} to {self.action_dim} dimensions")
                    else:
                        # Pad with zeros
                        padded_action = np.zeros(self.action_dim, dtype=np.float32)
                        padded_action[:action_numpy.size] = action_numpy.flatten()
                        action_numpy = padded_action
                        logger.info(f"Padded action from {action_numpy.size} to {self.action_dim} dimensions")
                
                return action_numpy, {
                    "agent": self.active_agent,
                    "uncertainty": uncertainty_estimate,
                    "epsilon": self.epsilon,
                    "regime": regime.value
                }
                
        except Exception as e:
            logger.warning(f"PyTorch policy failed: {e}, using fallback actions")
            
            # Enhanced error handling with dimension information
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                logger.error(f"❌ Model dimension mismatch: {e}")
                logger.error(f"🔍 Expected state dim: {self.expected_state_dim}, Actual: {self.actual_state_dim}")
                logger.error(f"🔍 Model input layer expects: {self.expected_state_dim + 16} (state + regime embedding)")
                
            # Generate meaningful random actions instead of zeros
            random_actions = np.random.uniform(-0.2, 0.2, self.action_dim).astype(np.float32)
            return random_actions, {"agent": "fallback_error", "uncertainty": 0.5, "error": str(e)}
    
    def update_performance(self, agent_type: str, performance_metrics: Dict[str, float]):
        """Update performance tracking for specified agent."""
        if agent_type == "stable":
            self.stable_performance.append(performance_metrics)
        else:
            self.learner_performance.append(performance_metrics)
    
    def should_switch_agents(self) -> bool:
        """Check if we should switch active agents based on performance."""
        
        if len(self.stable_performance) < self.config.performance_evaluation_window // 2:
            return False
            
        if len(self.learner_performance) < self.config.performance_evaluation_window // 2:
            return False
        
        # Calculate recent performance averages
        stable_returns = [p.get('return', 0) for p in list(self.stable_performance)[-50:]]
        learner_returns = [p.get('return', 0) for p in list(self.learner_performance)[-50:]]
        
        stable_avg = np.mean(stable_returns) if stable_returns else 0
        learner_avg = np.mean(learner_returns) if learner_returns else 0
        
        # Switch if learner significantly outperforms stable
        performance_diff = learner_avg - stable_avg
        threshold = self.config.learner_performance_threshold
        
        should_switch = performance_diff > threshold
        
        if should_switch:
            new_agent = "learner" if self.active_agent == "stable" else "stable"
            logger.info(f"🔄 Switching from {self.active_agent} to {new_agent} agent "
                       f"(performance diff: {performance_diff:.4f})")
            self.active_agent = new_agent
            
        return should_switch
    
    def update_stable_policy(self) -> bool:
        """Update stable policy if enough time has passed and learner is performing well."""
        
        time_since_update = datetime.now() - self.last_stable_update
        if time_since_update < self.config.stable_update_frequency:
            return False
            
        # Only update if learner has been performing well
        if len(self.learner_performance) >= 50:
            learner_returns = [p.get('return', 0) for p in list(self.learner_performance)[-50:]]
            if np.mean(learner_returns) > 0:  # Positive average return
                logger.info("📈 Updating stable policy from successful learner")
                if TORCH_AVAILABLE:
                    self.stable_policy.load_state_dict(self.learner_policy.state_dict())
                self.last_stable_update = datetime.now()
                return True
                
        return False

class OnlineRLTradingSystem:
    """Main online RL trading system with safe continuous learning."""
    
    def __init__(self, symbols: List[str], config: OnlineLearningConfig, device: str = 'cpu', alpaca_client=None):
        self.symbols = symbols
        self.config = config
        self.device = device
        self.alpaca_client = alpaca_client  # For dynamic symbol updates
        
        # Dynamic state and action dimensions based on actual symbols
        self.state_dim = len(symbols) * 12  # 12 features per symbol (price, volume, indicators, etc.)
        self.action_dim = len(symbols)      # One action per symbol
        
        logger.info(f"🎯 Online RL System initialized: {len(symbols)} symbols, "
                   f"state_dim={self.state_dim}, action_dim={self.action_dim}")
        
        # Core components
        self.dual_agent = DualAgentSystem(self.state_dim, self.action_dim, config, device, symbols)
        self.replay_buffer = PrioritizedReplayBuffer(config.buffer_size, config.priority_alpha, config.priority_beta)
        self.safety_monitor = SafetyMonitor(config.safety_constraints)
        
        # Training state
        self.training_active = False
        self.last_batch_update = datetime.now()
        self.last_safety_check = datetime.now()
        self.last_symbol_update = datetime.now()
        
        # Performance tracking
        self.performance_history = deque(maxlen=10000)
        self.training_stats = {
            'total_updates': 0,
            'safety_violations': 0,
            'agent_switches': 0,
            'stable_policy_updates': 0,
            'average_return': 0.0,
            'average_sharpe': 0.0
        }
        
        # Regime detection (simplified)
        self.current_regime = MarketRegime.UNKNOWN
        
        # Background training thread
        self.training_thread = None
        self.stop_training_event = threading.Event()
        
    def detect_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """Detect current market regime from market data."""
        
        # Simplified regime detection based on volatility and trend
        try:
            volatility = market_data.get('volatility', 0.02)
            trend = market_data.get('trend', 0.0)  # Positive = bullish, Negative = bearish
            
            if volatility > 0.05:  # High volatility
                if trend > 0.01:
                    return MarketRegime.BULL_HIGH_VOL
                elif trend < -0.01:
                    return MarketRegime.BEAR_HIGH_VOL
                else:
                    return MarketRegime.VOLATILE
            else:  # Low volatility
                if trend > 0.005:
                    return MarketRegime.BULL_LOW_VOL
                elif trend < -0.005:
                    return MarketRegime.BEAR_LOW_VOL
                else:
                    return MarketRegime.SIDEWAYS
                    
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return MarketRegime.UNKNOWN
    
    async def process_market_step(self, 
                                market_state: np.ndarray,
                                market_data: Dict[str, Any],
                                previous_action: np.ndarray = None,
                                previous_reward: float = 0.0,
                                deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single market step and return action."""
        
        # Detect current regime
        self.current_regime = self.detect_market_regime(market_data)
        
        # Get uncertainty estimate (simplified)
        uncertainty = self._estimate_uncertainty(market_state, market_data)
        
        # Select action
        action, action_info = self.dual_agent.select_action(
            market_state, 
            self.current_regime, 
            deterministic=deterministic,
            uncertainty_estimate=uncertainty
        )
        
        # CRITICAL FIX: Enhanced experience storage with validation
        if previous_action is not None:
            try:
                # Validate previous action dimensions
                if isinstance(previous_action, np.ndarray) and previous_action.shape[0] != self.action_dim:
                    logger.warning(f"Action dimension mismatch: {previous_action.shape[0]} != {self.action_dim}, reshaping")
                    if previous_action.shape[0] < self.action_dim:
                        # Pad with zeros
                        padded_action = np.zeros(self.action_dim)
                        padded_action[:previous_action.shape[0]] = previous_action
                        previous_action = padded_action
                    else:
                        # Truncate
                        previous_action = previous_action[:self.action_dim]
                
                # Create experience with proper validation
                experience = Experience(
                    state=self._last_state if hasattr(self, '_last_state') else market_state.copy(),
                    action=previous_action.copy() if isinstance(previous_action, np.ndarray) else np.array(previous_action),
                    reward=float(previous_reward),
                    next_state=market_state.copy(),
                    done=False,
                    info={
                        'regime': self.current_regime.value,
                        'uncertainty': uncertainty,
                        'timestamp': datetime.now().isoformat(),
                        'step': getattr(self, '_step_counter', 0)
                    }
                )
                
                # Store in replay buffer with validation
                self.replay_buffer.push(
                    experience, 
                    uncertainty=uncertainty,
                    regime=self.current_regime
                )
                
                # Log experience storage for debugging
                if getattr(self, '_step_counter', 0) % 20 == 0:
                    logger.info(f"🧠 Experience stored: Step {getattr(self, '_step_counter', 0)}, "
                               f"Reward={previous_reward:.4f}, Buffer={len(self.replay_buffer.buffer)}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to store experience: {e}")
                logger.error(f"   Previous action shape: {previous_action.shape if hasattr(previous_action, 'shape') else 'no shape'}")
                logger.error(f"   Market state shape: {market_state.shape if hasattr(market_state, 'shape') else 'no shape'}")
        else:
            # Log when no previous action (first step)
            if getattr(self, '_step_counter', 0) <= 1:
                logger.info(f"🎯 First step - no previous action to store (step {getattr(self, '_step_counter', 0)})")
        
        # Store current state for next iteration
        self._last_state = market_state.copy()
        
        # CRITICAL FIX: Enhanced performance tracking with proper logging
        performance_metrics = {
            'return': previous_reward,
            'timestamp': datetime.now(),
            'regime': self.current_regime.value,
            'uncertainty': uncertainty,
            'agent': action_info.get('agent', 'unknown'),
            'step': getattr(self, '_step_counter', 0),
            'portfolio_value': market_data.get('portfolio_value', 0),
            'buffer_size': len(self.replay_buffer.buffer)
        }
        
        # Increment step counter
        if not hasattr(self, '_step_counter'):
            self._step_counter = 0
        self._step_counter += 1
        
        # Always append to performance history for tracking
        self.performance_history.append(performance_metrics)
        
        # Update dual agent performance tracking
        self.dual_agent.update_performance(action_info['agent'], performance_metrics)
        
        # Log performance periodically for monitoring
        if self._step_counter % 10 == 0:
            logger.info(f"📊 Step {self._step_counter}: Return={previous_reward:.4f}, "
                       f"Agent={performance_metrics['agent']}, Regime={self.current_regime.value}, "
                       f"Buffer={len(self.replay_buffer.buffer)}, History={len(self.performance_history)}")
        
        # Check for agent switching
        self.dual_agent.should_switch_agents()
        
        # Check safety violations
        current_portfolio_value = market_data.get('portfolio_value', 100000)
        violations = self.safety_monitor.check_safety_violations(
            performance_metrics, 
            current_portfolio_value
        )
        
        if any(violations.values()):
            logger.warning(f"🚨 Safety violations detected: {violations}")
            self.training_stats['safety_violations'] += 1
            
            # Switch to stable policy in case of violations
            if self.dual_agent.active_agent != "stable":
                self.dual_agent.active_agent = "stable"
                logger.info("🛡️ Switched to stable policy due to safety violations")
            
            # Apply aggressive loss mitigation for specific violations
            if violations.get('min_sharpe', False):
                logger.warning("🚨 CRITICAL: Low Sharpe ratio detected - implementing loss mitigation")
                # Signal to reduce all positions by 20% to cut losses
                market_data['loss_mitigation_signal'] = {
                    'action': 'reduce_positions',
                    'reduction_factor': 0.2,
                    'reason': 'low_sharpe_ratio'
                }
            
            if violations.get('max_drawdown', False):
                logger.warning("🚨 CRITICAL: Maximum drawdown exceeded - emergency position reduction")
                # Signal to reduce all positions by 50% in emergency
                market_data['loss_mitigation_signal'] = {
                    'action': 'emergency_reduction',
                    'reduction_factor': 0.5,
                    'reason': 'max_drawdown'
                }
            
            if violations.get('daily_loss', False):
                logger.warning("🚨 CRITICAL: Daily loss limit exceeded - halt new purchases")
                # Signal to halt all buying, only allow selling
                market_data['loss_mitigation_signal'] = {
                    'action': 'halt_buying',
                    'reduction_factor': 0.0,
                    'reason': 'daily_loss_limit'
                }
        
        # Trigger background training if needed
        await self._maybe_trigger_training()
        
        # Validate and ensure action has correct dimensions
        if action.shape[0] != len(self.symbols):
            logger.warning(f"Action dimension mismatch in process_market_step: got {action.shape[0]}, expected {len(self.symbols)}")
            if action.shape[0] < len(self.symbols):
                # Pad with zeros
                padded_action = np.zeros(len(self.symbols), dtype=action.dtype)
                padded_action[:action.shape[0]] = action
                action = padded_action
            else:
                # Truncate
                action = action[:len(self.symbols)]
        
        return action, {
            **action_info,
            'regime': self.current_regime.value,
            'uncertainty': uncertainty,
            'safety_violations': violations
        }
    
    def _estimate_uncertainty(self, state: np.ndarray, market_data: Dict[str, Any]) -> float:
        """Estimate uncertainty in current market conditions."""
        
        # Simple uncertainty estimation based on volatility and missing data
        volatility = market_data.get('volatility', 0.02)
        data_completeness = market_data.get('data_completeness', 1.0)
        
        base_uncertainty = volatility * 2.0  # Scale volatility
        data_uncertainty = (1.0 - data_completeness) * 0.5
        
        return min(1.0, base_uncertainty + data_uncertainty)
    
    async def _maybe_trigger_training(self):
        """Check if we should trigger a training update."""
        
        time_since_update = datetime.now() - self.last_batch_update
        buffer_size = len(self.replay_buffer.buffer)
        
        # CRITICAL FIX: More aggressive training triggers to ensure learning happens
        should_train = (
            # Regular schedule trigger
            (time_since_update >= self.config.batch_update_frequency and 
             buffer_size >= self.config.min_replay_size) or
            # Force training when buffer gets full to prevent loss of experiences
            buffer_size >= self.config.buffer_size * 0.8 or
            # Regular small batch updates to keep learning active
            (buffer_size >= max(32, self.config.min_replay_size // 4) and 
             time_since_update >= timedelta(minutes=30))
        )
        
        if should_train and not self.training_active:
            logger.info(f"🎓 Triggering training: buffer_size={buffer_size}, time_since_update={time_since_update}")
            await self._perform_batch_update()
    
    async def _perform_batch_update(self):
        """Perform a batch update of the learning agent."""
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - skipping training")
            return
            
        self.training_active = True
        logger.info("🎓 Starting batch update...")
        
        try:
            # Determine effective batch size
            available_experiences = len(self.replay_buffer.buffer)
            effective_batch_size = min(self.config.min_batch_size, available_experiences)
            
            logger.info(f"🎓 Batch update: {available_experiences} available, using batch size {effective_batch_size}")
            
            # Sample batch
            experiences, indices, weights = self.replay_buffer.sample(effective_batch_size)
            
            if not experiences:
                logger.warning("⚠️ No experiences sampled for training")
                return
            
            # CRITICAL FIX: Validate and convert experiences to tensors safely
            try:
                # Extract and validate data
                states_list = []
                actions_list = []
                rewards_list = []
                next_states_list = []
                
                for i, e in enumerate(experiences):
                    # Validate state dimensions
                    if hasattr(e.state, 'shape') and e.state.shape[0] != self.state_dim:
                        logger.warning(f"Experience {i}: state dimension mismatch {e.state.shape[0]} != {self.state_dim}")
                        # Pad or truncate state
                        if e.state.shape[0] < self.state_dim:
                            padded_state = np.zeros(self.state_dim)
                            padded_state[:e.state.shape[0]] = e.state
                            states_list.append(padded_state)
                        else:
                            states_list.append(e.state[:self.state_dim])
                    else:
                        states_list.append(e.state)
                    
                    # Validate action dimensions  
                    if hasattr(e.action, 'shape') and e.action.shape[0] != self.action_dim:
                        logger.warning(f"Experience {i}: action dimension mismatch {e.action.shape[0]} != {self.action_dim}")
                        # Pad or truncate action
                        if e.action.shape[0] < self.action_dim:
                            padded_action = np.zeros(self.action_dim)
                            padded_action[:e.action.shape[0]] = e.action
                            actions_list.append(padded_action)
                        else:
                            actions_list.append(e.action[:self.action_dim])
                    else:
                        actions_list.append(e.action)
                    
                    # Validate next_state dimensions
                    if hasattr(e.next_state, 'shape') and e.next_state.shape[0] != self.state_dim:
                        if e.next_state.shape[0] < self.state_dim:
                            padded_next_state = np.zeros(self.state_dim)
                            padded_next_state[:e.next_state.shape[0]] = e.next_state
                            next_states_list.append(padded_next_state)
                        else:
                            next_states_list.append(e.next_state[:self.state_dim])
                    else:
                        next_states_list.append(e.next_state)
                    
                    rewards_list.append(float(e.reward))
                
                # Convert to tensors
                states = torch.FloatTensor(states_list).to(self.device)
                actions = torch.FloatTensor(actions_list).to(self.device)
                rewards = torch.FloatTensor(rewards_list).to(self.device)
                next_states = torch.FloatTensor(next_states_list).to(self.device)
                weights_tensor = torch.FloatTensor(weights).to(self.device)
                
                logger.debug(f"Batch tensors: states={states.shape}, actions={actions.shape}, rewards={rewards.shape}")
                
            except Exception as tensor_error:
                logger.error(f"❌ Failed to create training tensors: {tensor_error}")
                logger.error(f"   State dim: {self.state_dim}, Action dim: {self.action_dim}")
                if experiences:
                    logger.error(f"   Sample experience state shape: {experiences[0].state.shape if hasattr(experiences[0].state, 'shape') else 'no shape'}")
                    logger.error(f"   Sample experience action shape: {experiences[0].action.shape if hasattr(experiences[0].action, 'shape') else 'no shape'}")
                raise tensor_error
            
            # Regime conditioning (simplified - use current regime for all)  
            if isinstance(self.current_regime, int):
                regime_id = self.current_regime
            else:
                regime_id = list(MarketRegime).index(self.current_regime)
            regime_ids = torch.LongTensor([regime_id] * len(experiences)).to(self.device)
            
            # Policy gradient update
            self.dual_agent.learner_optimizer.zero_grad()
            
            means, stds = self.dual_agent.learner_policy(states, regime_ids)
            dist = Normal(means, stds)
            
            # Calculate policy loss (simplified REINFORCE)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            policy_loss = -(log_probs * rewards * weights_tensor).mean()
            
            # Add entropy regularization
            entropy = dist.entropy().sum(dim=-1).mean()
            policy_loss = policy_loss - 0.01 * entropy
            
            # Add L2 regularization
            l2_reg = sum(p.pow(2.0).sum() for p in self.dual_agent.learner_policy.parameters())
            policy_loss = policy_loss + self.config.l2_regularization * l2_reg
            
            policy_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.dual_agent.learner_policy.parameters(), 1.0)
            
            self.dual_agent.learner_optimizer.step()
            
            # Update priorities in replay buffer
            with torch.no_grad():
                td_errors = abs(rewards).cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)
            
            # Update training stats
            self.training_stats['total_updates'] += 1
            self.last_batch_update = datetime.now()
            
            # Check for stable policy update
            if self.dual_agent.update_stable_policy():
                self.training_stats['stable_policy_updates'] += 1
            
            logger.info(f"✅ Batch update completed (loss: {policy_loss.item():.4f})")
            
        except Exception as e:
            logger.error(f"❌ Batch update failed: {e}")
            
        finally:
            self.training_active = False
    
    async def force_training_update(self, min_batch_size: int = 16):
        """Force a training update regardless of normal triggers."""
        buffer_size = len(self.replay_buffer.buffer)
        
        if buffer_size < min_batch_size:
            logger.warning(f"⚠️ Insufficient experiences for forced training: {buffer_size} < {min_batch_size}")
            return False
        
        if self.training_active:
            logger.info("🔄 Training already active, skipping forced update")
            return False
        
        logger.info(f"🚀 FORCE TRAINING: Initiating batch update with {buffer_size} experiences")
        await self._perform_batch_update()
        return True
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics with comprehensive performance tracking."""
        
        # CRITICAL FIX: Enhanced performance metrics calculation
        performance_window = 100  # Last 100 steps for performance calculation
        
        # Calculate performance metrics from history
        if len(self.performance_history) > 0:
            recent_performance = list(self.performance_history)[-performance_window:]
            recent_returns = [p['return'] for p in recent_performance]
            
            self.training_stats['average_return'] = np.mean(recent_returns) if recent_returns else 0.0
            self.training_stats['total_return'] = np.sum(recent_returns) if recent_returns else 0.0
            self.training_stats['return_volatility'] = np.std(recent_returns) if len(recent_returns) > 1 else 0.0
            
            if len(recent_returns) > 10 and np.std(recent_returns) > 0:
                self.training_stats['average_sharpe'] = (
                    np.mean(recent_returns) / np.std(recent_returns) * np.sqrt(252)
                )
            else:
                self.training_stats['average_sharpe'] = 0.0
                
            # Calculate win rate
            winning_trades = sum(1 for r in recent_returns if r > 0)
            self.training_stats['win_rate'] = winning_trades / len(recent_returns) if recent_returns else 0.0
            
            # Track performance by agent
            agent_performance = {}
            for p in recent_performance:
                agent = p.get('agent', 'unknown')
                if agent not in agent_performance:
                    agent_performance[agent] = []
                agent_performance[agent].append(p['return'])
            
            for agent, returns in agent_performance.items():
                avg_return = np.mean(returns) if returns else 0.0
                self.training_stats[f'{agent}_avg_return'] = avg_return
                self.training_stats[f'{agent}_trades'] = len(returns)
        else:
            # Initialize empty stats if no history
            self.training_stats.update({
                'average_return': 0.0,
                'total_return': 0.0,
                'return_volatility': 0.0,
                'average_sharpe': 0.0,
                'win_rate': 0.0
            })
        
        # Add current system state
        current_stats = {
            **self.training_stats,
            'active_agent': self.dual_agent.active_agent,
            'current_regime': self.current_regime.value,
            'buffer_size': len(self.replay_buffer.buffer),
            'buffer_capacity': self.config.buffer_size,
            'buffer_utilization': len(self.replay_buffer.buffer) / self.config.buffer_size,
            'stable_performance_samples': len(self.dual_agent.stable_performance),
            'learner_performance_samples': len(self.dual_agent.learner_performance),
            'training_active': self.training_active,
            'performance_history_size': len(self.performance_history),
            'last_update_time': self.last_batch_update.isoformat() if self.last_batch_update else None,
            'last_symbol_update': self.last_symbol_update.isoformat() if hasattr(self, 'last_symbol_update') else None,
            'current_symbols': self.symbols.copy(),
            'symbol_count': len(self.symbols),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'system_initialized': True,
            'step_counter': getattr(self, '_step_counter', 0)
        }
        
        return current_stats
    
    def update_symbols_from_portfolio(self, force_update: bool = False) -> bool:
        """Dynamically update RL symbols from current portfolio positions."""
        if not self.alpaca_client:
            logger.warning("No Alpaca client available for dynamic symbol updates")
            return False
            
        # Check if update is needed (hourly or forced)
        now = datetime.now()
        if not force_update and (now - self.last_symbol_update).total_seconds() < 3600:  # 1 hour
            return False
            
        try:
            # Get current portfolio positions
            positions = self.alpaca_client.get_positions()
            
            # Extract symbols with meaningful positions (> $100 value)
            portfolio_symbols = []
            for pos in positions:
                if abs(pos['market_value']) > 100:  # Minimum $100 position
                    portfolio_symbols.append(pos['symbol'])
            
            # Ensure minimum symbol count (add core symbols if needed)
            core_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
            for symbol in core_symbols:
                if symbol not in portfolio_symbols and len(portfolio_symbols) < 10:
                    portfolio_symbols.append(symbol)
            
            # Sort for consistency
            portfolio_symbols = sorted(list(set(portfolio_symbols)))
            
            # Check if symbols changed
            if set(portfolio_symbols) == set(self.symbols):
                self.last_symbol_update = now
                return False  # No change needed
            
            logger.info(f"🔄 Updating RL symbols: {len(self.symbols)} → {len(portfolio_symbols)}")
            logger.info(f"Old symbols: {self.symbols}")
            logger.info(f"New symbols: {portfolio_symbols}")
            
            # Update symbols and dimensions
            old_symbols = self.symbols.copy()
            self.symbols = portfolio_symbols
            
            # Recalculate dimensions
            new_state_dim = len(portfolio_symbols) * 12
            new_action_dim = len(portfolio_symbols)
            
            if new_state_dim != self.state_dim or new_action_dim != self.action_dim:
                logger.info(f"🎯 Updating dimensions: state {self.state_dim}→{new_state_dim}, action {self.action_dim}→{new_action_dim}")
                self.state_dim = new_state_dim
                self.action_dim = new_action_dim
                
                # Reinitialize dual agent system with new dimensions
                logger.info("🔄 Reinitializing dual agent system with new dimensions")
                self.dual_agent = DualAgentSystem(self.state_dim, self.action_dim, self.config, self.device, self.symbols)
                
                # Clear experience buffer (old experiences have wrong dimensions)
                logger.info("🗑️ Clearing experience buffer due to dimension change")
                self.replay_buffer = PrioritizedReplayBuffer(self.config.buffer_size, self.config.priority_alpha, self.config.priority_beta)
            
            self.last_symbol_update = now
            
            # Log symbol changes
            added_symbols = set(portfolio_symbols) - set(old_symbols)
            removed_symbols = set(old_symbols) - set(portfolio_symbols)
            
            if added_symbols:
                logger.info(f"➕ Added symbols: {list(added_symbols)}")
            if removed_symbols:
                logger.info(f"➖ Removed symbols: {list(removed_symbols)}")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to update symbols from portfolio: {e}")
            return False
    
    def save_system_state(self, filepath: str):
        """Save complete system state with version control."""
        
        state = {
            'config': asdict(self.config),
            'symbols': self.symbols,
            'training_stats': self.training_stats,
            'current_regime': self.current_regime.value,
            'performance_history': list(self.performance_history)[-1000:]  # Last 1000 entries
        }
        
        # Save models using version manager
        if TORCH_AVAILABLE and VERSION_MANAGER_AVAILABLE:
            try:
                self.dual_agent.save_models(self.training_stats)
                logger.info("✅ Models saved with version control")
            except Exception as e:
                logger.warning(f"⚠️ Version-controlled model save failed: {e}")
                # Fallback to old method
                torch.save({
                    'stable_policy': self.dual_agent.stable_policy.state_dict(),
                    'learner_policy': self.dual_agent.learner_policy.state_dict(),
                    'stable_optimizer': self.dual_agent.stable_optimizer.state_dict(),
                    'learner_optimizer': self.dual_agent.learner_optimizer.state_dict()
                }, filepath.replace('.json', '_models.pt'))
                logger.info("💾 Models saved using fallback method")
        elif TORCH_AVAILABLE:
            # Fallback for when version manager is not available
            torch.save({
                'stable_policy': self.dual_agent.stable_policy.state_dict(),
                'learner_policy': self.dual_agent.learner_policy.state_dict(),
                'stable_optimizer': self.dual_agent.stable_optimizer.state_dict(),
                'learner_optimizer': self.dual_agent.learner_optimizer.state_dict()
            }, filepath.replace('.json', '_models.pt'))
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"💾 System state saved to {filepath}")
    
    def load_system_state(self, filepath: str):
        """Load system state from file with compatibility checking."""
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore training stats
            self.training_stats.update(state.get('training_stats', {}))
            self.current_regime = MarketRegime(state.get('current_regime', 'unknown'))
            
            # Check for symbol compatibility
            saved_symbols = state.get('symbols', [])
            if saved_symbols != self.symbols:
                logger.info(f"🔄 Symbol list changed: {len(saved_symbols)} → {len(self.symbols)} symbols")
                logger.info(f"   Old: {saved_symbols}")
                logger.info(f"   New: {self.symbols}")
                
                # The version manager in DualAgentSystem already handled model loading
                # during initialization, so no additional action needed here
            
            # Legacy model loading (fallback)
            model_path = filepath.replace('.json', '_models.pt')
            if TORCH_AVAILABLE and Path(model_path).exists() and not VERSION_MANAGER_AVAILABLE:
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    # Only try loading if dimensions match
                    stable_policy_dict = checkpoint.get('stable_policy', {})
                    if self._check_model_compatibility(stable_policy_dict):
                        self.dual_agent.stable_policy.load_state_dict(stable_policy_dict)
                        self.dual_agent.learner_policy.load_state_dict(checkpoint['learner_policy'])
                        self.dual_agent.stable_optimizer.load_state_dict(checkpoint['stable_optimizer'])
                        self.dual_agent.learner_optimizer.load_state_dict(checkpoint['learner_optimizer'])
                        logger.info("✅ Successfully loaded legacy model states")
                    else:
                        logger.warning("⚠️ Legacy model incompatible - using fresh models")
                except Exception as model_error:
                    logger.warning(f"⚠️ Legacy model loading failed: {model_error}")
            
            logger.info(f"✅ System state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load system state: {e}")
    
    def _check_model_compatibility(self, model_state_dict: Dict) -> bool:
        """Check if saved model is compatible with current architecture."""
        try:
            if not model_state_dict:
                return False
                
            # Check input layer compatibility (includes regime embedding)
            input_weight = model_state_dict.get('network.0.weight')
            if input_weight is not None:
                expected_shape = (256, self.state_dim + 16)  # +16 for regime embedding
                if input_weight.shape != expected_shape:
                    logger.info(f"Input layer mismatch: {input_weight.shape} != {expected_shape}")
                    return False
            
            # Check output layer compatibility (action_dim * 2 for mean and log_std)
            output_weight = model_state_dict.get('network.6.weight')
            if output_weight is not None:
                expected_shape = (self.action_dim * 2, 256)
                if output_weight.shape != expected_shape:
                    logger.info(f"Output layer mismatch: {output_weight.shape} != {expected_shape}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Compatibility check failed: {e}")
            return False

# Factory function
def create_online_rl_system(symbols: List[str], alpaca_client=None, **kwargs) -> OnlineRLTradingSystem:
    """Create online RL trading system."""
    
    config = OnlineLearningConfig(**kwargs)
    device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
    
    logger.info(f"🚀 Creating online RL system for {len(symbols)} symbols on {device}")
    
    return OnlineRLTradingSystem(symbols, config, device, alpaca_client=alpaca_client)


if __name__ == "__main__":
    # Test the online RL system
    import asyncio
    
    async def test_online_system():
        # Create system
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        system = create_online_rl_system(
            symbols=symbols,
            batch_update_frequency=timedelta(minutes=10),  # Fast for testing
            stable_update_frequency=timedelta(hours=1)
        )
        
        # Simulate market steps
        print("🧪 Testing online RL system...")
        
        for step in range(50):
            # Generate dummy market data
            market_state = np.random.randn(system.state_dim)
            market_data = {
                'volatility': np.random.uniform(0.01, 0.06),
                'trend': np.random.uniform(-0.02, 0.02),
                'portfolio_value': 100000 + step * 100,
                'data_completeness': 1.0
            }
            
            # Process step
            action, info = await system.process_market_step(
                market_state, 
                market_data,
                previous_action=np.random.randn(system.action_dim) if step > 0 else None,
                previous_reward=np.random.uniform(-0.01, 0.01)
            )
            
            if step % 10 == 0:
                stats = system.get_training_stats()
                print(f"Step {step}: Agent={info['agent']}, Regime={info['regime']}, "
                      f"Updates={stats['total_updates']}, Buffer={stats['buffer_size']}")
        
        print("✅ Test completed successfully!")
        
        # Print final stats
        final_stats = system.get_training_stats()
        print(f"Final stats: {final_stats}")
    
    # Run test
    asyncio.run(test_online_system())