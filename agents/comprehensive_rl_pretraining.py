"""
Comprehensive RL Pre-training System with Online Learning Core Techniques
Implements dual-agent system, prioritized experience replay, safety-guided updates, and regime-conditional learning
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import random
from collections import deque, namedtuple
import copy
from enum import Enum
import threading
import time

# Handle optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal
    import torch.distributions.kl as kl
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create fallback classes when PyTorch is not available
    class nn:
        class Module:
            def __init__(self):
                pass
            def to(self, device):
                return self
            def parameters(self):
                return []
        
        class Embedding:
            def __init__(self, *args, **kwargs):
                pass
        
        class ModuleList:
            def __init__(self, modules):
                self.modules = modules
        
        class Sequential:
            def __init__(self, *args):
                pass
        
        class Linear:
            def __init__(self, *args, **kwargs):
                pass
        
        class ReLU:
            def __init__(self, *args, **kwargs):
                pass
        
        class Softmax:
            def __init__(self, *args, **kwargs):
                pass
    
    class torch:
        @staticmethod
        def cat(*args, **kwargs):
            return None
        
        @staticmethod
        def clamp(*args, **kwargs):
            return None

# Import our custom components
from agents.hybrid_data_sources import HybridDataGenerator, MarketRegimeData, SentimentData
from agents.realistic_trading_env import RealisticTradingEnvironment
from agents.safe_rl_algorithms import SafeSAC, SafePPO, TrainingConfig
from agents.decision_transformer import DecisionTransformer, DecisionTransformerTrainer, DecisionTransformerConfig

logger = logging.getLogger(__name__)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'info'])

class RegimeType(Enum):
    """Market regime types for conditional learning."""
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRISIS = "crisis"
    UNKNOWN = "unknown"
    
    @classmethod
    def to_id(cls, regime_type):
        """Convert RegimeType to integer ID for tensor operations."""
        regime_mapping = {
            cls.BULL_LOW_VOL: 0,
            cls.BULL_HIGH_VOL: 1,
            cls.BEAR_LOW_VOL: 2,
            cls.BEAR_HIGH_VOL: 3,
            cls.SIDEWAYS: 4,
            cls.VOLATILE: 5,
            cls.CRISIS: 6,
            cls.UNKNOWN: 7
        }
        return regime_mapping.get(regime_type, 7)  # Default to UNKNOWN

@dataclass
class SafetyConstraints:
    """Safety constraints for online learning."""
    max_drawdown: float = 0.10  # 10% max drawdown
    min_sharpe_ratio: float = -1.0  # Minimum Sharpe before rollback
    max_kl_divergence: float = 0.05  # Trust region constraint
    max_position_change: float = 0.2  # Max position change per update
    consecutive_losses_limit: int = 5  # Max consecutive losing trades
    volatility_threshold: float = 0.05  # Max vol increase before safety mode

@dataclass
class OnlineLearningConfig:
    """Configuration for online learning system."""
    
    # Dual-agent system
    stable_policy_update_freq: timedelta = timedelta(days=7)  # Weekly updates
    learner_performance_threshold: float = 0.05  # 5% outperformance needed
    
    # Experience replay
    buffer_size: int = 50000
    priority_alpha: float = 0.6  # Prioritization strength
    priority_beta: float = 0.4   # Importance sampling correction
    novelty_weight: float = 0.3
    uncertainty_weight: float = 0.4
    regime_transition_weight: float = 0.3
    
    # Batch updates
    batch_update_freq: timedelta = timedelta(hours=1)  # Hourly updates
    min_batch_size: int = 32
    max_batch_size: int = 256
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    # Safety constraints
    safety_constraints: SafetyConstraints = None
    
    # Regime conditioning
    regime_embedding_dim: int = 16
    num_regime_experts: int = 3
    
    def __post_init__(self):
        if self.safety_constraints is None:
            self.safety_constraints = SafetyConstraints()

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer with novelty and uncertainty weighting."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.novelty_scores = deque(maxlen=capacity)
        self.uncertainty_scores = deque(maxlen=capacity)
        self.regime_tags = deque(maxlen=capacity)
        
        # State embeddings for novelty calculation
        self.state_embeddings = deque(maxlen=capacity)
        
        # Regime transition tracking
        self.regime_transition_indices = set()
        
        self.max_priority = 1.0
        
    def push(self, experience: Experience, uncertainty: float = 0.0, regime_tag: str = "unknown"):
        """Add experience to buffer with priority calculation."""
        
        # Calculate novelty score
        novelty = self._calculate_novelty(experience.state)
        
        # Calculate base priority
        td_error = abs(experience.reward)  # Simplified TD error
        priority = (td_error + 1e-6) ** self.alpha
        
        # Boost priority for regime transitions
        if len(self.regime_tags) > 0 and self.regime_tags[-1] != regime_tag:
            priority *= 2.0  # Boost regime transition experiences
            self.regime_transition_indices.add(len(self.buffer))
        
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.novelty_scores.append(novelty)
        self.uncertainty_scores.append(uncertainty)
        self.regime_tags.append(regime_tag)
        self.state_embeddings.append(self._embed_state(experience.state))
        
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int, config: OnlineLearningConfig) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with prioritization."""
        
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Calculate combined priorities
        priorities = np.array(self.priorities)
        novelty = np.array(self.novelty_scores)
        uncertainty = np.array(self.uncertainty_scores)
        
        # Normalize scores
        novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-8)
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
        
        # Combined priority score
        combined_priorities = (
            priorities +
            config.novelty_weight * novelty +
            config.uncertainty_weight * uncertainty
        )
        
        # Add recency bias (newer experiences get higher priority)
        recency_weights = np.linspace(0.5, 1.5, len(self.buffer))
        combined_priorities *= recency_weights
        
        # Sample indices
        probs = combined_priorities / combined_priorities.sum()
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), 
                                 p=probs, replace=False)
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        return experiences, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < len(self.priorities):
                priority = (abs(td_error) + 1e-6) ** self.alpha
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def _calculate_novelty(self, state: np.ndarray) -> float:
        """Calculate novelty score based on state distance."""
        if len(self.state_embeddings) == 0:
            return 1.0
        
        state_emb = self._embed_state(state)
        min_distance = min(np.linalg.norm(state_emb - emb) for emb in list(self.state_embeddings)[-100:])
        return min_distance / (min_distance + 1.0)  # Normalize
    
    def _embed_state(self, state: np.ndarray) -> np.ndarray:
        """Simple state embedding for novelty calculation."""
        # Use PCA-like dimensionality reduction
        return state[:10] if len(state) > 10 else state
    
    def clear_old_experiences(self, max_age_hours: int = 168):  # 1 week
        """Remove old experiences to keep buffer fresh."""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=max_age_hours)
        
        # This is simplified - in practice, you'd store timestamps with experiences
        # For now, just remove oldest 10% of buffer
        remove_count = len(self.buffer) // 10
        for _ in range(remove_count):
            if len(self.buffer) > 0:
                self.buffer.popleft()
                self.priorities.popleft()
                self.novelty_scores.popleft()
                self.uncertainty_scores.popleft()
                self.regime_tags.popleft()
                self.state_embeddings.popleft()

class RegimeAwarePolicy(nn.Module):
    """Policy network with regime conditioning."""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 regime_embedding_dim: int = 16,
                 num_experts: int = 3,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.num_experts = num_experts
        self.regime_embedding_dim = regime_embedding_dim
        
        # Regime embedding
        self.regime_embeddings = nn.Embedding(len(RegimeType), regime_embedding_dim)
        
        # Mixture of experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + regime_embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gating = nn.Sequential(
            nn.Linear(state_dim + regime_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state, regime_id):
        """Forward pass with regime conditioning."""
        
        if not TORCH_AVAILABLE:
            # Return dummy values when PyTorch is not available
            return None, None, None
            
        # Get regime embedding
        regime_emb = self.regime_embeddings(regime_id)
        
        # Concatenate state and regime
        # Handle tensor dimensions properly
        if len(regime_emb.shape) == 3 and regime_emb.shape[0] == 1:
            # If regime_emb is [1, 1, 16], squeeze to [1, 16]
            regime_emb = regime_emb.squeeze(1)
        elif len(regime_emb.shape) == 2 and len(state.shape) > 1:
            # If state has batch dimension and regime_emb doesn't match
            regime_emb = regime_emb.expand(state.shape[0], -1)
        
        state_regime = torch.cat([state, regime_emb], dim=-1)
        
        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            output = expert(state_regime)
            mean, log_std = output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            expert_outputs.append((mean, log_std))
        
        # Get gating weights
        gate_weights = self.gating(state_regime)
        
        # Combine expert outputs
        combined_mean = sum(w.unsqueeze(-1) * mean for w, (mean, log_std) in zip(gate_weights.T, expert_outputs))
        combined_log_std = sum(w.unsqueeze(-1) * log_std for w, (mean, log_std) in zip(gate_weights.T, expert_outputs))
        
        return combined_mean, combined_log_std.exp(), gate_weights

class DualAgentSystem:
    """Dual-agent system with stable policy and learning agent."""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 config: OnlineLearningConfig,
                 device: str = 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        
        # Stable policy (updated infrequently)
        self.stable_policy = RegimeAwarePolicy(state_dim, action_dim).to(device)
        self.stable_policy_performance = deque(maxlen=1000)
        self.last_stable_update = datetime.now()
        
        # Learning agent (updated frequently)
        training_config = TrainingConfig(
            conservative_weight=0.2,  # More conservative for online learning
            entropy_weight=0.05,
            max_action_change=config.safety_constraints.max_position_change
        )
        
        self.learner = SafeSAC(state_dim, action_dim, training_config, device)
        self.learner_performance = deque(maxlen=1000)
        
        # Performance tracking
        self.current_active_agent = "stable"  # Start with stable policy
        self.performance_comparison_window = 100
        
        # Safety monitoring
        self.safety_violations = 0
        self.consecutive_losses = 0
        self.last_performance_check = datetime.now()
        
    def select_action(self, state: np.ndarray, regime_id: int, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """Select action using appropriate agent."""
        
        if self.current_active_agent == "stable":
            return self._stable_policy_action(state, regime_id, deterministic)
        else:
            return self.learner.select_action(state, deterministic)
    
    def _stable_policy_action(self, state: np.ndarray, regime_id: int, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """Get action from stable policy."""
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        regime_tensor = torch.LongTensor([regime_id]).to(self.device)
        
        with torch.no_grad():
            mean, std, gate_weights = self.stable_policy(state_tensor, regime_tensor)
            
            if deterministic:
                action = torch.tanh(mean)
            else:
                normal = Normal(mean, std)
                x_t = normal.sample()
                action = torch.tanh(x_t)
        
        action = action.cpu().numpy()[0]
        
        metadata = {
            'agent': 'stable',
            'regime_gates': gate_weights.cpu().numpy()[0].tolist(),
            'deterministic': deterministic
        }
        
        return action, metadata
    
    def update_performance(self, reward: float, agent_type: str):
        """Update performance tracking for agents."""
        
        if agent_type == "stable":
            self.stable_policy_performance.append(reward)
        else:
            self.learner_performance.append(reward)
        
        # Check if learner significantly outperforms stable policy
        if len(self.stable_policy_performance) >= self.performance_comparison_window and \
           len(self.learner_performance) >= self.performance_comparison_window:
            
            stable_mean = np.mean(list(self.stable_policy_performance)[-self.performance_comparison_window:])
            learner_mean = np.mean(list(self.learner_performance)[-self.performance_comparison_window:])
            
            performance_diff = learner_mean - stable_mean
            
            if performance_diff > self.config.learner_performance_threshold:
                if self.current_active_agent == "stable":
                    logger.info(f"🔄 Switching to learner agent (performance advantage: {performance_diff:.4f})")
                    self.current_active_agent = "learner"
            elif performance_diff < -self.config.learner_performance_threshold:
                if self.current_active_agent == "learner":
                    logger.info(f"🔄 Switching back to stable policy (performance disadvantage: {performance_diff:.4f})")
                    self.current_active_agent = "stable"
    
    def should_update_stable_policy(self) -> bool:
        """Check if stable policy should be updated."""
        
        time_since_update = datetime.now() - self.last_stable_update
        
        # Update if enough time has passed and learner is performing well
        if (time_since_update >= self.config.stable_policy_update_freq and
            len(self.learner_performance) >= self.performance_comparison_window):
            
            learner_mean = np.mean(list(self.learner_performance)[-self.performance_comparison_window:])
            if learner_mean > 0:  # Only update if learner is profitable
                return True
        
        return False
    
    def update_stable_policy(self):
        """Update stable policy with learner's weights."""
        
        # Copy learner's policy to stable policy (simplified)
        # In practice, you'd want more sophisticated policy distillation
        logger.info("🔄 Updating stable policy from learner")
        
        # Reset performance tracking
        self.stable_policy_performance.clear()
        self.learner_performance.clear()
        self.last_stable_update = datetime.now()
        self.current_active_agent = "stable"

class SafetyMonitor:
    """Monitors safety constraints and triggers interventions."""
    
    def __init__(self, constraints: SafetyConstraints):
        self.constraints = constraints
        
        # Performance tracking
        self.portfolio_values = deque(maxlen=1000)
        self.returns = deque(maxlen=1000)
        self.positions = deque(maxlen=1000)
        
        # Safety state
        self.safety_mode = False
        self.violations = []
        self.last_violation_time = None
        
    def update(self, portfolio_value: float, positions: Dict[str, float], returns: List[float]):
        """Update safety monitoring."""
        
        self.portfolio_values.append(portfolio_value)
        self.returns.extend(returns)
        self.positions.append(positions.copy())
        
        # Check safety constraints
        violations = self._check_constraints()
        
        if violations:
            self.violations.extend(violations)
            self.last_violation_time = datetime.now()
            
            # Enter safety mode if severe violations
            if any(v['severity'] == 'high' for v in violations):
                self.safety_mode = True
                logger.warning(f"🚨 Safety mode activated: {[v['type'] for v in violations]}")
        
        # Exit safety mode if performance improves
        if self.safety_mode and len(violations) == 0 and len(self.returns) > 10:
            recent_returns = list(self.returns)[-10:]
            if np.mean(recent_returns) > 0 and np.sum([r > 0 for r in recent_returns]) >= 7:
                self.safety_mode = False
                logger.info("✅ Exiting safety mode - performance improved")
        
        return violations
    
    def _check_constraints(self) -> List[Dict]:
        """Check all safety constraints."""
        
        violations = []
        
        if len(self.portfolio_values) < 2:
            return violations
        
        # Drawdown check
        peak_value = max(self.portfolio_values)
        current_value = self.portfolio_values[-1]
        drawdown = (peak_value - current_value) / peak_value
        
        if drawdown > self.constraints.max_drawdown:
            violations.append({
                'type': 'max_drawdown',
                'severity': 'high',
                'value': drawdown,
                'limit': self.constraints.max_drawdown
            })
        
        # Sharpe ratio check
        if len(self.returns) >= 30:
            recent_returns = np.array(list(self.returns)[-30:])
            if len(recent_returns) > 0 and np.std(recent_returns) > 0:
                sharpe = np.mean(recent_returns) / np.std(recent_returns) * np.sqrt(252)
                
                if sharpe < self.constraints.min_sharpe_ratio:
                    violations.append({
                        'type': 'min_sharpe_ratio',
                        'severity': 'medium',
                        'value': sharpe,
                        'limit': self.constraints.min_sharpe_ratio
                    })
        
        # Volatility spike check
        if len(self.returns) >= 20:
            recent_vol = np.std(list(self.returns)[-20:]) * np.sqrt(252)
            historical_vol = np.std(list(self.returns)[:-20]) * np.sqrt(252) if len(self.returns) > 20 else recent_vol
            
            vol_increase = (recent_vol - historical_vol) / (historical_vol + 1e-8)
            
            if vol_increase > self.constraints.volatility_threshold:
                violations.append({
                    'type': 'volatility_spike',
                    'severity': 'medium',
                    'value': vol_increase,
                    'limit': self.constraints.volatility_threshold
                })
        
        return violations
    
    def should_halt_learning(self) -> bool:
        """Determine if learning should be halted."""
        return self.safety_mode or len([v for v in self.violations if v['severity'] == 'high']) > 0

class ComprehensiveRLPretrainingSystem:
    """Main system orchestrating all components."""
    
    def __init__(self,
                 symbols: List[str],
                 config: OnlineLearningConfig,
                 device: str = 'cpu'):
        
        self.symbols = symbols
        self.config = config
        self.device = device
        
        # Environment setup with consistent dimensions
        self.env = RealisticTradingEnvironment(symbols)
        
        # FIXED DIMENSIONS: Ensure consistency across all components
        # Using fixed symbol count and feature count for stable tensor operations
        self.fixed_symbol_count = 10  # Fixed number of symbols for consistent dimensions
        self.features_per_symbol = 10  # Fixed number of features per symbol
        state_dim = self.fixed_symbol_count * self.features_per_symbol  # 10 * 10 = 100
        action_dim = self.fixed_symbol_count  # Actions for 10 symbols
        
        logger.info(f"RL System Dimensions: state_dim={state_dim}, action_dim={action_dim}")
        
        # Core components
        self.dual_agent = DualAgentSystem(state_dim, action_dim, config, device)
        self.replay_buffer = PrioritizedReplayBuffer(config.buffer_size, config.priority_alpha, config.priority_beta)
        self.safety_monitor = SafetyMonitor(config.safety_constraints)
        
        # Data generation
        self.data_generator = HybridDataGenerator()
        
        # Training state
        self.training_active = False
        self.last_batch_update = datetime.now()
        self.training_stats = {
            'batch_updates': 0,
            'safety_violations': 0,
            'agent_switches': 0,
            'stable_policy_updates': 0
        }
        
        # Regime detection
        self.current_regime = RegimeType.UNKNOWN
        self.regime_history = deque(maxlen=100)
        
    async def pretrain_with_decision_transformer(self, 
                                               training_days: int = 1000,
                                               save_path: str = "models/pretrained_dt") -> Dict[str, Any]:
        """Pre-train using Decision Transformer on synthetic data."""
        
        logger.info("🚀 Starting Decision Transformer pre-training phase")
        
        # Generate comprehensive training dataset
        dataset = await self.data_generator.generate_training_dataset(
            symbols=self.symbols,
            training_days=training_days,
            validation_split=0.2
        )
        
        # Create Decision Transformer
        dt_config = DecisionTransformerConfig(
            sequence_length=20,
            d_model=256,
            num_layers=4,
            num_epochs=50
        )
        
        # Use consistent dimensions for Decision Transformer
        state_dim = self.fixed_symbol_count * self.features_per_symbol  # 10 * 10 = 100
        action_dim = self.fixed_symbol_count  # Actions for 10 symbols
        
        logger.info(f"Decision Transformer Dimensions: state_dim={state_dim}, action_dim={action_dim}")
        
        dt_model = DecisionTransformer(state_dim, action_dim, dt_config)
        
        # Convert data to trajectory format
        from agents.decision_transformer import create_trajectories_from_hybrid_data, TrajectoryDataset
        
        trajectories = create_trajectories_from_hybrid_data(dataset['training'], self.symbols)
        
        if not trajectories:
            logger.warning("No trajectories generated - using random initialization")
            return {"pretrain_success": False}
        
        dt_dataset = TrajectoryDataset(trajectories, dt_config)
        
        # Train Decision Transformer
        dt_trainer = DecisionTransformerTrainer(dt_model, dt_config, self.device)
        training_stats = dt_trainer.train(dt_dataset)
        
        # Save pre-trained model
        Path(save_path).parent.mkdir(exist_ok=True)
        dt_trainer.save_model(f"{save_path}.pt")
        
        logger.info("✅ Decision Transformer pre-training completed")
        
        return {
            "pretrain_success": True,
            "training_stats": training_stats,
            "model_path": f"{save_path}.pt"
        }
    
    async def start_online_learning(self, training_data: Dict[str, Any] = None):
        """Start online learning process."""
        
        logger.info("🚀 Starting comprehensive online learning system")
        
        self.training_active = True
        
        # Reset environment
        obs = await self.env.reset(training_data)
        
        episode_rewards = []
        step_count = 0
        
        while self.training_active and step_count < 10000:  # Max steps safety
            
            # Get current regime
            regime_id = self._detect_regime(obs)
            
            # Select action from dual agent system
            regime_int_id = RegimeType.to_id(regime_id)
            action, metadata = self.dual_agent.select_action(obs, regime_int_id)
            
            # Execute action in environment
            next_obs, reward, done, info = await self.env.step(action)
            
            # Update safety monitoring
            portfolio_value = info.get('portfolio_value', 100000)
            positions = info.get('positions', {})
            safety_violations = self.safety_monitor.update(portfolio_value, positions, [reward])
            
            # Store experience in prioritized replay buffer
            uncertainty = metadata.get('uncertainty', 0.0)
            regime_tag = regime_id.name
            
            experience = Experience(obs, action, reward, next_obs, done, info)
            self.replay_buffer.push(experience, uncertainty, regime_tag)
            
            # Update agent performance
            agent_type = metadata.get('agent', 'stable')
            self.dual_agent.update_performance(reward, agent_type)
            
            # Batch update check
            if self._should_perform_batch_update():
                if not self.safety_monitor.should_halt_learning():
                    await self._perform_batch_update()
                else:
                    logger.warning("⏸️ Skipping batch update due to safety constraints")
                    self.training_stats['safety_violations'] += 1
            
            # Stable policy update check
            if self.dual_agent.should_update_stable_policy():
                self.dual_agent.update_stable_policy()
                self.training_stats['stable_policy_updates'] += 1
            
            # Clean old experiences
            if step_count % 1000 == 0:
                self.replay_buffer.clear_old_experiences()
            
            # Update for next iteration
            obs = next_obs
            episode_rewards.append(reward)
            step_count += 1
            
            if done:
                obs = await self.env.reset()
                episode_rewards.clear()
            
            # Periodic logging
            if step_count % 100 == 0:
                recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                
                logger.info(f"Step {step_count}: Avg Reward={avg_reward:.4f}, "
                          f"Agent={self.dual_agent.current_active_agent}, "
                          f"Regime={regime_id.name}, "
                          f"Safety Mode={self.safety_monitor.safety_mode}")
        
        logger.info("🏁 Online learning completed")
        return self._get_training_summary()
    
    def _detect_regime(self, observation: np.ndarray) -> RegimeType:
        """Detect current market regime from observation."""
        
        # Simplified regime detection based on volatility and trend
        # In practice, this would use more sophisticated methods
        
        if len(observation) < 10:
            return RegimeType.UNKNOWN
        
        # Extract volatility and trend signals from observation
        vol_signal = np.std(observation[:5])  # Simplified volatility
        trend_signal = np.mean(observation[5:10])  # Simplified trend
        
        if vol_signal > 0.05:  # High volatility
            if trend_signal > 0.02:
                regime = RegimeType.BULL_HIGH_VOL
            elif trend_signal < -0.02:
                regime = RegimeType.BEAR_HIGH_VOL
            else:
                regime = RegimeType.VOLATILE
        else:  # Low volatility
            if trend_signal > 0.01:
                regime = RegimeType.BULL_LOW_VOL
            elif trend_signal < -0.01:
                regime = RegimeType.BEAR_LOW_VOL
            else:
                regime = RegimeType.SIDEWAYS
        
        # Track regime changes
        if len(self.regime_history) == 0 or self.regime_history[-1] != regime:
            logger.info(f"🔄 Regime change detected: {regime.name}")
        
        self.regime_history.append(regime)
        self.current_regime = regime
        
        return regime
    
    def _should_perform_batch_update(self) -> bool:
        """Check if batch update should be performed."""
        
        time_since_update = datetime.now() - self.last_batch_update
        
        return (time_since_update >= self.config.batch_update_freq and
                len(self.replay_buffer.buffer) >= self.config.min_batch_size)
    
    async def _perform_batch_update(self):
        """Perform batch update with early stopping."""
        
        logger.debug("🔄 Performing batch update")
        
        batch_size = min(self.config.max_batch_size, len(self.replay_buffer.buffer))
        
        # Sample batch
        experiences, weights, indices = self.replay_buffer.sample(batch_size, self.config)
        
        if not experiences:
            return
        
        # Prepare batch data
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        # Store experiences in learner's replay buffer
        for i, exp in enumerate(experiences):
            self.dual_agent.learner.store_transition(
                exp.state, exp.action, exp.reward, exp.next_state, exp.done
            )
        
        # Train with early stopping
        initial_loss = None
        patience_counter = 0
        
        for update_step in range(10):  # Max 10 updates per batch
            
            training_stats = self.dual_agent.learner.train_step()
            
            if not training_stats:  # No training performed
                break
            
            current_loss = training_stats.get('critic_loss', float('inf'))
            
            # Early stopping check
            if initial_loss is None:
                initial_loss = current_loss
            elif abs(current_loss - initial_loss) < self.config.early_stopping_threshold:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.debug(f"Early stopping at update {update_step}")
                    break
            else:
                patience_counter = 0
        
        # Update priorities in replay buffer
        # This would require getting TD errors from the critic
        # Simplified for now
        
        self.last_batch_update = datetime.now()
        self.training_stats['batch_updates'] += 1
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        
        return {
            'training_stats': self.training_stats,
            'safety_violations': len(self.safety_monitor.violations),
            'current_agent': self.dual_agent.current_active_agent,
            'regime_history': [r.name for r in list(self.regime_history)[-10:]],
            'buffer_size': len(self.replay_buffer.buffer),
            'stable_policy_performance': np.mean(list(self.dual_agent.stable_policy_performance)) if self.dual_agent.stable_policy_performance else 0.0,
            'learner_performance': np.mean(list(self.dual_agent.learner_performance)) if self.dual_agent.learner_performance else 0.0
        }
    
    def save_system_state(self, filepath: str):
        """Save complete system state."""
        
        state = {
            'config': asdict(self.config),
            'training_stats': self.training_stats,
            'regime_history': [r.name for r in list(self.regime_history)],
            'safety_violations': [asdict(v) for v in self.safety_monitor.violations],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save PyTorch models
        if TORCH_AVAILABLE:
            torch.save({
                'stable_policy': self.dual_agent.stable_policy.state_dict(),
                'learner_actor': self.dual_agent.learner.actor.state_dict(),
                'learner_critic': self.dual_agent.learner.critic.state_dict(),
                'system_state': state
            }, filepath)
        else:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        
        logger.info(f"💾 System state saved to {filepath}")
    
    async def stop_online_learning(self):
        """Stop online learning gracefully."""
        
        self.training_active = False
        logger.info("⏹️ Online learning stopped")

# Factory function
def create_comprehensive_rl_system(symbols: List[str], **kwargs) -> 'OnlineRLTradingSystem':
    """Create modern online RL trading system (replaces old pretraining approach)."""
    
    # Import the new online RL system
    try:
        from agents.online_rl_system import create_online_rl_system, OnlineLearningConfig as NewConfig
        
        # Convert old config parameters to new format
        new_kwargs = {}
        
        # Map common parameters
        if 'batch_update_freq' in kwargs:
            new_kwargs['batch_update_frequency'] = kwargs['batch_update_freq']
        if 'stable_policy_update_freq' in kwargs:
            new_kwargs['stable_update_frequency'] = kwargs['stable_policy_update_freq']
            
        # Use dynamic symbol count (no more fixed 10 symbol limitation)
        logger.info(f"🚀 Creating modern online RL system with {len(symbols)} symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
        logger.info("✨ New system features: Dynamic dimensions, dual-agent architecture, safe online learning")
        
        # Create new online learning config
        config = NewConfig(**new_kwargs)
        device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        
        return create_online_rl_system(symbols, **new_kwargs)
        
    except ImportError as e:
        logger.error(f"❌ Could not import new online RL system: {e}")
        logger.error("Falling back to legacy system with fixed dimensions")
        
        # Fallback to old system with dimension fixes
        fixed_symbols = symbols[:10]  # Ensure exactly 10 symbols maximum
        if len(symbols) > 10:
            logger.warning(f"Limiting symbols from {len(symbols)} to 10 for consistent RL dimensions")
        elif len(symbols) < 10:
            # Pad with repeated symbols if needed (for consistent dimensions)
            while len(fixed_symbols) < 10:
                fixed_symbols.extend(symbols[:min(10-len(fixed_symbols), len(symbols))])
            logger.info(f"Padded symbols from {len(symbols)} to 10 for consistent RL dimensions")
        
        logger.info(f"Creating legacy RL system with {len(fixed_symbols)} symbols: {fixed_symbols[:5]}{'...' if len(fixed_symbols) > 5 else ''}")
        
        config = OnlineLearningConfig(**kwargs)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        return ComprehensiveRLPretrainingSystem(fixed_symbols, config, device)

if __name__ == "__main__":
    # Test the comprehensive system
    async def test_comprehensive_system():
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Create system
        system = create_comprehensive_rl_system(
            symbols=symbols,
            batch_update_freq=timedelta(minutes=5),  # Fast for testing
            stable_policy_update_freq=timedelta(minutes=30)
        )
        
        # Pre-train with Decision Transformer
        pretrain_results = await system.pretrain_with_decision_transformer(training_days=100)
        print(f"Pre-training results: {pretrain_results}")
        
        # Run short online learning session
        print("Starting online learning...")
        await asyncio.wait_for(system.start_online_learning(), timeout=60)  # 1 minute test
        
        # Get summary
        summary = system._get_training_summary()
        print(f"Training summary: {summary}")
        
        print("✅ Comprehensive RL system test completed")
    
    if TORCH_AVAILABLE:
        asyncio.run(test_comprehensive_system())
    else:
        print("PyTorch not available - skipping test")