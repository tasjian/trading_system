#!/usr/bin/env python3
"""
Long/Short Regime-Aware Policy for RL Trading

Extends the existing RegimeAwarePolicy to support both long and short positions
with regime-conditional behavior and sophisticated constraint handling.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, Independent
    import torch.distributions.kl as kl
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch not available, using fallback implementations")

# Import existing regime definitions
from agents.comprehensive_rl_pretraining import RegimeType

logger = logging.getLogger(__name__)

class ActionSpace(Enum):
    """Action space configurations for different training phases."""
    LONG_ONLY = "long_only"          # [0, 1] for long-only training
    CONSTRAINED_SHORT = "constrained"  # [-0.5, 1] for initial short training
    FULL_RANGE = "full"              # [-1, 1] for full long/short training


@dataclass
class PolicyConfig:
    """Configuration for the long/short regime-aware policy."""
    # Architecture parameters
    state_dim: int = 100
    action_dim: int = 20
    hidden_dim: int = 256
    num_experts: int = 3
    regime_embedding_dim: int = 16
    
    # Action space configuration
    action_space: ActionSpace = ActionSpace.FULL_RANGE
    position_caps: Dict[str, float] = None  # Per-symbol position caps
    
    # Constraint parameters
    max_gross_exposure: float = 1.6
    max_net_exposure: float = 0.4
    min_net_exposure: float = -0.2
    max_position_size: float = 0.10
    
    # Regime-specific parameters
    regime_action_scaling: Dict[str, Dict[str, float]] = None
    regime_volatility_adjustment: bool = True
    
    # Regularization
    entropy_coeff: float = 0.01
    action_smoothing: float = 0.1
    constraint_penalty: float = 10.0
    
    def __post_init__(self):
        if self.position_caps is None:
            self.position_caps = {}
        
        if self.regime_action_scaling is None:
            # Default regime-based action scaling
            self.regime_action_scaling = {
                'bull': {'long_scale': 1.0, 'short_scale': 0.3},      # Prefer longs in bull
                'bear': {'long_scale': 0.5, 'short_scale': 1.0},     # Prefer shorts in bear  
                'sideways': {'long_scale': 0.8, 'short_scale': 0.6}, # Balanced in sideways
                'high_vol': {'long_scale': 0.6, 'short_scale': 0.8}  # Prefer shorts in high vol
            }


class RegimeAwarePolicy(nn.Module):
    """
    Actor-Critic Policy with regime awareness for trading actions.
    Integrates the simplified actor-critic structure with existing regime logic.
    """
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs):
        raise NotImplementedError

    def act(self, obs):
        x = self.actor(obs)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        action = torch.tanh(dist.rsample())  # squash to [-1, 1]
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob, self.critic(obs)

    def evaluate(self, obs, action):
        x = self.actor(obs)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(obs)
        return log_prob, entropy, value


class LongShortRegimeAwarePolicy(nn.Module):
    """
    Extended regime-aware policy supporting long and short positions.
    
    Key features:
    - Actions in [-1, +1] representing portfolio allocation fractions
    - Regime-conditional expert networks
    - Built-in constraint enforcement
    - Flexible action space for curriculum training
    - Stochastic policy with proper exploration
    - Integrated Actor-Critic architecture
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            logger.error("PyTorch required for LongShortRegimeAwarePolicy")
            return
        
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts
        
        # Regime embedding
        self.regime_embeddings = nn.Embedding(len(RegimeType), config.regime_embedding_dim)
        
        # Input dimension including regime embedding
        input_dim = self.state_dim + config.regime_embedding_dim
        
        # Enhanced Actor-Critic architecture with regime awareness
        self.actor_backbone = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

        # Critic network (enhanced from actor-critic pattern)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        
        # Mixture of experts for regime-conditional behavior
        self.experts = nn.ModuleList([
            self._create_expert_network(input_dim)
            for _ in range(self.num_experts)
        ])
        
        # Gating network to select experts
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Constraint enforcement layer
        self.constraint_layer = ConstraintEnforcementLayer(config)
        
        # Value function head (for actor-critic methods) - kept for backward compatibility
        self.value_head = self.critic
        
        logger.info(f"Initialized LongShortRegimeAwarePolicy with {self.num_experts} experts")
        logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        logger.info("Enhanced with Actor-Critic architecture integration")
    
    def _create_expert_network(self, input_dim: int) -> nn.Module:
        """Create an individual expert network."""
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            # Output mean and log_std for each action
            nn.Linear(self.hidden_dim // 2, self.action_dim * 2)
        )
    
    def forward(self, state: torch.Tensor, regime_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the policy.
        
        Args:
            state: State tensor [batch_size, state_dim]
            regime_id: Regime identifier [batch_size]
            
        Returns:
            action_mean: Mean action values [batch_size, action_dim]
            action_std: Action standard deviations [batch_size, action_dim]  
            gating_weights: Expert gating weights [batch_size, num_experts]
            value: State value estimate [batch_size, 1]
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available")
            return None, None, None, None
        
        batch_size = state.shape[0]
        
        # Get regime embeddings
        regime_emb = self.regime_embeddings(regime_id)  # [batch_size, regime_embedding_dim]
        
        # Concatenate state and regime embedding
        combined_input = torch.cat([state, regime_emb], dim=-1)  # [batch_size, input_dim]
        
        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(combined_input)  # [batch_size, action_dim * 2]
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, action_dim * 2]
        
        # Get gating weights
        gating_weights = self.gating_network(combined_input)  # [batch_size, num_experts]
        
        # Mix expert outputs using gating weights
        weighted_output = torch.sum(
            expert_outputs * gating_weights.unsqueeze(-1), dim=1
        )  # [batch_size, action_dim * 2]
        
        # Split into mean and log_std
        action_mean, action_log_std = torch.chunk(weighted_output, 2, dim=-1)
        
        # Apply activation functions
        action_mean = torch.tanh(action_mean)  # Initial range [-1, 1]
        action_log_std = torch.clamp(action_log_std, -20, 2)  # Clamp log_std for stability
        action_std = torch.exp(action_log_std)
        
        # Apply regime-specific scaling
        action_mean, action_std = self._apply_regime_scaling(
            action_mean, action_std, regime_id
        )
        
        # Apply action space constraints based on current training phase
        action_mean, action_std = self._apply_action_space_constraints(
            action_mean, action_std
        )
        
        # Get value estimate
        value = self.value_head(combined_input)
        
        return action_mean, action_std, gating_weights, value
    
    def act(self, obs, regime_id=None):
        """
        Actor-Critic style action sampling compatible with your integration.
        
        Args:
            obs: Observation tensor
            regime_id: Optional regime identifier (will use default if None)
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: Value estimate
        """
        if regime_id is None:
            # Default to neutral regime if not specified
            regime_id = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        
        # Protect input from NaN/inf
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Get regime embeddings
        regime_emb = self.regime_embeddings(regime_id)
        combined_input = torch.cat([obs, regime_emb], dim=-1)
        
        # Actor forward pass with NaN protection
        x = self.actor_backbone(combined_input)
        mu = self.mu_head(x)
        
        # Handle NaN/inf in mu
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)
        
        std = torch.exp(torch.clamp(self.log_std, -20, 2))  # Clamp for stability
        dist = Normal(mu, std)
        action = torch.tanh(dist.rsample())  # squash to [-1, 1]
        log_prob = dist.log_prob(action).sum(axis=-1)
        
        # Critic forward pass
        value = self.critic(combined_input)
        
        return action, log_prob, value
    
    def evaluate(self, obs, action, regime_id=None):
        """
        Evaluate actions for training (Actor-Critic style).
        
        Args:
            obs: Observation tensor
            action: Action tensor to evaluate
            regime_id: Optional regime identifier
            
        Returns:
            log_prob: Log probability of action
            entropy: Policy entropy
            value: Value estimate
        """
        if regime_id is None:
            regime_id = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        
        # Protect input from NaN/inf
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Get regime embeddings
        regime_emb = self.regime_embeddings(regime_id)
        combined_input = torch.cat([obs, regime_emb], dim=-1)
        
        # Actor evaluation with NaN protection
        x = self.actor_backbone(combined_input)
        mu = self.mu_head(x)
        
        # Handle NaN/inf in mu
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)
        
        std = torch.exp(torch.clamp(self.log_std, -20, 2))  # Clamp for stability
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        
        # Critic evaluation
        value = self.critic(combined_input)
        
        return log_prob, entropy, value
    
    def _apply_regime_scaling(self, action_mean: torch.Tensor, action_std: torch.Tensor, 
                             regime_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply regime-specific action scaling."""
        if not self.config.regime_volatility_adjustment:
            return action_mean, action_std
        
        batch_size = action_mean.shape[0]
        device = action_mean.device
        
        # Create scaling tensors
        long_scale = torch.ones(batch_size, device=device)
        short_scale = torch.ones(batch_size, device=device)
        
        # Apply regime-specific scaling
        for i, regime_val in enumerate(regime_id.cpu().numpy()):
            regime_name = list(RegimeType)[regime_val].name.lower()
            scaling = self.config.regime_action_scaling.get(regime_name, {})
            
            long_scale[i] = scaling.get('long_scale', 1.0)
            short_scale[i] = scaling.get('short_scale', 1.0)
        
        # Apply scaling to positive (long) and negative (short) actions separately
        positive_mask = action_mean > 0
        negative_mask = action_mean < 0
        
        scaled_mean = action_mean.clone()
        scaled_mean[positive_mask] *= long_scale.unsqueeze(-1).expand_as(action_mean)[positive_mask]
        scaled_mean[negative_mask] *= short_scale.unsqueeze(-1).expand_as(action_mean)[negative_mask]
        
        # Scale standard deviation proportionally (less aggressive)
        avg_scale = (long_scale + short_scale) / 2
        scaled_std = action_std * avg_scale.unsqueeze(-1).expand_as(action_std)
        
        return scaled_mean, scaled_std
    
    def _apply_action_space_constraints(self, action_mean: torch.Tensor, 
                                      action_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply action space constraints based on training phase."""
        
        if self.config.action_space == ActionSpace.LONG_ONLY:
            # Constrain to [0, 1] for long-only training
            action_mean = torch.sigmoid(action_mean)  # Map to [0, 1]
            # Reduce std near boundaries
            action_std = action_std * (action_mean * (1 - action_mean) + 0.1)
            
        elif self.config.action_space == ActionSpace.CONSTRAINED_SHORT:
            # Constrain to [-0.5, 1] for initial short training
            action_mean = torch.sigmoid(action_mean) * 1.5 - 0.5  # Map to [-0.5, 1]
            # Adjust std based on position in range
            range_factor = torch.clamp((action_mean + 0.5) / 1.5, 0.1, 1.0)
            action_std = action_std * range_factor
            
        elif self.config.action_space == ActionSpace.FULL_RANGE:
            # Full range [-1, 1] - action_mean already in this range from tanh
            pass
        
        return action_mean, action_std
    
    def get_action(self, state: torch.Tensor, regime_id: torch.Tensor, 
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample actions from the policy.
        
        Args:
            state: State tensor
            regime_id: Regime identifier
            deterministic: Whether to use deterministic (mean) actions
            
        Returns:
            actions: Sampled actions
            log_probs: Log probabilities of actions
            info: Additional information dictionary
        """
        action_mean, action_std, gating_weights, value = self.forward(state, regime_id)
        
        if deterministic:
            actions = action_mean
            # For deterministic actions, we still need log_probs for some algorithms
            dist = Normal(action_mean, action_std)
            log_probs = dist.log_prob(actions).sum(-1)
        else:
            # Create normal distribution and sample
            dist = Normal(action_mean, action_std)
            actions = dist.rsample()  # Reparameterized sampling
            log_probs = dist.log_prob(actions).sum(-1)
        
        # Apply hard constraints via projection
        constrained_actions = self.constraint_layer.apply_constraints(
            actions, state, regime_id
        )
        
        # Adjust log_probs if actions were modified by constraints
        if not torch.allclose(actions, constrained_actions):
            # Recompute log_probs for constrained actions
            adjusted_dist = Normal(action_mean, action_std * 0.5)  # Tighter distribution
            log_probs = adjusted_dist.log_prob(constrained_actions).sum(-1)
        
        info = {
            'action_mean': action_mean,
            'action_std': action_std,
            'gating_weights': gating_weights,
            'value': value,
            'entropy': dist.entropy().sum(-1),
            'regime_id': regime_id
        }
        
        return constrained_actions, log_probs, info
    
    def evaluate_actions(self, state: torch.Tensor, regime_id: torch.Tensor, 
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training (used in PPO updates).
        
        Args:
            state: State tensor
            regime_id: Regime identifier  
            actions: Actions to evaluate
            
        Returns:
            log_probs: Log probabilities of actions
            entropy: Policy entropy
            value: State value estimates
        """
        action_mean, action_std, gating_weights, value = self.forward(state, regime_id)
        
        # Create distribution and evaluate
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        return log_probs, entropy, value.squeeze(-1)
    
    def get_regime_importance(self, state: torch.Tensor, regime_id: torch.Tensor) -> torch.Tensor:
        """Get the importance weights of different experts for interpretability."""
        regime_emb = self.regime_embeddings(regime_id)
        combined_input = torch.cat([state, regime_emb], dim=-1)
        gating_weights = self.gating_network(combined_input)
        return gating_weights
    
    # =================== ONLINE LEARNING INTEGRATION ===================
    
    def update_online(self, experiences: List[Dict], optimizer: torch.optim.Optimizer, device: str = "cpu"):
        """
        Online learning update method for continuous improvement.
        
        Args:
            experiences: List of experience dictionaries with states, actions, rewards, etc.
            optimizer: PyTorch optimizer for gradient updates
            device: Device to run computations on
        """
        if not experiences:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}
        
        # Convert experiences to batched tensors
        states = torch.stack([torch.tensor(exp['state'], dtype=torch.float32) for exp in experiences]).to(device)
        actions = torch.stack([torch.tensor(exp['action'], dtype=torch.float32) for exp in experiences]).to(device)
        rewards = torch.tensor([exp['reward'] for exp in experiences], dtype=torch.float32).to(device)
        old_log_probs = torch.tensor([exp['log_prob'] for exp in experiences], dtype=torch.float32).to(device)
        
        # Get regime IDs (default to neutral if not provided)
        regime_ids = torch.tensor([exp.get('regime_id', 0) for exp in experiences], dtype=torch.long).to(device)
        
        # Forward pass through current policy
        log_probs, entropy, values = self.evaluate(states, actions, regime_ids)
        values = values.squeeze(-1)
        
        # Calculate advantages using simple TD error
        advantages = rewards - values.detach()
        returns = rewards
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO-style policy loss with clipping
        ratio = torch.exp(log_probs - old_log_probs)
        clip_range = 0.2  # Standard PPO clip range
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value function loss
        value_loss = torch.nn.functional.mse_loss(values, returns)
        
        # Entropy loss for exploration
        entropy_loss = -entropy.mean()
        
        # Combined loss
        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)  # Gradient clipping
        optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(), 
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item()
        }
    
    def get_online_compatible_state_dict(self) -> Dict[str, Any]:
        """
        Return state dict compatible with online learning system.
        
        Returns:
            Dictionary with stable_policy and learner_policy state dicts
        """
        current_state = self.state_dict()
        return {
            'stable_policy': current_state,
            'learner_policy': current_state.copy(),  # Start with same weights
            'metadata': {
                'model_architecture': 'LongShortRegimeAwarePolicy',
                'action_space': self.config.action_space.value,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'online_learning_enabled': True
            }
        }
    
    def load_from_online_state_dict(self, state_dict: Dict[str, Any]) -> bool:
        """
        Load model from online learning compatible state dict.
        
        Args:
            state_dict: State dict from online learning system
            
        Returns:
            True if loading was successful
        """
        try:
            # Try to load from stable_policy first, fallback to direct load
            if 'stable_policy' in state_dict:
                self.load_state_dict(state_dict['stable_policy'])
            else:
                self.load_state_dict(state_dict)
            return True
        except Exception as e:
            logger.warning(f"Failed to load from online state dict: {e}")
            return False
    
    def get_performance_metrics(self, recent_experiences: List[Dict]) -> Dict[str, float]:
        """
        Calculate performance metrics from recent experiences.
        
        Args:
            recent_experiences: List of recent experience dictionaries
            
        Returns:
            Dictionary of performance metrics
        """
        if not recent_experiences:
            return {}
        
        rewards = [exp['reward'] for exp in recent_experiences]
        
        metrics = {
            'mean_reward': float(torch.tensor(rewards).mean()),
            'reward_std': float(torch.tensor(rewards).std()),
            'total_episodes': len(recent_experiences),
            'sharpe_ratio': float(torch.tensor(rewards).mean() / (torch.tensor(rewards).std() + 1e-8))
        }
        
        # Calculate regime-specific performance if available
        regime_rewards = {}
        for exp in recent_experiences:
            regime_id = exp.get('regime_id', 0)
            if regime_id not in regime_rewards:
                regime_rewards[regime_id] = []
            regime_rewards[regime_id].append(exp['reward'])
        
        for regime_id, regime_reward_list in regime_rewards.items():
            metrics[f'regime_{regime_id}_mean_reward'] = float(torch.tensor(regime_reward_list).mean())
            metrics[f'regime_{regime_id}_count'] = len(regime_reward_list)
        
        return metrics
    
    def adapt_to_regime_distribution(self, regime_distribution: Dict[int, float]):
        """
        Adapt the policy based on observed regime distribution.
        
        Args:
            regime_distribution: Dictionary mapping regime IDs to their observed frequencies
        """
        # This could be used to adjust regime-specific parameters
        # For now, we'll just log the distribution
        total_observations = sum(regime_distribution.values())
        if total_observations > 0:
            for regime_id, count in regime_distribution.items():
                frequency = count / total_observations
                logger.debug(f"Regime {regime_id} frequency: {frequency:.3f}")
    
    def get_expert_utilization(self, states: torch.Tensor, regime_ids: torch.Tensor) -> Dict[str, float]:
        """
        Get expert utilization statistics for interpretability.
        
        Args:
            states: Batch of states
            regime_ids: Batch of regime IDs
            
        Returns:
            Dictionary with expert utilization statistics
        """
        with torch.no_grad():
            gating_weights = self.get_regime_importance(states, regime_ids)
            
            # Calculate average weights across batch
            avg_weights = gating_weights.mean(dim=0)
            
            return {
                f'expert_{i}_utilization': float(avg_weights[i])
                for i in range(self.num_experts)
            }


class ConstraintEnforcementLayer(nn.Module):
    """Layer to enforce portfolio constraints on actions."""
    
    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
    
    def apply_constraints(self, actions: torch.Tensor, state: torch.Tensor, 
                         regime_id: torch.Tensor) -> torch.Tensor:
        """Apply hard portfolio constraints to actions."""
        constrained_actions = actions.clone()
        
        for i in range(actions.shape[0]):  # Iterate over batch
            sample_actions = constrained_actions[i].detach().cpu().numpy()
            
            # Apply individual position size constraints
            sample_actions = np.clip(
                sample_actions,
                -self.config.max_position_size,
                self.config.max_position_size
            )
            
            # Apply gross exposure constraint
            gross_exposure = np.sum(np.abs(sample_actions))
            if gross_exposure > self.config.max_gross_exposure:
                scale_factor = self.config.max_gross_exposure / gross_exposure
                sample_actions *= scale_factor
            
            # Apply net exposure constraints
            net_exposure = np.sum(sample_actions)
            if net_exposure > self.config.max_net_exposure:
                excess = net_exposure - self.config.max_net_exposure
                # Reduce long positions proportionally
                long_positions = np.maximum(sample_actions, 0)
                if np.sum(long_positions) > 0:
                    reduction = excess / np.sum(long_positions)
                    sample_actions = np.where(
                        sample_actions > 0,
                        sample_actions * (1 - reduction),
                        sample_actions
                    )
            
            elif net_exposure < self.config.min_net_exposure:
                shortfall = self.config.min_net_exposure - net_exposure
                # Reduce short positions proportionally  
                short_positions = np.minimum(sample_actions, 0)
                if np.sum(np.abs(short_positions)) > 0:
                    reduction = abs(shortfall) / np.sum(np.abs(short_positions))
                    sample_actions = np.where(
                        sample_actions < 0,
                        sample_actions * (1 - reduction),
                        sample_actions
                    )
            
            constrained_actions[i] = torch.tensor(
                sample_actions, dtype=actions.dtype, device=actions.device
            )
        
        return constrained_actions


def create_long_short_policy(symbols: List[str], 
                           state_dim: int,
                           action_space: ActionSpace = ActionSpace.FULL_RANGE,
                           policy_type: str = "enhanced",
                           **kwargs) -> Union[LongShortRegimeAwarePolicy, RegimeAwarePolicy]:
    """
    Factory function to create a properly configured long/short policy.
    
    Args:
        symbols: List of trading symbols
        state_dim: Dimension of state observations
        action_space: Action space configuration for training phase
        policy_type: "enhanced" for LongShortRegimeAwarePolicy, "simple" for RegimeAwarePolicy
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured policy instance
    """
    action_dim = len(symbols)
    
    if policy_type == "simple":
        # Create simple RegimeAwarePolicy
        # Calculate input dimension including regime embedding
        input_dim = state_dim + 16  # Default regime embedding dim
        return RegimeAwarePolicy(obs_dim=input_dim, act_dim=action_dim, **kwargs)
    else:
        # Create enhanced LongShortRegimeAwarePolicy
        config = PolicyConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            **kwargs
        )
        return LongShortRegimeAwarePolicy(config)


# Utility functions for curriculum training
def transition_action_space(policy: LongShortRegimeAwarePolicy, 
                          new_action_space: ActionSpace) -> None:
    """Transition policy to a new action space during curriculum training."""
    logger.info(f"Transitioning policy action space to {new_action_space.value}")
    policy.config.action_space = new_action_space


def get_action_space_for_phase(phase: int) -> ActionSpace:
    """Get appropriate action space for training phase."""
    if phase == 0:
        return ActionSpace.LONG_ONLY
    elif phase == 1:
        return ActionSpace.CONSTRAINED_SHORT
    else:
        return ActionSpace.FULL_RANGE