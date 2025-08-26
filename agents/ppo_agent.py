#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Agent for Trading
Optimized for curriculum retraining with the new reward system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    
    # Network architecture
    hidden_sizes: List[int] = None  # Will default to [256, 256, 128]
    activation: str = "relu"
    dropout_rate: float = 0.1
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training parameters
    batch_size: int = 64
    ppo_epochs: int = 4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    
    # Exploration
    exploration_noise: float = 0.1
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 256, 128]

class PPOActorCritic(nn.Module):
    """
    PPO Actor-Critic network for trading decisions.
    
    Outputs both policy (actor) and value function (critic) predictions.
    Designed for discrete action spaces (buy/sell/hold decisions).
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig):
        """
        Initialize PPO Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space (market features, positions, etc.)
            action_dim: Dimension of action space (number of discrete actions)
            config: PPO configuration
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Activation function
        if config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "tanh":
            self.activation = nn.Tanh()
        elif config.activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # Shared feature extractor
        self.shared_layers = self._build_shared_network()
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(config.hidden_sizes[-1], config.hidden_sizes[-1] // 2),
            self.activation,
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_sizes[-1] // 2, action_dim)
        )
        
        # Critic head (value function)
        self.critic_head = nn.Sequential(
            nn.Linear(config.hidden_sizes[-1], config.hidden_sizes[-1] // 2),
            self.activation,
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_sizes[-1] // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized PPO Actor-Critic: {state_dim} -> {config.hidden_sizes} -> {action_dim}")
    
    def _build_shared_network(self) -> nn.Module:
        """Build shared feature extraction network."""
        
        layers = []
        input_dim = self.state_dim
        
        for hidden_size in self.config.hidden_sizes:
            layers.extend([
                nn.Linear(input_dim, hidden_size),
                self.activation,
                nn.Dropout(self.config.dropout_rate)
            ])
            input_dim = hidden_size
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize network weights using Xavier/Kaiming initialization."""
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.activation == "relu":
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor-critic network.
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (action_logits, state_value)
        """
        
        # Shared feature extraction
        features = self.shared_layers(state)
        
        # Actor output (action logits)
        action_logits = self.actor_head(features)
        
        # Critic output (state value)
        state_value = self.critic_head(features)
        
        return action_logits, state_value
    
    def get_action_and_value(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value for a given state.
        
        Args:
            state: State tensor [batch_size, state_dim]
            deterministic: If True, use argmax instead of sampling
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        
        action_logits, value = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            # Deterministic action (for evaluation)
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1))).squeeze(-1)
        else:
            # Stochastic action (for training)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities, values, and entropy for given states and actions.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size]
            
        Returns:
            Tuple of (log_prob, value, entropy)
        """
        
        action_logits, value = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        return log_prob, value.squeeze(-1), entropy

class PPOAgent:
    """
    PPO agent for trading with curriculum retraining support.
    
    Implements Proximal Policy Optimization with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Adaptive learning rate and exploration
    - Curriculum-aware training phases
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig, device: str = "cpu"):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            config: PPO configuration
            device: Torch device ("cpu" or "cuda")
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(device)
        
        # Initialize actor-critic network
        self.actor_critic = PPOActorCritic(state_dim, action_dim, config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), 
            lr=config.learning_rate,
            eps=1e-5
        )
        
        # Initialize learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.99
        )
        
        # Training statistics
        self.training_step = 0
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        
        logger.info(f"Initialized PPO agent on device: {self.device}")
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Get action for given state.
        
        Args:
            state: State array
            deterministic: If True, use deterministic policy
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.actor_critic.get_action_and_value(
                state_tensor, deterministic=deterministic
            )
        
        return action.item(), log_prob.item(), value.item()
    
    def update(self, rollout_buffer: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Update PPO policy using rollout buffer.
        
        Args:
            rollout_buffer: Dictionary containing:
                - states: [num_steps, state_dim]
                - actions: [num_steps]
                - rewards: [num_steps]
                - values: [num_steps]
                - log_probs: [num_steps]
                - dones: [num_steps]
                
        Returns:
            Dictionary with training metrics
        """
        
        # Convert to tensors
        states = torch.FloatTensor(rollout_buffer['states']).to(self.device)
        actions = torch.LongTensor(rollout_buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_buffer['log_probs']).to(self.device)
        rewards = torch.FloatTensor(rollout_buffer['rewards']).to(self.device)
        values = torch.FloatTensor(rollout_buffer['values']).to(self.device)
        dones = torch.FloatTensor(rollout_buffer['dones']).to(self.device)
        
        # Calculate advantages and returns using GAE
        advantages, returns = self._calculate_gae(rewards, values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        # Multiple epochs over the data
        for epoch in range(self.config.ppo_epochs):
            # Create random mini-batches
            batch_indices = self._get_batch_indices(len(states))
            
            for batch_idx in batch_indices:
                # Get batch data
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Evaluate current policy
                new_log_probs, new_values, entropy = self.actor_critic.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Calculate policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio
                )
                policy_loss = -torch.min(
                    ratio * batch_advantages, 
                    clipped_ratio * batch_advantages
                ).mean()
                
                # Calculate value loss
                if self.config.value_coeff > 0:
                    value_loss = F.mse_loss(new_values, batch_returns)
                else:
                    value_loss = torch.tensor(0.0, device=self.device)
                
                # Calculate entropy bonus
                entropy_loss = entropy.mean()
                
                # Total loss
                total_loss = (
                    policy_loss + 
                    self.config.value_coeff * value_loss - 
                    self.config.entropy_coeff * entropy_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), 
                    self.config.max_grad_norm
                )
                
                self.optimizer.step()
                
                # Accumulate losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
        
        # Update learning rate
        self.lr_scheduler.step()
        
        # Update training statistics
        self.training_step += 1
        avg_policy_loss = total_policy_loss / (self.config.ppo_epochs * len(batch_indices))
        avg_value_loss = total_value_loss / (self.config.ppo_epochs * len(batch_indices))
        avg_entropy = total_entropy / (self.config.ppo_epochs * len(batch_indices))
        
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropies.append(avg_entropy)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'training_step': self.training_step
        }
    
    def _calculate_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                      dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Reward tensor [num_steps]
            values: Value tensor [num_steps]
            dones: Done tensor [num_steps]
            
        Returns:
            Tuple of (advantages, returns)
        """
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        advantage = 0.0
        next_value = 0.0
        
        # Calculate GAE backwards through time
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            
            # GAE calculation
            advantage = delta + self.config.gamma * self.config.lambda_gae * next_non_terminal * advantage
            advantages[t] = advantage
            
            # Returns for value function update
            returns[t] = advantage + values[t]
        
        return advantages, returns
    
    def _get_batch_indices(self, num_samples: int) -> List[torch.Tensor]:
        """Get random batch indices for mini-batch training."""
        
        indices = torch.randperm(num_samples, device=self.device)
        batch_indices = []
        
        for i in range(0, num_samples, self.config.batch_size):
            batch_end = min(i + self.config.batch_size, num_samples)
            batch_indices.append(indices[i:batch_end])
        
        return batch_indices
    
    def save_model(self, filepath: str):
        """Save model checkpoint."""
        
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, filepath)
        
        logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint."""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        
        logger.info(f"Model loaded from: {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        
        if len(self.policy_losses) == 0:
            return {}
        
        recent_window = min(100, len(self.policy_losses))
        
        return {
            'training_step': self.training_step,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'recent_policy_loss': np.mean(self.policy_losses[-recent_window:]),
            'recent_value_loss': np.mean(self.value_losses[-recent_window:]),
            'recent_entropy': np.mean(self.entropies[-recent_window:]),
            'total_updates': len(self.policy_losses)
        }
    
    def set_exploration_noise(self, noise_level: float):
        """Set exploration noise level for curriculum training."""
        self.config.exploration_noise = noise_level
        logger.info(f"Updated exploration noise to: {noise_level}")
    
    def adjust_learning_rate(self, new_lr: float):
        """Adjust learning rate for curriculum phases."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        logger.info(f"Updated learning rate to: {new_lr}")

# Factory function for easy agent creation
def create_ppo_agent(state_dim: int, action_dim: int, 
                    device: str = "cpu", **config_kwargs) -> PPOAgent:
    """
    Factory function to create PPO agent with custom configuration.
    
    Args:
        state_dim: State space dimension
        action_dim: Action space dimension  
        device: Torch device
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured PPO agent
    """
    
    config = PPOConfig(**config_kwargs)
    return PPOAgent(state_dim, action_dim, config, device)

# Example usage
if __name__ == "__main__":
    # Example configuration
    state_dim = 100  # Market features, portfolio state, etc.
    action_dim = 5   # Buy strong, buy weak, hold, sell weak, sell strong
    
    # Create agent
    agent = create_ppo_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=[512, 512, 256],
        learning_rate=1e-4,
        clip_ratio=0.2
    )
    
    print(f"Created PPO agent with {sum(p.numel() for p in agent.actor_critic.parameters())} parameters")