"""
Safe, Generalizable RL Algorithms for Trading
Implements PPO and SAC with conservative regularization, entropy constraints, and uncertainty estimation
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path
import random
from collections import deque
import copy

# Handle optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal, Categorical
    import torch.distributions as distributions
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for safe RL training."""
    
    # General parameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 0.005  # Soft update rate
    
    # Conservative regularization
    conservative_weight: float = 0.1
    entropy_target: float = -3.0  # Target entropy for SAC
    entropy_weight: float = 0.01  # PPO entropy bonus
    
    # Safety constraints
    max_action_change: float = 0.1  # Maximum change per step
    position_limit: float = 0.2  # Maximum position size
    drawdown_limit: float = 0.15  # Maximum drawdown before stopping
    
    # Uncertainty estimation
    ensemble_size: int = 3
    dropout_rate: float = 0.1
    uncertainty_threshold: float = 0.5  # Threshold for conservative actions
    
    # Training stability
    gradient_clip: float = 0.5
    target_update_freq: int = 2
    save_freq: int = 1000

class EnsembleNetwork(nn.Module):
    """Ensemble of networks for uncertainty estimation."""
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dim: int = 256,
                 ensemble_size: int = 3,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.ensemble_size = ensemble_size
        self.dropout_rate = dropout_rate
        
        # Create ensemble of networks
        self.networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), 
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(ensemble_size)
        ])
    
    def forward(self, x, return_uncertainty=False):
        """Forward pass through ensemble."""
        outputs = []
        
        for network in self.networks:
            outputs.append(network(x))
        
        outputs = torch.stack(outputs)  # [ensemble_size, batch_size, output_dim]
        
        if return_uncertainty:
            mean = outputs.mean(dim=0)
            std = outputs.std(dim=0)
            return mean, std
        else:
            return outputs.mean(dim=0)

class ConservativeCritic(nn.Module):
    """Conservative Q-learning critic with uncertainty estimation."""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 ensemble_size: int = 3,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.ensemble_size = ensemble_size
        
        # Twin Q-networks with ensemble for each
        self.q1_ensemble = EnsembleNetwork(
            state_dim + action_dim, 1, hidden_dim, ensemble_size, dropout_rate
        )
        self.q2_ensemble = EnsembleNetwork(
            state_dim + action_dim, 1, hidden_dim, ensemble_size, dropout_rate
        )
        
        # Conservative Q-learning parameters
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable conservatism
        
    def forward(self, state, action, return_uncertainty=False):
        """Forward pass through conservative critic."""
        
        sa = torch.cat([state, action], dim=-1)
        
        if return_uncertainty:
            q1_mean, q1_std = self.q1_ensemble(sa, return_uncertainty=True)
            q2_mean, q2_std = self.q2_ensemble(sa, return_uncertainty=True)
            
            # Conservative estimate uses lower bound
            q1_conservative = q1_mean - self.alpha * q1_std
            q2_conservative = q2_mean - self.alpha * q2_std
            
            uncertainty = (q1_std + q2_std) / 2
            
            return q1_conservative, q2_conservative, uncertainty
        else:
            q1 = self.q1_ensemble(sa)
            q2 = self.q2_ensemble(sa)
            return q1, q2

class SafeActor(nn.Module):
    """Safe actor network with action constraints."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.max_action = max_action
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Mean and log std for continuous actions
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Safety constraints
        self.min_log_std = -20
        self.max_log_std = 2
        
    def forward(self, state):
        """Forward pass through actor."""
        
        features = self.network(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clamp log std for numerical stability
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = log_std.exp()
        
        return mean, std
    
    def sample_action(self, state, deterministic=False, return_log_prob=False):
        """Sample action with safety constraints."""
        
        mean, std = self.forward(state)
        
        if deterministic:
            action = torch.tanh(mean) * self.max_action
            log_prob = None
        else:
            # Create normal distribution
            normal = Normal(mean, std)
            x_t = normal.rsample()  # Reparameterization trick
            
            # Apply tanh transformation
            action = torch.tanh(x_t) * self.max_action
            
            if return_log_prob:
                # Correct log prob for tanh transformation
                log_prob = normal.log_prob(x_t)
                log_prob -= torch.log(self.max_action * (1 - action.pow(2) / (self.max_action ** 2)) + 1e-6)
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            else:
                log_prob = None
        
        return action, log_prob

class SafeSAC:
    """Safe Soft Actor-Critic with conservative regularization."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: TrainingConfig,
                 device: str = 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        
        # Networks
        self.actor = SafeActor(state_dim, action_dim).to(device)
        self.critic = ConservativeCritic(state_dim, action_dim, 
                                       ensemble_size=config.ensemble_size).to(device)
        self.target_critic = ConservativeCritic(state_dim, action_dim,
                                              ensemble_size=config.ensemble_size).to(device)
        
        # Copy parameters to target
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # Automatic entropy tuning
        self.target_entropy = config.entropy_target
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.learning_rate)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=config.buffer_size)
        
        # Training tracking
        self.total_steps = 0
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'alpha_loss': [],
            'entropy': [],
            'uncertainty': []
        }
        
        logger.info(f"✅ Safe SAC initialized with {config.ensemble_size} ensemble critics")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """Select action with uncertainty estimation."""
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.actor.sample_action(state_tensor, deterministic, return_log_prob=True)
            
            # Get uncertainty estimate
            _, _, uncertainty = self.critic(state_tensor, action, return_uncertainty=True)
            uncertainty = uncertainty.mean().item()
        
        action = action.cpu().numpy()[0]
        
        # Apply conservative scaling based on uncertainty
        if uncertainty > self.config.uncertainty_threshold:
            action = action * 0.5  # Reduce action magnitude when uncertain
            logger.debug(f"High uncertainty ({uncertainty:.3f}) - conservative action scaling applied")
        
        metadata = {
            'uncertainty': uncertainty,
            'log_prob': log_prob.item() if log_prob is not None else None,
            'deterministic': deterministic
        }
        
        return action, metadata
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        transition = (state, action, reward, next_state, done)
        self.replay_buffer.append(transition)
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.config.batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.BoolTensor(done).unsqueeze(1).to(self.device)
        
        # Train critic
        critic_loss = self._train_critic(state, action, reward, next_state, done)
        
        # Train actor
        actor_loss, entropy = self._train_actor(state)
        
        # Train alpha (entropy coefficient)
        alpha_loss = self._train_alpha(entropy)
        
        # Update target networks
        if self.total_steps % self.config.target_update_freq == 0:
            self._soft_update_target()
        
        self.total_steps += 1
        
        # Update training stats
        stats = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss, 
            'alpha_loss': alpha_loss,
            'entropy': entropy,
            'alpha': self.log_alpha.exp().item()
        }
        
        for key, value in stats.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        return stats
    
    def _train_critic(self, state, action, reward, next_state, done):
        """Train critic with conservative regularization."""
        
        with torch.no_grad():
            # Sample next actions
            next_action, next_log_prob = self.actor.sample_action(next_state, return_log_prob=True)
            
            # Conservative target Q values
            target_q1, target_q2, _ = self.target_critic(next_state, next_action, return_uncertainty=True)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_q = reward + (1 - done.float()) * self.config.gamma * target_q
        
        # Current Q values
        current_q1, current_q2 = self.critic(state, action)
        
        # Conservative Q-learning loss
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        # Conservative regularization - penalize overestimation
        with torch.no_grad():
            # Sample random actions for conservative regularization
            random_actions = torch.FloatTensor(action.shape).uniform_(-1, 1).to(self.device)
        
        q1_random, q2_random = self.critic(state, random_actions)
        conservative_loss = (q1_random.mean() + q2_random.mean()) * self.config.conservative_weight
        
        critic_loss = q1_loss + q2_loss + conservative_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _train_actor(self, state):
        """Train actor with entropy regularization."""
        
        action, log_prob = self.actor.sample_action(state, return_log_prob=True)
        q1, q2, uncertainty = self.critic(state, action, return_uncertainty=True)
        min_q = torch.min(q1, q2)
        
        # Actor loss with uncertainty penalty
        uncertainty_penalty = uncertainty.mean() * 0.1  # Penalize high uncertainty actions
        actor_loss = -(min_q - self.log_alpha.exp() * log_prob - uncertainty_penalty).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
        self.actor_optimizer.step()
        
        return actor_loss.item(), log_prob.mean().item()
    
    def _train_alpha(self, entropy):
        """Train entropy coefficient."""
        
        alpha_loss = -(self.log_alpha * (entropy + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return alpha_loss.item()
    
    def _soft_update_target(self):
        """Soft update target networks."""
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

class SafePPO:
    """Safe Proximal Policy Optimization with conservative regularization."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: TrainingConfig,
                 device: str = 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        
        # Networks
        self.actor = SafeActor(state_dim, action_dim).to(device)
        self.critic = EnsembleNetwork(state_dim, 1, ensemble_size=config.ensemble_size).to(device)
        
        # Optimizers
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=config.learning_rate
        )
        
        # Experience buffer for PPO
        self.trajectory_buffer = []
        
        # Training tracking
        self.total_steps = 0
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'uncertainty': [],
            'kl_divergence': []
        }
        
        logger.info(f"✅ Safe PPO initialized with {config.ensemble_size} ensemble critics")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """Select action with uncertainty estimation."""
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.actor.sample_action(state_tensor, deterministic, return_log_prob=True)
            value, uncertainty = self.critic(state_tensor, return_uncertainty=True)
            uncertainty = uncertainty.mean().item()
        
        action = action.cpu().numpy()[0]
        
        # Conservative scaling for high uncertainty
        if uncertainty > self.config.uncertainty_threshold:
            action = action * 0.5
            logger.debug(f"High uncertainty ({uncertainty:.3f}) - conservative action scaling applied")
        
        metadata = {
            'uncertainty': uncertainty,
            'log_prob': log_prob.item() if log_prob is not None else None,
            'value': value.item(),
            'deterministic': deterministic
        }
        
        return action, metadata
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """Store transition for PPO training."""
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        }
        self.trajectory_buffer.append(transition)
    
    def train_step(self) -> Dict[str, float]:
        """Perform PPO training update."""
        
        if len(self.trajectory_buffer) < self.config.batch_size:
            return {}
        
        # Calculate advantages
        states, actions, rewards, values, log_probs, advantages, returns = self._process_trajectory()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        stats = self._ppo_update(states, actions, old_log_probs, advantages, returns)
        
        # Clear trajectory buffer
        self.trajectory_buffer.clear()
        self.total_steps += 1
        
        return stats
    
    def _process_trajectory(self):
        """Process trajectory to calculate advantages and returns."""
        
        states = [t['state'] for t in self.trajectory_buffer]
        actions = [t['action'] for t in self.trajectory_buffer]
        rewards = [t['reward'] for t in self.trajectory_buffer]
        values = [t['value'] for t in self.trajectory_buffer]
        log_probs = [t['log_prob'] for t in self.trajectory_buffer]
        dones = [t['done'] for t in self.trajectory_buffer]
        
        # Calculate returns and advantages using GAE
        advantages = []
        returns = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i]
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.config.gamma * next_value - values[i]
            gae = delta + self.config.gamma * 0.95 * gae  # GAE with λ=0.95
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return states, actions, rewards, values, log_probs, advantages, returns
    
    def _ppo_update(self, states, actions, old_log_probs, advantages, returns):
        """Perform PPO policy and value updates."""
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl_div = 0
        
        # Multiple epochs of updates
        for epoch in range(4):  # PPO typically uses 4 epochs
            
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy and value estimates
                _, new_log_probs = self.actor.sample_action(batch_states, return_log_prob=True)
                values, uncertainty = self.critic(batch_states, return_uncertainty=True)
                
                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with uncertainty penalty
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                uncertainty_penalty = uncertainty.mean() * 0.1
                value_loss += uncertainty_penalty
                
                # Entropy bonus for exploration
                entropy = -(new_log_probs * torch.exp(new_log_probs)).mean()
                entropy_loss = -self.config.entropy_weight * entropy
                
                # KL divergence for early stopping
                kl_div = (batch_old_log_probs - new_log_probs).mean()
                
                # Total loss
                total_loss = policy_loss + value_loss + entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.gradient_clip
                )
                self.optimizer.step()
                
                # Accumulate stats
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl_div += kl_div.item()
                
                # Early stopping on large KL divergence
                if kl_div > 0.01:
                    break
        
        # Average stats
        num_updates = min(4, len(states) // self.config.batch_size)
        stats = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'kl_divergence': total_kl_div / num_updates
        }
        
        # Update training stats
        for key, value in stats.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        return stats

# Factory functions
def create_safe_sac_agent(state_dim: int, action_dim: int, **kwargs) -> SafeSAC:
    """Create Safe SAC agent with default configuration."""
    config = TrainingConfig(**kwargs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SafeSAC(state_dim, action_dim, config, device)

def create_safe_ppo_agent(state_dim: int, action_dim: int, **kwargs) -> SafePPO:
    """Create Safe PPO agent with default configuration."""
    config = TrainingConfig(**kwargs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SafePPO(state_dim, action_dim, config, device)

if __name__ == "__main__":
    # Test safe RL algorithms
    def test_safe_rl():
        state_dim = 30  # Example state dimension
        action_dim = 5   # Example action dimension (5 stocks)
        
        # Test Safe SAC
        sac_agent = create_safe_sac_agent(state_dim, action_dim)
        
        # Test action selection
        test_state = np.random.randn(state_dim)
        action, metadata = sac_agent.select_action(test_state)
        
        print(f"SAC Action shape: {action.shape}")
        print(f"SAC Metadata: {metadata}")
        
        # Test Safe PPO
        ppo_agent = create_safe_ppo_agent(state_dim, action_dim)
        
        action, metadata = ppo_agent.select_action(test_state)
        print(f"PPO Action shape: {action.shape}")
        print(f"PPO Metadata: {metadata}")
        
        print("✅ Safe RL algorithms test completed")
    
    if TORCH_AVAILABLE:
        test_safe_rl()
    else:
        print("PyTorch not available - skipping test")