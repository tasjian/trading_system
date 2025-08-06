"""
FinRL-based Reinforcement Learning Agent with Short Selling and Limit Orders
Implements advanced RL algorithms for portfolio management with enhanced trading features.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import pickle
from pathlib import Path

from agents.finrl_enhanced_env import EnhancedFinRLTradingEnv, OrderSide, OrderType

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for RL agent training and execution."""
    
    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 3
    dropout_rate: float = 0.1
    
    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005   # Soft update rate
    
    # PPO-specific parameters
    ppo_epochs: int = 10
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    
    # Experience replay
    buffer_size: int = 100000
    min_buffer_size: int = 1000
    
    # Exploration
    exploration_noise: float = 0.1
    noise_decay: float = 0.995
    min_noise: float = 0.01
    
    # Training schedule
    update_frequency: int = 4
    target_update_frequency: int = 1000
    
    # Risk management
    max_position_size: float = 0.2
    risk_penalty_weight: float = 0.1
    
    # Model saving
    save_frequency: int = 1000
    model_path: str = "models/finrl_agent"


class ActorNetwork(nn.Module):
    """Actor network for continuous action space."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # Action mean and log std
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_log_std = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        features = self.feature_net(state)
        
        action_mean = torch.tanh(self.action_mean(features))  # [-1, 1] range
        action_log_std = torch.clamp(self.action_log_std(features), -20, 2)
        
        return action_mean, action_log_std
    
    def get_action(self, state, deterministic=False):
        """Get action from the policy."""
        action_mean, action_log_std = self.forward(state)
        
        if deterministic:
            return action_mean
        
        action_std = torch.exp(action_log_std)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        
        # Clamp actions to [-1, 1]
        action = torch.clamp(action, -1.0, 1.0)
        
        return action
    
    def get_log_prob(self, state, action):
        """Get log probability of action."""
        action_mean, action_log_std = self.forward(state)
        action_std = torch.exp(action_log_std)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return log_prob


class CriticNetwork(nn.Module):
    """Critic network for value estimation."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        features = self.feature_net(state)
        value = self.value_head(features)
        return value


class ExperienceBuffer:
    """Experience replay buffer for RL training."""
    
    def __init__(self, buffer_size: int = 100000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done, log_prob=None, value=None):
        """Add experience to buffer."""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        """Sample batch of experiences."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        return self._process_batch(batch)
    
    def get_recent(self, n: int):
        """Get n most recent experiences."""
        return self._process_batch(list(self.buffer)[-n:])
    
    def _process_batch(self, batch):
        """Process batch into tensors."""
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.FloatTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch])
        dones = torch.BoolTensor([exp['done'] for exp in batch])
        
        result = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
        
        # Add optional fields if available
        if batch[0].get('log_prob') is not None:
            result['log_probs'] = torch.FloatTensor([exp['log_prob'] for exp in batch])
        
        if batch[0].get('value') is not None:
            result['values'] = torch.FloatTensor([exp['value'] for exp in batch])
        
        return result
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class FinRLAgent:
    """
    Advanced FinRL-based RL agent with support for short selling and limit orders.
    
    Features:
    - PPO (Proximal Policy Optimization) algorithm
    - Continuous action space for position sizing
    - Risk-aware reward shaping
    - Short selling and limit order capabilities
    - Advanced portfolio management
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 config: RLConfig = None,
                 device: str = None):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or RLConfig()
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"FinRL Agent using device: {self.device}")
        
        # Initialize networks
        self.actor = ActorNetwork(
            state_dim, action_dim, 
            self.config.hidden_dim, self.config.num_layers
        ).to(self.device)
        
        self.critic = CriticNetwork(
            state_dim, 
            self.config.hidden_dim, self.config.num_layers
        ).to(self.device)
        
        # Target networks for stability (optional)
        self.target_critic = CriticNetwork(
            state_dim,
            self.config.hidden_dim, self.config.num_layers
        ).to(self.device)
        
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.config.learning_rate
        )
        
        # Experience buffer
        self.buffer = ExperienceBuffer(self.config.buffer_size)
        
        # Training state
        self.training_step = 0
        self.episode_count = 0
        self.current_noise = self.config.exploration_noise
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        
        # Model directory
        self.model_dir = Path(self.config.model_path)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ FinRL Agent initialized")
        logger.info(f"   State dim: {state_dim}, Action dim: {action_dim}")
        logger.info(f"   Networks: Actor/Critic with {self.config.num_layers} layers")
    
    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict]:
        """Get action from the policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if training:
                action = self.actor.get_action(state_tensor, deterministic=False)
                log_prob = self.actor.get_log_prob(state_tensor, action)
                value = self.critic(state_tensor)
                
                # Add exploration noise
                if self.current_noise > 0:
                    noise = torch.randn_like(action) * self.current_noise
                    action = torch.clamp(action + noise, -1.0, 1.0)
                
                action_np = action.cpu().numpy().flatten()
                
                metadata = {
                    'log_prob': log_prob.cpu().numpy().flatten()[0],
                    'value': value.cpu().numpy().flatten()[0],
                    'noise_level': self.current_noise
                }
            else:
                action = self.actor.get_action(state_tensor, deterministic=True)
                action_np = action.cpu().numpy().flatten()
                metadata = {'deterministic': True}
        
        return action_np, metadata
    
    def update_policy(self, batch_size: int = None) -> Dict[str, float]:
        """Update policy using PPO algorithm."""
        if len(self.buffer) < self.config.min_buffer_size:
            return {}
        
        batch_size = batch_size or self.config.batch_size
        
        # Get recent experiences for PPO
        recent_experiences = self.buffer.get_recent(min(len(self.buffer), batch_size * 4))
        
        # Calculate advantages and returns
        advantages, returns = self._compute_advantages_and_returns(recent_experiences)
        
        # PPO updates
        policy_losses = []
        value_losses = []
        entropies = []
        
        for epoch in range(self.config.ppo_epochs):
            # Sample mini-batches
            indices = torch.randperm(len(recent_experiences['states']))
            
            for start_idx in range(0, len(indices), batch_size):
                end_idx = min(start_idx + batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                states = recent_experiences['states'][batch_indices]
                actions = recent_experiences['actions'][batch_indices]
                old_log_probs = recent_experiences['log_probs'][batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Compute current policy output
                current_log_probs = self.actor.get_log_prob(states, actions)
                current_values = self.critic(states).squeeze()
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(current_log_probs.squeeze() - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                
                policy_loss1 = ratio * batch_advantages
                policy_loss2 = clipped_ratio * batch_advantages
                policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
                
                # Value loss
                value_loss = F.mse_loss(current_values, batch_returns)
                
                # Entropy bonus for exploration
                action_mean, action_log_std = self.actor(states)
                entropy = (action_log_std + 0.5 * np.log(2 * np.pi * np.e)).sum(dim=-1).mean()
                
                # Total loss
                total_loss = (policy_loss + 
                             self.config.value_loss_coeff * value_loss - 
                             self.config.entropy_coeff * entropy)
                
                # Update actor
                self.actor_optimizer.zero_grad()
                policy_loss_with_entropy = policy_loss - self.config.entropy_coeff * entropy
                policy_loss_with_entropy.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()
                
                # Track losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
        
        # Update target networks
        if self.training_step % self.config.target_update_frequency == 0:
            self._soft_update_target_networks()
        
        # Decay exploration noise
        self.current_noise = max(
            self.config.min_noise,
            self.current_noise * self.config.noise_decay
        )
        
        self.training_step += 1
        
        # Return training metrics
        metrics = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'noise_level': self.current_noise,
            'training_step': self.training_step
        }
        
        self.training_losses.append(metrics)
        
        return metrics
    
    def _compute_advantages_and_returns(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and returns using GAE (Generalized Advantage Estimation)."""
        states = batch['states']
        rewards = batch['rewards']
        dones = batch['dones']
        values = batch['values']
        
        # Compute next values
        with torch.no_grad():
            next_values = torch.zeros_like(values)
            next_values[:-1] = values[1:]
            next_values[dones] = 0.0
        
        # Compute TD errors and advantages
        td_errors = rewards + self.config.gamma * next_values - values
        
        advantages = torch.zeros_like(rewards)
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                advantage = 0
            advantage = td_errors[t] + self.config.gamma * 0.95 * advantage  # GAE with lambda=0.95
            advantages[t] = advantage
        
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _soft_update_target_networks(self):
        """Soft update target networks."""
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    def add_experience(self, state, action, reward, next_state, done, metadata=None):
        """Add experience to replay buffer."""
        log_prob = metadata.get('log_prob') if metadata else None
        value = metadata.get('value') if metadata else None
        
        self.buffer.add(state, action, reward, next_state, done, log_prob, value)
    
    def train_episode(self, env: EnhancedFinRLTradingEnv, max_steps: int = None) -> Dict:
        """Train for one episode."""
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_info = {}
        
        max_steps = max_steps or 1000
        
        for step in range(max_steps):
            # Get action
            action, metadata = self.get_action(state, training=True)
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Add to buffer
            self.add_experience(state, action, reward, next_state, done, metadata)
            
            # Update policy
            if len(self.buffer) >= self.config.min_buffer_size and step % self.config.update_frequency == 0:
                training_metrics = self.update_policy()
                if training_metrics:
                    episode_info.update(training_metrics)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # Update episode stats
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_count += 1
        
        episode_info.update({
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'episode_count': self.episode_count,
            'buffer_size': len(self.buffer)
        })
        
        # Save model periodically
        if self.episode_count % self.config.save_frequency == 0:
            self.save_model(f"episode_{self.episode_count}")
        
        return episode_info
    
    def evaluate_episode(self, env: EnhancedFinRLTradingEnv, max_steps: int = None) -> Dict:
        """Evaluate policy for one episode (no training)."""
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        trades = []
        
        max_steps = max_steps or 1000
        
        for step in range(max_steps):
            # Get deterministic action
            action, _ = self.get_action(state, training=False)
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            # Track significant trades
            if info.get('trades_executed', 0) > 0:
                trades.append({
                    'step': step,
                    'action': action.tolist(),
                    'portfolio_value': info['portfolio_value'],
                    'positions': info['positions']
                })
            
            if done:
                break
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'final_portfolio_value': info.get('portfolio_value', 0),
            'total_trades': len(trades),
            'trades': trades
        }
    
    def save_model(self, checkpoint_name: str = None):
        """Save model checkpoints."""
        checkpoint_name = checkpoint_name or f"checkpoint_{self.training_step}"
        
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'config': self.config.__dict__,
            'performance_stats': {
                'episode_rewards': self.episode_rewards[-100:],  # Last 100 episodes
                'episode_lengths': self.episode_lengths[-100:],
                'training_losses': self.training_losses[-100:]
            }
        }
        
        checkpoint_path = self.model_dir / f"{checkpoint_name}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Model saved: {checkpoint_path}")
    
    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            self.training_step = checkpoint.get('training_step', 0)
            self.episode_count = checkpoint.get('episode_count', 0)
            
            # Load performance stats if available
            perf_stats = checkpoint.get('performance_stats', {})
            self.episode_rewards = perf_stats.get('episode_rewards', [])
            self.episode_lengths = perf_stats.get('episode_lengths', [])
            self.training_losses = perf_stats.get('training_losses', [])
            
            logger.info(f"Model loaded: {checkpoint_path}")
            logger.info(f"  Training step: {self.training_step}")
            logger.info(f"  Episode count: {self.episode_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
        
        return {
            'total_episodes': len(self.episode_rewards),
            'total_training_steps': self.training_step,
            'average_reward': np.mean(recent_rewards),
            'best_reward': np.max(self.episode_rewards),
            'recent_average_reward': np.mean(recent_rewards[-10:]) if len(recent_rewards) >= 10 else np.mean(recent_rewards),
            'average_episode_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
            'current_noise_level': self.current_noise,
            'buffer_utilization': len(self.buffer) / self.config.buffer_size
        }
    
    def get_trading_strategy_summary(self, env: EnhancedFinRLTradingEnv) -> Dict:
        """Analyze the learned trading strategy."""
        if len(self.buffer) < 100:
            return {"message": "Not enough experience for analysis"}
        
        # Sample recent experiences
        recent_batch = self.buffer.get_recent(100)
        
        # Analyze action patterns
        actions = recent_batch['actions'].numpy()
        rewards = recent_batch['rewards'].numpy()
        
        # Calculate action statistics
        action_stats = {}
        for i, symbol in enumerate(env.symbols):
            symbol_actions = actions[:, i]
            action_stats[symbol] = {
                'mean_position': np.mean(symbol_actions),
                'position_std': np.std(symbol_actions),
                'long_frequency': np.mean(symbol_actions > 0.1),
                'short_frequency': np.mean(symbol_actions < -0.1),
                'hold_frequency': np.mean(np.abs(symbol_actions) <= 0.1)
            }
        
        return {
            'action_statistics': action_stats,
            'average_reward': np.mean(rewards),
            'reward_volatility': np.std(rewards),
            'profitable_trades': np.mean(rewards > 0),
            'analysis_sample_size': len(actions)
        }


# Utility functions
def create_finrl_agent(env: EnhancedFinRLTradingEnv, config: RLConfig = None) -> FinRLAgent:
    """Create FinRL agent for given environment."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    return FinRLAgent(state_dim, action_dim, config)


def train_finrl_agent(agent: FinRLAgent, 
                     env: EnhancedFinRLTradingEnv,
                     num_episodes: int = 1000,
                     eval_frequency: int = 100) -> Dict:
    """Train FinRL agent with evaluation."""
    
    training_history = []
    evaluation_history = []
    
    logger.info(f"🚀 Starting FinRL agent training for {num_episodes} episodes")
    
    for episode in range(num_episodes):
        # Training episode
        train_info = agent.train_episode(env)
        training_history.append(train_info)
        
        # Evaluation episode
        if episode % eval_frequency == 0:
            eval_info = agent.evaluate_episode(env)
            evaluation_history.append({
                'episode': episode,
                **eval_info
            })
            
            logger.info(f"Episode {episode}:")
            logger.info(f"  Train Reward: {train_info['episode_reward']:.4f}")
            logger.info(f"  Eval Reward: {eval_info['episode_reward']:.4f}")
            logger.info(f"  Portfolio Value: ${eval_info['final_portfolio_value']:,.2f}")
            logger.info(f"  Buffer Size: {train_info['buffer_size']}")
    
    # Final statistics
    performance_stats = agent.get_performance_stats()
    strategy_summary = agent.get_trading_strategy_summary(env)
    
    results = {
        'training_history': training_history,
        'evaluation_history': evaluation_history,
        'performance_stats': performance_stats,
        'strategy_summary': strategy_summary,
        'final_model_path': str(agent.model_dir / f"final_episode_{num_episodes}.pt")
    }
    
    # Save final model
    agent.save_model(f"final_episode_{num_episodes}")
    
    logger.info("✅ FinRL agent training completed")
    logger.info(f"   Average reward: {performance_stats.get('average_reward', 0):.4f}")
    logger.info(f"   Best reward: {performance_stats.get('best_reward', 0):.4f}")
    
    return results


if __name__ == "__main__":
    # Test the FinRL agent
    import yfinance as yf
    from agents.finrl_enhanced_env import create_enhanced_finrl_env
    
    def test_finrl_agent():
        # Create test environment
        symbols = ["AAPL", "MSFT"]
        env = create_enhanced_finrl_env(symbols)
        
        # Generate test data
        data_dict = {}
        for symbol in symbols:
            stock_data = yf.download(symbol, period="2mo", interval="1d")
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                data_dict[f"{symbol}_{col.lower()}"] = stock_data[col].values
        
        test_data = pd.DataFrame(data_dict)
        test_data.fillna(method='ffill', inplace=True)
        
        env.set_data(test_data)
        
        # Create agent with test configuration
        config = RLConfig(
            hidden_dim=128,
            num_layers=2,
            batch_size=32,
            buffer_size=10000,
            min_buffer_size=100
        )
        
        agent = create_finrl_agent(env, config)
        
        print("Testing FinRL Agent")
        print(f"State dim: {agent.state_dim}, Action dim: {agent.action_dim}")
        
        # Test single episode
        print("\nRunning test episode...")
        train_info = agent.train_episode(env, max_steps=50)
        print(f"Episode reward: {train_info['episode_reward']:.4f}")
        print(f"Episode length: {train_info['episode_length']}")
        
        # Test evaluation
        print("\nRunning evaluation episode...")
        eval_info = agent.evaluate_episode(env, max_steps=50)
        print(f"Eval reward: {eval_info['episode_reward']:.4f}")
        print(f"Final portfolio: ${eval_info['final_portfolio_value']:,.2f}")
        print(f"Total trades: {eval_info['total_trades']}")
        
        # Test save/load
        print("\nTesting save/load...")
        agent.save_model("test_checkpoint")
        
        # Create new agent and load
        agent2 = create_finrl_agent(env, config)
        success = agent2.load_model(agent.model_dir / "test_checkpoint.pt")
        print(f"Model load success: {success}")
        
        # Performance summary
        perf_stats = agent.get_performance_stats()
        print(f"\nPerformance stats: {perf_stats}")
        
        print("✅ FinRL Agent test completed")
    
    test_finrl_agent()