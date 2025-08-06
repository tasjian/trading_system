"""
Enhanced FinRL Agent for Intelligent Limit Order Placement
Extends the base FinRL agent to handle 3D action spaces for sophisticated limit order strategies.

Action Space: [position_action, order_type_action, price_offset_action] per symbol
- position_action: [-1, 1] position sizing and direction
- order_type_action: [-1, 1] order type selection (market vs limit)
- price_offset_action: [-1, 1] price offset for limit orders
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from agents.finrl_rl_agent import FinRLAgent, RLConfig, ActorNetwork, CriticNetwork

logger = logging.getLogger(__name__)


class LimitOrderActorNetwork(nn.Module):
    """
    Enhanced Actor Network for 3D action space (position, order_type, price_offset).
    Uses separate heads for different action dimensions to better learn specialized policies.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim  # This is symbol_count * 3
        self.symbol_count = action_dim // 3
        
        # Shared feature extraction layers
        self.shared_layers = nn.ModuleList()
        input_dim = state_dim
        
        for i in range(num_layers - 1):
            self.shared_layers.append(nn.Linear(input_dim, hidden_dim))
            self.shared_layers.append(nn.LayerNorm(hidden_dim))
            self.shared_layers.append(nn.ReLU())
            self.shared_layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        
        # Separate heads for different action types
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.symbol_count),
            nn.Tanh()  # Position actions in [-1, 1]
        )
        
        self.order_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.symbol_count),
            nn.Tanh()  # Order type actions in [-1, 1]
        )
        
        self.price_offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.symbol_count),
            nn.Tanh()  # Price offset actions in [-1, 1]
        )
        
        # Separate log_std parameters for each action type
        self.position_log_std = nn.Parameter(torch.zeros(self.symbol_count))
        self.order_type_log_std = nn.Parameter(torch.zeros(self.symbol_count))
        self.price_offset_log_std = nn.Parameter(torch.zeros(self.symbol_count))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=0.5)  # Smaller gain for stability
            torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        # Handle NaN inputs
        state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Shared feature extraction
        x = state
        for layer in self.shared_layers:
            x = layer(x)
            # Check for NaN after each layer
            if torch.isnan(x).any():
                x = torch.nan_to_num(x, nan=0.0)
        
        # Separate action heads
        position_mean = self.position_head(x)
        order_type_mean = self.order_type_head(x)
        price_offset_mean = self.price_offset_head(x)
        
        # Handle NaN in means
        position_mean = torch.nan_to_num(position_mean, nan=0.0)
        order_type_mean = torch.nan_to_num(order_type_mean, nan=0.0)
        price_offset_mean = torch.nan_to_num(price_offset_mean, nan=0.0)
        
        # Combine means
        combined_mean = torch.cat([
            position_mean, order_type_mean, price_offset_mean
        ], dim=-1)
        
        # Combine log_std
        combined_log_std = torch.cat([
            self.position_log_std.expand_as(position_mean),
            self.order_type_log_std.expand_as(order_type_mean), 
            self.price_offset_log_std.expand_as(price_offset_mean)
        ], dim=-1)
        
        # Clamp log_std for stability
        combined_log_std = torch.clamp(combined_log_std, -5, 2)  # More conservative range
        
        return combined_mean, combined_log_std
    
    def get_action_and_log_prob(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Create normal distribution
        dist = Normal(mean, std)
        
        # Sample action and get log probability
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clamp actions to valid range
        action = torch.clamp(action, -1.0, 1.0)
        
        return action, log_prob


class LimitOrderCriticNetwork(nn.Module):
    """Enhanced Critic Network for limit order environment with richer state features."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        self.layers = nn.ModuleList()
        input_dim = state_dim
        
        for i in range(num_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if i < num_layers - 1:  # No normalization on final layer
                self.layers.append(nn.LayerNorm(hidden_dim))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        x = state
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:  # Last layer
                x = F.relu(x)  # ReLU after final linear layer
        
        value = self.value_head(x)
        return value


class LimitOrderFinRLAgent(FinRLAgent):
    """
    Enhanced FinRL Agent for intelligent limit order placement.
    
    Features:
    - 3D action space handling
    - Specialized networks for different action types
    - Enhanced reward processing for execution quality
    - Limit order specific performance metrics
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: RLConfig = None):
        # Initialize base class first
        super().__init__(state_dim, action_dim, config)
        
        # Override networks with limit order specific versions
        self.actor = LimitOrderActorNetwork(
            state_dim, action_dim, 
            self.config.hidden_dim, self.config.num_layers
        ).to(self.device)
        
        self.critic = LimitOrderCriticNetwork(
            state_dim, self.config.hidden_dim, self.config.num_layers
        ).to(self.device)
        
        # Update optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.learning_rate
        )
        
        # Limit order specific metrics
        self.limit_order_metrics = {
            'successful_limit_orders': 0,
            'total_limit_orders': 0,
            'avg_execution_quality': 0.0,
            'price_improvement_total': 0.0
        }
        
        logger.info(f"✅ Limit Order FinRL Agent initialized")
        logger.info(f"   Enhanced networks: Actor/Critic with specialized heads")
        logger.info(f"   Action dimensions: {action_dim} (3D per symbol)")
    
    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict]:
        """Enhanced action selection with 3D action interpretation."""
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if training:
                action, log_prob = self.actor.get_action_and_log_prob(state_tensor)
                action = action.squeeze(0).cpu().numpy()
                
                # Store for training
                self._last_log_prob = log_prob.item()
            else:
                mean, log_std = self.actor(state_tensor)
                action = mean.squeeze(0).cpu().numpy()
                self._last_log_prob = 0.0
            
            # Interpret 3D actions
            symbol_count = len(action) // 3
            actions_3d = action.reshape(symbol_count, 3)
            
            # Calculate confidence based on action magnitudes
            position_confidence = np.mean(np.abs(actions_3d[:, 0]))
            order_type_confidence = np.mean(np.abs(actions_3d[:, 1]))
            price_offset_confidence = np.mean(np.abs(actions_3d[:, 2]))
            
            overall_confidence = (position_confidence + order_type_confidence + price_offset_confidence) / 3
            
            metadata = {
                'confidence': float(overall_confidence),
                'log_prob': self._last_log_prob,
                'position_actions': actions_3d[:, 0].tolist(),
                'order_type_actions': actions_3d[:, 1].tolist(),
                'price_offset_actions': actions_3d[:, 2].tolist(),
                'action_interpretation': self._interpret_actions(actions_3d)
            }
            
            return action, metadata
    
    def _interpret_actions(self, actions_3d: np.ndarray) -> List[Dict]:
        """Interpret 3D actions into human-readable strategy descriptions."""
        interpretations = []
        
        for i, (pos_action, order_type_action, price_offset_action) in enumerate(actions_3d):
            # Position interpretation
            if abs(pos_action) < 0.1:
                position_desc = "hold"
            elif pos_action > 0:
                size_desc = "large" if pos_action > 0.5 else "small"
                position_desc = f"{size_desc} long"
            else:
                size_desc = "large" if abs(pos_action) > 0.5 else "small"
                position_desc = f"{size_desc} short"
            
            # Order type interpretation
            if order_type_action < -0.3:
                order_desc = "market order"
            elif abs(order_type_action) < 0.3:
                order_desc = "limit order"
            else:
                order_desc = "aggressive limit"
            
            # Price offset interpretation
            if abs(price_offset_action) < 0.2:
                price_desc = "at market"
            elif price_offset_action > 0:
                price_desc = "passive pricing"
            else:
                price_desc = "aggressive pricing"
            
            interpretations.append({
                'symbol_index': i,
                'position': position_desc,
                'order_type': order_desc,
                'pricing': price_desc,
                'raw_actions': [float(pos_action), float(order_type_action), float(price_offset_action)]
            })
        
        return interpretations
    
    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                      next_state: np.ndarray, done: bool, metadata: Dict = None):
        """Enhanced experience storage with limit order metrics."""
        
        # Call parent method
        super().add_experience(state, action, reward, next_state, done, metadata)
        
        # Track limit order specific metrics
        if metadata:
            # Update limit order performance tracking
            if 'execution_quality' in metadata:
                self.limit_order_metrics['avg_execution_quality'] = (
                    0.9 * self.limit_order_metrics['avg_execution_quality'] + 
                    0.1 * metadata['execution_quality']
                )
            
            if 'price_improvement' in metadata:
                self.limit_order_metrics['price_improvement_total'] += metadata['price_improvement']
            
            # Track action type usage
            action_interpretation = metadata.get('action_interpretation', [])
            for interp in action_interpretation:
                if 'limit' in interp.get('order_type', ''):
                    self.limit_order_metrics['total_limit_orders'] += 1
                    if reward > 0:  # Simplistic success metric
                        self.limit_order_metrics['successful_limit_orders'] += 1
    
    def update_policy(self, batch_size: int = None) -> Optional[Dict[str, float]]:
        """Enhanced policy update with limit order considerations."""
        
        # Use parent update method
        training_metrics = super().update_policy(batch_size)
        
        if training_metrics:
            # Add limit order specific metrics
            limit_success_rate = (
                self.limit_order_metrics['successful_limit_orders'] / 
                max(1, self.limit_order_metrics['total_limit_orders'])
            )
            
            training_metrics.update({
                'limit_order_success_rate': limit_success_rate,
                'avg_execution_quality': self.limit_order_metrics['avg_execution_quality'],
                'total_price_improvement': self.limit_order_metrics['price_improvement_total']
            })
        
        return training_metrics
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Enhanced performance statistics with limit order metrics."""
        
        base_stats = super().get_performance_stats()
        
        # Add limit order specific stats
        limit_stats = {
            'limit_order_usage_rate': (
                self.limit_order_metrics['total_limit_orders'] / 
                max(1, len(self.buffer))
            ),
            'limit_order_success_rate': (
                self.limit_order_metrics['successful_limit_orders'] / 
                max(1, self.limit_order_metrics['total_limit_orders'])
            ),
            'avg_execution_quality': self.limit_order_metrics['avg_execution_quality'],
            'total_price_improvement': self.limit_order_metrics['price_improvement_total']
        }
        
        base_stats.update(limit_stats)
        return base_stats
    
    def save_model(self, checkpoint_name: str = None):
        """Enhanced model saving with limit order metrics."""
        super().save_model(checkpoint_name)
        
        # Save limit order specific state
        if checkpoint_name:
            limit_order_path = self.model_dir / f"{checkpoint_name}_limit_order_metrics.json"
            with open(limit_order_path, 'w') as f:
                import json
                json.dump(self.limit_order_metrics, f, indent=2)
    
    def load_model(self, checkpoint_path: str) -> bool:
        """Enhanced model loading with limit order metrics."""
        success = super().load_model(checkpoint_path)
        
        if success:
            # Try to load limit order metrics
            try:
                limit_order_path = str(checkpoint_path).replace('.pt', '_limit_order_metrics.json')
                with open(limit_order_path, 'r') as f:
                    import json
                    self.limit_order_metrics = json.load(f)
                logger.info("Limit order metrics loaded successfully")
            except FileNotFoundError:
                logger.info("No limit order metrics file found, using defaults")
        
        return success


# Factory functions
def create_limit_order_finrl_agent(environment, config: RLConfig = None) -> LimitOrderFinRLAgent:
    """Create a limit order FinRL agent with optimal configuration."""
    
    if config is None:
        config = RLConfig(
            hidden_dim=512,  # Larger network for complex 3D action space
            num_layers=4,
            learning_rate=1e-4,  # Lower learning rate for stability
            batch_size=128,
            ppo_epochs=15,
            clip_epsilon=0.15,
            entropy_coeff=0.02  # Higher entropy for exploration
        )
    
    state_dim = environment.observation_space.shape[0]
    action_dim = environment.action_space.shape[0]
    
    agent = LimitOrderFinRLAgent(state_dim, action_dim, config)
    
    logger.info(f"✅ Limit Order FinRL Agent created")
    logger.info(f"   State dim: {state_dim}, Action dim: {action_dim}")
    logger.info(f"   Network: {config.hidden_dim}x{config.num_layers} with specialized heads")
    
    return agent


def train_limit_order_finrl_agent(agent: LimitOrderFinRLAgent, 
                                 environment,
                                 num_episodes: int = 1000,
                                 eval_frequency: int = 100) -> Dict[str, Any]:
    """Enhanced training function for limit order agent with specialized metrics."""
    
    logger.info(f"🚀 Starting Limit Order FinRL agent training for {num_episodes} episodes")
    
    episode_rewards = []
    execution_quality_history = []
    limit_order_usage_history = []
    
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        state = environment.reset()
        episode_reward = 0
        episode_limit_orders = 0
        episode_executions = 0
        step_count = 0
        
        while True:
            # Get action from agent
            action, metadata = agent.get_action(state, training=True)
            
            # Execute step
            next_state, reward, done, info = environment.step(action)
            
            # Enhanced metadata with execution info
            enhanced_metadata = metadata.copy()
            enhanced_metadata.update({
                'execution_quality': info.get('execution_quality', 0.5),
                'price_improvement': info.get('price_improvement', 0.0),
                'limit_orders_placed': info.get('limit_orders_placed', 0),
                'limit_orders_filled': info.get('limit_orders_filled', 0)
            })
            
            # Add experience
            agent.add_experience(state, action, reward, next_state, done, enhanced_metadata)
            
            episode_reward += reward
            episode_limit_orders += info.get('limit_orders_placed', 0)
            episode_executions += info.get('trades_executed', 0) + info.get('limit_orders_filled', 0)
            step_count += 1
            
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        execution_quality_history.append(info.get('execution_quality', 0.5))
        
        # Calculate limit order usage rate
        limit_usage_rate = episode_limit_orders / max(1, episode_executions)
        limit_order_usage_history.append(limit_usage_rate)
        
        # Train agent
        if len(agent.buffer) >= agent.config.min_buffer_size:
            training_metrics = agent.update_policy()
        
        # Logging and evaluation
        if episode % eval_frequency == 0:
            avg_reward = np.mean(episode_rewards[-eval_frequency:])
            avg_quality = np.mean(execution_quality_history[-eval_frequency:])
            avg_limit_usage = np.mean(limit_order_usage_history[-eval_frequency:])
            
            logger.info(f"Episode {episode}:")
            logger.info(f"  Avg Reward: {avg_reward:.4f}")
            logger.info(f"  Avg Execution Quality: {avg_quality:.3f}")
            logger.info(f"  Limit Order Usage: {avg_limit_usage:.2%}")
            logger.info(f"  Portfolio Value: ${info['portfolio_value']:,.2f}")
            logger.info(f"  Buffer Size: {len(agent.buffer)}")
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save_model(f"best_episode_{episode}")
                logger.info(f"  🏆 New best model saved!")
    
    # Final model save
    agent.save_model(f"final_episode_{num_episodes}")
    
    # Comprehensive results
    results = {
        'episode_rewards': episode_rewards,
        'execution_quality_history': execution_quality_history,
        'limit_order_usage_history': limit_order_usage_history,
        'best_reward': best_reward,
        'final_performance': agent.get_performance_stats(),
        'training_completed': True
    }
    
    logger.info("✅ Limit Order FinRL agent training completed")
    logger.info(f"   Best reward: {best_reward:.4f}")
    logger.info(f"   Final execution quality: {execution_quality_history[-1]:.3f}")
    logger.info(f"   Final limit order usage: {limit_order_usage_history[-1]:.2%}")
    
    return results


if __name__ == "__main__":
    # Test the limit order agent
    from agents.finrl_limit_order_env import create_limit_order_finrl_env
    import pandas as pd
    
    def test_limit_order_agent():
        # Create environment and agent
        symbols = ["AAPL", "MSFT", "GOOGL"]
        env = create_limit_order_finrl_env(symbols)
        
        # Generate test data
        test_data = {}
        for symbol in symbols:
            prices = np.random.uniform(100, 200, 50)
            volumes = np.random.uniform(1000000, 5000000, 50)
            
            test_data[f"{symbol}_close"] = prices
            test_data[f"{symbol}_volume"] = volumes
        
        df = pd.DataFrame(test_data)
        env.set_data(df)
        
        # Create agent
        agent = create_limit_order_finrl_agent(env)
        
        print("Testing Limit Order FinRL Agent")
        print(f"State dim: {agent.state_dim}")
        print(f"Action dim: {agent.action_dim}")
        print(f"Device: {agent.device}")
        
        # Test training
        results = train_limit_order_finrl_agent(
            agent, env, 
            num_episodes=20,
            eval_frequency=10
        )
        
        print(f"\nTraining Results:")
        print(f"Best reward: {results['best_reward']:.4f}")
        print(f"Final stats: {results['final_performance']}")
        
        env.close()
        print("✅ Limit Order Agent test completed")
    
    test_limit_order_agent()