"""
Online Reinforcement Learning Agent
Implements continuous learning and adaptation during live/paper trading
"""

import numpy as np
import pandas as pd
import logging
import asyncio
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import warnings
import os
from pathlib import Path

# Core RL components (will work with or without external dependencies)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available - using simplified online learning")

from agents.rl_trading_env import TradingState, MarketRegime
from agents.llm_rl_integration import EnhancedTradingState, LLMStateEnricher

logger = logging.getLogger(__name__)


class OnlineLearningMode(Enum):
    """Online learning operation modes."""
    EXPLORATION = "exploration"     # High exploration for learning
    EXPLOITATION = "exploitation"   # Low exploration for performance
    SAFETY = "safety"              # Minimal risk during volatility
    ADAPTATION = "adaptation"      # Moderate exploration during regime shifts


@dataclass
class ExperienceTransition:
    """Single experience transition for replay buffer."""
    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: datetime
    
    # Enhanced features for financial RL
    market_regime: MarketRegime
    volatility: float
    sentiment_score: float
    confidence: float
    
    # Metadata
    portfolio_value: float
    position: float
    trade_executed: bool


@dataclass
class OnlineLearningMetrics:
    """Metrics for tracking online learning performance."""
    # Learning metrics
    total_updates: int
    avg_loss: float
    policy_stability: float
    exploration_entropy: float
    
    # Performance metrics
    live_sharpe: float
    rolling_return: float
    max_drawdown: float
    regret_vs_benchmark: float
    
    # Risk metrics
    position_turnover: float
    risk_adjusted_return: float
    var_95: float
    
    # Adaptation metrics
    regime_adaptations: int
    checkpoint_rollbacks: int
    safety_overrides: int


class PrioritizedExperienceReplay:
    """
    Prioritized experience replay buffer for online learning.
    """
    
    def __init__(self, 
                 capacity: int = 10000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 epsilon: float = 1e-6):
        
        self.capacity = capacity
        self.alpha = alpha  # Prioritization strength
        self.beta = beta    # Importance sampling correction
        self.epsilon = epsilon
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
        # Time-based importance weighting
        self.time_decay = 0.99
        
    def add(self, experience: ExperienceTransition, priority: float = None):
        """Add experience with priority."""
        
        if priority is None:
            # Default priority is maximum (for new experiences)
            priority = max(self.priorities) if self.priorities else 1.0
        
        self.buffer.append(experience)
        self.priorities.append(priority)
        
    def sample(self, batch_size: int) -> Tuple[List[ExperienceTransition], np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights."""
        
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Convert to numpy for efficient sampling
        priorities = np.array(self.priorities)
        
        # Add time decay to priorities (more recent = higher priority)
        time_weights = np.array([self.time_decay ** (len(self.buffer) - i - 1) 
                                for i in range(len(self.buffer))])
        priorities = priorities * time_weights
        
        # Calculate probabilities
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        return experiences, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def get_recent_experiences(self, n: int) -> List[ExperienceTransition]:
        """Get n most recent experiences."""
        return list(self.buffer)[-n:] if len(self.buffer) >= n else list(self.buffer)
    
    def clear_old_experiences(self, max_age_hours: float = 24):
        """Remove experiences older than max_age_hours."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Filter experiences and priorities
        valid_indices = []
        for i, exp in enumerate(self.buffer):
            if exp.timestamp > cutoff_time:
                valid_indices.append(i)
        
        if valid_indices:
            # Keep only valid experiences
            valid_experiences = [self.buffer[i] for i in valid_indices]
            valid_priorities = [self.priorities[i] for i in valid_indices]
            
            self.buffer.clear()
            self.priorities.clear()
            
            self.buffer.extend(valid_experiences)
            self.priorities.extend(valid_priorities)


class SimpleNeuralPolicy:
    """
    Lightweight neural network policy for online learning.
    Works with or without PyTorch.
    """
    
    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        if TORCH_AVAILABLE:
            self._init_torch_policy()
        else:
            self._init_simple_policy()
    
    def _init_torch_policy(self):
        """Initialize PyTorch-based policy."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Tanh()  # Output in [-1, 1]
        ).to(self.device)
        
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        ).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.use_torch = True
    
    def _init_simple_policy(self):
        """Initialize simple linear policy without PyTorch."""
        # Simple linear policy: action = W * state + b
        self.actor_weights = np.random.randn(self.state_dim, self.action_dim) * 0.1
        self.actor_bias = np.zeros(self.action_dim)
        
        # Simple value function: V = W * state + b
        self.critic_weights = np.random.randn(self.state_dim, 1) * 0.1
        self.critic_bias = np.zeros(1)
        
        self.learning_rate = 0.001
        self.use_torch = False
    
    def get_action(self, state: np.ndarray, exploration_noise: float = 0.1) -> Tuple[float, float]:
        """Get action and value estimate."""
        
        if self.use_torch:
            return self._get_action_torch(state, exploration_noise)
        else:
            return self._get_action_simple(state, exploration_noise)
    
    def _get_action_torch(self, state: np.ndarray, exploration_noise: float) -> Tuple[float, float]:
        """PyTorch-based action selection."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0, 0]
            value = self.critic(state_tensor).cpu().numpy()[0, 0]
        
        # Add exploration noise
        if exploration_noise > 0:
            action += np.random.normal(0, exploration_noise)
            action = np.clip(action, -1, 1)
        
        return float(action), float(value)
    
    def _get_action_simple(self, state: np.ndarray, exploration_noise: float) -> Tuple[float, float]:
        """Simple linear policy action selection."""
        action = np.dot(state, self.actor_weights).item() + self.actor_bias[0]
        action = np.tanh(action)  # Keep in [-1, 1]
        
        value = np.dot(state, self.critic_weights).item() + self.critic_bias[0]
        
        # Add exploration noise
        if exploration_noise > 0:
            action += np.random.normal(0, exploration_noise)
            action = np.clip(action, -1, 1)
        
        return float(action), float(value)
    
    def update(self, experiences: List[ExperienceTransition], weights: np.ndarray = None) -> Dict[str, float]:
        """Update policy based on experiences."""
        
        if self.use_torch:
            return self._update_torch(experiences, weights)
        else:
            return self._update_simple(experiences, weights)
    
    def _update_torch(self, experiences: List[ExperienceTransition], weights: np.ndarray) -> Dict[str, float]:
        """PyTorch-based policy update."""
        if not experiences:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}
        
        # Prepare batch
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        actions = torch.FloatTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)
        
        if weights is not None:
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            weights = torch.ones(len(experiences)).to(self.device)
        
        # Critic update (TD error)
        current_values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        target_values = rewards + 0.99 * next_values * (~dones)
        
        critic_loss = ((current_values - target_values.detach()) ** 2 * weights).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update (policy gradient)
        predicted_actions = self.actor(states).squeeze()
        advantages = (target_values - current_values).detach()
        
        actor_loss = -(predicted_actions * advantages * weights).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }
    
    def _update_simple(self, experiences: List[ExperienceTransition], weights: np.ndarray) -> Dict[str, float]:
        """Simple gradient-based policy update."""
        if not experiences:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}
        
        # Prepare data
        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        
        if weights is None:
            weights = np.ones(len(experiences))
        
        # Simple TD learning for critic
        current_values = np.dot(states, self.critic_weights).flatten() + self.critic_bias[0]
        next_values = np.dot(next_states, self.critic_weights).flatten() + self.critic_bias[0]
        target_values = rewards + 0.99 * next_values
        
        critic_error = current_values - target_values
        critic_loss = np.mean((critic_error ** 2) * weights)
        
        # Update critic
        critic_grad_w = np.dot(states.T, (critic_error * weights).reshape(-1, 1)) / len(experiences)
        critic_grad_b = np.mean(critic_error * weights)
        
        self.critic_weights -= self.learning_rate * critic_grad_w
        self.critic_bias[0] -= self.learning_rate * critic_grad_b
        
        # Simple policy gradient for actor
        advantages = target_values - current_values
        actor_grad_w = np.dot(states.T, (actions * advantages * weights).reshape(-1, 1)) / len(experiences)
        actor_grad_b = np.mean(actions * advantages * weights)
        
        self.actor_weights += self.learning_rate * actor_grad_w  # Gradient ascent
        self.actor_bias[0] += self.learning_rate * actor_grad_b
        
        return {
            'actor_loss': -np.mean(actions * advantages * weights),  # Negative for gradient ascent
            'critic_loss': critic_loss
        }
    
    def save(self, filepath: str):
        """Save policy parameters."""
        if self.use_torch:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            }, filepath)
        else:
            np.savez(filepath,
                    actor_weights=self.actor_weights,
                    actor_bias=self.actor_bias,
                    critic_weights=self.critic_weights,
                    critic_bias=self.critic_bias)
    
    def load(self, filepath: str):
        """Load policy parameters."""
        if self.use_torch and filepath.endswith('.pt'):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        elif filepath.endswith('.npz'):
            data = np.load(filepath)
            self.actor_weights = data['actor_weights']
            self.actor_bias = data['actor_bias']
            self.critic_weights = data['critic_weights']
            self.critic_bias = data['critic_bias']


class OnlineRLAgent:
    """
    Online Reinforcement Learning Agent for continuous adaptation during live trading.
    """
    
    def __init__(self,
                 symbol: str,
                 state_dim: int = 32,  # Enhanced state dimension
                 learning_mode: OnlineLearningMode = OnlineLearningMode.EXPLOITATION,
                 buffer_capacity: int = 10000,
                 update_frequency: int = 100,
                 checkpoint_frequency: int = 1000):
        
        self.symbol = symbol
        self.state_dim = state_dim
        self.learning_mode = learning_mode
        self.update_frequency = update_frequency
        self.checkpoint_frequency = checkpoint_frequency
        
        # Core components
        self.policy = SimpleNeuralPolicy(state_dim)
        self.replay_buffer = PrioritizedExperienceReplay(buffer_capacity)
        self.state_enricher = LLMStateEnricher()
        
        # Performance tracking
        self.step_count = 0
        self.update_count = 0
        self.metrics_history: List[OnlineLearningMetrics] = []
        
        # Safety and stability
        self.performance_tracker = deque(maxlen=100)  # Recent returns
        self.policy_checkpoints: Dict[str, Any] = {}
        self.last_checkpoint_step = 0
        
        # Exploration parameters
        self.exploration_schedules = {
            OnlineLearningMode.EXPLORATION: 0.3,
            OnlineLearningMode.EXPLOITATION: 0.05,
            OnlineLearningMode.SAFETY: 0.0,
            OnlineLearningMode.ADAPTATION: 0.15
        }
        
        # Risk management
        self.safety_thresholds = {
            'max_drawdown': -0.10,
            'min_sharpe': -1.0,
            'max_position_change': 0.5
        }
        
        self.last_position = 0.0
        self.cumulative_return = 0.0
        self.rolling_volatility = 0.02
        
        logger.info(f"✅ Online RL Agent initialized for {symbol}")
        logger.info(f"📊 State dimension: {state_dim}, Mode: {learning_mode.value}")
    
    async def get_action(self, 
                        enhanced_state: EnhancedTradingState,
                        current_price: float) -> Tuple[float, float, Dict[str, Any]]:
        """Get action with online learning adaptation."""
        
        # Convert enhanced state to array
        state_array = self._enhanced_state_to_array(enhanced_state)
        
        # Get exploration noise based on current mode
        exploration_noise = self.exploration_schedules[self.learning_mode]
        
        # Adjust exploration based on market conditions
        if enhanced_state.llm_market_state.risk_level == "high":
            exploration_noise *= 0.5  # Reduce exploration in high risk
        elif enhanced_state.llm_market_state.volatility_forecast > 0.4:
            exploration_noise *= 0.3  # Reduce exploration in high volatility
        
        # Get action from policy
        action, value_estimate = self.policy.get_action(state_array, exploration_noise)
        
        # Apply safety constraints
        action = self._apply_safety_constraints(action, enhanced_state)
        
        # Calculate confidence based on value estimate and market conditions
        confidence = self._calculate_action_confidence(value_estimate, enhanced_state)
        
        # Prepare metadata
        metadata = {
            'value_estimate': value_estimate,
            'exploration_noise': exploration_noise,
            'learning_mode': self.learning_mode.value,
            'step_count': self.step_count,
            'safety_constrained': action != self.policy.get_action(state_array, 0)[0]
        }
        
        self.step_count += 1
        return action, confidence, metadata
    
    async def learn_from_experience(self,
                                  prev_state: EnhancedTradingState,
                                  action: float,
                                  reward: float,
                                  current_state: EnhancedTradingState,
                                  done: bool,
                                  metadata: Dict[str, Any]):
        """Learn from a single experience transition."""
        
        # Create experience transition
        experience = ExperienceTransition(
            state=self._enhanced_state_to_array(prev_state),
            action=action,
            reward=reward,
            next_state=self._enhanced_state_to_array(current_state),
            done=done,
            timestamp=datetime.now(),
            market_regime=current_state.llm_market_state.regime_prediction,
            volatility=current_state.llm_market_state.volatility_forecast,
            sentiment_score=current_state.llm_market_state.overall_score,
            confidence=current_state.llm_market_state.confidence,
            portfolio_value=current_state.base_state.portfolio_value,
            position=current_state.base_state.position,
            trade_executed=metadata.get('trade_executed', False)
        )
        
        # Add to replay buffer with priority based on TD error
        priority = abs(reward) + 1.0  # Simple priority heuristic
        self.replay_buffer.add(experience, priority)
        
        # Update performance tracking
        self.performance_tracker.append(reward)
        self.cumulative_return += reward
        
        # Check if it's time to update
        if (self.step_count % self.update_frequency == 0 and 
            len(self.replay_buffer.buffer) >= self.update_frequency):
            
            await self._perform_policy_update()
        
        # Check if it's time to create checkpoint
        if self.step_count % self.checkpoint_frequency == 0:
            await self._create_checkpoint()
        
        # Check if performance degradation requires rollback
        if await self._should_rollback_policy():
            await self._rollback_to_checkpoint()
    
    async def _perform_policy_update(self):
        """Perform a policy update using replay buffer."""
        
        batch_size = min(64, len(self.replay_buffer.buffer))
        experiences, weights, indices = self.replay_buffer.sample(batch_size)
        
        # Update policy
        losses = self.policy.update(experiences, weights)
        
        # Calculate TD errors for priority updates
        td_errors = []
        for exp in experiences:
            _, value = self.policy.get_action(exp.state, 0.0)
            _, next_value = self.policy.get_action(exp.next_state, 0.0)
            td_error = abs(exp.reward + 0.99 * next_value * (1 - exp.done) - value)
            td_errors.append(td_error)
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, np.array(td_errors))
        
        self.update_count += 1
        
        logger.debug(f"Policy update {self.update_count}: Actor loss={losses['actor_loss']:.6f}, "
                    f"Critic loss={losses['critic_loss']:.6f}")
    
    async def _create_checkpoint(self):
        """Create a policy checkpoint."""
        
        checkpoint_name = f"checkpoint_{self.step_count}"
        
        # Calculate recent performance metrics
        recent_returns = list(self.performance_tracker)[-100:]
        if recent_returns:
            avg_return = np.mean(recent_returns)
            sharpe = avg_return / (np.std(recent_returns) + 1e-8) * np.sqrt(252)
        else:
            avg_return = 0.0
            sharpe = 0.0
        
        checkpoint_data = {
            'step_count': self.step_count,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe,
            'cumulative_return': self.cumulative_return,
            'timestamp': datetime.now().isoformat()
        }
        
        self.policy_checkpoints[checkpoint_name] = checkpoint_data
        
        # Save policy parameters
        checkpoint_path = f"checkpoints/{self.symbol}_{checkpoint_name}.{'pt' if TORCH_AVAILABLE else 'npz'}"
        os.makedirs("checkpoints", exist_ok=True)
        self.policy.save(checkpoint_path)
        
        self.last_checkpoint_step = self.step_count
        
        logger.info(f"💾 Created checkpoint: {checkpoint_name} (Return: {avg_return:.4f}, Sharpe: {sharpe:.3f})")
    
    async def _should_rollback_policy(self) -> bool:
        """Check if policy should be rolled back due to poor performance."""
        
        if len(self.performance_tracker) < 50:
            return False
        
        recent_returns = list(self.performance_tracker)[-50:]
        recent_avg_return = np.mean(recent_returns)
        recent_sharpe = recent_avg_return / (np.std(recent_returns) + 1e-8) * np.sqrt(252)
        
        # Check safety thresholds
        drawdown = min(0, min(np.cumsum(recent_returns)))
        
        if (drawdown < self.safety_thresholds['max_drawdown'] or 
            recent_sharpe < self.safety_thresholds['min_sharpe']):
            
            logger.warning(f"⚠️ Performance degradation detected: DD={drawdown:.2%}, Sharpe={recent_sharpe:.3f}")
            return True
        
        return False
    
    async def _rollback_to_checkpoint(self):
        """Rollback to the best performing checkpoint."""
        
        if not self.policy_checkpoints:
            logger.warning("⚠️ No checkpoints available for rollback")
            return
        
        # Find best checkpoint by Sharpe ratio
        best_checkpoint = max(self.policy_checkpoints.items(), 
                            key=lambda x: x[1]['sharpe_ratio'])
        
        checkpoint_name, checkpoint_data = best_checkpoint
        
        # Load the checkpoint
        checkpoint_path = f"checkpoints/{self.symbol}_{checkpoint_name}.{'pt' if TORCH_AVAILABLE else 'npz'}"
        
        if os.path.exists(checkpoint_path):
            self.policy.load(checkpoint_path)
            logger.info(f"🔄 Rolled back to checkpoint: {checkpoint_name}")
            logger.info(f"📊 Checkpoint performance: Return={checkpoint_data['avg_return']:.4f}, "
                       f"Sharpe={checkpoint_data['sharpe_ratio']:.3f}")
        else:
            logger.error(f"❌ Checkpoint file not found: {checkpoint_path}")
    
    def _enhanced_state_to_array(self, enhanced_state: EnhancedTradingState) -> np.ndarray:
        """Convert enhanced trading state to array for policy input."""
        
        base_state = enhanced_state.base_state
        llm_state = enhanced_state.llm_market_state
        
        # Base trading features (25)
        base_features = np.array([
            # Market features (7)
            base_state.price / 1000.0,
            base_state.volume / 1e6,
            base_state.volatility,
            np.mean(base_state.returns[-5:]) if len(base_state.returns) >= 5 else 0.0,
            base_state.rsi / 100.0,
            base_state.macd,
            (base_state.price - base_state.bollinger_lower) / (base_state.bollinger_upper - base_state.bollinger_lower + 1e-8),
            
            # LLM features (6)
            llm_state.overall_score,
            llm_state.confidence,
            llm_state.news_sentiment,
            llm_state.social_sentiment,
            llm_state.earnings_sentiment,
            llm_state.sec_filings_sentiment,
            
            # Portfolio features (4)
            base_state.position,
            base_state.cash / 100000.0,
            base_state.portfolio_value / 100000.0,
            base_state.unrealized_pnl / 100000.0,
            
            # Risk features (3)
            np.clip(base_state.sharpe_ratio / 3.0, -1, 1),
            base_state.max_drawdown,
            base_state.var_95,
            
            # Regime features (2)
            base_state.regime.value / 3.0,
            llm_state.regime_confidence,
            
            # Time features (3)  
            base_state.hour_of_day / 24.0,
            base_state.day_of_week / 6.0,
            float(base_state.is_market_open)
        ], dtype=np.float32)
        
        # Enhanced features (7)
        enhanced_features = np.array([
            enhanced_state.sentiment_momentum,
            enhanced_state.consensus_strength,
            enhanced_state.news_impact_score,
            enhanced_state.fundamental_score,
            enhanced_state.technical_alignment,
            llm_state.trend_strength,
            1.0 if llm_state.risk_level == "high" else 0.5 if llm_state.risk_level == "medium" else 0.0
        ], dtype=np.float32)
        
        return np.concatenate([base_features, enhanced_features])
    
    def _apply_safety_constraints(self, action: float, enhanced_state: EnhancedTradingState) -> float:
        """Apply safety constraints to the action."""
        
        current_position = enhanced_state.base_state.position
        
        # Limit position changes
        max_change = self.safety_thresholds['max_position_change']
        position_change = action - current_position
        
        if abs(position_change) > max_change:
            action = current_position + np.sign(position_change) * max_change
        
        # Reduce action in high risk/volatility conditions
        if enhanced_state.llm_market_state.risk_level == "high":
            action *= 0.5
        elif enhanced_state.llm_market_state.volatility_forecast > 0.4:
            action *= 0.7
        
        # Ensure action is in valid range
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def _calculate_action_confidence(self, value_estimate: float, enhanced_state: EnhancedTradingState) -> float:
        """Calculate confidence in the action based on various factors."""
        
        # Base confidence from value estimate
        base_confidence = 1.0 / (1.0 + abs(value_estimate))  # Higher value = lower confidence (more uncertain)
        
        # Adjust for market conditions
        market_confidence = enhanced_state.llm_market_state.confidence
        consensus_strength = enhanced_state.consensus_strength
        
        # Combine factors
        combined_confidence = (base_confidence * 0.4 + 
                             market_confidence * 0.4 + 
                             consensus_strength * 0.2)
        
        # Reduce confidence in volatile conditions
        if enhanced_state.llm_market_state.volatility_forecast > 0.3:
            combined_confidence *= 0.8
        
        return np.clip(combined_confidence, 0.1, 0.9)
    
    def set_learning_mode(self, mode: OnlineLearningMode):
        """Set the online learning mode."""
        self.learning_mode = mode
        logger.info(f"🔄 Learning mode changed to: {mode.value}")
    
    def get_performance_metrics(self) -> OnlineLearningMetrics:
        """Get current performance metrics."""
        
        recent_returns = list(self.performance_tracker)[-100:] if self.performance_tracker else [0.0]
        
        # Calculate metrics
        avg_return = np.mean(recent_returns)
        volatility = np.std(recent_returns) + 1e-8
        sharpe = avg_return / volatility * np.sqrt(252)
        
        drawdown = min(0, min(np.cumsum(recent_returns))) if len(recent_returns) > 1 else 0.0
        
        return OnlineLearningMetrics(
            total_updates=self.update_count,
            avg_loss=0.0,  # Would need to track this
            policy_stability=1.0,  # Would need to calculate
            exploration_entropy=self.exploration_schedules[self.learning_mode],
            live_sharpe=sharpe,
            rolling_return=avg_return,
            max_drawdown=drawdown,
            regret_vs_benchmark=0.0,  # Would need benchmark
            position_turnover=0.0,  # Would need to track
            risk_adjusted_return=avg_return / (abs(drawdown) + 1e-8),
            var_95=np.percentile(recent_returns, 5) if len(recent_returns) > 10 else 0.0,
            regime_adaptations=0,  # Would need to track
            checkpoint_rollbacks=0,  # Would need to track
            safety_overrides=0  # Would need to track
        )
    
    async def close(self):
        """Clean up resources."""
        await self.state_enricher.close()
        logger.info(f"✅ Online RL Agent for {self.symbol} closed")


# Utility functions
def create_online_rl_agent(symbol: str, 
                          pretrained_model_path: str = None,
                          learning_mode: OnlineLearningMode = OnlineLearningMode.EXPLOITATION) -> OnlineRLAgent:
    """Create an online RL agent with optional pretrained model."""
    
    agent = OnlineRLAgent(
        symbol=symbol,
        learning_mode=learning_mode
    )
    
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        agent.policy.load(pretrained_model_path)
        logger.info(f"📥 Loaded pretrained model: {pretrained_model_path}")
    
    return agent


if __name__ == "__main__":
    # Test the online RL agent
    async def test_online_rl_agent():
        agent = create_online_rl_agent("AAPL", learning_mode=OnlineLearningMode.ADAPTATION)
        
        # Create mock enhanced state
        from agents.rl_trading_env import TradingState, MarketRegime
        from agents.llm_rl_integration import LLMMarketState, EnhancedTradingState, AnalysisDepth
        
        mock_base_state = TradingState(
            price=150.0, volume=1000000, volatility=0.2, returns=np.array([0.01, -0.005, 0.02]),
            rsi=55.0, macd=0.5, bollinger_upper=155.0, bollinger_lower=145.0,
            sentiment_score=0.3, sentiment_confidence=0.8, news_sentiment=0.2,
            social_sentiment=0.1, earnings_sentiment=0.4, sec_filings_sentiment=0.0,
            position=0.5, cash=50000, portfolio_value=100000, unrealized_pnl=1000,
            sharpe_ratio=1.2, max_drawdown=-0.05, var_95=-0.02,
            regime=MarketRegime.BULL, regime_confidence=0.7,
            hour_of_day=14, day_of_week=2, is_market_open=True
        )
        
        mock_llm_state = LLMMarketState(
            overall_sentiment="positive", overall_score=0.3, confidence=0.8,
            news_sentiment=0.2, social_sentiment=0.1, earnings_sentiment=0.4, sec_filings_sentiment=0.0,
            trend_direction="bullish", trend_strength=0.7, volatility_forecast=0.25,
            regime_prediction=MarketRegime.BULL, regime_confidence=0.7,
            risk_level="medium", risk_factors=["market volatility"], opportunities=["strong earnings"],
            key_themes=["tech growth"], upcoming_events=[], market_catalysts=["earnings"],
            analysis_timestamp=datetime.now(), data_freshness=2.0, analysis_depth=AnalysisDepth.STANDARD
        )
        
        enhanced_state = EnhancedTradingState(
            base_state=mock_base_state, llm_market_state=mock_llm_state,
            sentiment_momentum=0.1, consensus_strength=0.8, news_impact_score=0.6,
            fundamental_score=0.4, technical_alignment=0.7
        )
        
        # Test action generation
        action, confidence, metadata = await agent.get_action(enhanced_state, 150.0)
        print(f"Action: {action:.4f}, Confidence: {confidence:.4f}")
        print(f"Metadata: {metadata}")
        
        # Test learning from experience
        reward = 0.001  # 0.1% return
        await agent.learn_from_experience(
            enhanced_state, action, reward, enhanced_state, False, metadata
        )
        
        # Get performance metrics
        metrics = agent.get_performance_metrics()
        print(f"Performance metrics: {asdict(metrics)}")
        
        await agent.close()
    
    asyncio.run(test_online_rl_agent())