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
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

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
    max_drawdown: float = 0.05           # 5% max drawdown trigger
    min_sharpe_ratio: float = -0.5       # Minimum Sharpe before rollback
    max_kl_divergence: float = 0.02      # Trust region constraint (conservative)
    max_position_change: float = 0.1     # Max 10% position change per update
    consecutive_losses_limit: int = 5     # Max consecutive losing trades
    volatility_threshold: float = 0.03   # Max vol increase before safety mode
    daily_loss_limit: float = 0.02       # 2% daily loss limit

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
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
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
            current_drawdown = (peak_value - current_portfolio_value) / peak_value
            violations['max_drawdown'] = current_drawdown > self.constraints.max_drawdown
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
    
    def __init__(self, state_dim: int, action_dim: int, config: OnlineLearningConfig, device: str = 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - using dummy agent")
            return
        
        # Stable policy (conservative, updated infrequently)
        self.stable_policy = RegimeAwarePolicy(state_dim, action_dim).to(device)
        self.stable_optimizer = optim.Adam(self.stable_policy.parameters(), lr=0.0001)
        self.last_stable_update = datetime.now()
        
        # Learning agent (experimental, updated frequently)
        self.learner_policy = RegimeAwarePolicy(state_dim, action_dim).to(device)
        self.learner_optimizer = optim.Adam(self.learner_policy.parameters(), lr=0.0003)
        
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
        
    def sync_learner_to_stable(self):
        """Sync learner policy to stable policy."""
        if TORCH_AVAILABLE:
            self.learner_policy.load_state_dict(self.stable_policy.state_dict())
        
    def select_action(self, state: np.ndarray, regime: MarketRegime, 
                     deterministic: bool = False, uncertainty_estimate: float = 0.0) -> Tuple[np.ndarray, Dict]:
        """Select action using appropriate agent."""
        
        if not TORCH_AVAILABLE:
            # Return dummy action
            return np.zeros(self.action_dim), {"agent": "dummy", "uncertainty": 0.0}
        
        # Choose active policy
        policy = self.stable_policy if self.active_agent == "stable" else self.learner_policy
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            regime_tensor = torch.LongTensor([list(MarketRegime).index(regime)]).to(self.device)
            
            mean, std = policy(state_tensor, regime_tensor)
            
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
        
        return action.cpu().numpy(), {
            "agent": self.active_agent,
            "uncertainty": uncertainty_estimate,
            "epsilon": self.epsilon,
            "regime": regime.value
        }
    
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
    
    def __init__(self, symbols: List[str], config: OnlineLearningConfig, device: str = 'cpu'):
        self.symbols = symbols
        self.config = config
        self.device = device
        
        # Dynamic state and action dimensions based on actual symbols
        self.state_dim = len(symbols) * 12  # 12 features per symbol (price, volume, indicators, etc.)
        self.action_dim = len(symbols)      # One action per symbol
        
        logger.info(f"🎯 Online RL System initialized: {len(symbols)} symbols, "
                   f"state_dim={self.state_dim}, action_dim={self.action_dim}")
        
        # Core components
        self.dual_agent = DualAgentSystem(self.state_dim, self.action_dim, config, device)
        self.replay_buffer = PrioritizedReplayBuffer(config.buffer_size, config.priority_alpha, config.priority_beta)
        self.safety_monitor = SafetyMonitor(config.safety_constraints)
        
        # Training state
        self.training_active = False
        self.last_batch_update = datetime.now()
        self.last_safety_check = datetime.now()
        
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
        
        # Store experience if we have previous step
        if previous_action is not None:
            experience = Experience(
                state=self._last_state if hasattr(self, '_last_state') else market_state,
                action=previous_action,
                reward=previous_reward,
                next_state=market_state,
                done=False,
                info={}
            )
            
            self.replay_buffer.push(
                experience, 
                uncertainty=uncertainty,
                regime=self.current_regime
            )
        
        # Store current state for next iteration
        self._last_state = market_state.copy()
        
        # Update performance tracking
        performance_metrics = {
            'return': previous_reward,
            'timestamp': datetime.now(),
            'regime': self.current_regime.value,
            'uncertainty': uncertainty
        }
        
        self.performance_history.append(performance_metrics)
        self.dual_agent.update_performance(action_info['agent'], performance_metrics)
        
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
        
        # Trigger background training if needed
        await self._maybe_trigger_training()
        
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
        
        if (time_since_update >= self.config.batch_update_frequency and 
            len(self.replay_buffer.buffer) >= self.config.min_replay_size):
            
            if not self.training_active:
                await self._perform_batch_update()
    
    async def _perform_batch_update(self):
        """Perform a batch update of the learning agent."""
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - skipping training")
            return
            
        self.training_active = True
        logger.info("🎓 Starting batch update...")
        
        try:
            # Sample batch
            experiences, indices, weights = self.replay_buffer.sample(self.config.min_batch_size)
            
            # Convert to tensors
            states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
            actions = torch.FloatTensor([e.action for e in experiences]).to(self.device)
            rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
            next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
            weights_tensor = torch.FloatTensor(weights).to(self.device)
            
            # Regime conditioning (simplified - use current regime for all)
            regime_ids = torch.LongTensor([list(MarketRegime).index(self.current_regime)] * len(experiences)).to(self.device)
            
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
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        
        # Calculate performance metrics
        if len(self.performance_history) > 0:
            recent_returns = [p['return'] for p in list(self.performance_history)[-100:]]
            self.training_stats['average_return'] = np.mean(recent_returns) if recent_returns else 0.0
            
            if len(recent_returns) > 10 and np.std(recent_returns) > 0:
                self.training_stats['average_sharpe'] = (
                    np.mean(recent_returns) / np.std(recent_returns) * np.sqrt(252)
                )
        
        return {
            **self.training_stats,
            'active_agent': self.dual_agent.active_agent,
            'current_regime': self.current_regime.value,
            'buffer_size': len(self.replay_buffer.buffer),
            'stable_performance_samples': len(self.dual_agent.stable_performance),
            'learner_performance_samples': len(self.dual_agent.learner_performance),
            'training_active': self.training_active
        }
    
    def save_system_state(self, filepath: str):
        """Save complete system state."""
        
        state = {
            'config': asdict(self.config),
            'symbols': self.symbols,
            'training_stats': self.training_stats,
            'current_regime': self.current_regime.value,
            'performance_history': list(self.performance_history)[-1000:]  # Last 1000 entries
        }
        
        # Save model states if PyTorch is available
        if TORCH_AVAILABLE:
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
        """Load system state from file."""
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore training stats
            self.training_stats.update(state.get('training_stats', {}))
            self.current_regime = MarketRegime(state.get('current_regime', 'unknown'))
            
            # Load model states if available
            model_path = filepath.replace('.json', '_models.pt')
            if TORCH_AVAILABLE and Path(model_path).exists():
                checkpoint = torch.load(model_path)
                self.dual_agent.stable_policy.load_state_dict(checkpoint['stable_policy'])
                self.dual_agent.learner_policy.load_state_dict(checkpoint['learner_policy'])
                self.dual_agent.stable_optimizer.load_state_dict(checkpoint['stable_optimizer'])
                self.dual_agent.learner_optimizer.load_state_dict(checkpoint['learner_optimizer'])
            
            logger.info(f"✅ System state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load system state: {e}")

# Factory function
def create_online_rl_system(symbols: List[str], **kwargs) -> OnlineRLTradingSystem:
    """Create online RL trading system."""
    
    config = OnlineLearningConfig(**kwargs)
    device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
    
    logger.info(f"🚀 Creating online RL system for {len(symbols)} symbols on {device}")
    
    return OnlineRLTradingSystem(symbols, config, device)


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