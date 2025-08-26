#!/usr/bin/env python3
"""
Staged Curriculum Retraining System for RL Trading Agent
Implements phased retraining after reward policy changes using PPO fine-tuning
"""

import numpy as np
import logging
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from .unified_reward_calculator import UnifiedRewardCalculator, TradeMetrics, TradeType

logger = logging.getLogger(__name__)

class CurriculumPhase(Enum):
    """Stages of curriculum learning progression."""
    PHASE_1_PROFIT_STABILIZATION = "profit_stabilization"  # Focus only on profit component
    PHASE_2_LOSS_CUTTING = "loss_cutting"                  # Add loss-cutting discipline  
    PHASE_3_FULL_REWARD = "full_reward"                    # Complete reward with signals and costs
    PHASE_4_PRODUCTION = "production"                      # Full deployment mode

@dataclass
class CurriculumConfig:
    """Configuration for staged curriculum retraining."""
    
    # Phase duration and progression
    phase_1_episodes: int = 1000      # Profit-only stabilization episodes
    phase_2_episodes: int = 1500      # Add loss-cutting discipline 
    phase_3_episodes: int = 2000      # Full reward integration
    
    # Evaluation gates
    min_sharpe_threshold: float = 0.8   # Minimum Sharpe ratio to advance phases
    min_profit_rate: float = 0.6        # Minimum profitable trade percentage
    max_drawdown_threshold: float = 0.15 # Maximum drawdown before phase reset
    
    # PPO hyperparameters with curriculum scheduling
    initial_learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    lr_decay_factor: float = 0.8        # LR decay between phases
    
    initial_exploration_rate: float = 0.4
    min_exploration_rate: float = 0.1
    exploration_decay_rate: float = 0.95 # Per-episode decay
    
    # PPO-specific parameters
    ppo_epochs: int = 4
    batch_size: int = 64
    clip_ratio: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    
    # Evaluation and checkpointing
    eval_episodes: int = 100
    checkpoint_frequency: int = 200
    performance_window: int = 50        # Episodes to average for phase advancement
    
    # Reward component weights progression
    phase_1_alpha: float = 1.0    # Profit-only (other components = 0)
    phase_2_alpha: float = 1.0    # Keep profit focus
    phase_2_beta: float = 0.3     # Start with conservative loss-cutting
    phase_3_alpha: float = 1.0    # Final profit weight
    phase_3_beta: float = 0.5     # Full loss-cutting discipline
    phase_3_gamma: float = 0.3    # Signal alignment
    phase_3_delta: float = 0.2    # Transaction costs

@dataclass
class PhaseMetrics:
    """Performance metrics for curriculum phase evaluation."""
    phase: CurriculumPhase
    episode_count: int
    total_return: float
    sharpe_ratio: float
    profit_rate: float
    max_drawdown: float
    avg_reward_per_episode: float
    volatility: float
    trade_success_rate: float
    
    # Component-specific metrics
    profit_component_avg: float = 0.0
    loss_cutting_component_avg: float = 0.0
    signal_alignment_avg: float = 0.0
    transaction_cost_avg: float = 0.0
    
    # Learning metrics
    learning_rate: float = 0.0
    exploration_rate: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    
    # Advancement criteria
    meets_advancement_criteria: bool = False
    advancement_reason: Optional[str] = None

class CurriculumRetrainer:
    """
    Staged curriculum retraining system for RL trading agent.
    
    Implements the following curriculum:
    1. Phase 1: Profit-only stabilization (R_t = α·(r_t/σ))  
    2. Phase 2: Add loss-cutting discipline (R_t = α·(r_t/σ) + β·L_t)
    3. Phase 3: Full reward integration (R_t = α·(r_t/σ) + β·L_t + γ·S_t - δ·C_t)
    4. Phase 4: Production deployment with all extensions
    """
    
    def __init__(self, 
                 config: CurriculumConfig,
                 agent_model: nn.Module,
                 trading_environment: Any,  # Your trading environment
                 checkpoint_dir: str = "./checkpoints/curriculum"):
        """
        Initialize curriculum retrainer.
        
        Args:
            config: Curriculum configuration
            agent_model: PPO agent neural network model
            trading_environment: Trading environment for training
            checkpoint_dir: Directory for saving checkpoints
        """
        self.config = config
        self.agent_model = agent_model
        self.trading_env = trading_environment
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize curriculum state
        self.current_phase = CurriculumPhase.PHASE_1_PROFIT_STABILIZATION
        self.phase_episode_count = 0
        self.total_episode_count = 0
        self.phase_metrics_history: List[PhaseMetrics] = []
        
        # Initialize optimizers and schedulers
        self.optimizer = optim.Adam(agent_model.parameters(), lr=config.initial_learning_rate)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        
        # Current training parameters
        self.current_learning_rate = config.initial_learning_rate
        self.current_exploration_rate = config.initial_exploration_rate
        
        # Reward calculator for each phase
        self.reward_calculators = self._setup_phase_reward_calculators()
        
        # Performance tracking
        self.episode_rewards: List[float] = []
        self.episode_returns: List[float] = []
        self.episode_trades: List[int] = []
        
        logger.info(f"Initialized CurriculumRetrainer with {len(self.reward_calculators)} phases")
    
    def _setup_phase_reward_calculators(self) -> Dict[CurriculumPhase, UnifiedRewardCalculator]:
        """Setup reward calculators for each curriculum phase."""
        
        calculators = {}
        
        # Phase 1: Profit-only (R_t = α·(r_t/σ))
        calculators[CurriculumPhase.PHASE_1_PROFIT_STABILIZATION] = UnifiedRewardCalculator(
            alpha=self.config.phase_1_alpha,    # Profit component only
            beta=0.0,                           # No loss-cutting yet
            gamma=0.0,                          # No signal alignment yet
            delta=0.0,                          # No transaction costs yet
            lambda_cash=0.0,                    # No cash management yet
            lambda_regime=0.0,                  # No regime awareness yet
            lambda_risk=0.0                     # No risk management yet
        )
        
        # Phase 2: Add loss-cutting discipline (R_t = α·(r_t/σ) + β·L_t)
        calculators[CurriculumPhase.PHASE_2_LOSS_CUTTING] = UnifiedRewardCalculator(
            alpha=self.config.phase_2_alpha,
            beta=self.config.phase_2_beta,      # Start introducing loss-cutting
            gamma=0.0,                          # Still no signal alignment
            delta=0.0,                          # Still no transaction costs
            lambda_cash=0.2,                    # Light cash management
            lambda_regime=0.0,
            lambda_risk=0.0
        )
        
        # Phase 3: Full reward integration (R_t = α·(r_t/σ) + β·L_t + γ·S_t - δ·C_t)
        calculators[CurriculumPhase.PHASE_3_FULL_REWARD] = UnifiedRewardCalculator(
            alpha=self.config.phase_3_alpha,
            beta=self.config.phase_3_beta,      # Full loss-cutting discipline
            gamma=self.config.phase_3_gamma,    # Add signal alignment
            delta=self.config.phase_3_delta,    # Add transaction cost awareness
            lambda_cash=0.4,                    # Full cash management
            lambda_regime=0.1,                  # Add regime awareness
            lambda_risk=0.2                     # Add risk management
        )
        
        # Phase 4: Production (same as Phase 3 but with production settings)
        calculators[CurriculumPhase.PHASE_4_PRODUCTION] = UnifiedRewardCalculator(
            alpha=self.config.phase_3_alpha,
            beta=self.config.phase_3_beta,
            gamma=self.config.phase_3_gamma,
            delta=self.config.phase_3_delta,
            lambda_cash=0.4,
            lambda_regime=0.1,
            lambda_risk=0.2
        )
        
        return calculators
    
    async def run_curriculum_retraining(self) -> Dict[str, Any]:
        """
        Execute complete staged curriculum retraining.
        
        Returns:
            Dict with final metrics and training summary
        """
        logger.info("Starting staged curriculum retraining...")
        training_start_time = time.time()
        
        try:
            # Phase 1: Profit Stabilization
            logger.info("=== PHASE 1: PROFIT STABILIZATION ===")
            phase_1_metrics = await self._train_phase(
                CurriculumPhase.PHASE_1_PROFIT_STABILIZATION, 
                self.config.phase_1_episodes
            )
            
            # Phase 2: Loss-Cutting Integration
            logger.info("=== PHASE 2: LOSS-CUTTING DISCIPLINE ===")
            phase_2_metrics = await self._train_phase(
                CurriculumPhase.PHASE_2_LOSS_CUTTING,
                self.config.phase_2_episodes
            )
            
            # Phase 3: Full Reward Integration
            logger.info("=== PHASE 3: FULL REWARD INTEGRATION ===")
            phase_3_metrics = await self._train_phase(
                CurriculumPhase.PHASE_3_FULL_REWARD,
                self.config.phase_3_episodes
            )
            
            # Phase 4: Production Validation
            logger.info("=== PHASE 4: PRODUCTION VALIDATION ===")
            self.current_phase = CurriculumPhase.PHASE_4_PRODUCTION
            production_metrics = await self._evaluate_phase(self.config.eval_episodes)
            
            # Final evaluation and summary
            training_duration = time.time() - training_start_time
            final_summary = self._generate_training_summary(training_duration)
            
            logger.info(f"Curriculum retraining completed in {training_duration:.1f}s")
            return final_summary
            
        except Exception as e:
            logger.error(f"Curriculum retraining failed: {e}")
            raise
    
    async def _train_phase(self, phase: CurriculumPhase, max_episodes: int) -> PhaseMetrics:
        """
        Train agent in specific curriculum phase.
        
        Args:
            phase: Current curriculum phase
            max_episodes: Maximum episodes for this phase
            
        Returns:
            PhaseMetrics with performance results
        """
        logger.info(f"Training phase {phase.value} for up to {max_episodes} episodes")
        
        self.current_phase = phase
        self.phase_episode_count = 0
        current_reward_calculator = self.reward_calculators[phase]
        
        # Phase-specific training loop
        phase_rewards = []
        phase_returns = []
        recent_performance = []
        
        while self.phase_episode_count < max_episodes:
            # Run training episode
            episode_reward, episode_return, episode_metrics = await self._train_episode(
                current_reward_calculator
            )
            
            phase_rewards.append(episode_reward)
            phase_returns.append(episode_return)
            recent_performance.append(episode_return)
            
            # Keep only recent performance for evaluation
            if len(recent_performance) > self.config.performance_window:
                recent_performance.pop(0)
            
            self.phase_episode_count += 1
            self.total_episode_count += 1
            
            # Update learning parameters
            self._update_training_parameters()
            
            # Checkpoint regularly
            if self.phase_episode_count % self.config.checkpoint_frequency == 0:
                await self._save_checkpoint(phase, episode_metrics)
            
            # Evaluate phase advancement criteria
            if (len(recent_performance) >= self.config.performance_window and 
                self.phase_episode_count >= self.config.performance_window):
                
                phase_metrics = self._evaluate_phase_performance(
                    phase, recent_performance, phase_rewards, phase_returns
                )
                
                if phase_metrics.meets_advancement_criteria:
                    logger.info(f"Phase {phase.value} completed successfully: {phase_metrics.advancement_reason}")
                    self.phase_metrics_history.append(phase_metrics)
                    return phase_metrics
                
                # Check for phase reset conditions
                if phase_metrics.max_drawdown > self.config.max_drawdown_threshold:
                    logger.warning(f"Excessive drawdown {phase_metrics.max_drawdown:.3f}, resetting phase")
                    await self._reset_phase(phase)
        
        # Phase completed by episode limit
        final_phase_metrics = self._evaluate_phase_performance(
            phase, recent_performance, phase_rewards, phase_returns
        )
        self.phase_metrics_history.append(final_phase_metrics)
        
        logger.info(f"Phase {phase.value} completed with {self.phase_episode_count} episodes")
        return final_phase_metrics
    
    async def _train_episode(self, reward_calculator: UnifiedRewardCalculator) -> Tuple[float, float, Dict]:
        """
        Execute single training episode with PPO updates.
        
        Args:
            reward_calculator: Reward calculator for current phase
            
        Returns:
            Tuple of (episode_reward, episode_return, episode_metrics)
        """
        
        # Initialize episode
        state = await self.trading_env.reset()
        episode_reward = 0.0
        episode_return = 0.0
        episode_trades = 0
        
        # PPO trajectory storage
        states, actions, rewards, values, log_probs = [], [], [], [], []
        
        done = False
        step = 0
        
        while not done:
            # Get action from current policy
            with torch.no_grad():
                action, log_prob, value = self._get_action(state)
            
            # Execute action in environment
            next_state, base_reward, done, info = await self.trading_env.step(action)
            
            # Calculate curriculum-specific reward
            trade_metrics = self._extract_trade_metrics(state, action, next_state, info)
            reward_components = reward_calculator.calculate_reward(trade_metrics)
            curriculum_reward = reward_components.total_reward
            
            # Store trajectory
            states.append(state)
            actions.append(action)
            rewards.append(curriculum_reward)
            values.append(value)
            log_probs.append(log_prob)
            
            # Update metrics
            episode_reward += curriculum_reward
            if 'portfolio_return' in info:
                episode_return = info['portfolio_return']
            if info.get('trade_executed', False):
                episode_trades += 1
            
            state = next_state
            step += 1
        
        # PPO policy update
        policy_loss, value_loss = self._ppo_update(
            states, actions, rewards, values, log_probs
        )
        
        episode_metrics = {
            'episode_reward': episode_reward,
            'episode_return': episode_return,
            'episode_trades': episode_trades,
            'episode_steps': step,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'learning_rate': self.current_learning_rate,
            'exploration_rate': self.current_exploration_rate
        }
        
        return episode_reward, episode_return, episode_metrics
    
    def _get_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Get action from current policy with exploration."""
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Forward pass through policy network
        action_logits, value = self.agent_model(state_tensor)
        
        # Clip logits to prevent extreme values
        action_logits = torch.clamp(action_logits, -10, 10)
        
        # Add exploration noise based on current exploration rate
        if np.random.random() < self.current_exploration_rate:
            # Exploration: sample from uniform distribution
            action = np.random.randint(0, action_logits.shape[1])
        else:
            # Exploitation: sample from policy distribution with numerical stability
            action_probs = torch.softmax(action_logits, dim=1)
            # Add small epsilon to prevent zero probabilities
            action_probs = action_probs + 1e-8
            action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
            
            # Check for NaN/inf before sampling
            if torch.any(torch.isnan(action_probs)) or torch.any(torch.isinf(action_probs)):
                logger.warning("Invalid probabilities detected, using random action")
                action = np.random.randint(0, action_logits.shape[1])
            else:
                action = torch.multinomial(action_probs, 1).item()
        
        # Get log probability and value for PPO with numerical stability
        action_probs = torch.softmax(action_logits, dim=1) + 1e-8
        action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
        log_prob = torch.log(action_probs[0, action] + 1e-8)
        
        return action, log_prob, value.squeeze()
    
    def _ppo_update(self, states: List, actions: List, rewards: List, 
                   values: List, old_log_probs: List) -> Tuple[float, float]:
        """Perform PPO policy update."""
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        values_tensor = torch.FloatTensor(values)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs)
        
        # Calculate advantages using GAE
        advantages = self._calculate_gae_advantages(rewards_tensor, values_tensor)
        returns = advantages + values_tensor
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO policy update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for _ in range(self.config.ppo_epochs):
            # Create random batches
            batch_indices = np.random.permutation(len(states))
            
            for i in range(0, len(states), self.config.batch_size):
                batch_idx = batch_indices[i:i + self.config.batch_size]
                
                batch_states = states_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_log_probs = old_log_probs_tensor[batch_idx]
                
                # Forward pass
                action_logits, values_pred = self.agent_model(batch_states)
                
                # Clip logits for numerical stability
                action_logits = torch.clamp(action_logits, -10, 10)
                
                # Calculate new log probabilities with stability
                action_probs = torch.softmax(action_logits, dim=1) + 1e-8
                action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
                new_log_probs = torch.log(action_probs.gather(1, batch_actions.unsqueeze(1)) + 1e-8).squeeze()
                
                # PPO loss components with clipping
                ratio = torch.exp(torch.clamp(new_log_probs - batch_old_log_probs, -10, 10))
                clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                
                # Fix value loss dimension mismatch
                values_pred_flat = values_pred.squeeze()
                if values_pred_flat.dim() == 0:
                    values_pred_flat = values_pred_flat.unsqueeze(0)
                if batch_returns.dim() == 0:
                    batch_returns = batch_returns.unsqueeze(0)
                value_loss = nn.MSELoss()(values_pred_flat, batch_returns)
                
                entropy_loss = -(action_probs * torch.log(action_probs + 1e-8)).sum(1).mean()
                
                # Combined loss
                total_loss = (policy_loss + 
                             self.config.value_coeff * value_loss - 
                             self.config.entropy_coeff * entropy_loss)
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent_model.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        
        return total_policy_loss, total_value_loss
    
    def _calculate_gae_advantages(self, rewards: torch.Tensor, values: torch.Tensor, 
                                 gamma: float = 0.99, lambda_gae: float = 0.95) -> torch.Tensor:
        """Calculate Generalized Advantage Estimation (GAE)."""
        
        advantages = torch.zeros_like(rewards)
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantage = delta + gamma * lambda_gae * advantage
            advantages[t] = advantage
        
        return advantages
    
    def _extract_trade_metrics(self, state: np.ndarray, action: int, 
                              next_state: np.ndarray, info: Dict) -> TradeMetrics:
        """Extract trade metrics from environment step for reward calculation."""
        
        # Extract basic metrics from info dict
        trade_metrics = TradeMetrics(
            realized_return=info.get('realized_return', 0.0),
            unrealized_return=info.get('unrealized_return', 0.0),
            position_change=info.get('position_change', 0.0),
            signal_confidence=info.get('signal_confidence', 0.0),
            trade_direction=info.get('trade_direction', 0.0),
            signal_direction=info.get('signal_direction', 0.0),
            volatility=info.get('volatility', 0.02),
            transaction_cost=info.get('transaction_cost', 0.0),
            
            portfolio_value_t=info.get('portfolio_value_t', 100000.0),
            portfolio_value_t1=info.get('portfolio_value_t1', 100000.0),
            trade_type=TradeType(info.get('trade_type', 'hold')),
            position_size=info.get('position_size', 0.0),
            current_price=info.get('current_price', 0.0),
            
            # Cash management
            current_cash=info.get('current_cash', 0.0),
            buying_power=info.get('buying_power', 0.0),
            unrealized_pnl_pct=info.get('unrealized_pnl_pct', 0.0),
            is_underperformer=info.get('is_underperformer', False),
            
            # Market context
            sentiment_score=info.get('sentiment_score', 0.0),
            regime_alignment=info.get('regime_alignment', 0.0)
        )
        
        return trade_metrics
    
    def _update_training_parameters(self):
        """Update learning rate and exploration rate based on curriculum schedule."""
        
        # Learning rate decay
        self.lr_scheduler.step()
        self.current_learning_rate = self.optimizer.param_groups[0]['lr']
        
        # Exploration rate decay
        self.current_exploration_rate = max(
            self.config.min_exploration_rate,
            self.current_exploration_rate * self.config.exploration_decay_rate
        )
    
    def _evaluate_phase_performance(self, phase: CurriculumPhase, recent_returns: List[float],
                                   phase_rewards: List[float], phase_returns: List[float]) -> PhaseMetrics:
        """Evaluate whether phase meets advancement criteria."""
        
        # Calculate performance metrics
        avg_return = np.mean(recent_returns)
        volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.001
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0
        
        # Calculate profit rate (percentage of profitable episodes)
        profit_rate = sum(1 for r in recent_returns if r > 0) / len(recent_returns)
        
        # Calculate maximum drawdown
        cumulative_returns = np.cumsum(recent_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) / (running_max[-1] + 1e-8) if len(drawdowns) > 0 else 0.0
        
        # Check advancement criteria
        meets_criteria = (
            sharpe_ratio >= self.config.min_sharpe_threshold and
            profit_rate >= self.config.min_profit_rate and
            max_drawdown <= self.config.max_drawdown_threshold
        )
        
        advancement_reason = None
        if meets_criteria:
            advancement_reason = f"Sharpe: {sharpe_ratio:.3f}, Profit rate: {profit_rate:.3f}, Drawdown: {max_drawdown:.3f}"
        
        return PhaseMetrics(
            phase=phase,
            episode_count=self.phase_episode_count,
            total_return=sum(phase_returns),
            sharpe_ratio=sharpe_ratio,
            profit_rate=profit_rate,
            max_drawdown=max_drawdown,
            avg_reward_per_episode=np.mean(phase_rewards),
            volatility=volatility,
            trade_success_rate=profit_rate,
            learning_rate=self.current_learning_rate,
            exploration_rate=self.current_exploration_rate,
            meets_advancement_criteria=meets_criteria,
            advancement_reason=advancement_reason
        )
    
    async def _evaluate_phase(self, num_episodes: int) -> PhaseMetrics:
        """Evaluate current phase without training updates."""
        
        logger.info(f"Evaluating {self.current_phase.value} for {num_episodes} episodes")
        
        eval_returns = []
        eval_rewards = []
        current_calculator = self.reward_calculators[self.current_phase]
        
        # Run evaluation episodes
        for episode in range(num_episodes):
            state = await self.trading_env.reset()
            episode_reward = 0.0
            episode_return = 0.0
            done = False
            
            while not done:
                # Get action without exploration
                with torch.no_grad():
                    action, _, _ = self._get_action(state)
                    # Override exploration for evaluation
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_logits, _ = self.agent_model(state_tensor)
                    action = torch.argmax(action_logits, dim=1).item()
                
                next_state, _, done, info = await self.trading_env.step(action)
                
                # Calculate reward
                trade_metrics = self._extract_trade_metrics(state, action, next_state, info)
                reward_components = current_calculator.calculate_reward(trade_metrics)
                episode_reward += reward_components.total_reward
                
                if 'portfolio_return' in info:
                    episode_return = info['portfolio_return']
                
                state = next_state
            
            eval_returns.append(episode_return)
            eval_rewards.append(episode_reward)
        
        # Calculate evaluation metrics
        return self._evaluate_phase_performance(
            self.current_phase, eval_returns, eval_rewards, eval_returns
        )
    
    async def _reset_phase(self, phase: CurriculumPhase):
        """Reset phase due to poor performance."""
        
        logger.warning(f"Resetting phase {phase.value}")
        
        # Reset episode counters
        self.phase_episode_count = 0
        
        # Reset learning parameters
        self.current_learning_rate = self.config.initial_learning_rate
        self.current_exploration_rate = self.config.initial_exploration_rate
        
        # Reload last good checkpoint if available
        checkpoint_path = self.checkpoint_dir / f"phase_{phase.value}_best.pt"
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.agent_model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    async def _save_checkpoint(self, phase: CurriculumPhase, metrics: Dict):
        """Save training checkpoint."""
        
        checkpoint = {
            'phase': phase.value,
            'episode': self.phase_episode_count,
            'total_episodes': self.total_episode_count,
            'model_state': self.agent_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'learning_rate': self.current_learning_rate,
            'exploration_rate': self.current_exploration_rate,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / f"phase_{phase.value}_ep_{self.phase_episode_count}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        logger.debug(f"Saved checkpoint: {checkpoint_path}")
    
    def _generate_training_summary(self, training_duration: float) -> Dict[str, Any]:
        """Generate comprehensive training summary."""
        
        summary = {
            'training_duration_seconds': training_duration,
            'total_episodes': self.total_episode_count,
            'phases_completed': len(self.phase_metrics_history),
            'final_phase': self.current_phase.value,
            'phase_metrics': [asdict(pm) for pm in self.phase_metrics_history],
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate overall performance progression
        if len(self.phase_metrics_history) > 0:
            final_metrics = self.phase_metrics_history[-1]
            summary.update({
                'final_sharpe_ratio': final_metrics.sharpe_ratio,
                'final_profit_rate': final_metrics.profit_rate,
                'final_max_drawdown': final_metrics.max_drawdown,
                'total_return': final_metrics.total_return
            })
        
        # Save summary to file
        summary_path = self.checkpoint_dir / f"training_summary_{int(time.time())}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Training summary saved: {summary_path}")
        return summary

# Example usage for testing the curriculum retrainer
if __name__ == "__main__":
    # This would be used with your actual PPO model and trading environment
    pass