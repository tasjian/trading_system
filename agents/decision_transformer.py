"""
Decision Transformer for Supervised Pre-training
Implements offline RL using transformer architecture for trading agents
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
from pathlib import Path
import random
import math

# Handle optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DecisionTransformerConfig:
    """Configuration for Decision Transformer."""
    
    # Model architecture
    d_model: int = 256  # Transformer dimension
    nhead: int = 8      # Number of attention heads
    num_layers: int = 6 # Number of transformer layers
    dim_feedforward: int = 1024
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    sequence_length: int = 20  # Context length
    
    # Decision Transformer specific
    max_return: float = 1.0  # Maximum return-to-go
    return_scale: float = 1000.0  # Scale returns for better training
    
    # Training setup
    num_epochs: int = 100
    warmup_steps: int = 10000
    gradient_clip: float = 0.25
    
    # Evaluation
    target_return: float = 0.1  # Target return for evaluation

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class DecisionTransformer(nn.Module):
    """Decision Transformer for offline RL in trading."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: DecisionTransformerConfig):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Token embeddings
        self.state_embedding = nn.Linear(state_dim, config.d_model)
        self.action_embedding = nn.Linear(action_dim, config.d_model)
        self.return_embedding = nn.Linear(1, config.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, config.num_layers)
        
        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        """
        Forward pass of Decision Transformer.
        
        Args:
            states: [batch_size, seq_len, state_dim]
            actions: [batch_size, seq_len, action_dim]
            returns_to_go: [batch_size, seq_len, 1]
            timesteps: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Embed tokens
        state_embeddings = self.state_embedding(states)
        action_embeddings = self.action_embedding(actions)
        return_embeddings = self.return_embedding(returns_to_go)
        
        # Interleave tokens: [return, state, action, return, state, action, ...]
        # This creates the GPT-style sequence
        token_embeddings = torch.zeros(
            (batch_size, seq_len * 3, self.config.d_model),
            dtype=torch.float32,
            device=states.device
        )
        
        token_embeddings[:, ::3, :] = return_embeddings  # returns
        token_embeddings[:, 1::3, :] = state_embeddings  # states
        token_embeddings[:, 2::3, :] = action_embeddings # actions
        
        # Add positional encoding
        token_embeddings = self.pos_encoding(token_embeddings.transpose(0, 1)).transpose(0, 1)
        
        # Layer normalization
        token_embeddings = self.layer_norm(token_embeddings)
        
        # Create attention mask for interleaved tokens
        if attention_mask is not None:
            # Expand mask to match interleaved sequence
            expanded_mask = torch.zeros(
                (batch_size, seq_len * 3),
                dtype=torch.bool,
                device=attention_mask.device
            )
            expanded_mask[:, ::3] = attention_mask    # returns
            expanded_mask[:, 1::3] = attention_mask   # states  
            expanded_mask[:, 2::3] = attention_mask   # actions
        else:
            expanded_mask = None
        
        # Transformer forward pass
        transformer_outputs = self.transformer(
            token_embeddings,
            src_key_padding_mask=~expanded_mask if expanded_mask is not None else None
        )
        
        # Extract action predictions from state tokens
        state_outputs = transformer_outputs[:, 1::3, :]  # Get state positions
        action_predictions = self.action_head(state_outputs)
        
        return action_predictions
    
    def get_action(self, states, actions, returns_to_go, timesteps):
        """Get action for the last timestep (inference mode)."""
        
        # Ensure inputs are the right shape
        if len(states.shape) == 2:
            states = states.unsqueeze(0)
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(0)
        if len(returns_to_go.shape) == 2:
            returns_to_go = returns_to_go.unsqueeze(0)
        if len(timesteps.shape) == 1:
            timesteps = timesteps.unsqueeze(0)
        
        with torch.no_grad():
            action_predictions = self.forward(states, actions, returns_to_go, timesteps)
            return action_predictions[0, -1]  # Return last action

class TrajectoryDataset:
    """Dataset for Decision Transformer training."""
    
    def __init__(self, trajectories: List[Dict], config: DecisionTransformerConfig):
        self.trajectories = trajectories
        self.config = config
        self.sequence_length = config.sequence_length
        
        # Preprocess trajectories
        self.processed_trajectories = self._process_trajectories()
        
        logger.info(f"Created trajectory dataset with {len(self.processed_trajectories)} sequences")
    
    def _process_trajectories(self):
        """Process raw trajectories into training sequences."""
        
        processed = []
        
        for traj in self.trajectories:
            states = np.array(traj['states'])
            actions = np.array(traj['actions'])
            rewards = np.array(traj['rewards'])
            
            # Calculate returns-to-go
            returns_to_go = []
            cumulative_return = 0
            
            for i in reversed(range(len(rewards))):
                cumulative_return += rewards[i]
                returns_to_go.insert(0, cumulative_return)
            
            returns_to_go = np.array(returns_to_go)
            
            # Normalize returns
            returns_to_go = returns_to_go / self.config.return_scale
            
            # Create sequences - ensure minimum trajectory length
            if len(states) < self.sequence_length:
                continue  # Skip short trajectories
                
            for i in range(len(states) - self.sequence_length + 1):
                seq_states = states[i:i + self.sequence_length]
                seq_actions = actions[i:i + self.sequence_length]
                seq_returns = returns_to_go[i:i + self.sequence_length]
                seq_timesteps = np.arange(i, i + self.sequence_length)
                
                # Ensure all sequences have exactly the right length
                if (len(seq_states) == self.sequence_length and 
                    len(seq_actions) == self.sequence_length and 
                    len(seq_returns) == self.sequence_length):
                    processed.append({
                        'states': seq_states,
                        'actions': seq_actions,
                        'returns_to_go': seq_returns,
                        'timesteps': seq_timesteps
                    })
        
        return processed
    
    def __len__(self):
        return len(self.processed_trajectories)
    
    def __getitem__(self, idx):
        item = self.processed_trajectories[idx]
        return (
            torch.FloatTensor(item['states']),
            torch.FloatTensor(item['actions']),
            torch.FloatTensor(item['returns_to_go']).unsqueeze(-1),
            torch.LongTensor(item['timesteps'])
        )
    
    def get_dataloader(self, batch_size: int, shuffle: bool = True):
        """Create PyTorch DataLoader."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DataLoader")
        
        from torch.utils.data import DataLoader
        
        def collate_fn(batch):
            states, actions, returns_to_go, timesteps = zip(*batch)
            
            # Pad sequences to the same length
            max_len = max(s.size(0) for s in states)
            
            padded_states = []
            padded_actions = []
            padded_returns = []
            padded_timesteps = []
            
            for s, a, r, t in zip(states, actions, returns_to_go, timesteps):
                pad_len = max_len - s.size(0)
                if pad_len > 0:
                    # Pad with zeros
                    padded_states.append(torch.cat([s, torch.zeros(pad_len, s.size(1))]))
                    padded_actions.append(torch.cat([a, torch.zeros(pad_len, a.size(1))]))
                    padded_returns.append(torch.cat([r, torch.zeros(pad_len, r.size(1))]))
                    padded_timesteps.append(torch.cat([t, torch.zeros(pad_len, dtype=torch.long)]))
                else:
                    padded_states.append(s)
                    padded_actions.append(a)
                    padded_returns.append(r)
                    padded_timesteps.append(t)
            
            return (
                torch.stack(padded_states),
                torch.stack(padded_actions),
                torch.stack(padded_returns),
                torch.stack(padded_timesteps)
            )
        
        return DataLoader(
            dataset=list(range(len(self))),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda indices: collate_fn([self[i] for i in indices])
        )

class DecisionTransformerTrainer:
    """Trainer for Decision Transformer."""
    
    def __init__(self,
                 model: DecisionTransformer,
                 config: DecisionTransformerConfig,
                 device: str = 'cpu'):
        
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer with learning rate scheduling
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                (step + 1) / config.warmup_steps,  # Warmup
                1.0
            )
        )
        
        # Training tracking
        self.training_stats = {
            'losses': [],
            'action_accuracy': [],
            'learning_rates': []
        }
        
        self.step_count = 0
    
    def train(self, dataset: TrajectoryDataset) -> Dict[str, List[float]]:
        """Train the Decision Transformer."""
        
        dataloader = dataset.get_dataloader(self.config.batch_size, shuffle=True)
        
        self.model.train()
        epoch_losses = []
        epoch_accuracies = []
        
        logger.info(f"Starting Decision Transformer training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = 0
            
            for batch_idx, (states, actions, returns_to_go, timesteps) in enumerate(dataloader):
                states = states.to(self.device)
                actions = actions.to(self.device)
                returns_to_go = returns_to_go.to(self.device)
                timesteps = timesteps.to(self.device)
                
                # Forward pass
                action_predictions = self.model(states, actions, returns_to_go, timesteps)
                
                # Loss: MSE between predicted and actual actions
                loss = F.mse_loss(action_predictions, actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Statistics
                epoch_loss += loss.item()
                
                # Action accuracy (within 10% of target)
                action_diff = torch.abs(action_predictions - actions)
                accuracy = (action_diff < 0.1).float().mean().item()
                epoch_accuracy += accuracy
                
                num_batches += 1
                self.step_count += 1
                
                if batch_idx % 10 == 0:
                    logger.debug(f"Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}, Accuracy={accuracy:.3f}")
            
            # Average epoch statistics
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(avg_accuracy)
            
            # Update training stats
            self.training_stats['losses'].append(avg_loss)
            self.training_stats['action_accuracy'].append(avg_accuracy)
            self.training_stats['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.3f}")
        
        logger.info("✅ Decision Transformer training completed")
        
        return {
            'losses': epoch_losses,
            'accuracies': epoch_accuracies
        }
    
    def evaluate(self, test_dataset: TrajectoryDataset) -> Dict[str, float]:
        """Evaluate the trained model."""
        
        self.model.eval()
        test_dataloader = test_dataset.get_dataloader(self.config.batch_size, shuffle=False)
        
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for states, actions, returns_to_go, timesteps in test_dataloader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                returns_to_go = returns_to_go.to(self.device)
                timesteps = timesteps.to(self.device)
                
                action_predictions = self.model(states, actions, returns_to_go, timesteps)
                
                loss = F.mse_loss(action_predictions, actions)
                total_loss += loss.item()
                
                action_diff = torch.abs(action_predictions - actions)
                accuracy = (action_diff < 0.1).float().mean().item()
                total_accuracy += accuracy
                
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        logger.info(f"Evaluation: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.3f}")
        
        return {
            'test_loss': avg_loss,
            'test_accuracy': avg_accuracy
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats,
            'step_count': self.step_count
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        self.step_count = checkpoint.get('step_count', 0)
        
        logger.info(f"Model loaded from {filepath}")

def create_trajectories_from_hybrid_data(hybrid_dataset: Dict[str, Any],
                                       symbols: List[str]) -> List[Dict]:
    """Convert hybrid dataset to trajectory format for Decision Transformer."""
    
    trajectories = []
    market_data = hybrid_dataset['market_data']
    
    if len(market_data) < 20:  # Need sufficient length for sequences
        return trajectories
    
    # Create features for each timestep
    states = []
    actions = []
    rewards = []
    
    for i, day_data in enumerate(market_data):
        # Create state representation
        state_features = []
        for symbol in symbols:
            if symbol in day_data:
                data = day_data[symbol]
                sentiment = data.get('sentiment', {})
                orderbook = data.get('orderbook', {})
                
                features = [
                    data.get('price', 0.0) / 100.0,
                    data.get('return', 0.0) * 100,
                    data.get('volume', 0.0) / 1000000,
                    sentiment.get('overall_sentiment', 0.0),
                    sentiment.get('confidence', 0.5),
                    orderbook.get('spread', 0.01) * 1000,
                    orderbook.get('imbalance', 0.0)
                ]
                state_features.extend(features)
            else:
                state_features.extend([0.0] * 7)  # Default features
        
        states.append(state_features)
        
        # Generate synthetic actions based on sentiment
        action = []
        for symbol in symbols:
            if symbol in day_data:
                sentiment = day_data[symbol].get('sentiment', {})
                overall_sentiment = sentiment.get('overall_sentiment', 0.0)
                confidence = sentiment.get('confidence', 0.5)
                
                # Convert sentiment to action with some noise
                base_action = overall_sentiment * confidence
                noise = np.random.normal(0, 0.1)
                action.append(np.clip(base_action + noise, -1.0, 1.0))
            else:
                action.append(0.0)
        
        actions.append(action)
        
        # Calculate reward based on returns
        if i > 0:
            portfolio_return = 0
            for j, symbol in enumerate(symbols):
                if symbol in day_data and symbol in market_data[i-1]:
                    price_return = day_data[symbol].get('return', 0.0)
                    position = actions[i-1][j]  # Previous action
                    portfolio_return += position * price_return
            
            rewards.append(portfolio_return)
        else:
            rewards.append(0.0)
    
    # Create trajectory
    trajectory = {
        'states': states,
        'actions': actions[:-1],  # Remove last action (no next state)
        'rewards': rewards[1:]    # Remove first reward (no previous action)
    }
    
    trajectories.append(trajectory)
    return trajectories

# Factory function
def create_decision_transformer(state_dim: int, action_dim: int, **kwargs) -> Tuple[DecisionTransformer, DecisionTransformerConfig]:
    """Create Decision Transformer model and config."""
    
    config = DecisionTransformerConfig(**kwargs)
    model = DecisionTransformer(state_dim, action_dim, config)
    
    return model, config

if __name__ == "__main__":
    # Test Decision Transformer
    def test_decision_transformer():
        if not TORCH_AVAILABLE:
            print("PyTorch not available - skipping test")
            return
        
        state_dim = 21  # 3 symbols * 7 features
        action_dim = 3  # 3 symbols
        
        # Create model
        model, config = create_decision_transformer(state_dim, action_dim)
        
        # Create dummy trajectory
        dummy_trajectory = {
            'states': np.random.randn(50, state_dim).tolist(),
            'actions': np.random.randn(49, action_dim).tolist(),
            'rewards': np.random.randn(49).tolist()
        }
        
        # Create dataset
        dataset = TrajectoryDataset([dummy_trajectory], config)
        print(f"Dataset size: {len(dataset)}")
        
        # Create trainer
        trainer = DecisionTransformerTrainer(model, config)
        
        # Quick training test
        config.num_epochs = 2  # Quick test
        stats = trainer.train(dataset)
        
        print(f"Training completed. Final loss: {stats['losses'][-1]:.4f}")
        print("✅ Decision Transformer test completed")
    
    test_decision_transformer()