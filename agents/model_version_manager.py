"""
Model Version Manager for RL Trading System

Handles model versioning, compatibility checks, and automatic migration
to prevent dimension mismatch errors and preserve training progress.
"""

import torch
import torch.nn as nn
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import hashlib
import pickle
import copy

logger = logging.getLogger(__name__)

class ModelVersionManager:
    """Manages model versions and handles compatibility issues."""
    
    def __init__(self, base_path: str = "models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Model metadata storage
        self.metadata_file = self.base_path / "model_metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load model metadata from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            self.metadata = {}
    
    def save_metadata(self):
        """Save model metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
    
    def save_model_with_metadata(self, 
                                model_state_dict: Dict[str, Any],
                                symbols: List[str],
                                state_dim: int,
                                action_dim: int,
                                training_stats: Dict = None) -> str:
        """Save model with comprehensive metadata."""
        
        # Generate version ID
        version_id = self._generate_version_id(symbols, state_dim, action_dim)
        model_path = self.base_path / f"model_{version_id}.pt"
        
        # Create metadata
        metadata = {
            "version_id": version_id,
            "timestamp": datetime.now(),
            "symbols": symbols,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "model_architecture": "RegimeAwarePolicy",
            "pytorch_version": torch.__version__,
            "training_stats": training_stats or {},
            "file_path": str(model_path)
        }
        
        # Save model
        torch.save(model_state_dict, model_path)
        
        # Update metadata registry
        self.metadata[version_id] = metadata
        self.save_metadata()
        
        logger.info(f"📦 Saved model version {version_id} with metadata")
        return str(model_path)
    
    def load_compatible_model(self, 
                             symbols: List[str],
                             state_dim: int,
                             action_dim: int) -> Optional[Dict[str, Any]]:
        """Load a compatible model or migrate from closest match."""
        
        # Validate input dimensions
        validation = self.validate_model_architecture(state_dim, action_dim)
        if not validation["valid"]:
            logger.error(f"Cannot load model with invalid dimensions: {validation['issues']}")
            return self.create_fresh_compatible_model(state_dim, action_dim)
        
        # First try exact match
        version_id = self._generate_version_id(symbols, state_dim, action_dim)
        if version_id in self.metadata:
            try:
                model_path = Path(self.metadata[version_id]["file_path"])
                if model_path.exists():
                    model_dict = torch.load(model_path, map_location='cpu')
                    logger.info(f"✅ Loaded exact match model {version_id}")
                    return model_dict
            except Exception as e:
                logger.warning(f"Could not load exact match: {e}")
        
        # Try to find compatible model for migration
        compatible_version = self._find_compatible_version(symbols, state_dim, action_dim)
        if compatible_version:
            try:
                source_model = torch.load(
                    Path(self.metadata[compatible_version]["file_path"]), 
                    map_location='cpu'
                )
                
                # Attempt migration
                migrated_model = self._migrate_model(
                    source_model,
                    self.metadata[compatible_version],
                    state_dim,
                    action_dim
                )
                
                if migrated_model:
                    logger.info(f"🔄 Successfully migrated model from {compatible_version}")
                    # Add migration metadata
                    migrated_model['metadata'] = {
                        'migrated_from': compatible_version,
                        'original_dimensions': {
                            'state_dim': self.metadata[compatible_version].get('state_dim'),
                            'action_dim': self.metadata[compatible_version].get('action_dim')
                        },
                        'target_dimensions': {'state_dim': state_dim, 'action_dim': action_dim},
                        'migration_timestamp': datetime.now(),
                        'architecture': self.metadata[compatible_version].get('model_architecture', 'unknown')
                    }
                    return migrated_model
                    
            except Exception as e:
                logger.warning(f"Migration failed: {e}")
        
        # If no compatible model found, create fresh one
        logger.info("🆕 No compatible model found, creating fresh compatible model")
        return self.create_fresh_compatible_model(state_dim, action_dim)
    
    def _generate_version_id(self, symbols: List[str], state_dim: int, action_dim: int) -> str:
        """Generate unique version ID based on model configuration."""
        config_str = f"{sorted(symbols)}_{state_dim}_{action_dim}"
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def _find_compatible_version(self, 
                                symbols: List[str],
                                state_dim: int,
                                action_dim: int) -> Optional[str]:
        """Find most compatible existing version for migration."""
        
        best_version = None
        best_score = 0
        
        for version_id, metadata in self.metadata.items():
            score = self._calculate_compatibility_score(
                metadata, symbols, state_dim, action_dim
            )
            
            if score > best_score:
                best_score = score
                best_version = version_id
        
        # Only consider versions with reasonable compatibility
        if best_score > 0.3:
            return best_version
        
        return None
    
    def _calculate_compatibility_score(self,
                                      metadata: Dict,
                                      symbols: List[str],
                                      state_dim: int,
                                      action_dim: int) -> float:
        """Calculate compatibility score between models."""
        score = 0.0
        
        # Symbol overlap (40% weight)
        stored_symbols = set(metadata.get("symbols", []))
        current_symbols = set(symbols)
        if stored_symbols and current_symbols:
            symbol_overlap = len(stored_symbols & current_symbols) / len(stored_symbols | current_symbols)
            score += symbol_overlap * 0.4
        
        # Action dimension compatibility (30% weight)
        stored_action_dim = metadata.get("action_dim", 0)
        if stored_action_dim == action_dim:
            score += 0.3
        elif stored_action_dim > 0:
            # Partial compatibility if dimensions are close
            ratio = min(stored_action_dim, action_dim) / max(stored_action_dim, action_dim)
            if ratio > 0.7:
                score += ratio * 0.15
        
        # State dimension compatibility (30% weight)
        stored_state_dim = metadata.get("state_dim", 0)
        if stored_state_dim == state_dim:
            score += 0.3
        elif stored_state_dim > 0:
            # Check if we can migrate
            ratio = min(stored_state_dim, state_dim) / max(stored_state_dim, state_dim)
            if ratio > 0.5:  # At least 50% dimensional overlap
                score += ratio * 0.2
        
        return score
    
    def _migrate_model(self,
                      source_model: Dict[str, Any],
                      source_metadata: Dict,
                      target_state_dim: int,
                      target_action_dim: int) -> Optional[Dict[str, Any]]:
        """Migrate model from source dimensions to target dimensions."""
        
        try:
            migrated_model = {}
            
            for key, state_dict in source_model.items():
                if key in ['stable_policy', 'learner_policy']:
                    migrated_state_dict = self._migrate_policy_state_dict(
                        state_dict,
                        source_metadata.get("state_dim", 0),
                        source_metadata.get("action_dim", 0),
                        target_state_dim,
                        target_action_dim
                    )
                    
                    if migrated_state_dict:
                        migrated_model[key] = migrated_state_dict
                    else:
                        logger.warning(f"Could not migrate {key}")
                        return None
                else:
                    # Copy optimizer states as-is (they'll adapt automatically)
                    migrated_model[key] = state_dict
            
            return migrated_model if migrated_model else None
            
        except Exception as e:
            logger.error(f"Migration error: {e}")
            return None
    
    def _migrate_policy_state_dict(self,
                                  source_state_dict: Dict[str, torch.Tensor],
                                  source_state_dim: int,
                                  source_action_dim: int,
                                  target_state_dim: int,
                                  target_action_dim: int) -> Optional[Dict[str, torch.Tensor]]:
        """Migrate policy network state dict to new dimensions."""
        
        try:
            migrated_dict = {}
            
            for name, param in source_state_dict.items():
                if name == 'network.0.weight':
                    # Input layer: resize to new state dimensions
                    migrated_dict[name] = self._resize_input_layer(
                        param, source_state_dim, target_state_dim
                    )
                elif name == 'network.0.bias':
                    # Input layer bias stays the same
                    migrated_dict[name] = param.clone()
                elif name == 'network.6.weight':
                    # Output layer: resize to new action dimensions
                    migrated_dict[name] = self._resize_output_layer(
                        param, source_action_dim, target_action_dim
                    )
                elif name == 'network.6.bias':
                    # Output layer bias: resize to new action dimensions
                    migrated_dict[name] = self._resize_output_bias(
                        param, source_action_dim, target_action_dim
                    )
                else:
                    # Copy other parameters as-is
                    migrated_dict[name] = param.clone()
            
            return migrated_dict
            
        except Exception as e:
            logger.error(f"State dict migration error: {e}")
            return None
    
    def _resize_input_layer(self,
                           weight: torch.Tensor,
                           source_dim: int,
                           target_dim: int) -> torch.Tensor:
        """Resize input layer weight matrix."""
        
        # Current weight shape: [hidden_dim, source_state_dim + 16]
        # Target shape: [hidden_dim, target_state_dim + 16]
        
        if source_dim == target_dim:
            return weight.clone()
        
        hidden_dim = weight.shape[0]
        source_full_dim = source_dim + 16  # +16 for regime embedding
        target_full_dim = target_dim + 16
        
        # Create new weight matrix
        new_weight = torch.zeros(hidden_dim, target_full_dim)
        
        if target_dim > source_dim:
            # Expanding: copy existing weights and initialize new ones
            new_weight[:, :source_full_dim] = weight
            # Initialize new weights with Xavier uniform
            torch.nn.init.xavier_uniform_(new_weight[:, source_full_dim:])
        else:
            # Contracting: keep most important features
            new_weight = weight[:, :target_full_dim].clone()
        
        return new_weight
    
    def _resize_output_layer(self,
                            weight: torch.Tensor,
                            source_action_dim: int,
                            target_action_dim: int) -> torch.Tensor:
        """Resize output layer weight matrix."""
        
        if source_action_dim == target_action_dim:
            return weight.clone()
        
        hidden_dim = weight.shape[1]
        # Output layer produces action_dim * 2 (mean and log_std)
        source_output_dim = source_action_dim * 2
        target_output_dim = target_action_dim * 2
        
        # Create new weight matrix
        new_weight = torch.zeros(target_output_dim, hidden_dim)
        
        if target_action_dim > source_action_dim:
            # Expanding: copy existing and initialize new
            new_weight[:source_output_dim, :] = weight
            torch.nn.init.xavier_uniform_(new_weight[source_output_dim:, :])
        else:
            # Contracting: keep most important actions
            new_weight = weight[:target_output_dim, :].clone()
        
        return new_weight
    
    def _resize_output_bias(self,
                           bias: torch.Tensor,
                           source_action_dim: int,
                           target_action_dim: int) -> torch.Tensor:
        """Resize output layer bias vector."""
        
        if source_action_dim == target_action_dim:
            return bias.clone()
        
        source_output_dim = source_action_dim * 2
        target_output_dim = target_action_dim * 2
        
        new_bias = torch.zeros(target_output_dim)
        
        if target_action_dim > source_action_dim:
            # Expanding: copy existing and zero-initialize new
            new_bias[:source_output_dim] = bias
        else:
            # Contracting: keep most important
            new_bias = bias[:target_output_dim].clone()
        
        return new_bias
    
    def cleanup_old_versions(self, keep_count: int = 5):
        """Clean up old model versions, keeping only the most recent."""
        try:
            # Sort by timestamp
            versions = sorted(
                self.metadata.items(),
                key=lambda x: x[1].get("timestamp", datetime.min),
                reverse=True
            )
            
            # Remove old versions
            for version_id, metadata in versions[keep_count:]:
                model_path = Path(metadata.get("file_path", ""))
                if model_path.exists():
                    model_path.unlink()
                
                del self.metadata[version_id]
                logger.info(f"🗑️ Cleaned up old model version {version_id}")
            
            self.save_metadata()
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def validate_model_architecture(self, state_dim: int, action_dim: int) -> Dict[str, Any]:
        """Validate model architecture and suggest fixes for common issues."""
        validation_result = {
            "valid": True,
            "issues": [],
            "suggestions": [],
            "expected_dimensions": {
                "input_layer": (256, state_dim + 16),  # Hidden units x (state + regime)
                "output_layer": (action_dim * 2, 256)  # (mean + log_std) x hidden units
            }
        }
        
        # Check for common dimension issues
        if state_dim <= 0:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Invalid state dimension: {state_dim}")
            validation_result["suggestions"].append("State dimension must be positive")
            
        if action_dim <= 0:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Invalid action dimension: {action_dim}")
            validation_result["suggestions"].append("Action dimension must be positive")
            
        # Check for extreme dimensions that might cause issues
        if state_dim > 10000:
            validation_result["issues"].append(f"Very large state dimension: {state_dim}")
            validation_result["suggestions"].append("Consider feature reduction or normalization")
            
        if action_dim > 1000:
            validation_result["issues"].append(f"Very large action dimension: {action_dim}")
            validation_result["suggestions"].append("Consider action space reduction")
            
        # Check for dimension ratios that might be problematic
        if state_dim / action_dim > 100:
            validation_result["issues"].append(f"High state/action ratio: {state_dim}/{action_dim}")
            validation_result["suggestions"].append("Model may overfit; consider regularization")
            
        return validation_result
        
    def create_fresh_compatible_model(self, state_dim: int, action_dim: int) -> Dict[str, Any]:
        """Create a fresh model state dict compatible with given dimensions."""
        try:
            # Validate dimensions first
            validation = self.validate_model_architecture(state_dim, action_dim)
            if not validation["valid"]:
                logger.error(f"Cannot create model with invalid dimensions: {validation['issues']}")
                return None
                
            # Create a minimal compatible model structure
            hidden_dim = 256
            input_dim = state_dim + 16  # +16 for regime embedding
            output_dim = action_dim * 2  # mean and log_std
            
            fresh_model = {}
            
            # Create policy state dict with proper initialization
            policy_state = {
                # Regime embedding
                'regime_embedding.weight': torch.randn(len(self._get_market_regimes()), 16) * 0.1,
                
                # Network layers
                'network.0.weight': torch.zeros(hidden_dim, input_dim),
                'network.0.bias': torch.zeros(hidden_dim),
                'network.2.weight': torch.zeros(hidden_dim, hidden_dim),
                'network.2.bias': torch.zeros(hidden_dim),
                'network.4.weight': torch.zeros(hidden_dim, hidden_dim),
                'network.4.bias': torch.zeros(hidden_dim),
                'network.6.weight': torch.zeros(output_dim, hidden_dim),
                'network.6.bias': torch.zeros(output_dim)
            }
            
            # Initialize weights with Xavier uniform
            for name, param in policy_state.items():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
            
            fresh_model['stable_policy'] = policy_state
            fresh_model['learner_policy'] = copy.deepcopy(policy_state)
            
            logger.info(f"🆕 Created fresh compatible model: {state_dim}D state → {action_dim}D action")
            return fresh_model
            
        except Exception as e:
            logger.error(f"Fresh model creation failed: {e}")
            return None
    
    def _get_market_regimes(self) -> List[str]:
        """Get list of market regime types for embedding initialization."""
        return [
            "bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol",
            "sideways", "volatile", "crisis", "unknown"
        ]
        
    def get_migration_statistics(self) -> Dict[str, Any]:
        """Get statistics about model migrations and compatibility."""
        stats = {
            "total_models": len(self.metadata),
            "successful_migrations": 0,
            "failed_migrations": 0,
            "dimension_ranges": {
                "state_dim": {"min": float('inf'), "max": 0},
                "action_dim": {"min": float('inf'), "max": 0}
            },
            "architecture_types": {},
            "model_ages": []
        }
        
        for version_id, metadata in self.metadata.items():
            # Track dimension ranges
            state_dim = metadata.get("state_dim", 0)
            action_dim = metadata.get("action_dim", 0)
            
            if state_dim > 0:
                stats["dimension_ranges"]["state_dim"]["min"] = min(stats["dimension_ranges"]["state_dim"]["min"], state_dim)
                stats["dimension_ranges"]["state_dim"]["max"] = max(stats["dimension_ranges"]["state_dim"]["max"], state_dim)
                
            if action_dim > 0:
                stats["dimension_ranges"]["action_dim"]["min"] = min(stats["dimension_ranges"]["action_dim"]["min"], action_dim)
                stats["dimension_ranges"]["action_dim"]["max"] = max(stats["dimension_ranges"]["action_dim"]["max"], action_dim)
                
            # Track architecture types
            arch_type = metadata.get("model_architecture", "unknown")
            stats["architecture_types"][arch_type] = stats["architecture_types"].get(arch_type, 0) + 1
            
            # Calculate model age
            if "timestamp" in metadata:
                try:
                    if isinstance(metadata["timestamp"], str):
                        timestamp = datetime.fromisoformat(metadata["timestamp"])
                    else:
                        timestamp = metadata["timestamp"]
                    age_hours = (datetime.now() - timestamp).total_seconds() / 3600
                    stats["model_ages"].append(age_hours)
                except:
                    pass
        
        # Handle empty ranges
        if stats["dimension_ranges"]["state_dim"]["min"] == float('inf'):
            stats["dimension_ranges"]["state_dim"] = {"min": 0, "max": 0}
        if stats["dimension_ranges"]["action_dim"]["min"] == float('inf'):
            stats["dimension_ranges"]["action_dim"] = {"min": 0, "max": 0}
            
        return stats

# Global instance
_version_manager = None

def get_version_manager() -> ModelVersionManager:
    """Get global version manager instance."""
    global _version_manager
    if _version_manager is None:
        _version_manager = ModelVersionManager()
    return _version_manager