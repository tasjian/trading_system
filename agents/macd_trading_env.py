#!/usr/bin/env python3
"""
MACD-Enhanced FinRL Trading Environment

This module extends the standard FinRL StockTradingEnv with MACD technical indicators:
1. Adds MACD line, Signal line, and MACD histogram to observation space
2. Implements MACD-based gating for buy/sell actions
3. Provides enhanced technical analysis for better trading decisions

MACD Logic:
- MACD < Signal: Bearish - forbid buy actions
- MACD > Signal: Bullish - forbid sell/short actions
- This prevents counter-trend trades and improves risk management
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Any

# Add FinRL to path
import sys
import os
finrl_path = '/Users/zac/Desktop/02_PROJECTS/FinRL'
if finrl_path not in sys.path:
    sys.path.insert(0, finrl_path)

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

logger = logging.getLogger(__name__)

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Compute MACD indicator and add columns to dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
        
    Returns:
        DataFrame with MACD columns added
    """
    df = df.copy()
    
    # Calculate EMAs
    df["EMA_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["EMA_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line
    df["MACD_line"] = df["EMA_fast"] - df["EMA_slow"]
    
    # Calculate Signal line
    df["Signal_line"] = df["MACD_line"].ewm(span=signal, adjust=False).mean()
    
    # Calculate MACD histogram
    df["MACD_hist"] = df["MACD_line"] - df["Signal_line"]
    
    # Fill any NaN values
    df = df.fillna(0)
    
    logger.info(f"✅ Added MACD indicators: fast={fast}, slow={slow}, signal={signal}")
    return df


class MACDTradingEnv(StockTradingEnv):
    """
    FinRL environment with MACD integrated:
      1. Adds MACD values to the observation/state space.
      2. Uses MACD as a gating mechanism for buy/sell actions.
      3. Provides enhanced technical analysis capabilities.
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        """
        Initialize MACD-enhanced trading environment.
        
        Args:
            df: Market data DataFrame
            **kwargs: Additional arguments for StockTradingEnv
        """
        # Ensure MACD is computed before passing to parent
        logger.info("🔧 Enhancing market data with MACD indicators...")
        df_with_macd = add_macd(df.copy())
        
        # Store original state space dimension
        self.original_state_space = kwargs.get('state_space', 0)
        
        # Extend state space to include 3 MACD features per stock
        if 'stock_dim' in kwargs:
            macd_features = 3 * kwargs['stock_dim']  # MACD_line, Signal_line, MACD_hist
            kwargs['state_space'] = self.original_state_space + macd_features
            logger.info(f"📊 Extended state space: {self.original_state_space} + {macd_features} = {kwargs['state_space']}")
        
        # Initialize parent with MACD-enhanced data
        super().__init__(df=df_with_macd, **kwargs)
        
        # Track MACD gating statistics
        self.macd_gated_buys = 0
        self.macd_gated_sells = 0
        self.total_steps = 0
        
        logger.info("✅ MACD Trading Environment initialized")

    def _get_observation(self) -> np.ndarray:
        """Extend state with MACD values for all stocks."""
        try:
            # Get base observation from parent
            obs = super()._get_observation()
            
            # Add MACD features for each stock
            macd_features = []
            
            for i in range(self.stock_dim):
                try:
                    # Get current row for this stock
                    current_row = self.df.iloc[self.current_step]
                    
                    # Extract MACD values
                    macd_line = current_row.get("MACD_line", 0.0) if "MACD_line" in current_row else 0.0
                    signal_line = current_row.get("Signal_line", 0.0) if "Signal_line" in current_row else 0.0
                    macd_hist = current_row.get("MACD_hist", 0.0) if "MACD_hist" in current_row else 0.0
                    
                    macd_features.extend([macd_line, signal_line, macd_hist])
                    
                except Exception as e:
                    logger.warning(f"MACD feature extraction failed for stock {i}: {e}")
                    macd_features.extend([0.0, 0.0, 0.0])
            
            # Concatenate to observation vector
            extended_obs = np.append(obs, macd_features)
            
            return extended_obs.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ MACD observation failed: {e}")
            # Fallback to base observation with zero MACD features
            obs = super()._get_observation()
            macd_features = [0.0] * (3 * self.stock_dim)
            return np.append(obs, macd_features).astype(np.float32)

    def step(self, actions: List[float]) -> tuple:
        """
        Override step to include MACD gating.
        - If MACD < Signal -> forbid buys (bearish signal)
        - If MACD > Signal -> forbid sells/shorts (bullish signal)
        
        Args:
            actions: List of actions for each stock
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        try:
            current_row = self.df.iloc[self.current_step]
            macd_line = current_row.get("MACD_line", 0.0)
            signal_line = current_row.get("Signal_line", 0.0)
            
            # Apply MACD gating
            gated_actions = []
            for i, action in enumerate(actions):
                original_action = action
                
                # MACD gating logic
                if macd_line < signal_line and action > 0:
                    # Bearish signal - disallow buy → replace with hold
                    gated_actions.append(0.0)
                    self.macd_gated_buys += 1
                    logger.debug(f"MACD gated buy action: {action:.3f} → 0.0 (bearish)")
                    
                elif macd_line > signal_line and action < 0:
                    # Bullish signal - disallow sell/short → replace with hold
                    gated_actions.append(0.0)
                    self.macd_gated_sells += 1
                    logger.debug(f"MACD gated sell action: {action:.3f} → 0.0 (bullish)")
                    
                else:
                    # Allow action
                    gated_actions.append(action)
            
            self.total_steps += 1
            
            # Execute step with gated actions
            obs, reward, done, info = super().step(gated_actions)
            
            # Add MACD info to info dict
            info['macd_info'] = {
                'macd_line': macd_line,
                'signal_line': signal_line,
                'macd_hist': current_row.get("MACD_hist", 0.0),
                'gated_buys': self.macd_gated_buys,
                'gated_sells': self.macd_gated_sells,
                'gating_rate': (self.macd_gated_buys + self.macd_gated_sells) / max(self.total_steps, 1)
            }
            
            return obs, reward, done, info
            
        except Exception as e:
            logger.error(f"❌ MACD step failed: {e}")
            # Fallback to parent step without gating
            return super().step(actions)

    def reset(self) -> np.ndarray:
        """Reset environment and MACD statistics."""
        # Reset MACD statistics
        self.macd_gated_buys = 0
        self.macd_gated_sells = 0
        self.total_steps = 0
        
        # Call parent reset
        obs = super().reset()
        
        return obs

    def get_macd_stats(self) -> dict:
        """Get MACD gating statistics."""
        return {
            'gated_buys': self.macd_gated_buys,
            'gated_sells': self.macd_gated_sells,
            'total_steps': self.total_steps,
            'buy_gating_rate': self.macd_gated_buys / max(self.total_steps, 1),
            'sell_gating_rate': self.macd_gated_sells / max(self.total_steps, 1),
            'overall_gating_rate': (self.macd_gated_buys + self.macd_gated_sells) / max(self.total_steps, 1)
        }

logger.info("✅ MACD Trading Environment module loaded")