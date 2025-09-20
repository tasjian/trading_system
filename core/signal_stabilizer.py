#!/usr/bin/env python3
"""
Signal Stabilization Module
Reduces signal noise and prevents whipsaw trading through averaging and confidence filtering
"""

import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SignalHistory:
    """Track signal history for stability analysis."""
    symbol: str
    timestamp: datetime
    action: str
    confidence: float
    strength: float
    source: str  # 'finrl', 'sentiment', etc.
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class SignalStabilizer:
    """
    Stabilizes trading signals by averaging over time and filtering low-confidence signals.
    Prevents rapid signal changes that lead to churning.
    """
    
    def __init__(self, 
                 history_periods: int = 3,
                 min_confidence_threshold: float = 0.75,
                 signal_consistency_threshold: float = 0.8,
                 db_path: str = "data/signal_history.db"):
        
        self.history_periods = history_periods
        self.min_confidence = min_confidence_threshold
        self.consistency_threshold = signal_consistency_threshold
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # In-memory signal history
        self.signal_history: Dict[str, List[SignalHistory]] = {}
        
        # Initialize database
        self._init_database()
        self._load_recent_signals()
        
        logger.info(f"📊 Signal Stabilizer initialized")
        logger.info(f"   History periods: {history_periods}")
        logger.info(f"   Min confidence: {min_confidence_threshold}")
        logger.info(f"   Consistency threshold: {signal_consistency_threshold}")
    
    def _init_database(self):
        """Initialize SQLite database for signal tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_history (
                    symbol TEXT,
                    timestamp TEXT,
                    action TEXT,
                    confidence REAL,
                    strength REAL,
                    source TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
                ON signal_history(symbol, timestamp)
            """)
    
    def _load_recent_signals(self):
        """Load recent signals from database for continuity."""
        cutoff_time = datetime.now() - timedelta(hours=2)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT symbol, timestamp, action, confidence, strength, source
                FROM signal_history 
                WHERE datetime(timestamp) >= datetime(?)
                ORDER BY symbol, timestamp
            """, (cutoff_time.isoformat(),))
            
            for row in cursor.fetchall():
                symbol, timestamp_str, action, confidence, strength, source = row
                timestamp = datetime.fromisoformat(timestamp_str)
                
                if symbol not in self.signal_history:
                    self.signal_history[symbol] = []
                
                self.signal_history[symbol].append(SignalHistory(
                    symbol=symbol,
                    timestamp=timestamp,
                    action=action,
                    confidence=confidence,
                    strength=strength,
                    source=source
                ))
        
        # Trim to history_periods
        for symbol in self.signal_history:
            self.signal_history[symbol] = self.signal_history[symbol][-self.history_periods:]
        
        total_signals = sum(len(signals) for signals in self.signal_history.values())
        if total_signals > 0:
            logger.info(f"📈 Loaded {total_signals} recent signals for {len(self.signal_history)} symbols")
    
    def add_signal(self, 
                   symbol: str, 
                   action: str, 
                   confidence: float, 
                   strength: float = 1.0, 
                   source: str = "finrl"):
        """Add a new signal to the history."""
        
        signal = SignalHistory(
            symbol=symbol,
            timestamp=datetime.now(),
            action=action.upper(),
            confidence=confidence,
            strength=strength,
            source=source
        )
        
        # Add to in-memory history
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        
        self.signal_history[symbol].append(signal)
        
        # Keep only recent history
        self.signal_history[symbol] = self.signal_history[symbol][-self.history_periods:]
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO signal_history 
                (symbol, timestamp, action, confidence, strength, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, signal.timestamp.isoformat(), action, confidence, strength, source))
        
        logger.debug(f"📊 {symbol}: Added {action} signal (confidence: {confidence:.2f})")
    
    def get_stabilized_signal(self, symbol: str) -> Optional[Dict]:
        """
        Get stabilized signal for a symbol based on recent history.
        
        Returns:
            Dict with stabilized signal or None if signal not stable enough
        """
        if symbol not in self.signal_history:
            return None
            
        recent_signals = self.signal_history[symbol]
        
        # Bootstrap mode: Allow high-confidence single signals
        if len(recent_signals) == 1:
            single_signal = recent_signals[0]
            if single_signal.confidence >= 0.80:  # Lower threshold for single signals
                logger.info(f"📊 {symbol}: Bootstrap mode - single signal ({single_signal.confidence:.2f})")
                return {
                    "symbol": symbol,
                    "action": single_signal.action.lower(),  # Normalize to lowercase
                    "confidence": single_signal.confidence,
                    "strength": single_signal.strength,
                    "consistency": 1.0,
                    "history_count": 1,
                    "signal_age_minutes": 0,
                    "bootstrap_mode": True
                }
        
        # Need at least 2 signals for stability analysis
        if len(recent_signals) < 2:
            return None
        
        # Calculate signal consistency
        actions = [s.action for s in recent_signals]
        confidences = [s.confidence for s in recent_signals]
        strengths = [s.strength for s in recent_signals]
        
        # Check action consistency
        unique_actions = set(actions)
        if len(unique_actions) > 1:
            # Mixed signals - check if trend is emerging
            latest_action = actions[-1]
            latest_count = sum(1 for a in actions[-2:] if a == latest_action)
            if latest_count < 2:
                logger.debug(f"📊 {symbol}: Mixed signals, no clear trend")
                return None
        
        # Calculate averaged metrics
        avg_confidence = np.mean(confidences)
        avg_strength = np.mean(strengths)
        
        # Apply confidence filtering
        if avg_confidence < self.min_confidence:
            logger.debug(f"📊 {symbol}: Low confidence ({avg_confidence:.2f} < {self.min_confidence})")
            return None
        
        # Determine dominant action
        dominant_action = max(set(actions), key=actions.count)
        action_frequency = actions.count(dominant_action) / len(actions)
        
        if action_frequency < self.consistency_threshold:
            logger.debug(f"📊 {symbol}: Inconsistent action ({action_frequency:.2f} < {self.consistency_threshold})")
            return None
        
        # Calculate signal age (time since first signal in history)
        signal_age_minutes = (datetime.now() - recent_signals[0].timestamp).total_seconds() / 60
        
        stabilized_signal = {
            "symbol": symbol,
            "action": dominant_action.lower(),  # Normalize to lowercase
            "confidence": avg_confidence,
            "strength": avg_strength,
            "consistency": action_frequency,
            "history_count": len(recent_signals),
            "signal_age_minutes": signal_age_minutes,
            "is_stable": True
        }
        
        logger.info(f"✅ {symbol}: Stable {dominant_action} signal - confidence: {avg_confidence:.2f}, consistency: {action_frequency:.2f}")
        
        return stabilized_signal
    
    def filter_signals(self, raw_signals: List[Dict]) -> List[Dict]:
        """
        Filter a list of raw signals through stabilization process.
        
        Args:
            raw_signals: List of signal dictionaries
            
        Returns:
            List of stabilized signals that pass filtering criteria
        """
        # First, add all signals to history
        for signal in raw_signals:
            symbol = signal.get('symbol')
            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 0.5)
            strength = signal.get('strength', 1.0)
            source = signal.get('source', 'finrl')
            
            if symbol and action.upper() != 'HOLD':
                self.add_signal(symbol, action, confidence, strength, source)
        
        # Then get stabilized signals
        stabilized_signals = []
        processed_symbols = set()
        
        for signal in raw_signals:
            symbol = signal.get('symbol')
            if symbol in processed_symbols:
                continue
            
            stabilized = self.get_stabilized_signal(symbol)
            if stabilized:
                # Convert back to signal format
                stable_signal = {
                    "symbol": symbol,
                    "action": stabilized["action"],
                    "confidence": stabilized["confidence"],
                    "strength": stabilized["strength"],
                    "quantity": signal.get('quantity', 0),  # Preserve original quantity
                    "reasoning": f"Stabilized signal (consistency: {stabilized['consistency']:.2f})",
                    "stabilization_data": stabilized
                }
                stabilized_signals.append(stable_signal)
                processed_symbols.add(symbol)
        
        logger.info(f"📊 Signal Stabilization: {len(raw_signals)} raw → {len(stabilized_signals)} stable signals")
        
        return stabilized_signals
    
    def get_signal_stats(self, symbol: str = None) -> Dict:
        """Get signal statistics for analysis."""
        if symbol:
            if symbol not in self.signal_history:
                return {"symbol": symbol, "no_data": True}
            
            signals = self.signal_history[symbol]
            actions = [s.action for s in signals]
            confidences = [s.confidence for s in signals]
            
            return {
                "symbol": symbol,
                "signal_count": len(signals),
                "avg_confidence": round(np.mean(confidences), 2) if confidences else 0,
                "action_distribution": {action: actions.count(action) for action in set(actions)},
                "latest_action": actions[-1] if actions else None,
                "time_span_minutes": (signals[-1].timestamp - signals[0].timestamp).total_seconds() / 60 if len(signals) > 1 else 0
            }
        else:
            # Overall stats
            total_signals = sum(len(signals) for signals in self.signal_history.values())
            active_symbols = len(self.signal_history)
            
            all_confidences = []
            for signals in self.signal_history.values():
                all_confidences.extend([s.confidence for s in signals])
            
            return {
                "total_signals": total_signals,
                "active_symbols": active_symbols,
                "avg_confidence": round(np.mean(all_confidences), 2) if all_confidences else 0,
                "history_periods": self.history_periods,
                "min_confidence_threshold": self.min_confidence,
                "consistency_threshold": self.consistency_threshold
            }
    
    def clear_old_signals(self, hours: int = 24):
        """Clean up old signals from database."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM signal_history 
                WHERE datetime(timestamp) < datetime(?)
            """, (cutoff_time.isoformat(),))
            
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} old signals (older than {hours} hours)")

# Global instance
signal_stabilizer = SignalStabilizer()

__all__ = ['SignalStabilizer', 'SignalHistory', 'signal_stabilizer']