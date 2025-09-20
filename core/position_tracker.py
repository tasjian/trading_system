#!/usr/bin/env python3
"""
Position Stability Tracker
Prevents excessive churning by enforcing minimum holding periods and signal confidence
"""

import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PositionEntry:
    """Track when a position was entered and at what price."""
    symbol: str
    entry_time: datetime
    entry_price: float
    quantity: float
    entry_type: str  # 'long', 'short'
    confidence: float
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['entry_time'] = self.entry_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PositionEntry':
        data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        return cls(**data)

class PositionStabilityTracker:
    """
    Tracks position entry/exit times to prevent churning and enforce holding periods.
    """
    
    def __init__(self, 
                 min_holding_period_minutes: int = 30,
                 db_path: str = "data/position_stability.db"):
        self.min_holding_period = timedelta(minutes=min_holding_period_minutes)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # In-memory tracking for current session
        self.position_entries: Dict[str, PositionEntry] = {}
        self.recently_exited: Dict[str, datetime] = {}
        
        # Initialize database
        self._init_database()
        self._load_current_positions()
        
        logger.info(f"🕒 Position Stability Tracker initialized")
        logger.info(f"   Minimum holding period: {min_holding_period_minutes} minutes")
        logger.info(f"   Database: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database for position tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS position_entries (
                    symbol TEXT,
                    entry_time TEXT,
                    exit_time TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    entry_type TEXT,
                    confidence REAL,
                    holding_period_minutes REAL,
                    profit_loss REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_entry_time 
                ON position_entries(symbol, entry_time)
            """)
    
    def _load_current_positions(self):
        """Load any existing open positions from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT symbol, entry_time, entry_price, quantity, entry_type, confidence
                FROM position_entries 
                WHERE exit_time IS NULL
            """)
            
            for row in cursor.fetchall():
                symbol, entry_time_str, entry_price, quantity, entry_type, confidence = row
                entry_time = datetime.fromisoformat(entry_time_str)
                
                self.position_entries[symbol] = PositionEntry(
                    symbol=symbol,
                    entry_time=entry_time,
                    entry_price=entry_price,
                    quantity=quantity,
                    entry_type=entry_type,
                    confidence=confidence
                )
        
        if self.position_entries:
            logger.info(f"📊 Loaded {len(self.position_entries)} existing positions from database")
    
    def can_exit_position(self, symbol: str, current_price: float = None) -> Tuple[bool, str]:
        """
        Check if a position can be exited based on holding period and other criteria.
        
        Returns:
            (can_exit, reason)
        """
        if symbol not in self.position_entries:
            return True, "No position to exit"
        
        entry = self.position_entries[symbol]
        time_held = datetime.now() - entry.entry_time
        
        # Check minimum holding period
        if time_held < self.min_holding_period:
            remaining = self.min_holding_period - time_held
            remaining_minutes = remaining.total_seconds() / 60
            return False, f"Minimum holding period not met (need {remaining_minutes:.1f} more minutes)"
        
        # Check for recent exit (prevent immediate re-entry)
        if symbol in self.recently_exited:
            last_exit = self.recently_exited[symbol]
            if datetime.now() - last_exit < timedelta(minutes=10):
                return False, "Recently exited position, cooling off period active"
        
        # Additional checks can be added here (stop-loss, take-profit, etc.)
        
        return True, "Position can be exited"
    
    def can_enter_position(self, symbol: str, entry_type: str = "long") -> Tuple[bool, str]:
        """
        Check if a new position can be entered.
        
        Args:
            symbol: Stock symbol
            entry_type: 'long' or 'short'
        
        Returns:
            (can_enter, reason)
        """
        # Check if position already exists
        if symbol in self.position_entries:
            existing_type = self.position_entries[symbol].entry_type
            if existing_type == entry_type:
                return False, f"Already have {entry_type} position in {symbol}"
            # Allow opposite direction (long vs short)
        
        # Check for recent exit
        if symbol in self.recently_exited:
            last_exit = self.recently_exited[symbol]
            if datetime.now() - last_exit < timedelta(minutes=5):
                return False, "Recently exited, cooling off period active"
        
        return True, "Position entry allowed"
    
    def record_position_entry(self, 
                            symbol: str, 
                            entry_price: float, 
                            quantity: float, 
                            entry_type: str = "long",
                            confidence: float = 0.5):
        """Record a new position entry."""
        
        # Create position entry
        entry = PositionEntry(
            symbol=symbol,
            entry_time=datetime.now(),
            entry_price=entry_price,
            quantity=quantity,
            entry_type=entry_type,
            confidence=confidence
        )
        
        self.position_entries[symbol] = entry
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO position_entries 
                (symbol, entry_time, entry_price, quantity, entry_type, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, entry.entry_time.isoformat(), entry_price, quantity, entry_type, confidence))
        
        # Remove from recently exited
        self.recently_exited.pop(symbol, None)
        
        logger.info(f"📈 {symbol}: Recorded {entry_type} position entry - {quantity} shares @ ${entry_price:.2f}")
    
    def record_position_exit(self, 
                           symbol: str, 
                           exit_price: float, 
                           exit_reason: str = "rebalance"):
        """Record position exit and calculate metrics."""
        
        if symbol not in self.position_entries:
            logger.warning(f"⚠️ Attempted to exit non-existent position: {symbol}")
            return
        
        entry = self.position_entries[symbol]
        exit_time = datetime.now()
        holding_period = exit_time - entry.entry_time
        holding_minutes = holding_period.total_seconds() / 60
        
        # Calculate P&L
        if entry.entry_type == "long":
            profit_loss = (exit_price - entry.entry_price) * entry.quantity
        else:  # short
            profit_loss = (entry.entry_price - exit_price) * entry.quantity
        
        # Update database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE position_entries 
                SET exit_time = ?, exit_price = ?, holding_period_minutes = ?, profit_loss = ?
                WHERE symbol = ? AND exit_time IS NULL
            """, (exit_time.isoformat(), exit_price, holding_minutes, profit_loss, symbol))
        
        # Move to recently exited
        self.recently_exited[symbol] = exit_time
        del self.position_entries[symbol]
        
        logger.info(f"📉 {symbol}: Recorded position exit - held {holding_minutes:.1f} min, P&L: ${profit_loss:.2f}")
    
    def get_position_metrics(self, symbol: str) -> Optional[Dict]:
        """Get current position metrics."""
        if symbol not in self.position_entries:
            return None
        
        entry = self.position_entries[symbol]
        time_held = datetime.now() - entry.entry_time
        
        return {
            "symbol": symbol,
            "entry_time": entry.entry_time.isoformat(),
            "entry_price": entry.entry_price,
            "quantity": entry.quantity,
            "entry_type": entry.entry_type,
            "confidence": entry.confidence,
            "minutes_held": time_held.total_seconds() / 60,
            "can_exit": self.can_exit_position(symbol)[0]
        }
    
    def get_trading_stats(self, days: int = 7) -> Dict:
        """Get trading statistics for recent period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Completed trades
            cursor = conn.execute("""
                SELECT COUNT(*), AVG(holding_period_minutes), SUM(profit_loss), 
                       AVG(profit_loss), COUNT(CASE WHEN profit_loss > 0 THEN 1 END)
                FROM position_entries 
                WHERE exit_time IS NOT NULL 
                AND datetime(entry_time) >= datetime(?)
            """, (cutoff_date.isoformat(),))
            
            row = cursor.fetchone()
            total_trades, avg_holding, total_pnl, avg_pnl, winning_trades = row
            
            return {
                "period_days": days,
                "total_trades": total_trades or 0,
                "average_holding_minutes": round(avg_holding or 0, 1),
                "total_pnl": round(total_pnl or 0, 2),
                "average_pnl_per_trade": round(avg_pnl or 0, 2),
                "winning_trades": winning_trades or 0,
                "win_rate": round((winning_trades or 0) / max(total_trades or 1, 1) * 100, 1),
                "current_open_positions": len(self.position_entries)
            }

# Global instance
position_tracker = PositionStabilityTracker()

__all__ = ['PositionStabilityTracker', 'PositionEntry', 'position_tracker']