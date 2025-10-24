#!/usr/bin/env python3
"""
Cost-Aware Trading Logic
Prevents unprofitable trades by analyzing transaction costs vs expected returns
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TransactionCost:
    """Transaction cost breakdown for a trade."""
    symbol: str
    quantity: float
    price: float
    notional_value: float
    spread_cost: float
    commission_cost: float
    total_cost: float
    cost_percentage: float

@dataclass
class TradeAnalysis:
    """Analysis of trade profitability vs costs."""
    symbol: str
    action: str
    current_price: float
    target_price: float
    quantity: float
    expected_return: float
    transaction_cost: float
    net_expected_return: float
    is_profitable: bool
    profit_margin: float
    reasoning: str

class CostAwareRebalancer:
    """
    Analyzes transaction costs vs expected returns to prevent unprofitable trades.
    """
    
    def __init__(self, 
                 commission_per_trade: float = 0.0,  # Alpaca is commission-free
                 typical_spread_pct: float = 0.002,  # 0.2% typical bid-ask spread
                 min_profit_threshold: float = 0.005,  # 0.5% minimum profit to justify trade
                 min_position_value: float = 100.0,  # Minimum $100 position
                 db_path: str = "data/transaction_costs.db"):
        
        self.commission_per_trade = commission_per_trade
        self.typical_spread_pct = typical_spread_pct
        self.min_profit_threshold = min_profit_threshold
        self.min_position_value = min_position_value
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"💰 Cost-Aware Rebalancer initialized")
        logger.info(f"   Commission per trade: ${commission_per_trade:.2f}")
        logger.info(f"   Typical spread: {typical_spread_pct:.1%}")
        logger.info(f"   Min profit threshold: {min_profit_threshold:.1%}")
        logger.info(f"   Min position value: ${min_position_value:.2f}")
    
    def _init_database(self):
        """Initialize SQLite database for transaction cost tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transaction_costs (
                    symbol TEXT,
                    timestamp TEXT,
                    action TEXT,
                    quantity REAL,
                    price REAL,
                    notional_value REAL,
                    spread_cost REAL,
                    commission_cost REAL,
                    total_cost REAL,
                    cost_percentage REAL,
                    expected_return REAL,
                    net_return REAL,
                    was_profitable BOOLEAN,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
                ON transaction_costs(symbol, timestamp)
            """)
    
    def calculate_transaction_cost(self, 
                                 symbol: str, 
                                 quantity: float, 
                                 price: float,
                                 action: str = "buy") -> TransactionCost:
        """Calculate detailed transaction cost for a trade."""
        
        notional_value = abs(quantity) * price
        
        # Spread cost - typically paid on both entry and exit
        # For new positions: pay half spread on entry, expect to pay half on exit
        # For exit positions: pay half spread on exit (already paid on entry)
        if action.lower() in ["buy", "short"]:
            spread_cost = notional_value * self.typical_spread_pct  # Full round-trip cost
        else:  # sell, cover
            spread_cost = notional_value * (self.typical_spread_pct / 2)  # Exit cost only
        
        # Commission cost
        commission_cost = self.commission_per_trade
        
        # Total cost
        total_cost = spread_cost + commission_cost
        cost_percentage = total_cost / notional_value if notional_value > 0 else 0
        
        return TransactionCost(
            symbol=symbol,
            quantity=quantity,
            price=price,
            notional_value=notional_value,
            spread_cost=spread_cost,
            commission_cost=commission_cost,
            total_cost=total_cost,
            cost_percentage=cost_percentage
        )
    
    def estimate_spread_cost(self, symbol: str, current_price: float) -> float:
        """Estimate bid-ask spread cost based on symbol characteristics."""
        
        # Dynamic spread estimation based on price and volatility
        if current_price < 5.0:
            # Penny stocks have wider spreads
            estimated_spread_pct = 0.01  # 1%
        elif current_price < 20.0:
            # Small-cap stocks
            estimated_spread_pct = 0.005  # 0.5%
        elif current_price < 100.0:
            # Mid-cap stocks
            estimated_spread_pct = 0.002  # 0.2%
        else:
            # Large-cap stocks
            estimated_spread_pct = 0.001  # 0.1%
        
        # Adjust for market hours (spreads widen after hours)
        now = datetime.now()
        if now.hour < 9 or now.hour >= 16:  # Rough market hours check
            estimated_spread_pct *= 2  # Double spread after hours
        
        return estimated_spread_pct
    
    def analyze_trade_profitability(self, 
                                  symbol: str, 
                                  action: str, 
                                  quantity: float, 
                                  current_price: float,
                                  expected_return_pct: float = None,
                                  target_price: float = None) -> TradeAnalysis:
        """
        Analyze if a trade is likely to be profitable after transaction costs.
        
        Args:
            symbol: Stock symbol
            action: 'buy', 'sell', 'short', 'cover'
            quantity: Number of shares
            current_price: Current market price
            expected_return_pct: Expected return percentage (optional)
            target_price: Target price (alternative to expected_return_pct)
        
        Returns:
            TradeAnalysis with profitability assessment
        """
        
        # Estimate transaction cost
        cost = self.calculate_transaction_cost(symbol, quantity, current_price, action)
        
        # Calculate expected return
        if target_price is not None:
            if action.lower() in ["buy", "long"]:
                expected_return_pct = (target_price - current_price) / current_price
            elif action.lower() in ["short", "sell_short"]:
                expected_return_pct = (current_price - target_price) / current_price
            else:  # sell, cover
                expected_return_pct = 0  # Already determined by market
        elif expected_return_pct is None:
            # No expected return provided - use minimum threshold
            expected_return_pct = self.min_profit_threshold
        
        expected_return = cost.notional_value * expected_return_pct
        net_expected_return = expected_return - cost.total_cost
        
        # Profitability assessment
        is_profitable = net_expected_return > 0
        
        # Calculate profit margin above transaction costs
        profit_margin = (expected_return - cost.total_cost) / cost.notional_value if cost.notional_value > 0 else 0
        
        # Determine reasoning
        if not is_profitable:
            if expected_return <= 0:
                reasoning = f"Expected return ({expected_return_pct:.1%}) is negative"
            else:
                reasoning = f"Transaction cost ({cost.cost_percentage:.1%}) exceeds expected return ({expected_return_pct:.1%})"
        elif profit_margin < self.min_profit_threshold:
            reasoning = f"Profit margin ({profit_margin:.1%}) below minimum threshold ({self.min_profit_threshold:.1%})"
            is_profitable = False
        else:
            reasoning = f"Profitable: {profit_margin:.1%} margin after {cost.cost_percentage:.1%} transaction cost"
        
        return TradeAnalysis(
            symbol=symbol,
            action=action,
            current_price=current_price,
            target_price=target_price or current_price * (1 + expected_return_pct),
            quantity=quantity,
            expected_return=expected_return,
            transaction_cost=cost.total_cost,
            net_expected_return=net_expected_return,
            is_profitable=is_profitable,
            profit_margin=profit_margin,
            reasoning=reasoning
        )
    
    def filter_profitable_trades(self, 
                                signals: List[Dict],
                                current_prices: Dict[str, float] = None) -> List[Dict]:
        """
        Filter signals to only include profitable trades.
        
        Args:
            signals: List of trading signals
            current_prices: Dict of symbol -> current price (optional)
            
        Returns:
            List of profitable signals
        """
        profitable_signals = []
        cost_analyses = []
        
        for signal in signals:
            symbol = signal.get("symbol")
            action = signal.get("action", "buy").lower()
            quantity = signal.get("quantity", 0)
            confidence = signal.get("confidence", 0.5)
            
            if not symbol or quantity <= 0:
                continue
            
            # Get current price
            if current_prices and symbol in current_prices:
                current_price = current_prices[symbol]
            else:
                # Try to get price from signal or use a default
                current_price = signal.get("price", 100.0)  # Default for testing
            
            # Skip very small positions
            notional_value = quantity * current_price
            if notional_value < self.min_position_value:
                logger.debug(f"💰 {symbol}: Skipping small position (${notional_value:.2f} < ${self.min_position_value})")
                continue
            
            # Estimate expected return based on confidence
            # Higher confidence signals get higher expected return estimates
            if action in ["buy", "long"]:
                expected_return_pct = confidence * 0.02  # 0-2% based on confidence
            elif action in ["short", "sell_short"]:
                expected_return_pct = confidence * 0.015  # 0-1.5% for shorts
            else:  # sell, cover - assume current market return
                expected_return_pct = 0.001  # Small positive for exits
            
            # Analyze profitability
            analysis = self.analyze_trade_profitability(
                symbol, action, quantity, current_price, expected_return_pct
            )
            
            cost_analyses.append(analysis)
            
            if analysis.is_profitable:
                profitable_signals.append(signal)
                logger.info(f"💰 {symbol}: {action.upper()} approved - {analysis.reasoning}")
            else:
                logger.info(f"💸 {symbol}: {action.upper()} rejected - {analysis.reasoning}")
        
        # Log summary
        total_signals = len(signals)
        profitable_count = len(profitable_signals)
        total_expected_costs = sum(a.transaction_cost for a in cost_analyses)
        total_expected_returns = sum(a.net_expected_return for a in cost_analyses if a.is_profitable)
        
        logger.info(f"💰 Cost Analysis Summary:")
        logger.info(f"   Signals: {profitable_count}/{total_signals} profitable")
        logger.info(f"   Total transaction costs: ${total_expected_costs:.2f}")
        logger.info(f"   Expected net returns: ${total_expected_returns:.2f}")
        
        # Store analysis in database
        self._store_cost_analysis(cost_analyses)
        
        return profitable_signals
    
    def _store_cost_analysis(self, analyses: List[TradeAnalysis]):
        """Store cost analysis results in database."""
        with sqlite3.connect(self.db_path) as conn:
            for analysis in analyses:
                conn.execute("""
                    INSERT INTO transaction_costs 
                    (symbol, timestamp, action, quantity, price, notional_value,
                     spread_cost, commission_cost, total_cost, cost_percentage,
                     expected_return, net_return, was_profitable)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis.symbol,
                    datetime.now().isoformat(),
                    analysis.action,
                    analysis.quantity,
                    analysis.current_price,
                    analysis.quantity * analysis.current_price,
                    analysis.transaction_cost * 0.8,  # Estimate spread portion
                    analysis.transaction_cost * 0.2,  # Estimate commission portion
                    analysis.transaction_cost,
                    (analysis.transaction_cost / (analysis.quantity * analysis.current_price)),
                    analysis.expected_return,
                    analysis.net_expected_return,
                    analysis.is_profitable
                ))
    
    def get_cost_statistics(self, days: int = 7) -> Dict:
        """Get transaction cost statistics for analysis."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*), AVG(cost_percentage), SUM(total_cost),
                       SUM(expected_return), SUM(net_return),
                       COUNT(CASE WHEN was_profitable = 1 THEN 1 END)
                FROM transaction_costs 
                WHERE datetime(timestamp) >= datetime(?)
            """, (cutoff_date.isoformat(),))
            
            row = cursor.fetchone()
            total_trades, avg_cost_pct, total_costs, total_returns, total_net, profitable_trades = row
            
            return {
                "period_days": days,
                "total_analyses": total_trades or 0,
                "average_cost_percentage": round(avg_cost_pct or 0, 4),
                "total_transaction_costs": round(total_costs or 0, 2),
                "total_expected_returns": round(total_returns or 0, 2),
                "total_net_returns": round(total_net or 0, 2),
                "profitable_trades": profitable_trades or 0,
                "profitability_rate": round((profitable_trades or 0) / max(total_trades or 1, 1) * 100, 1),
                "cost_efficiency": round((total_net or 0) / max(total_costs or 1, 1), 2)
            }

# Global instance
cost_aware_rebalancer = CostAwareRebalancer()

__all__ = ['CostAwareRebalancer', 'TransactionCost', 'TradeAnalysis', 'cost_aware_rebalancer']