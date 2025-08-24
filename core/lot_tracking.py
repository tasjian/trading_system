#!/usr/bin/env python3
"""
Specific Lot Tracking System

A comprehensive lot-level position tracking system that maintains detailed records
of individual purchase lots for precise cost basis calculation, tax lot management,
and specific identification for tax-loss harvesting.

Key Features:
- FIFO, HIFO, and Specific Identification lot accounting methods
- Real-time cost basis calculation and adjustment
- Detailed transaction history and audit trail
- Integration with wash sale monitoring
- Support for stock splits, dividends, and corporate actions
- Persistent storage with data integrity validation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Literal, Union
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import json
import uuid
import sqlite3
from pathlib import Path
import pandas as pd

from tools.alpaca_client import alpaca_client
from config.settings import settings

logger = logging.getLogger(__name__)

class LotAccounting(Enum):
    """Lot accounting methods for cost basis calculation."""
    FIFO = "fifo"                    # First In, First Out
    LIFO = "lifo"                    # Last In, First Out  
    HIFO = "hifo"                    # Highest In, First Out (tax optimization)
    LOFO = "lofo"                    # Lowest In, First Out
    SPECIFIC_ID = "specific_id"      # Specific Identification
    AVERAGE_COST = "average_cost"    # Average Cost (for mutual funds)

class TransactionType(Enum):
    """Types of transactions affecting lots."""
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"
    SPLIT = "split"
    SPINOFF = "spinoff"
    MERGER = "merger"
    RETURN_OF_CAPITAL = "return_of_capital"

@dataclass
class TaxLot:
    """Represents a specific tax lot (purchase) of a security."""
    lot_id: str
    symbol: str
    purchase_date: datetime
    quantity: Decimal
    cost_basis_per_share: Decimal
    total_cost_basis: Decimal
    current_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    
    # Transaction details
    order_id: Optional[str] = None
    commission_fees: Decimal = Decimal("0.00")
    other_fees: Decimal = Decimal("0.00")
    
    # Corporate actions tracking
    split_adjusted: bool = False
    dividend_adjustments: Decimal = Decimal("0.00")
    
    # Tax classification
    is_long_term: bool = field(init=False)
    holding_period_days: int = field(init=False)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.holding_period_days = (datetime.now() - self.purchase_date).days
        self.is_long_term = self.holding_period_days >= 365
        
        # Ensure decimal precision
        self.quantity = self.quantity.quantize(Decimal('0.000001'))
        self.cost_basis_per_share = self.cost_basis_per_share.quantize(Decimal('0.01'))
        self.total_cost_basis = self.total_cost_basis.quantize(Decimal('0.01'))
    
    def update_market_value(self, current_price: Decimal):
        """Update current market value and unrealized P&L."""
        self.current_price = current_price.quantize(Decimal('0.01'))
        current_value = self.quantity * self.current_price
        self.unrealized_pnl = current_value - self.total_cost_basis
        self.updated_at = datetime.now()

@dataclass
class LotTransaction:
    """Records a transaction affecting tax lots."""
    transaction_id: str
    symbol: str
    transaction_type: TransactionType
    transaction_date: datetime
    quantity: Decimal
    price_per_share: Optional[Decimal] = None
    
    # For sales: which lots were used
    lots_affected: List[str] = field(default_factory=list)
    realized_pnl: Optional[Decimal] = None
    
    # Fees and adjustments
    commission_fees: Decimal = Decimal("0.00")
    other_fees: Decimal = Decimal("0.00")
    
    # External references
    order_id: Optional[str] = None
    alpaca_order_id: Optional[str] = None
    
    # Audit trail
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PositionSummary:
    """Summary of all lots for a symbol."""
    symbol: str
    total_quantity: Decimal
    total_cost_basis: Decimal
    average_cost_basis: Decimal
    current_market_value: Optional[Decimal] = None
    total_unrealized_pnl: Optional[Decimal] = None
    
    # Lot breakdown
    lot_count: int = 0
    lots: List[TaxLot] = field(default_factory=list)
    
    # Tax implications
    short_term_quantity: Decimal = Decimal("0")
    long_term_quantity: Decimal = Decimal("0")
    short_term_cost_basis: Decimal = Decimal("0")
    long_term_cost_basis: Decimal = Decimal("0")
    
    # Performance metrics
    oldest_lot_date: Optional[datetime] = None
    newest_lot_date: Optional[datetime] = None
    weighted_holding_period: int = 0

class LotTrackingDatabase:
    """SQLite database for persistent lot tracking storage."""
    
    def __init__(self, db_path: str = "data/lot_tracking.db"):
        """Initialize database connection and create tables."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Tax lots table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS tax_lots (
                        lot_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        purchase_date TIMESTAMP NOT NULL,
                        quantity DECIMAL(15,6) NOT NULL,
                        cost_basis_per_share DECIMAL(10,2) NOT NULL,
                        total_cost_basis DECIMAL(15,2) NOT NULL,
                        current_price DECIMAL(10,2),
                        unrealized_pnl DECIMAL(15,2),
                        order_id TEXT,
                        commission_fees DECIMAL(10,2) DEFAULT 0.00,
                        other_fees DECIMAL(10,2) DEFAULT 0.00,
                        split_adjusted BOOLEAN DEFAULT FALSE,
                        dividend_adjustments DECIMAL(10,2) DEFAULT 0.00,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for tax_lots table
                conn.execute('CREATE INDEX IF NOT EXISTS idx_tax_lots_symbol ON tax_lots(symbol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_tax_lots_purchase_date ON tax_lots(purchase_date)')
                
                # Lot transactions table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS lot_transactions (
                        transaction_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        transaction_type TEXT NOT NULL,
                        transaction_date TIMESTAMP NOT NULL,
                        quantity DECIMAL(15,6) NOT NULL,
                        price_per_share DECIMAL(10,2),
                        lots_affected TEXT,  -- JSON array of lot IDs
                        realized_pnl DECIMAL(15,2),
                        commission_fees DECIMAL(10,2) DEFAULT 0.00,
                        other_fees DECIMAL(10,2) DEFAULT 0.00,
                        order_id TEXT,
                        alpaca_order_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for lot_transactions table
                conn.execute('CREATE INDEX IF NOT EXISTS idx_lot_transactions_symbol ON lot_transactions(symbol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_lot_transactions_date ON lot_transactions(transaction_date)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_lot_transactions_type ON lot_transactions(transaction_type)')
                
                # Position summaries cache table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS position_summaries (
                        symbol TEXT PRIMARY KEY,
                        total_quantity DECIMAL(15,6),
                        total_cost_basis DECIMAL(15,2),
                        average_cost_basis DECIMAL(10,2),
                        current_market_value DECIMAL(15,2),
                        total_unrealized_pnl DECIMAL(15,2),
                        lot_count INTEGER,
                        short_term_quantity DECIMAL(15,6),
                        long_term_quantity DECIMAL(15,6),
                        oldest_lot_date TIMESTAMP,
                        newest_lot_date TIMESTAMP,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create index for position_summaries table
                conn.execute('CREATE INDEX IF NOT EXISTS idx_position_summaries_symbol ON position_summaries(symbol)')
                
                conn.commit()
                logger.info("Lot tracking database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing lot tracking database: {e}")
            raise

class SpecificLotTracker:
    """
    Comprehensive specific lot tracking system for tax-efficient portfolio management.
    
    This system maintains detailed records of every purchase (tax lot) and provides
    sophisticated cost basis calculation using various accounting methods including
    specific identification for optimal tax-loss harvesting.
    """
    
    def __init__(self, accounting_method: LotAccounting = LotAccounting.HIFO):
        """Initialize the lot tracker."""
        self.accounting_method = accounting_method
        self.db = LotTrackingDatabase()
        
        # In-memory cache for performance
        self.lots_cache: Dict[str, List[TaxLot]] = {}
        self.position_summaries: Dict[str, PositionSummary] = {}
        
        # Cache management
        self.cache_expiry_minutes = 5
        self.last_cache_update: Dict[str, datetime] = {}
        
        logger.info(f"Specific Lot Tracker initialized with {accounting_method.value} accounting")
    
    async def record_purchase(self, 
                            symbol: str, 
                            quantity: Decimal, 
                            price_per_share: Decimal,
                            transaction_date: Optional[datetime] = None,
                            order_id: Optional[str] = None,
                            commission_fees: Decimal = Decimal("0.00")) -> str:
        """
        Record a new purchase creating a new tax lot.
        
        Args:
            symbol: Stock symbol
            quantity: Shares purchased
            price_per_share: Purchase price per share
            transaction_date: Purchase date (defaults to now)
            order_id: Associated order ID
            commission_fees: Commission and fees
            
        Returns:
            Lot ID of the created tax lot
        """
        try:
            transaction_date = transaction_date or datetime.now()
            lot_id = f"{symbol}_{transaction_date.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Calculate total cost including fees
            total_cost = (quantity * price_per_share) + commission_fees
            
            # Create new tax lot
            tax_lot = TaxLot(
                lot_id=lot_id,
                symbol=symbol,
                purchase_date=transaction_date,
                quantity=quantity,
                cost_basis_per_share=price_per_share,
                total_cost_basis=total_cost,
                order_id=order_id,
                commission_fees=commission_fees
            )
            
            # Store in database
            await self._store_lot(tax_lot)
            
            # Record transaction
            transaction = LotTransaction(
                transaction_id=f"BUY_{lot_id}",
                symbol=symbol,
                transaction_type=TransactionType.BUY,
                transaction_date=transaction_date,
                quantity=quantity,
                price_per_share=price_per_share,
                lots_affected=[lot_id],
                commission_fees=commission_fees,
                order_id=order_id
            )
            
            await self._store_transaction(transaction)
            
            # Update cache
            self._invalidate_cache(symbol)
            
            logger.info(f"✅ Recorded purchase: {quantity} shares of {symbol} @ ${price_per_share} (Lot: {lot_id})")
            return lot_id
            
        except Exception as e:
            logger.error(f"Error recording purchase for {symbol}: {e}")
            raise
    
    async def record_sale(self, 
                        symbol: str, 
                        quantity: Decimal,
                        price_per_share: Decimal,
                        transaction_date: Optional[datetime] = None,
                        lot_selection_method: Optional[LotAccounting] = None,
                        specific_lot_ids: Optional[List[str]] = None,
                        order_id: Optional[str] = None,
                        commission_fees: Decimal = Decimal("0.00")) -> Tuple[List[str], Decimal]:
        """
        Record a sale and determine which lots are sold based on accounting method.
        
        Args:
            symbol: Stock symbol
            quantity: Shares sold
            price_per_share: Sale price per share
            transaction_date: Sale date
            lot_selection_method: Override default accounting method
            specific_lot_ids: Specific lots to sell (for SPECIFIC_ID method)
            order_id: Associated order ID
            commission_fees: Commission and fees
            
        Returns:
            Tuple of (lot_ids_sold, realized_pnl)
        """
        try:
            transaction_date = transaction_date or datetime.now()
            selection_method = lot_selection_method or self.accounting_method
            
            # Get available lots for this symbol
            available_lots = await self.get_lots_by_symbol(symbol)
            if not available_lots:
                raise ValueError(f"No lots available for {symbol}")
            
            # Calculate total available quantity
            total_available = sum(lot.quantity for lot in available_lots)
            if quantity > total_available:
                raise ValueError(f"Insufficient shares: requested {quantity}, available {total_available}")
            
            # Select lots to sell based on accounting method
            if selection_method == LotAccounting.SPECIFIC_ID and specific_lot_ids:
                lots_to_sell = [lot for lot in available_lots if lot.lot_id in specific_lot_ids]
            else:
                lots_to_sell = self._select_lots_for_sale(available_lots, quantity, selection_method)
            
            # Execute the sale
            lots_sold, total_realized_pnl = await self._execute_lot_sale(
                lots_to_sell, quantity, price_per_share, transaction_date, commission_fees
            )
            
            # Record transaction
            transaction = LotTransaction(
                transaction_id=f"SELL_{symbol}_{transaction_date.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                transaction_type=TransactionType.SELL,
                transaction_date=transaction_date,
                quantity=quantity,
                price_per_share=price_per_share,
                lots_affected=[lot.lot_id for lot in lots_sold],
                realized_pnl=total_realized_pnl,
                commission_fees=commission_fees,
                order_id=order_id
            )
            
            await self._store_transaction(transaction)
            
            # Update cache
            self._invalidate_cache(symbol)
            
            lot_ids_sold = [lot.lot_id for lot in lots_sold]
            logger.info(f"✅ Recorded sale: {quantity} shares of {symbol} @ ${price_per_share}")
            logger.info(f"   Lots sold: {len(lot_ids_sold)}, Realized P&L: ${total_realized_pnl:,.2f}")
            
            return lot_ids_sold, total_realized_pnl
            
        except Exception as e:
            logger.error(f"Error recording sale for {symbol}: {e}")
            raise
    
    async def get_lots_by_symbol(self, symbol: str, refresh_cache: bool = False) -> List[TaxLot]:
        """Get all tax lots for a specific symbol."""
        try:
            # Check cache first
            if not refresh_cache and symbol in self.lots_cache:
                if self._is_cache_valid(symbol):
                    return self.lots_cache[symbol]
            
            # Load from database
            with sqlite3.connect(self.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM tax_lots 
                    WHERE symbol = ? AND quantity > 0 
                    ORDER BY purchase_date
                ''', (symbol,))
                
                lots = []
                for row in cursor:
                    lot = TaxLot(
                        lot_id=row['lot_id'],
                        symbol=row['symbol'],
                        purchase_date=datetime.fromisoformat(row['purchase_date']),
                        quantity=Decimal(str(row['quantity'])),
                        cost_basis_per_share=Decimal(str(row['cost_basis_per_share'])),
                        total_cost_basis=Decimal(str(row['total_cost_basis'])),
                        current_price=Decimal(str(row['current_price'])) if row['current_price'] else None,
                        unrealized_pnl=Decimal(str(row['unrealized_pnl'])) if row['unrealized_pnl'] else None,
                        order_id=row['order_id'],
                        commission_fees=Decimal(str(row['commission_fees'])),
                        other_fees=Decimal(str(row['other_fees'])),
                        split_adjusted=bool(row['split_adjusted']),
                        dividend_adjustments=Decimal(str(row['dividend_adjustments'])),
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at'])
                    )
                    lots.append(lot)
                
                # Update cache
                self.lots_cache[symbol] = lots
                self.last_cache_update[symbol] = datetime.now()
                
                return lots
                
        except Exception as e:
            logger.error(f"Error getting lots for {symbol}: {e}")
            return []
    
    async def get_position_summary(self, symbol: str, update_market_prices: bool = True) -> PositionSummary:
        """Get comprehensive position summary for a symbol."""
        try:
            lots = await self.get_lots_by_symbol(symbol)
            if not lots:
                return PositionSummary(
                    symbol=symbol,
                    total_quantity=Decimal("0"),
                    total_cost_basis=Decimal("0"),
                    average_cost_basis=Decimal("0")
                )
            
            # Update market prices if requested
            if update_market_prices:
                current_price = await self._get_current_price(symbol)
                if current_price:
                    for lot in lots:
                        lot.update_market_value(current_price)
            
            # Calculate summary metrics
            total_quantity = sum(lot.quantity for lot in lots)
            total_cost_basis = sum(lot.total_cost_basis for lot in lots)
            average_cost_basis = total_cost_basis / total_quantity if total_quantity > 0 else Decimal("0")
            
            # Current market value and unrealized P&L
            current_market_value = None
            total_unrealized_pnl = None
            
            if lots[0].current_price is not None:
                current_market_value = sum(lot.quantity * lot.current_price for lot in lots)
                total_unrealized_pnl = current_market_value - total_cost_basis
            
            # Tax classification
            short_term_quantity = Decimal("0")
            long_term_quantity = Decimal("0")
            short_term_cost_basis = Decimal("0")
            long_term_cost_basis = Decimal("0")
            
            for lot in lots:
                if lot.is_long_term:
                    long_term_quantity += lot.quantity
                    long_term_cost_basis += lot.total_cost_basis
                else:
                    short_term_quantity += lot.quantity
                    short_term_cost_basis += lot.total_cost_basis
            
            # Date range and weighted holding period
            purchase_dates = [lot.purchase_date for lot in lots]
            oldest_date = min(purchase_dates)
            newest_date = max(purchase_dates)
            
            # Calculate weighted average holding period
            total_value_days = sum(lot.total_cost_basis * lot.holding_period_days for lot in lots)
            weighted_holding_period = int(total_value_days / total_cost_basis) if total_cost_basis > 0 else 0
            
            summary = PositionSummary(
                symbol=symbol,
                total_quantity=total_quantity,
                total_cost_basis=total_cost_basis,
                average_cost_basis=average_cost_basis,
                current_market_value=current_market_value,
                total_unrealized_pnl=total_unrealized_pnl,
                lot_count=len(lots),
                lots=lots,
                short_term_quantity=short_term_quantity,
                long_term_quantity=long_term_quantity,
                short_term_cost_basis=short_term_cost_basis,
                long_term_cost_basis=long_term_cost_basis,
                oldest_lot_date=oldest_date,
                newest_lot_date=newest_date,
                weighted_holding_period=weighted_holding_period
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating position summary for {symbol}: {e}")
            return PositionSummary(
                symbol=symbol,
                total_quantity=Decimal("0"),
                total_cost_basis=Decimal("0"),
                average_cost_basis=Decimal("0")
            )
    
    async def calculate_tax_impact(self, 
                                 symbol: str, 
                                 sale_quantity: Decimal,
                                 accounting_method: LotAccounting) -> Dict[str, Union[Decimal, List[Dict]]]:
        """
        Calculate tax impact of a potential sale using different accounting methods.
        
        Args:
            symbol: Stock symbol
            sale_quantity: Quantity to sell
            accounting_method: Accounting method to use
            
        Returns:
            Dict with tax impact analysis
        """
        try:
            lots = await self.get_lots_by_symbol(symbol)
            if not lots:
                return {"error": "No lots available"}
            
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return {"error": "Cannot get current price"}
            
            # Select lots that would be sold
            selected_lots = self._select_lots_for_sale(lots, sale_quantity, accounting_method)
            
            # Calculate tax implications
            short_term_gain = Decimal("0")
            long_term_gain = Decimal("0")
            total_proceeds = Decimal("0")
            total_cost_basis = Decimal("0")
            
            lot_details = []
            
            remaining_to_sell = sale_quantity
            for lot in selected_lots:
                if remaining_to_sell <= 0:
                    break
                
                quantity_from_lot = min(remaining_to_sell, lot.quantity)
                proceeds_from_lot = quantity_from_lot * current_price
                cost_basis_from_lot = quantity_from_lot * lot.cost_basis_per_share
                gain_from_lot = proceeds_from_lot - cost_basis_from_lot
                
                total_proceeds += proceeds_from_lot
                total_cost_basis += cost_basis_from_lot
                
                if lot.is_long_term:
                    long_term_gain += gain_from_lot
                else:
                    short_term_gain += gain_from_lot
                
                lot_details.append({
                    "lot_id": lot.lot_id,
                    "purchase_date": lot.purchase_date.isoformat(),
                    "quantity": float(quantity_from_lot),
                    "cost_basis": float(cost_basis_from_lot),
                    "proceeds": float(proceeds_from_lot),
                    "gain_loss": float(gain_from_lot),
                    "is_long_term": lot.is_long_term,
                    "holding_days": lot.holding_period_days
                })
                
                remaining_to_sell -= quantity_from_lot
            
            total_gain = short_term_gain + long_term_gain
            
            return {
                "accounting_method": accounting_method.value,
                "total_proceeds": total_proceeds,
                "total_cost_basis": total_cost_basis,
                "total_gain_loss": total_gain,
                "short_term_gain_loss": short_term_gain,
                "long_term_gain_loss": long_term_gain,
                "lot_details": lot_details,
                "lots_affected": len(lot_details)
            }
            
        except Exception as e:
            logger.error(f"Error calculating tax impact for {symbol}: {e}")
            return {"error": str(e)}
    
    def _select_lots_for_sale(self, available_lots: List[TaxLot], quantity_to_sell: Decimal, method: LotAccounting) -> List[TaxLot]:
        """Select which lots to sell based on accounting method."""
        if method == LotAccounting.FIFO:
            # First In, First Out
            return sorted(available_lots, key=lambda x: x.purchase_date)
        
        elif method == LotAccounting.LIFO:
            # Last In, First Out
            return sorted(available_lots, key=lambda x: x.purchase_date, reverse=True)
        
        elif method == LotAccounting.HIFO:
            # Highest cost In, First Out (tax loss harvesting optimization)
            return sorted(available_lots, key=lambda x: x.cost_basis_per_share, reverse=True)
        
        elif method == LotAccounting.LOFO:
            # Lowest cost In, First Out
            return sorted(available_lots, key=lambda x: x.cost_basis_per_share)
        
        else:
            # Default to FIFO
            return sorted(available_lots, key=lambda x: x.purchase_date)
    
    async def _store_lot(self, lot: TaxLot):
        """Store a tax lot in the database."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO tax_lots (
                        lot_id, symbol, purchase_date, quantity, cost_basis_per_share,
                        total_cost_basis, current_price, unrealized_pnl, order_id,
                        commission_fees, other_fees, split_adjusted, dividend_adjustments,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    lot.lot_id, lot.symbol, lot.purchase_date.isoformat(), 
                    float(lot.quantity), float(lot.cost_basis_per_share),
                    float(lot.total_cost_basis), 
                    float(lot.current_price) if lot.current_price else None,
                    float(lot.unrealized_pnl) if lot.unrealized_pnl else None,
                    lot.order_id, float(lot.commission_fees), float(lot.other_fees),
                    lot.split_adjusted, float(lot.dividend_adjustments),
                    lot.created_at.isoformat(), lot.updated_at.isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing lot {lot.lot_id}: {e}")
            raise
    
    async def _store_transaction(self, transaction: LotTransaction):
        """Store a lot transaction in the database."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute('''
                    INSERT INTO lot_transactions (
                        transaction_id, symbol, transaction_type, transaction_date,
                        quantity, price_per_share, lots_affected, realized_pnl,
                        commission_fees, other_fees, order_id, alpaca_order_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transaction.transaction_id, transaction.symbol, transaction.transaction_type.value,
                    transaction.transaction_date.isoformat(), float(transaction.quantity),
                    float(transaction.price_per_share) if transaction.price_per_share else None,
                    json.dumps(transaction.lots_affected),
                    float(transaction.realized_pnl) if transaction.realized_pnl else None,
                    float(transaction.commission_fees), float(transaction.other_fees),
                    transaction.order_id, transaction.alpaca_order_id,
                    transaction.created_at.isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing transaction {transaction.transaction_id}: {e}")
            raise
    
    async def _get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current market price for a symbol."""
        try:
            price = alpaca_client.get_current_price(symbol)
            return Decimal(str(price)) if price else None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cache is still valid for a symbol."""
        if symbol not in self.last_cache_update:
            return False
        
        cache_age = (datetime.now() - self.last_cache_update[symbol]).total_seconds() / 60
        return cache_age < self.cache_expiry_minutes
    
    def _invalidate_cache(self, symbol: str):
        """Invalidate cache for a symbol."""
        if symbol in self.lots_cache:
            del self.lots_cache[symbol]
        if symbol in self.last_cache_update:
            del self.last_cache_update[symbol]
        if symbol in self.position_summaries:
            del self.position_summaries[symbol]

# Global instance
lot_tracker = SpecificLotTracker()

# Convenience functions
async def record_purchase(symbol: str, quantity: Decimal, price_per_share: Decimal, **kwargs) -> str:
    """Record a stock purchase."""
    return await lot_tracker.record_purchase(symbol, quantity, price_per_share, **kwargs)

async def record_sale(symbol: str, quantity: Decimal, price_per_share: Decimal, **kwargs) -> Tuple[List[str], Decimal]:
    """Record a stock sale."""
    return await lot_tracker.record_sale(symbol, quantity, price_per_share, **kwargs)

async def get_position_summary(symbol: str, **kwargs) -> PositionSummary:
    """Get position summary for a symbol."""
    return await lot_tracker.get_position_summary(symbol, **kwargs)

__all__ = [
    'TaxLot', 'LotTransaction', 'PositionSummary', 'LotAccounting', 'TransactionType',
    'SpecificLotTracker', 'lot_tracker', 'record_purchase', 'record_sale', 'get_position_summary'
]