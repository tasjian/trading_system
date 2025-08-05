"""
Email Batching Manager for Trading Notifications

This module batches trading notifications within a 1-minute window to reduce
email spam and provide consolidated transaction summaries.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class BatchedTransaction:
    """Individual transaction for batching."""
    transaction_id: str
    symbol: str
    action: str
    quantity: float
    price: float
    total_value: float
    timestamp: datetime
    portfolio_value: float
    confidence: float
    reasoning: str
    agent_source: str


@dataclass
class TransactionBatch:
    """Batch of transactions within a time window."""
    batch_id: str
    start_time: datetime
    end_time: datetime
    transactions: List[BatchedTransaction] = field(default_factory=list)
    total_value: float = 0.0
    portfolio_value: float = 0.0
    
    def add_transaction(self, transaction: BatchedTransaction):
        """Add a transaction to the batch."""
        self.transactions.append(transaction)
        self.total_value += abs(transaction.total_value)
        self.portfolio_value = transaction.portfolio_value  # Use latest portfolio value
        
        # Update end time to latest transaction
        if transaction.timestamp > self.end_time:
            self.end_time = transaction.timestamp
    
    def get_summary(self) -> Dict:
        """Get batch summary statistics."""
        buy_orders = [t for t in self.transactions if t.action.lower() == 'buy']
        sell_orders = [t for t in self.transactions if t.action.lower() == 'sell']
        
        buy_value = sum(t.total_value for t in buy_orders)
        sell_value = sum(abs(t.total_value) for t in sell_orders)
        
        return {
            'total_transactions': len(self.transactions),
            'buy_orders': len(buy_orders),
            'sell_orders': len(sell_orders),
            'total_buy_value': buy_value,
            'total_sell_value': sell_value,
            'net_value': buy_value - sell_value,
            'duration_seconds': (self.end_time - self.start_time).total_seconds(),
            'symbols_traded': list(set(t.symbol for t in self.transactions)),
            'portfolio_value': self.portfolio_value
        }


class EmailBatchManager:
    """
    Manages batching of trading email notifications within 1-minute windows.
    """
    
    def __init__(self, batch_window_seconds: int = 60):
        """
        Initialize the email batch manager.
        
        Args:
            batch_window_seconds: Time window for batching emails (default: 60 seconds)
        """
        self.batch_window = timedelta(seconds=batch_window_seconds)
        self.active_batches: Dict[str, TransactionBatch] = {}
        self.pending_transactions: List[BatchedTransaction] = []
        self.batch_counter = 0
        self.lock = threading.Lock()
        
        # Start background task for processing batches
        self.batch_task = None
        self.running = False
        
    def start_batch_processor(self):
        """Start the background batch processing task."""
        if not self.running:
            self.running = True
            self.batch_task = asyncio.create_task(self._batch_processor())
            logger.info("Email batch processor started")
    
    def stop_batch_processor(self):
        """Stop the background batch processing task."""
        self.running = False
        if self.batch_task:
            self.batch_task.cancel()
            logger.info("Email batch processor stopped")
    
    async def add_transaction(self, symbol: str, action: str, quantity: float, 
                            price: float, portfolio_value: float, 
                            confidence: float, reasoning: str, 
                            agent_source: str) -> None:
        """
        Add a transaction to be batched for email notification.
        
        Args:
            symbol: Stock symbol
            action: Trading action (buy/sell)
            quantity: Number of shares
            price: Price per share
            portfolio_value: Current portfolio value
            confidence: AI confidence score
            reasoning: AI reasoning for the trade
            agent_source: Source agent that generated the trade
        """
        try:
            transaction = BatchedTransaction(
                transaction_id=f"txn_{int(datetime.now().timestamp())}_{symbol}",
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                total_value=quantity * price,
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
                confidence=confidence,
                reasoning=reasoning,
                agent_source=agent_source
            )
            
            with self.lock:
                self.pending_transactions.append(transaction)
                logger.info(f"Transaction added to batch queue: {action} {quantity} {symbol}")
            
        except Exception as e:
            logger.error(f"Error adding transaction to batch: {e}")
    
    async def _batch_processor(self):
        """Background task that processes transaction batches."""
        logger.info("Batch processor started")
        
        while self.running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                current_time = datetime.now()
                
                # Process pending transactions
                with self.lock:
                    if self.pending_transactions:
                        await self._process_pending_transactions(current_time)
                
                # Check for expired batches and send them
                await self._send_expired_batches(current_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(10)  # Wait before retrying
        
        # Send any remaining batches before shutdown
        await self._send_all_remaining_batches()
        logger.info("Batch processor stopped")
    
    async def _process_pending_transactions(self, current_time: datetime):
        """Process pending transactions and assign them to batches."""
        transactions_to_process = self.pending_transactions.copy()
        self.pending_transactions.clear()
        
        for transaction in transactions_to_process:
            # Find an existing batch within the time window
            batch_key = None
            for key, batch in self.active_batches.items():
                time_diff = abs((transaction.timestamp - batch.start_time).total_seconds())
                if time_diff <= self.batch_window.total_seconds():
                    batch_key = key
                    break
            
            # Create new batch if none found
            if batch_key is None:
                self.batch_counter += 1
                batch_key = f"batch_{self.batch_counter}_{int(transaction.timestamp.timestamp())}"
                self.active_batches[batch_key] = TransactionBatch(
                    batch_id=batch_key,
                    start_time=transaction.timestamp,
                    end_time=transaction.timestamp
                )
                logger.info(f"Created new transaction batch: {batch_key}")
            
            # Add transaction to batch
            self.active_batches[batch_key].add_transaction(transaction)
            logger.debug(f"Added transaction to batch {batch_key}: {transaction.symbol}")
    
    async def _send_expired_batches(self, current_time: datetime):
        """Send batches that have exceeded the time window."""
        expired_batches = []
        
        for batch_key, batch in self.active_batches.items():
            time_since_last = (current_time - batch.end_time).total_seconds()
            if time_since_last >= self.batch_window.total_seconds():
                expired_batches.append(batch_key)
        
        # Send expired batches
        for batch_key in expired_batches:
            batch = self.active_batches.pop(batch_key)
            await self._send_batch_email(batch)
    
    async def _send_all_remaining_batches(self):
        """Send all remaining batches during shutdown."""
        for batch_key, batch in list(self.active_batches.items()):
            await self._send_batch_email(batch)
        self.active_batches.clear()
    
    async def _send_batch_email(self, batch: TransactionBatch):
        """Send email notification for a transaction batch."""
        try:
            logger.info(f"Sending batch email for {batch.batch_id} with {len(batch.transactions)} transactions")
            
            # Import email function
            from notifications.email_webhooks import send_batch_transaction_email
            
            # Send batch email
            await send_batch_transaction_email(batch)
            
            logger.info(f"Batch email sent successfully: {batch.batch_id}")
            
        except Exception as e:
            logger.error(f"Failed to send batch email for {batch.batch_id}: {e}")
    
    def get_status(self) -> Dict:
        """Get current batching status."""
        with self.lock:
            return {
                'running': self.running,
                'active_batches': len(self.active_batches),
                'pending_transactions': len(self.pending_transactions),
                'batch_window_seconds': self.batch_window.total_seconds(),
                'batch_details': [
                    {
                        'batch_id': batch.batch_id,
                        'transactions': len(batch.transactions),
                        'start_time': batch.start_time.isoformat(),
                        'end_time': batch.end_time.isoformat(),
                        'total_value': batch.total_value
                    }
                    for batch in self.active_batches.values()
                ]
            }


# Global batch manager instance
email_batch_manager = EmailBatchManager()


async def send_batched_transaction_notification(
    symbol: str, action: str, quantity: float, price: float,
    portfolio_value: float, confidence: float, reasoning: str,
    agent_source: str
) -> None:
    """
    Send a transaction notification through the batching system.
    
    This function replaces direct email sending and adds transactions
    to the batching queue for consolidated email delivery.
    """
    await email_batch_manager.add_transaction(
        symbol=symbol,
        action=action,
        quantity=quantity,
        price=price,
        portfolio_value=portfolio_value,
        confidence=confidence,
        reasoning=reasoning,
        agent_source=agent_source
    )


# Auto-start the batch processor when module is imported
def _auto_start_batch_processor():
    """Auto-start the batch processor in a background task."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            email_batch_manager.start_batch_processor()
        else:
            email_batch_manager.start_batch_processor()
    except RuntimeError:
        # No event loop running, will start when needed
        pass


# Initialize batch processor when module loads
try:
    if not email_batch_manager.running:
        email_batch_manager.start_batch_processor()
except RuntimeError:
    # Event loop not running yet, will start when first transaction is added
    pass