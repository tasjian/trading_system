#!/usr/bin/env python3
"""
Complete Trading System Execution Script
This script runs the full AI trading system pipeline:
1. Portfolio construction using multi-agent system
2. Real trades execution on Alpaca (paper trading)
3. Email notifications for all transactions
4. Portfolio verification and reporting
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
from typing import List, Dict

# Add project root to Python path
sys.path.append('.')

# Import our trading system components
from agents.simplified_portfolio import build_simple_portfolio
from notifications.email_webhooks import EmailWebhookNotifier, TransactionAlert
from tools.alpaca_client import AlpacaClient
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSystemExecutor:
    """Main trading system execution engine."""
    
    def __init__(self):
        """Initialize the trading system executor."""
        self.config = settings
        self.alpaca_client = AlpacaClient()
        self.email_notifier = EmailWebhookNotifier()
        self.portfolio_value = 100000.0  # $100k portfolio
        self.max_positions = 6
        
    async def execute_full_trading_system(self) -> Dict:
        """Execute the complete trading system pipeline."""
        
        logger.info("🚀 Starting AI Trading System Execution")
        logger.info("=" * 60)
        
        results = {
            'success': False,
            'portfolio': None,
            'transactions': [],
            'errors': [],
            'total_invested': 0.0,
            'cash_remaining': 0.0
        }
        
        try:
            # Step 1: Get account information
            logger.info("1. Checking Alpaca account status...")
            account = self.alpaca_client.get_account_info()
            if account:
                logger.info(f"   ✓ Account active - Buying Power: ${account['buying_power']:,.2f}")
                logger.info(f"   ✓ Portfolio Value: ${account['portfolio_value']:,.2f}")
                logger.info(f"   ✓ Cash Available: ${account['cash']:,.2f}")
            else:
                logger.error("   ✗ Could not access Alpaca account")
                results['errors'].append("Alpaca account access failed")
                return results
            
            # Step 2: Define candidate stocks for portfolio
            candidate_symbols = [
                'AAPL',  # Apple
                'MSFT',  # Microsoft
                'GOOGL', # Google/Alphabet
                'AMZN',  # Amazon
                'TSLA',  # Tesla
                'NVDA',  # NVIDIA
                'META',  # Meta/Facebook
                'NFLX',  # Netflix
                'ADBE',  # Adobe
                'CRM'    # Salesforce
            ]
            
            # Step 3: Build optimal portfolio using AI agents
            logger.info("2. Constructing AI-optimized portfolio...")
            portfolio = await build_simple_portfolio(
                symbols=candidate_symbols,
                portfolio_value=self.portfolio_value,
                risk_level='moderate',
                max_positions=self.max_positions
            )
            
            logger.info(f"   ✓ Portfolio constructed successfully")
            logger.info(f"   ✓ Expected Return: {portfolio.expected_return:.2%}")
            logger.info(f"   ✓ Risk Level: {portfolio.risk_level:.2f}")
            logger.info(f"   ✓ Selected {len(portfolio.positions)} positions")
            
            results['portfolio'] = portfolio
            
            # Step 4: Execute trades on Alpaca
            logger.info("3. Executing trades on Alpaca Markets...")
            
            total_invested = 0.0
            transaction_count = 0
            
            for position in portfolio.positions:
                try:
                    # Calculate shares to buy based on position value and current price
                    shares_to_buy = max(1, int(position.value / position.price))
                    actual_cost = shares_to_buy * position.price
                    
                    logger.info(f"   → Placing BUY order: {shares_to_buy} shares of {position.symbol} at ${position.price:.2f}")
                    
                    # Place the order on Alpaca
                    order = self.alpaca_client.place_order(
                        symbol=position.symbol,
                        qty=shares_to_buy,
                        side='buy',
                        order_type='market',
                        time_in_force='day'
                    )
                    
                    if order:
                        logger.info(f"     ✓ Order placed successfully - Order ID: {order['id']}")
                        total_invested += actual_cost
                        transaction_count += 1
                        
                        # Create transaction alert for email notification
                        alert = TransactionAlert(
                            transaction_id=order['id'],
                            symbol=position.symbol,
                            action='buy',
                            quantity=shares_to_buy,
                            price=position.price,
                            total_value=actual_cost,
                            timestamp=datetime.now(),
                            portfolio_value=self.portfolio_value,
                            confidence=position.confidence,
                            reasoning=position.reasoning,
                            agent_source='simplified_portfolio',
                            metadata={
                                'weight': position.weight,
                                'order_id': order['id'],
                                'expected_return': portfolio.expected_return
                            }
                        )
                        
                        # Send email notification
                        logger.info(f"     → Sending email notification...")
                        try:
                            email_results = await self.email_notifier.send_transaction_notification(alert)
                            if any(email_results.values()):
                                logger.info(f"     ✓ Email notification sent successfully")
                            else:
                                logger.warning(f"     ⚠ Email notification configured but not sent (check credentials)")
                        except Exception as e:
                            logger.error(f"     ✗ Email notification failed: {e}")
                        
                        results['transactions'].append({
                            'symbol': position.symbol,
                            'shares': shares_to_buy,
                            'price': position.price,
                            'total': actual_cost,
                            'order_id': order['id']
                        })
                        
                    else:
                        logger.error(f"     ✗ Order failed for {position.symbol}")
                        results['errors'].append(f"Order failed for {position.symbol}")
                        
                except Exception as e:
                    logger.error(f"     ✗ Error trading {position.symbol}: {e}")
                    results['errors'].append(f"Error trading {position.symbol}: {str(e)}")
                    
                # Small delay between orders
                await asyncio.sleep(1)
            
            # Step 5: Calculate final results
            cash_remaining = self.portfolio_value - total_invested
            
            logger.info("4. Trading Execution Summary:")
            logger.info(f"   ✓ Total transactions executed: {transaction_count}")
            logger.info(f"   ✓ Total amount invested: ${total_invested:,.2f}")
            logger.info(f"   ✓ Cash remaining: ${cash_remaining:,.2f}")
            logger.info(f"   ✓ Portfolio utilization: {(total_invested/self.portfolio_value)*100:.1f}%")
            
            results['total_invested'] = total_invested
            results['cash_remaining'] = cash_remaining
            results['success'] = len(results['errors']) == 0
            
            logger.info("=" * 60)
            logger.info("🎉 AI Trading System Execution COMPLETED")
            
            if results['success']:
                logger.info("✅ All operations completed successfully!")
            else:
                logger.warning(f"⚠ Completed with {len(results['errors'])} errors")
            
            return results
            
        except Exception as e:
            logger.error(f"💥 Critical error in trading system: {e}")
            import traceback
            traceback.print_exc()
            results['errors'].append(f"Critical error: {str(e)}")
            return results

async def main():
    """Main execution function."""
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Initialize and run the trading system
    executor = TradingSystemExecutor()
    results = await executor.execute_full_trading_system()
    
    # Print final summary
    print("\n" + "="*60)
    print("🏁 FINAL EXECUTION RESULTS")
    print("="*60)
    
    if results['success']:
        print("✅ Status: SUCCESS")
    else:
        print("❌ Status: FAILED")
        print(f"Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  - {error}")
    
    if results['portfolio']:
        print(f"\n📊 Portfolio Performance:")
        print(f"  Expected Return: {results['portfolio'].expected_return:.2%}")
        print(f"  Risk Level: {results['portfolio'].risk_level:.2f}")
        print(f"  Confidence: {results['portfolio'].total_confidence:.2f}")
    
    print(f"\n💰 Financial Summary:")
    print(f"  Total Invested: ${results['total_invested']:,.2f}")
    print(f"  Cash Remaining: ${results['cash_remaining']:,.2f}")
    print(f"  Total Portfolio: ${results['total_invested'] + results['cash_remaining']:,.2f}")
    
    print(f"\n🔄 Transactions: {len(results['transactions'])}")
    for txn in results['transactions']:
        print(f"  - {txn['symbol']}: {txn['shares']} shares at ${txn['price']:.2f} = ${txn['total']:,.2f}")
    
    print("="*60)
    
    return results['success']

if __name__ == "__main__":
    # Run the complete trading system
    success = asyncio.run(main())
    sys.exit(0 if success else 1)