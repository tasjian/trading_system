#!/usr/bin/env python3
"""
Continuous Trading Engine with Portfolio Rebalancing

This engine runs continuously and automatically rebalances the portfolio based on:
1. Updated market data and AI analysis
2. Configurable rebalancing frequency 
3. Day trading compliance (PDT rules)
4. Risk management and position sizing
5. Aggressive trading stance with proper controls
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
import json

# Add project root to Python path
sys.path.append('.')

# Import trading system components
from agents.simplified_portfolio import build_simple_portfolio
from agents.advanced_portfolio_strategy import build_advanced_portfolio
from tools.alpaca_client import AlpacaClient
from notifications.email_webhooks import EmailWebhookNotifier, TransactionAlert
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continuous_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DayTradingCompliance:
    """Day trading compliance checker to respect PDT rules."""
    
    def __init__(self, alpaca_client: AlpacaClient):
        self.alpaca_client = alpaca_client
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()
        
    def check_account_status(self) -> Dict:
        """Check account status and day trading restrictions."""
        try:
            account_info = self.alpaca_client.get_account_info()
            
            # Reset daily count if new day
            today = datetime.now().date()
            if today > self.last_reset_date:
                self.daily_trade_count = 0
                self.last_reset_date = today
            
            return {
                'account_value': account_info['portfolio_value'],
                'day_trade_count': account_info['day_trade_count'],
                'pattern_day_trader': account_info['pattern_day_trader'],
                'trading_blocked': account_info['trading_blocked'],
                'can_day_trade': self._can_day_trade(account_info),
                'remaining_day_trades': self._remaining_day_trades(account_info)
            }
        except Exception as e:
            logger.error(f"Failed to check account status: {e}")
            return {'can_day_trade': False, 'error': str(e)}
    
    def _can_day_trade(self, account_info: Dict) -> bool:
        """Check if day trading is allowed based on PDT rules."""
        # PDT rules: Need $25k+ account value OR be flagged as PDT
        account_value = account_info['portfolio_value']
        is_pdt = account_info['pattern_day_trader']
        
        # If account has $25k+, day trading is allowed
        if account_value >= 25000:
            return True
        
        # If flagged as PDT with less than $25k, limited day trades
        if is_pdt:
            return account_info['day_trade_count'] < 3
        
        # Non-PDT accounts can make limited day trades
        return account_info['day_trade_count'] < 3
    
    def _remaining_day_trades(self, account_info: Dict) -> int:
        """Calculate remaining day trades for the account."""
        if account_info['portfolio_value'] >= 25000:
            return 999  # Unlimited for accounts with $25k+
        
        return max(0, 3 - account_info['day_trade_count'])

class ContinuousTradingEngine:
    """Main continuous trading engine with automatic rebalancing."""
    
    def __init__(self):
        """Initialize the continuous trading engine."""
        self.alpaca_client = AlpacaClient()
        self.email_notifier = EmailWebhookNotifier()
        self.compliance_checker = DayTradingCompliance(self.alpaca_client)
        
        # Trading parameters (aggressive stance)
        self.portfolio_value = 100000.0  # Target portfolio value
        self.candidate_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'ADBE', 'CRM', 'AMD', 'PLTR', 'PYPL', 'SNOW', 'COIN', 'RBLX',
            'U', 'SHOP', 'ROKU', 'ZM'  # Aggressive tech-focused portfolio (cleaned)
        ]
        self.max_positions = 8  # Aggressive: More positions for diversification
        self.risk_level = 'aggressive'
        
        # Rebalancing configuration
        self.rebalance_frequency = settings.rebalance_frequency  # seconds
        self.min_threshold = settings.min_rebalance_threshold    # 5% deviation
        
        # State tracking
        self.current_portfolio = None
        self.last_rebalance = None
        self.is_running = False
        self.total_rebalances = 0
        
        # Trading hours (US market: 9:30 AM - 4:00 PM ET)
        self.market_open = time(9, 30)  # 9:30 AM
        self.market_close = time(16, 0)  # 4:00 PM
        
    def is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        now = datetime.now().time()
        return self.market_open <= now <= self.market_close
    
    def is_weekday(self) -> bool:
        """Check if today is a weekday (Monday-Friday)."""
        return datetime.now().weekday() < 5
    
    async def get_current_positions(self) -> Dict:
        """Get current portfolio positions from Alpaca."""
        try:
            positions = self.alpaca_client.get_positions()
            portfolio = {
                'positions': positions,
                'total_value': sum(float(pos['market_value']) for pos in positions),
                'symbols': [pos['symbol'] for pos in positions]
            }
            logger.info(f"Current portfolio: {len(positions)} positions, ${portfolio['total_value']:,.2f} value")
            return portfolio
        except Exception as e:
            logger.error(f"Failed to get current positions: {e}")
            return {'positions': [], 'total_value': 0, 'symbols': []}
    
    async def build_target_portfolio(self) -> Dict:
        """Build target portfolio using AI agents."""
        try:
            logger.info("🧠 Building target portfolio with AI agents...")
            
            # Build optimal portfolio using advanced long/short strategies
            if self.risk_level == "aggressive":
                logger.info("🚀 Using advanced long/short portfolio strategy")
                portfolio_result = await build_advanced_portfolio(
                    symbols=self.candidate_symbols,
                    total_capital=self.portfolio_value,
                    risk_level=self.risk_level
                )
                portfolio = portfolio_result['portfolio']
                # Convert advanced positions to simple format for compatibility
                positions = []
                for pos in portfolio['positions']:
                    positions.append({
                        'symbol': pos.symbol,
                        'weight': pos.weight if pos.side == 'long' else -pos.weight,  # Negative for shorts
                        'expected_return': pos.expected_return,
                        'strategy': pos.strategy,
                        'side': pos.side
                    })
                portfolio = {
                    'positions': positions,
                    'expected_return': portfolio['expected_return'],
                    'risk': 0.15,  # Estimated portfolio risk
                    'metadata': {
                        'long_exposure': portfolio['long_exposure'],
                        'short_exposure': portfolio['short_exposure'],
                        'net_exposure': portfolio['net_exposure'],
                        'num_long': portfolio['num_long'],
                        'num_short': portfolio['num_short']
                    }
                }
            else:
                # Use simple long-only portfolio for conservative/moderate
                portfolio = await build_simple_portfolio(
                    symbols=self.candidate_symbols,
                    portfolio_value=self.portfolio_value,
                    risk_level=self.risk_level,
                    max_positions=self.max_positions
                )
            
            target = {
                'positions': portfolio['positions'],
                'expected_return': portfolio['expected_return'],
                'risk_level': self.risk_level,
                'total_confidence': 0.8,  # Default confidence
                'timestamp': datetime.now(),
                'metadata': portfolio.get('metadata', {})
            }
            
            num_positions = len(portfolio['positions'])
            expected_return = portfolio['expected_return']
            logger.info(f"🎯 Target portfolio: {num_positions} positions, "
                       f"{expected_return:.2%} expected return")
            
            # Log advanced portfolio metadata if available
            if 'metadata' in portfolio and 'long_exposure' in portfolio['metadata']:
                meta = portfolio['metadata']
                logger.info(f"📊 Long: {meta['num_long']}, Short: {meta['num_short']}, "
                           f"Net exposure: {meta['net_exposure']:.1%}")
            
            return target
            
        except Exception as e:
            logger.error(f"Failed to build target portfolio: {e}")
            return None
    
    async def calculate_rebalancing_trades(self, current: Dict, target: Dict) -> List[Dict]:
        """Calculate the trades needed to rebalance from current to target portfolio."""
        trades = []
        
        try:
            # Get current position symbols and weights
            current_symbols = set(current['symbols'])
            target_symbols = set(pos['symbol'] for pos in target['positions'])
            
            # Calculate target weights (handle both dict and object formats)
            target_weights = {}
            for pos in target['positions']:
                if isinstance(pos, dict):
                    symbol = pos['symbol']
                    weight = pos['weight']
                    side = pos.get('side', 'long')
                else:
                    symbol = pos.symbol
                    weight = pos.weight
                    side = getattr(pos, 'side', 'long')
                
                target_weights[symbol] = {
                    'weight': weight,
                    'side': side
                }
            
            # Calculate current weights
            current_weights = {}
            if current['total_value'] > 0:
                for pos in current['positions']:
                    weight = float(pos['market_value']) / current['total_value']
                    current_weights[pos['symbol']] = weight
            
            # Find symbols to sell (in current but not in target, or over-weighted)
            for symbol in current_symbols:
                current_weight = current_weights.get(symbol, 0)
                target_data = target_weights.get(symbol, {'weight': 0, 'side': 'long'})
                target_weight = target_data['weight']
                target_side = target_data['side']
                
                weight_diff = current_weight - target_weight
                
                if symbol not in target_symbols or weight_diff > self.min_threshold:
                    # Need to sell some or all of this position
                    current_pos = next(p for p in current['positions'] if p['symbol'] == symbol)
                    current_qty = float(current_pos['qty'])
                    
                    if symbol not in target_symbols:
                        # Sell entire position
                        sell_qty = current_qty
                    else:
                        # Sell excess amount
                        target_value = self.portfolio_value * target_weight
                        current_value = float(current_pos['market_value'])
                        excess_value = current_value - target_value
                        
                        if excess_value > 0:
                            sell_qty = min(current_qty, excess_value / float(current_pos['current_price']))
                        else:
                            sell_qty = 0
                    
                    if sell_qty > 0:
                        trades.append({
                            'symbol': symbol,
                            'side': 'sell',
                            'qty': int(sell_qty),
                            'reason': f'Rebalance: reduce {symbol} weight from {current_weight:.1%} to {target_weight:.1%}'
                        })
            
            # Find symbols to buy (in target but not in current, or under-weighted)
            for pos in target['positions']:
                # Handle both dict and object formats
                if isinstance(pos, dict):
                    symbol = pos['symbol']
                    target_weight = abs(pos['weight'])  # Use absolute weight for calculation
                    side = pos.get('side', 'long')
                    strategy = pos.get('strategy', 'unknown')
                else:
                    symbol = pos.symbol
                    target_weight = abs(pos.weight)
                    side = getattr(pos, 'side', 'long')
                    strategy = getattr(pos, 'strategy', 'unknown')
                
                current_weight = current_weights.get(symbol, 0)
                weight_diff = target_weight - current_weight
                
                if weight_diff > self.min_threshold:
                    # Need to buy more of this position
                    target_value = self.portfolio_value * target_weight
                    current_value = current_weights.get(symbol, 0) * current['total_value']
                    buy_value = target_value - current_value
                    
                    if buy_value > 0:
                        # Estimate share price (need to get real price from market data)
                        estimated_price = 100  # Default estimate - should be replaced with real price
                        
                        if isinstance(pos, dict) and 'price' in pos:
                            estimated_price = pos['price']
                        elif hasattr(pos, 'price'):
                            estimated_price = pos.price
                            
                        buy_qty = int(buy_value / estimated_price)
                        if buy_qty > 0:
                            # Determine trade side based on position type
                            trade_side = 'sell_short' if side == 'short' else 'buy'
                            
                            trades.append({
                                'symbol': symbol,
                                'side': trade_side,
                                'qty': buy_qty,
                                'price': estimated_price,
                                'strategy': strategy,
                                'position_side': side,
                                'reason': f'Rebalance: {"short" if side == "short" else "increase"} {symbol} weight from {current_weight:.1%} to {target_weight:.1%}'
                            })
            
            logger.info(f"📊 Calculated {len(trades)} rebalancing trades")
            return trades
            
        except Exception as e:
            logger.error(f"Failed to calculate rebalancing trades: {e}")
            return []
    
    async def execute_rebalancing_trades(self, trades: List[Dict]) -> Dict:
        """Execute the calculated rebalancing trades."""
        results = {
            'executed': 0,
            'failed': 0,
            'total_value': 0,
            'trades': []
        }
        
        # Check day trading compliance
        compliance = self.compliance_checker.check_account_status()
        if not compliance.get('can_day_trade', False):
            logger.warning("⚠️ Day trading restricted - limiting to position exits only")
            # Filter to only sell orders if day trading is restricted
            trades = [t for t in trades if t['side'] == 'sell']
        
        logger.info(f"🔄 Executing {len(trades)} rebalancing trades...")
        
        for trade in trades:
            try:
                # Place the order
                order = self.alpaca_client.place_order(
                    symbol=trade['symbol'],
                    qty=trade['qty'],
                    side=trade['side'],
                    order_type='market',
                    time_in_force='day'
                )
                
                if order:
                    trade_value = trade['qty'] * trade.get('price', 0)
                    logger.info(f"✅ {trade['side'].upper()} {trade['qty']} {trade['symbol']} - ${trade_value:,.2f}")
                    
                    results['executed'] += 1
                    results['total_value'] += trade_value
                    results['trades'].append({
                        'order_id': order['id'],
                        'symbol': trade['symbol'],
                        'side': trade['side'],
                        'qty': trade['qty'],
                        'value': trade_value,
                        'reason': trade['reason']
                    })
                    
                    # Send email notification
                    try:
                        alert = TransactionAlert(
                            transaction_id=order['id'],
                            symbol=trade['symbol'],
                            action=trade['side'],
                            quantity=trade['qty'],
                            price=trade.get('price', 0),
                            total_value=trade_value,
                            timestamp=datetime.now(),
                            portfolio_value=self.portfolio_value,
                            confidence=0.85,
                            reasoning=trade['reason'],
                            agent_source='continuous_rebalancing',
                            metadata={'rebalance_count': self.total_rebalances}
                        )
                        await self.email_notifier.send_transaction_notification(alert)
                    except Exception as e:
                        logger.warning(f"Email notification failed: {e}")
                
                else:
                    logger.error(f"❌ Failed to execute {trade['side']} {trade['qty']} {trade['symbol']}")
                    results['failed'] += 1
                
                # Small delay between trades
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error executing trade {trade}: {e}")
                results['failed'] += 1
        
        return results
    
    async def rebalance_portfolio(self) -> bool:
        """Perform a complete portfolio rebalancing."""
        try:
            logger.info("🔄 Starting portfolio rebalancing...")
            
            # Get current portfolio state
            current_portfolio = await self.get_current_positions()
            
            # Build target portfolio
            target_portfolio = await self.build_target_portfolio()
            if not target_portfolio:
                logger.error("Failed to build target portfolio")
                return False
            
            # Calculate required trades
            trades = await self.calculate_rebalancing_trades(current_portfolio, target_portfolio)
            
            if not trades:
                logger.info("✅ Portfolio already balanced - no trades needed")
                return True
            
            # Execute trades
            results = await self.execute_rebalancing_trades(trades)
            
            # Log results
            logger.info(f"📊 Rebalancing complete: {results['executed']} executed, "
                       f"{results['failed']} failed, ${results['total_value']:,.2f} traded")
            
            self.last_rebalance = datetime.now()
            self.total_rebalances += 1
            
            return results['executed'] > 0
            
        except Exception as e:
            logger.error(f"❌ Portfolio rebalancing failed: {e}")
            return False
    
    async def run_continuous_trading(self):
        """Main continuous trading loop."""
        logger.info("🚀 Starting Continuous Trading Engine")
        logger.info("=" * 60)
        logger.info(f"📊 Portfolio Value: ${self.portfolio_value:,.2f}")
        logger.info(f"🎯 Max Positions: {self.max_positions}")
        logger.info(f"📈 Risk Level: {self.risk_level}")
        logger.info(f"⏰ Rebalance Frequency: {self.rebalance_frequency} seconds")
        logger.info(f"📉 Rebalance Threshold: {self.min_threshold:.1%}")
        logger.info("=" * 60)
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Check if we should trade (market hours + weekdays)
                if not self.is_weekday():
                    logger.info("🏖️ Weekend - market closed, sleeping...")
                    await asyncio.sleep(3600)  # Sleep 1 hour on weekends
                    continue
                
                if not self.is_market_hours():
                    logger.info("🌙 After hours - sleeping until market opens...")
                    await asyncio.sleep(1800)  # Sleep 30 minutes after hours
                    continue
                
                # Check if it's time to rebalance
                now = datetime.now()
                if (self.last_rebalance is None or 
                    (now - self.last_rebalance).total_seconds() >= self.rebalance_frequency):
                    
                    logger.info(f"⏰ Rebalancing time reached (every {self.rebalance_frequency}s)")
                    
                    # Perform rebalancing
                    success = await self.rebalance_portfolio()
                    
                    if success:
                        logger.info(f"✅ Rebalancing successful (#{self.total_rebalances})")
                    else:
                        logger.warning("⚠️ Rebalancing had issues")
                
                # Sleep for a shorter interval to check conditions frequently
                await asyncio.sleep(300)  # Check every 5 minutes
                
        except KeyboardInterrupt:
            logger.info("⏸️ Received stop signal - shutting down gracefully...")
        except Exception as e:
            logger.error(f"❌ Critical error in trading loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            logger.info("🛑 Continuous Trading Engine stopped")
    
    def stop(self):
        """Stop the continuous trading engine."""
        logger.info("🛑 Stopping Continuous Trading Engine...")
        self.is_running = False

def signal_handler(engine):
    """Handle shutdown signals gracefully."""
    def handler(signum, frame):
        logger.info(f"📡 Received signal {signum}")
        engine.stop()
    return handler

async def main():
    """Main function to run the continuous trading engine."""
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Initialize the trading engine
    engine = ContinuousTradingEngine()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler(engine))
    signal.signal(signal.SIGTERM, signal_handler(engine))
    
    try:
        # Run the continuous trading engine
        await engine.run_continuous_trading()
    except Exception as e:
        logger.error(f"❌ Failed to start trading engine: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🤖 AI CONTINUOUS TRADING SYSTEM")
    print("=" * 50)
    print("⚡ Aggressive Trading Configuration")
    print("🔄 Automatic Portfolio Rebalancing")
    print("📊 Day Trading Compliance Included")
    print("=" * 50)
    
    # Run the trading engine
    asyncio.run(main())