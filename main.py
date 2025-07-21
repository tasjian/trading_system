"""Main application for the agentic trading system."""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional
import uuid

from config.settings import settings, validate_settings
from agents.workflow import trading_workflow
from tools.alpaca_client import alpaca_client
from compliance.regulatory_compliance import ensure_regulatory_compliance

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TradingSystemApp:
    """Main trading system application."""
    
    def __init__(self):
        """Initialize the trading system."""
        self.running = False
        self.session_id = str(uuid.uuid4())
        self.cycle_count = 0
        self.last_cycle_time = None
        
        # Validate configuration
        validate_settings()
        
        # Regulatory compliance validation
        logger.info("Performing regulatory compliance validation...")
        if not ensure_regulatory_compliance():
            raise ValueError("Regulatory compliance validation failed")
        
        # Safety checks
        self._perform_startup_checks()
        
        logger.info(f"Trading System initialized - Session: {self.session_id}")
        logger.info(f"Trading Mode: {settings.trading_mode}")
        logger.info(f"Max Position Size: {settings.max_position_size:.1%}")
        logger.info(f"Max Portfolio Risk: {settings.max_portfolio_risk:.1%}")
    
    def _perform_startup_checks(self):
        """Perform critical startup safety checks."""
        logger.info("Performing startup safety checks...")
        
        # Check 1: Verify paper trading mode
        if "paper" not in settings.alpaca_base_url.lower():
            logger.error("CRITICAL: Not using paper trading URL!")
            raise ValueError("System must use paper trading for safety")
        
        # Check 2: Test API connection
        try:
            account_info = alpaca_client.get_account_info()
            logger.info(f"Connected to account: {account_info['id']}")
            logger.info(f"Account status: {account_info['status']}")
            
            if account_info.get("account_blocked", False):
                raise ValueError("Account is blocked")
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}")
            raise
        
        # Check 3: Verify risk limits are reasonable
        if settings.max_position_size > 0.2:  # 20%
            logger.warning("Position size limit seems high!")
        
        if settings.max_portfolio_risk > 0.05:  # 5%
            logger.warning("Portfolio risk limit seems high!")
        
        # Check 4: Ensure required directories exist
        import os
        os.makedirs("logs", exist_ok=True)
        
        logger.info("All startup checks passed ✓")
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run_single_cycle(self, manual_trigger: str = "") -> dict:
        """Run a single trading cycle."""
        cycle_start = datetime.now()
        self.cycle_count += 1
        
        logger.info(f"=== Trading Cycle {self.cycle_count} ===")
        
        try:
            # Check if market is open
            if not alpaca_client.is_market_open():
                logger.info("Market is closed, skipping trading cycle")
                return {
                    "status": "skipped",
                    "reason": "market_closed",
                    "cycle": self.cycle_count
                }
            
            # Run the trading workflow
            input_message = manual_trigger or f"Automated trading cycle #{self.cycle_count}"
            
            result = await trading_workflow.run_cycle(
                session_id=self.session_id,
                input_message=input_message
            )
            
            # Log cycle results
            if result["status"] == "success":
                portfolio_value = result.get("portfolio_value", 0)
                positions = result.get("active_positions", 0)
                signals = result.get("signals_generated", 0)
                orders = result.get("orders_executed", 0)
                
                logger.info(f"Cycle completed: Portfolio=${portfolio_value:.2f}, "
                          f"Positions={positions}, Signals={signals}, Orders={orders}")
            else:
                logger.error(f"Cycle failed: {result.get('error', 'Unknown error')}")
            
            self.last_cycle_time = cycle_start
            return result
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return {
                "status": "error",
                "error": str(e),
                "cycle": self.cycle_count
            }
    
    async def run_continuous(self, cycle_interval: int = 300):
        """
        Run the trading system continuously.
        
        Args:
            cycle_interval: Seconds between trading cycles (default: 5 minutes)
        """
        logger.info(f"Starting continuous trading mode (interval: {cycle_interval}s)")
        self.running = True
        
        try:
            while self.running:
                # Run trading cycle
                await self.run_single_cycle()
                
                # Check for emergency stop conditions
                if await self._check_emergency_conditions():
                    logger.critical("Emergency conditions detected, stopping system")
                    break
                
                # Wait for next cycle
                if self.running:
                    logger.info(f"Waiting {cycle_interval} seconds until next cycle...")
                    await asyncio.sleep(cycle_interval)
                
        except asyncio.CancelledError:
            logger.info("Trading system cancelled")
        except Exception as e:
            logger.error(f"Error in continuous mode: {e}")
        finally:
            self.shutdown()
    
    async def _check_emergency_conditions(self) -> bool:
        """Check for emergency stop conditions."""
        try:
            account_info = alpaca_client.get_account_info()
            
            # Check 1: Account blocked
            if account_info.get("account_blocked", False):
                logger.critical("Account is blocked!")
                return True
            
            # Check 2: Excessive daily loss
            equity = account_info["equity"]
            initial_equity = 100000  # Assume initial value - should be stored properly
            
            if initial_equity > 0:
                daily_loss = (initial_equity - equity) / initial_equity
                if daily_loss > settings.max_daily_loss * 2:  # 2x normal limit
                    logger.critical(f"Excessive daily loss: {daily_loss:.2%}")
                    return True
            
            # Check 3: Too many day trades
            day_trades = account_info.get("day_trade_count", 0)
            if day_trades > settings.max_daily_trades:
                logger.critical(f"Too many day trades: {day_trades}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {e}")
            return True  # Err on the side of caution
    
    def shutdown(self):
        """Shutdown the trading system gracefully."""
        logger.info("Shutting down trading system...")
        self.running = False
        
        try:
            # Cancel all pending orders as safety measure
            if alpaca_client.cancel_all_orders():
                logger.info("All pending orders cancelled")
            
            # Log final portfolio state
            account_info = alpaca_client.get_account_info()
            positions = alpaca_client.get_positions()
            
            logger.info(f"Final portfolio value: ${account_info['equity']:.2f}")
            logger.info(f"Cash available: ${account_info['cash']:.2f}")
            logger.info(f"Active positions: {len(positions)}")
            
            if positions:
                logger.info("Active positions:")
                for pos in positions:
                    unrealized_pl = pos.get("unrealized_pl", 0)
                    logger.info(f"  {pos['symbol']}: {pos['qty']} shares, "
                              f"P&L: ${unrealized_pl:.2f}")
            
            logger.info(f"Trading session completed - Total cycles: {self.cycle_count}")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("Trading system shutdown complete")
    
    async def manual_trade(self, symbol: str, action: str, quantity: Optional[float] = None):
        """Execute a manual trade with safety checks."""
        logger.info(f"Manual trade request: {action} {symbol}")
        
        try:
            # Validate inputs
            if action.lower() not in ["buy", "sell", "close"]:
                raise ValueError("Action must be 'buy', 'sell', or 'close'")
            
            if action.lower() in ["buy", "sell"] and not quantity:
                raise ValueError("Quantity required for buy/sell orders")
            
            # Get asset info
            asset_info = alpaca_client.get_asset_info(symbol)
            if not asset_info.get("tradable", False):
                raise ValueError(f"{symbol} is not tradable")
            
            # Execute based on action
            if action.lower() == "close":
                result = alpaca_client.close_position(symbol)
                logger.info(f"Position closed: {symbol}")
            else:
                # Validate trade risk first
                from tools.trading_tools import validate_trade_risk
                validation = validate_trade_risk(symbol, quantity, action)
                
                if not validation.get("validation_passed", False):
                    logger.warning(f"Trade validation failed: {validation}")
                    return {"status": "rejected", "reason": "risk_validation_failed", "details": validation}
                
                # Place the order
                result = alpaca_client.place_order(
                    symbol=symbol,
                    qty=quantity,
                    side=action.lower()
                )
                logger.info(f"Order placed: {action} {quantity} {symbol}")
            
            return {"status": "success", "order": result}
            
        except Exception as e:
            logger.error(f"Manual trade error: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_status(self) -> dict:
        """Get current system status."""
        try:
            account_info = alpaca_client.get_account_info()
            positions = alpaca_client.get_positions()
            open_orders = alpaca_client.get_orders(status="open")
            
            return {
                "system": {
                    "running": self.running,
                    "session_id": self.session_id,
                    "cycle_count": self.cycle_count,
                    "last_cycle": self.last_cycle_time.isoformat() if self.last_cycle_time else None,
                    "trading_mode": settings.trading_mode
                },
                "account": {
                    "equity": account_info["equity"],
                    "cash": account_info["cash"],
                    "buying_power": account_info["buying_power"],
                    "day_trades": account_info.get("day_trade_count", 0)
                },
                "portfolio": {
                    "positions": len(positions),
                    "open_orders": len(open_orders),
                    "total_unrealized_pl": sum(pos.get("unrealized_pl", 0) for pos in positions)
                },
                "market": {
                    "open": alpaca_client.is_market_open()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"error": str(e)}


async def main():
    """Main application entry point."""
    app = TradingSystemApp()
    app._setup_signal_handlers()
    
    # Show startup information
    status = app.get_status()
    print("\n" + "="*60)
    print("🤖 AGENTIC TRADING SYSTEM STARTED")
    print("="*60)
    print(f"Session ID: {app.session_id}")
    print(f"Trading Mode: {settings.trading_mode.upper()}")
    print(f"Portfolio Value: ${status['account']['equity']:.2f}")
    print(f"Cash Available: ${status['account']['cash']:.2f}")
    print(f"Market Open: {'✓' if status['market']['open'] else '✗'}")
    print("="*60)
    print("\nCommands:")
    print("  'cycle' - Run single trading cycle")
    print("  'auto' - Start continuous trading")
    print("  'status' - Show system status")
    print("  'buy SYMBOL QTY' - Manual buy order")
    print("  'sell SYMBOL QTY' - Manual sell order") 
    print("  'close SYMBOL' - Close position")
    print("  'stop' - Stop system")
    print("  'help' - Show this help")
    print("\n")
    
    try:
        while True:
            command = input("trading> ").strip().lower()
            
            if command == "quit" or command == "exit" or command == "stop":
                break
                
            elif command == "cycle":
                result = await app.run_single_cycle("Manual cycle trigger")
                print(f"Cycle result: {result['status']}")
                
            elif command == "auto":
                print("Starting continuous trading mode...")
                print("Press Ctrl+C to stop")
                await app.run_continuous()
                break
                
            elif command == "status":
                status = app.get_status()
                print(f"\nSystem Status:")
                print(f"  Running: {status['system']['running']}")
                print(f"  Cycles: {status['system']['cycle_count']}")
                print(f"  Portfolio: ${status['account']['equity']:.2f}")
                print(f"  Positions: {status['portfolio']['positions']}")
                print(f"  Orders: {status['portfolio']['open_orders']}")
                print(f"  Market: {'Open' if status['market']['open'] else 'Closed'}")
                
            elif command.startswith("buy ") or command.startswith("sell "):
                parts = command.split()
                if len(parts) >= 3:
                    action, symbol, qty = parts[0], parts[1].upper(), float(parts[2])
                    result = await app.manual_trade(symbol, action, qty)
                    print(f"Trade result: {result['status']}")
                else:
                    print("Usage: buy/sell SYMBOL QUANTITY")
                    
            elif command.startswith("close "):
                parts = command.split()
                if len(parts) >= 2:
                    symbol = parts[1].upper()
                    result = await app.manual_trade(symbol, "close")
                    print(f"Close result: {result['status']}")
                else:
                    print("Usage: close SYMBOL")
                    
            elif command == "help":
                print("\nAvailable commands:")
                print("  cycle - Run single trading cycle")
                print("  auto - Start continuous trading")
                print("  status - Show system status")
                print("  buy SYMBOL QTY - Manual buy order")
                print("  sell SYMBOL QTY - Manual sell order")
                print("  close SYMBOL - Close position")
                print("  stop - Stop system")
                
            elif command:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
                
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        app.shutdown()
        print("\nGoodbye! 👋")


if __name__ == "__main__":
    asyncio.run(main())