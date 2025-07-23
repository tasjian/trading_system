#!/usr/bin/env python3
"""
High Frequency Trading Main Application

Entry point for high-frequency trading operations with $25,000 minimum balance
and unlimited day trading capabilities.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hft_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logger.info(f"🛑 Received signal {signum}, shutting down HFT system...")
    sys.exit(0)

async def main():
    """Main HFT application entry point."""
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 HIGH FREQUENCY TRADING SYSTEM")
    print("=" * 60)
    print("⚡ Unlimited day trading with $25,000+ balance")
    print("📈 Advanced order types and risk management")
    print("🎯 Cross-industry diversification")
    print("🔄 Real-time position management")
    print("=" * 60)
    
    try:
        # Import HFT components
        from hft.hft_engine import hft_engine
        from hft.hft_compliance import hft_compliance_manager
        from config.hft_settings import hft_settings
        from tools.alpaca_client import alpaca_client
        
        # Validate account and compliance
        logger.info("🔍 Performing initial compliance check...")
        
        account_info = alpaca_client.get_account_info()
        positions = {pos["symbol"]: pos for pos in alpaca_client.get_positions()}
        
        compliance_result = await hft_compliance_manager.perform_hft_compliance_check(
            account_balance=account_info["equity"],
            positions=positions,
            recent_trades=[]
        )
        
        if not compliance_result.compliant:
            logger.error("❌ HFT compliance check failed - cannot start trading")
            print("\n❌ COMPLIANCE FAILURE")
            for violation in compliance_result.violations:
                print(f"   • {violation}")
            
            if compliance_result.recommendations:
                print("\n💡 RECOMMENDATIONS:")
                for rec in compliance_result.recommendations:
                    print(f"   • {rec}")
            
            return
        
        logger.info("✅ HFT compliance check passed")
        
        # Display account status
        print(f"\n📊 ACCOUNT STATUS")
        print(f"   Balance: ${account_info['equity']:,.2f}")
        print(f"   Buying Power: ${account_info['buying_power']:,.2f}")
        print(f"   PDT Status: {'Yes' if account_info.get('pattern_day_trader', False) else 'No'}")
        print(f"   Active Positions: {len(positions)}")
        print(f"   Available Capital: ${hft_settings.get_available_trading_capital(account_info['equity']):,.2f}")
        
        # Display HFT settings
        print(f"\n⚙️  HFT CONFIGURATION")
        print(f"   Max Trades/Day: {hft_settings.max_trades_per_day}")
        print(f"   Max Position Size: {hft_settings.max_position_size:.1%}")
        print(f"   Max Positions: {hft_settings.max_positions}")
        print(f"   Stop Loss: {hft_settings.stop_loss_percent:.1%}")
        print(f"   Take Profit: {hft_settings.take_profit_percent:.1%}")
        print(f"   Min Time Between Trades: {hft_settings.min_time_between_trades}s")
        
        # Interactive menu
        while True:
            print(f"\n🎮 HFT TRADING MENU")
            print("=" * 40)
            print("1. Start HFT Trading (1 hour)")
            print("2. Start HFT Trading (Custom duration)")
            print("3. Check Account Status")
            print("4. View Compliance Report")
            print("5. View HFT Settings")
            print("6. Test HFT Signals")
            print("7. Exit")
            
            try:
                choice = input("\nSelect option (1-7): ").strip()
                
                if choice == "1":
                    print("\n🚀 Starting HFT trading for 1 hour...")
                    await hft_engine.start_hft_trading(duration_minutes=60)
                
                elif choice == "2":
                    try:
                        duration = int(input("Enter duration in minutes: "))
                        if duration <= 0 or duration > 480:  # Max 8 hours
                            print("❌ Duration must be between 1-480 minutes")
                            continue
                        
                        print(f"\n🚀 Starting HFT trading for {duration} minutes...")
                        await hft_engine.start_hft_trading(duration_minutes=duration)
                    except ValueError:
                        print("❌ Invalid duration")
                
                elif choice == "3":
                    await display_account_status()
                
                elif choice == "4":
                    await display_compliance_report()
                
                elif choice == "5":
                    display_hft_settings()
                
                elif choice == "6":
                    await test_hft_signals()
                
                elif choice == "7":
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid option")
                    
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Menu error: {e}")
                print(f"❌ Error: {e}")
    
    except Exception as e:
        logger.error(f"HFT application error: {e}")
        print(f"❌ Critical error: {e}")

async def display_account_status():
    """Display current account status."""
    try:
        from tools.alpaca_client import alpaca_client
        from config.hft_settings import hft_settings
        
        account_info = alpaca_client.get_account_info()
        positions = alpaca_client.get_positions()
        
        print(f"\n📊 DETAILED ACCOUNT STATUS")
        print("=" * 50)
        print(f"Account ID: {account_info['id']}")
        print(f"Account Status: {account_info['status']}")
        print(f"Equity: ${account_info['equity']:,.2f}")
        print(f"Cash: ${account_info['cash']:,.2f}")
        print(f"Buying Power: ${account_info['buying_power']:,.2f}")
        print(f"Day Trading Buying Power: ${account_info.get('daytrading_buying_power', 0):,.2f}")
        print(f"Day Trade Count: {account_info.get('day_trade_count', 0)}")
        print(f"Pattern Day Trader: {'Yes' if account_info.get('pattern_day_trader', False) else 'No'}")
        
        if positions:
            print(f"\n📈 ACTIVE POSITIONS ({len(positions)})")
            print("-" * 50)
            total_value = 0
            for pos in positions[:10]:  # Show first 10
                market_value = float(pos["market_value"])
                total_value += abs(market_value)
                pnl = float(pos["unrealized_pl"])
                print(f"{pos['symbol']:6} {pos['qty']:>8} shares ${market_value:>10,.2f} P&L: ${pnl:>8,.2f}")
            
            if len(positions) > 10:
                print(f"... and {len(positions) - 10} more positions")
            
            print(f"\nTotal Position Value: ${total_value:,.2f}")
            
            # Check compliance with HFT limits
            max_position_value = hft_settings.get_max_position_value(account_info['equity'])
            utilization = total_value / account_info['equity'] if account_info['equity'] > 0 else 0
            
            print(f"Account Utilization: {utilization:.1%}")
            print(f"Max Position Size: ${max_position_value:,.2f}")
        else:
            print("\n📈 No active positions")
        
        # Check minimum balance compliance
        min_balance_ok = account_info['equity'] >= hft_settings.minimum_account_balance
        balance_status = "✅ OK" if min_balance_ok else "❌ BELOW MINIMUM"
        print(f"\nMinimum Balance Check: {balance_status}")
        print(f"Required: ${hft_settings.minimum_account_balance:,.2f}")
        print(f"Available Capital: ${hft_settings.get_available_trading_capital(account_info['equity']):,.2f}")
        
    except Exception as e:
        logger.error(f"Account status error: {e}")
        print(f"❌ Error getting account status: {e}")

async def display_compliance_report():
    """Display HFT compliance report."""
    try:
        from hft.hft_compliance import hft_compliance_manager
        from tools.alpaca_client import alpaca_client
        
        print(f"\n🔍 HFT COMPLIANCE REPORT")
        print("=" * 50)
        
        # Get current compliance status
        account_info = alpaca_client.get_account_info()
        positions = {pos["symbol"]: pos for pos in alpaca_client.get_positions()}
        
        compliance_result = await hft_compliance_manager.perform_hft_compliance_check(
            account_balance=account_info["equity"],
            positions=positions,
            recent_trades=[]
        )
        
        status = "✅ COMPLIANT" if compliance_result.compliant else "❌ NON-COMPLIANT"
        print(f"Overall Status: {status}")
        print(f"Compliance Score: {compliance_result.compliance_score:.1%}")
        print(f"Checks Performed: {compliance_result.checks_performed}")
        
        if compliance_result.violations:
            print(f"\n❌ VIOLATIONS ({len(compliance_result.violations)}):")
            for i, violation in enumerate(compliance_result.violations, 1):
                print(f"   {i}. {violation}")
        
        if compliance_result.warnings:
            print(f"\n⚠️  WARNINGS ({len(compliance_result.warnings)}):")
            for i, warning in enumerate(compliance_result.warnings, 1):
                print(f"   {i}. {warning}")
        
        if compliance_result.recommendations:
            print(f"\n💡 RECOMMENDATIONS ({len(compliance_result.recommendations)}):")
            for i, rec in enumerate(compliance_result.recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Show compliance history summary
        summary = hft_compliance_manager.get_compliance_summary()
        if summary.get("total_checks", 0) > 0:
            print(f"\n📊 COMPLIANCE HISTORY")
            print(f"   Total Checks: {summary['total_checks']}")
            print(f"   Recent Average Score: {summary.get('recent_average_score', 0):.1%}")
            print(f"   Trend: {summary.get('compliance_trend', 'unknown').title()}")
            
            if summary.get("frequent_violations"):
                print(f"   Most Frequent Issues:")
                for violation, count in list(summary["frequent_violations"].items())[:3]:
                    print(f"     • {violation} ({count} times)")
        
    except Exception as e:
        logger.error(f"Compliance report error: {e}")
        print(f"❌ Error generating compliance report: {e}")

def display_hft_settings():
    """Display HFT configuration settings."""
    try:
        from config.hft_settings import hft_settings
        
        print(f"\n⚙️  HFT CONFIGURATION SETTINGS")
        print("=" * 50)
        
        print(f"ACCOUNT REQUIREMENTS:")
        print(f"   Minimum Balance: ${hft_settings.minimum_account_balance:,.2f}")
        print(f"   Cash Reserve: ${hft_settings.minimum_cash_reserve:,.2f}")
        print(f"   Max Utilization: {hft_settings.max_account_utilization:.1%}")
        
        print(f"\nTRADING FREQUENCY:")
        print(f"   Max Trades/Minute: {hft_settings.max_trades_per_minute}")
        print(f"   Max Trades/Hour: {hft_settings.max_trades_per_hour}")
        print(f"   Max Trades/Day: {hft_settings.max_trades_per_day}")
        print(f"   Min Time Between Trades: {hft_settings.min_time_between_trades}s")
        
        print(f"\nPOSITION MANAGEMENT:")
        print(f"   Max Positions: {hft_settings.max_positions}")
        print(f"   Min Position Size: ${hft_settings.min_position_size}")
        print(f"   Max Position Size: {hft_settings.max_position_size:.1%}")
        print(f"   Position Hold Time: {hft_settings.position_hold_time_min}-{hft_settings.position_hold_time_max} min")
        
        print(f"\nRISK MANAGEMENT:")
        print(f"   Max Daily Loss: {hft_settings.max_daily_loss:.1%}")
        print(f"   Max Drawdown: {hft_settings.max_drawdown:.1%}")
        print(f"   Stop Loss: {hft_settings.stop_loss_percent:.1%}")
        print(f"   Take Profit: {hft_settings.take_profit_percent:.1%}")
        
        print(f"\nTECHNICAL INDICATORS:")
        print(f"   Price Momentum Threshold: {hft_settings.price_momentum_threshold:.1%}")
        print(f"   Volume Spike Threshold: {hft_settings.volume_spike_threshold}x")
        print(f"   RSI Oversold: {hft_settings.rsi_oversold}")
        print(f"   RSI Overbought: {hft_settings.rsi_overbought}")
        
        print(f"\nTRADING UNIVERSE:")
        print(f"   Symbols: {len(hft_settings.tick_data_symbols)} stocks")
        print(f"   Top symbols: {', '.join(hft_settings.tick_data_symbols[:10])}")
        
    except Exception as e:
        logger.error(f"Settings display error: {e}")
        print(f"❌ Error displaying settings: {e}")

async def test_hft_signals():
    """Test HFT signal generation."""
    try:
        from hft.hft_engine import hft_engine
        
        print(f"\n🧪 TESTING HFT SIGNAL GENERATION")
        print("=" * 50)
        
        # Initialize engine
        if not await hft_engine.initialize_engine():
            print("❌ Failed to initialize HFT engine")
            return
        
        print("🔄 Generating test signals...")
        
        # Generate signals for a few symbols
        signals = await hft_engine._generate_hft_signals()
        
        if not signals:
            print("📊 No signals generated (market conditions may not be suitable)")
            return
        
        print(f"✅ Generated {len(signals)} signals:")
        
        for i, signal in enumerate(signals, 1):
            print(f"\n   Signal {i}:")
            print(f"     Symbol: {signal.symbol}")
            print(f"     Type: {signal.signal_type}")
            print(f"     Direction: {signal.direction}")
            print(f"     Strength: {signal.strength:.2f}")
            print(f"     Confidence: {signal.confidence:.2f}")
            print(f"     Price Target: ${signal.price_target:.2f}")
            print(f"     Stop Loss: ${signal.stop_loss:.2f}")
            print(f"     Time Horizon: {signal.time_horizon} minutes")
            
            # Check if signal would be executed
            would_execute = hft_engine._should_execute_signal(signal)
            execute_status = "✅ Would Execute" if would_execute else "❌ Would Skip"
            print(f"     Execution: {execute_status}")
        
        print(f"\n📈 Signal Quality Summary:")
        avg_strength = sum(s.strength for s in signals) / len(signals)
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        signal_types = set(s.signal_type for s in signals)
        
        print(f"   Average Strength: {avg_strength:.2f}")
        print(f"   Average Confidence: {avg_confidence:.2f}")
        print(f"   Signal Types: {', '.join(signal_types)}")
        
    except Exception as e:
        logger.error(f"Signal test error: {e}")
        print(f"❌ Error testing signals: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 HFT system interrupted")
    except Exception as e:
        logger.error(f"HFT main error: {e}")
        print(f"❌ Critical error: {e}")