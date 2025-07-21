#!/usr/bin/env python3
"""Demo script to showcase the agentic trading system capabilities."""

import asyncio
import json
from datetime import datetime
from config.settings import settings

async def run_demo():
    """Run a comprehensive demo of the trading system."""
    
    print("🎬 AGENTIC TRADING SYSTEM DEMO")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Trading Mode: {settings.trading_mode.upper()}")
    print("=" * 60)
    
    try:
        # Demo 1: Test API Connection
        print("\n📡 Demo 1: API Connection Test")
        print("-" * 30)
        
        from tools.alpaca_client import alpaca_client
        account_info = alpaca_client.get_account_info()
        
        print(f"✓ Connected to Alpaca account: {account_info['id']}")
        print(f"✓ Account status: {account_info['status']}")
        print(f"✓ Portfolio value: ${account_info['equity']:,.2f}")
        print(f"✓ Cash available: ${account_info['cash']:,.2f}")
        print(f"✓ Buying power: ${account_info['buying_power']:,.2f}")
        
        # Demo 2: Risk Assessment
        print("\n🛡️  Demo 2: Risk Assessment")
        print("-" * 30)
        
        from tools.risk_controls import risk_monitor
        assessment = risk_monitor.assess_portfolio_risk()
        
        print(f"✓ Overall risk level: {assessment['overall_risk'].value.upper()}")
        print(f"✓ Risk score: {assessment['risk_score']:.1f}/100")
        print(f"✓ Trading halted: {'Yes' if risk_monitor.is_trading_halted() else 'No'}")
        print(f"✓ Active alerts: {len(assessment['alerts'])}")
        
        if assessment['alerts']:
            print("   Alert details:")
            for alert in assessment['alerts'][:3]:  # Show first 3
                print(f"   - {alert.level.value}: {alert.message}")
        
        # Demo 3: Market Data Retrieval
        print("\n📊 Demo 3: Market Data Analysis")
        print("-" * 30)
        
        symbols = ["SPY", "AAPL", "MSFT"]
        for symbol in symbols:
            try:
                market_data = alpaca_client.get_market_data(symbol, limit=5)
                if not market_data.empty:
                    latest = market_data.iloc[-1]
                    change = latest["close"] - latest["open"]
                    change_pct = (change / latest["open"]) * 100
                    
                    print(f"✓ {symbol}: ${latest['close']:.2f} "
                          f"({change:+.2f}, {change_pct:+.2f}%)")
            except Exception as e:
                print(f"⚠️ {symbol}: Error getting data - {e}")
        
        # Demo 4: Technical Analysis
        print("\n📈 Demo 4: Technical Indicators")
        print("-" * 30)
        
        from tools.trading_tools import calculate_technical_indicators
        
        for symbol in ["SPY", "AAPL"]:
            indicators = calculate_technical_indicators(symbol, ["sma", "rsi"])
            if "error" not in indicators:
                sma_data = indicators["indicators"].get("sma", {})
                rsi_value = indicators["indicators"].get("rsi")
                
                print(f"✓ {symbol} Technical Analysis:")
                if sma_data.get("sma_20"):
                    print(f"   - SMA(20): ${sma_data['sma_20']:.2f}")
                if sma_data.get("sma_50"):
                    print(f"   - SMA(50): ${sma_data['sma_50']:.2f}")
                if rsi_value:
                    print(f"   - RSI(14): {rsi_value:.1f}")
            else:
                print(f"⚠️ {symbol}: {indicators['error']}")
        
        # Demo 5: Trading Workflow Simulation
        print("\n🤖 Demo 5: Agent Workflow Simulation")
        print("-" * 30)
        
        from agents.workflow import trading_workflow
        
        print("⚡ Running single trading cycle...")
        result = await trading_workflow.run_cycle(
            session_id="demo_session",
            input_message="Demo trading cycle - market analysis"
        )
        
        if result["status"] == "success":
            print("✓ Trading cycle completed successfully")
            print(f"✓ Portfolio value: ${result.get('portfolio_value', 0):,.2f}")
            print(f"✓ Active positions: {result.get('active_positions', 0)}")
            print(f"✓ Signals generated: {result.get('signals_generated', 0)}")
            print(f"✓ Orders executed: {result.get('orders_executed', 0)}")
        else:
            print(f"⚠️ Trading cycle encountered issues: {result.get('error', 'Unknown')}")
        
        # Demo 6: Risk Validation
        print("\n🔍 Demo 6: Trade Risk Validation")
        print("-" * 30)
        
        from tools.trading_tools import validate_trade_risk
        
        test_trades = [
            ("AAPL", 10, "buy"),
            ("SPY", 100, "buy"),
            ("MSFT", 5, "sell")
        ]
        
        for symbol, qty, side in test_trades:
            validation = validate_trade_risk(symbol, qty, side)
            if "error" not in validation:
                status = "✓ APPROVED" if validation["validation_passed"] else "❌ REJECTED"
                print(f"{status} {side.upper()} {qty} {symbol}")
                
                if not validation["validation_passed"]:
                    failed_checks = [
                        check_name for check_name, check_result in validation["checks"].items()
                        if not check_result.get("passed", True)
                    ]
                    print(f"   Failed checks: {', '.join(failed_checks)}")
            else:
                print(f"⚠️ {symbol}: {validation['error']}")
        
        # Demo 7: System Status Summary
        print("\n📋 Demo 7: System Status Summary")
        print("-" * 30)
        
        from main import TradingSystemApp
        
        # Create app instance for status
        app = TradingSystemApp()
        status = app.get_status()
        
        print("✓ System Status:")
        print(f"   - Session ID: {status['system']['session_id'][:8]}...")
        print(f"   - Trading mode: {status['system']['trading_mode']}")
        print(f"   - Market open: {'Yes' if status['market']['open'] else 'No'}")
        print(f"   - Portfolio positions: {status['portfolio']['positions']}")
        print(f"   - Open orders: {status['portfolio']['open_orders']}")
        print(f"   - Day trades used: {status['account']['day_trades']}")
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("🚀 The system is ready for operation!")
        print("📝 Run 'python main.py' to start interactive mode")
        print("📚 See README.md for detailed usage instructions")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("🔧 Please check your configuration and API keys")
        import traceback
        traceback.print_exc()

def main():
    """Main demo function."""
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")

if __name__ == "__main__":
    main()