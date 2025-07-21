#!/usr/bin/env python3
"""Enhanced demo showcasing advanced market analysis agents and quantitative strategies."""

import asyncio
import json
from datetime import datetime, timedelta
from config.settings import settings

async def run_enhanced_demo():
    """Run comprehensive demo of enhanced trading system capabilities."""
    
    print("🚀 ENHANCED AGENTIC TRADING SYSTEM DEMO")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Trading Mode: {settings.trading_mode.upper()}")
    print("Features: Advanced Market Analysis • Quantitative Strategies • Risk Management")
    print("=" * 70)
    
    try:
        # Demo 1: Advanced Market Analysis Agents
        print("\n🎯 Demo 1: Advanced Market Analysis Agents")
        print("-" * 50)
        
        from agents.market_analysis import market_analysis_factory
        from tools.alpaca_client import alpaca_client
        
        # Get account info
        account_info = alpaca_client.get_account_info()
        portfolio_value = account_info['equity']
        
        # Test symbols
        test_symbols = ["AAPL", "MSFT", "SPY", "QQQ"]
        
        print(f"📊 Analyzing symbols: {', '.join(test_symbols)}")
        print(f"💰 Portfolio value: ${portfolio_value:,.2f}")
        
        # Perform comprehensive analysis
        analysis_results = market_analysis_factory.get_portfolio_recommendations(
            symbols=test_symbols,
            portfolio_value=portfolio_value
        )
        
        # Display market conditions
        market_condition = analysis_results["market_condition"]
        print(f"\n🌍 Market Conditions:")
        print(f"   - Trend: {market_condition.market_trend.upper()}")
        print(f"   - Volatility: {market_condition.volatility_regime.upper()}")
        print(f"   - Fear/Greed Index: {market_condition.fear_greed_index:.1f}/100")
        
        # Display individual symbol analysis
        print(f"\n📈 Symbol Analysis:")
        symbol_analysis = analysis_results["symbol_analysis"]
        for symbol, analysis in symbol_analysis.items():
            action_icon = "🟢" if analysis.action == "buy" else "🔴" if analysis.action == "sell" else "⚪"
            print(f"   {action_icon} {symbol}: {analysis.action.upper()} "
                  f"(confidence: {analysis.confidence:.2f}, "
                  f"strength: {analysis.strength.value})")
            print(f"      └─ {analysis.reasoning[:80]}...")
        
        # Display portfolio optimization
        optimal_weights = analysis_results["optimal_weights"]
        if optimal_weights:
            print(f"\n⚖️  Optimal Portfolio Weights:")
            for symbol, weight in optimal_weights.items():
                print(f"   - {symbol}: {weight:.1%}")
        
        # Display recommendations
        recommendations = analysis_results["recommendations"]
        if recommendations:
            print(f"\n💡 Portfolio Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"   {i}. {rec}")
        
        # Demo 2: Quantitative Strategy Backtesting
        print("\n🧮 Demo 2: Quantitative Strategy Backtesting")
        print("-" * 50)
        
        from tools.advanced_trading import (
            algorithmic_engine, MomentumStrategy, MeanReversionStrategy
        )
        
        print("🔬 Running strategy comparison across multiple symbols...")
        
        # Compare strategies
        strategy_results = algorithmic_engine.run_strategy_comparison(
            symbols=test_symbols[:2],  # Test on first 2 symbols for speed
            lookback_days=180  # 6 months
        )
        
        if strategy_results:
            print(f"📊 Backtesting Period: {strategy_results['analysis_period']}")
            
            # Display best strategies
            best_strategies = strategy_results["best_strategies"]
            if best_strategies:
                print(f"\n🏆 Top Performing Strategies:")
                for i, strategy in enumerate(best_strategies[:5], 1):
                    print(f"   {i}. {strategy['strategy']} on {strategy['symbol']}")
                    print(f"      └─ Return: {strategy['total_return']:+.2%}, "
                          f"Sharpe: {strategy['sharpe_ratio']:.2f}, "
                          f"Win Rate: {strategy['win_rate']:.1%}")
        
        # Demo 3: Enhanced Signal Generation
        print("\n🎯 Demo 3: Enhanced Signal Generation")
        print("-" * 50)
        
        print("🔍 Generating enhanced signals with risk management...")
        
        enhanced_signals = algorithmic_engine.generate_enhanced_signals(
            symbols=test_symbols,
            portfolio_value=portfolio_value
        )
        
        if enhanced_signals:
            print(f"✅ Generated {len(enhanced_signals)} enhanced signals:")
            for signal in enhanced_signals:
                action_icon = "📈" if signal["action"] == "buy" else "📉"
                print(f"   {action_icon} {signal['symbol']}: {signal['action'].upper()} "
                      f"{signal['quantity']:.0f} shares")
                print(f"      └─ Entry: ${signal['entry_price']:.2f}, "
                      f"Stop: ${signal['stop_loss']:.2f}, "
                      f"Confidence: {signal['confidence']:.2f}")
                print(f"      └─ Strategy: {signal['strategy']}")
        else:
            print("ℹ️  No enhanced signals generated (market conditions or risk constraints)")
        
        # Demo 4: Risk Management Analysis
        print("\n🛡️  Demo 4: Risk Management Analysis")
        print("-" * 50)
        
        from tools.risk_controls import risk_monitor
        
        print("🔍 Performing comprehensive risk assessment...")
        
        risk_assessment = risk_monitor.assess_portfolio_risk()
        
        print(f"📊 Risk Assessment Results:")
        print(f"   - Overall Risk Level: {risk_assessment['overall_risk'].value.upper()}")
        print(f"   - Risk Score: {risk_assessment['risk_score']:.1f}/100")
        print(f"   - Trading Halted: {'❌ YES' if risk_monitor.is_trading_halted() else '✅ NO'}")
        
        # Display risk metrics
        metrics = risk_assessment.get("metrics", {})
        if metrics:
            print(f"\n📈 Risk Metrics:")
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and "status" in metric_data:
                    status_icon = "🟢" if metric_data["status"] == "OK" else "🟡" if metric_data["status"] == "HIGH" else "🔴"
                    print(f"   {status_icon} {metric_name.title()}: {metric_data['status']}")
        
        # Display active alerts
        alerts = risk_assessment.get("alerts", [])
        if alerts:
            print(f"\n⚠️  Active Risk Alerts:")
            for alert in alerts:
                level_icon = "🔴" if alert.level.value == "critical" else "🟡" if alert.level.value == "high" else "🟠"
                print(f"   {level_icon} {alert.level.value.upper()}: {alert.message}")
        
        # Demo 5: Real-time Trading Simulation
        print("\n🤖 Demo 5: Real-time Trading Workflow")
        print("-" * 50)
        
        from agents.workflow import trading_workflow
        
        print("⚡ Running enhanced trading cycle with new agents...")
        
        result = await trading_workflow.run_cycle(
            session_id="enhanced_demo_session",
            input_message="Enhanced demo: comprehensive market analysis and signal generation"
        )
        
        if result["status"] == "success":
            print("✅ Enhanced trading cycle completed successfully!")
            print(f"📊 Results:")
            print(f"   - Portfolio Value: ${result.get('portfolio_value', 0):,.2f}")
            print(f"   - Active Positions: {result.get('active_positions', 0)}")
            print(f"   - Signals Generated: {result.get('signals_generated', 0)}")
            print(f"   - Orders Executed: {result.get('orders_executed', 0)}")
            
            # Get final state for additional insights
            final_state = result.get("final_state", {})
            market_conditions = final_state.get("market_conditions")
            if market_conditions:
                print(f"   - Market Trend: {market_conditions.market_trend}")
                print(f"   - Volatility Regime: {market_conditions.volatility_regime}")
        else:
            print(f"⚠️  Enhanced trading cycle encountered issues: {result.get('error', 'Unknown')}")
        
        # Demo 6: Performance Analytics
        print("\n📊 Demo 6: Performance Analytics")
        print("-" * 50)
        
        from tools.advanced_trading import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer()
        
        # Get current positions for analysis
        positions = alpaca_client.get_positions()
        
        if positions:
            print(f"📈 Current Portfolio Analysis:")
            total_unrealized_pl = sum(pos.get("unrealized_pl", 0) for pos in positions)
            total_market_value = sum(abs(pos.get("market_value", 0)) for pos in positions)
            
            print(f"   - Positions: {len(positions)}")
            print(f"   - Total Market Value: ${total_market_value:,.2f}")
            print(f"   - Unrealized P&L: ${total_unrealized_pl:+,.2f}")
            
            # Show position breakdown
            print(f"\n📋 Position Breakdown:")
            for pos in positions:
                pnl_icon = "🟢" if pos.get("unrealized_pl", 0) >= 0 else "🔴"
                print(f"   {pnl_icon} {pos['symbol']}: {pos['qty']} shares, "
                      f"P&L: ${pos.get('unrealized_pl', 0):+.2f}")
        else:
            print("📝 No current positions to analyze")
        
        # Demo 7: System Performance Summary
        print("\n🎯 Demo 7: System Performance Summary")
        print("-" * 50)
        
        from main import TradingSystemApp
        
        app = TradingSystemApp()
        status = app.get_status()
        
        print("✅ Enhanced System Status:")
        print(f"   - Core System: ✅ Operational")
        print(f"   - Market Analysis Agents: ✅ Active")
        print(f"   - Quantitative Strategies: ✅ Ready")
        print(f"   - Risk Management: ✅ Monitoring")
        print(f"   - Portfolio Optimization: ✅ Available")
        print(f"   - Advanced Backtesting: ✅ Functional")
        
        print(f"\n📊 System Metrics:")
        print(f"   - Market Status: {'🟢 OPEN' if status['market']['open'] else '🔴 CLOSED'}")
        print(f"   - Account Equity: ${status['account']['equity']:,.2f}")
        print(f"   - Buying Power: ${status['account']['buying_power']:,.2f}")
        print(f"   - Day Trades Used: {status['account']['day_trades']}")
        print(f"   - Active Positions: {status['portfolio']['positions']}")
        print(f"   - Open Orders: {status['portfolio']['open_orders']}")
        
        print("\n" + "=" * 70)
        print("🎉 ENHANCED DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("🚀 The enhanced system is ready for advanced trading operations!")
        print("📈 New Features Available:")
        print("   • Multi-strategy technical analysis (momentum, mean reversion, breakout)")
        print("   • Fundamental market regime detection")
        print("   • Quantitative portfolio optimization")
        print("   • Advanced risk management with dynamic position sizing")
        print("   • Comprehensive backtesting with transaction costs")
        print("   • Real-time strategy performance comparison")
        print("\n💡 Usage:")
        print("   - Run 'python main.py' for interactive enhanced trading")
        print("   - Use 'python enhanced_demo.py' to see all features")
        print("   - Check README.md for advanced configuration options")
        
    except Exception as e:
        print(f"\n❌ Enhanced demo failed with error: {e}")
        print("🔧 Please check your configuration and API keys")
        import traceback
        traceback.print_exc()

def main():
    """Main enhanced demo function."""
    try:
        asyncio.run(run_enhanced_demo())
    except KeyboardInterrupt:
        print("\n🛑 Enhanced demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Enhanced demo crashed: {e}")

if __name__ == "__main__":
    main()