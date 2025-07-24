#!/usr/bin/env python3
"""
Simplified Main Trading Application

A clean, unified interface to the trading system that demonstrates
all major capabilities:
- Market intelligence and analysis
- Trading execution
- Portfolio management
- Strategy execution (including pairs trading)
- Performance monitoring

This replaces the complex multi-file system with a single, 
easy-to-use application.
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict
from datetime import datetime

# Core unified components
from core import (
    analyze_stock, analyze_portfolio,
    execute_signal, rebalance_portfolio, get_portfolio_metrics,
    market_intelligence, trading_engine,
    SignalType, ConfidenceLevel
)

# Enhanced market screener for stock selection
from enhanced_market_screener import enhanced_screener

# Pairs trading strategy
from strategies.pairs_trading import PairsStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simplified_trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SimplifiedTradingSystem:
    """
    Simplified trading system that demonstrates all capabilities
    in a clean, easy-to-understand interface.
    """
    
    def __init__(self):
        self.pairs_strategy = PairsStrategy()
        self.running = False
    
    async def run_market_analysis(self, symbols: List[str] = None) -> None:
        """Run comprehensive market analysis on selected stocks."""
        print("\n🔍 MARKET ANALYSIS")
        print("=" * 50)
        
        if not symbols:
            # Get diversified stock selection
            selection = enhanced_screener.get_diversified_stock_selection(10)
            symbols = []
            for industry_stocks in selection.values():
                symbols.extend(industry_stocks[:2])  # Top 2 from each industry
        
        # Analyze all symbols
        analyses = await analyze_portfolio(symbols)
        
        # Display results
        for analysis in analyses:
            signal_emoji = {
                'strong_buy': '🚀',
                'buy': '📈',
                'hold': '⏸️',
                'sell': '📉',
                'strong_sell': '🔴'
            }.get(analysis.signal.value, '❓')
            
            confidence_emoji = {
                'very_high': '⭐⭐⭐',
                'high': '⭐⭐',
                'medium': '⭐',
                'low': '🔸',
                'very_low': '🔹'
            }.get(analysis.confidence.value, '❓')
            
            print(f"{signal_emoji} {analysis.symbol:6} | {analysis.signal.value:11} | "
                  f"Score: {analysis.score:6.2f} | Confidence: {confidence_emoji}")
            
            if analysis.target_price:
                print(f"    💰 Current: ${analysis.price:.2f} → Target: ${analysis.target_price:.2f}")
            
            if analysis.reasoning:
                print(f"    💭 {analysis.reasoning[0]}")
            print()
        
        return analyses
    
    async def run_trading_signals(self, analyses: List = None) -> None:
        """Execute trading based on market analysis signals."""
        print("\n🎯 TRADING EXECUTION")
        print("=" * 50)
        
        if not analyses:
            # Run analysis first
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
            analyses = await analyze_portfolio(symbols)
        
        executed_orders = []
        
        for analysis in analyses:
            # Only trade on high-confidence signals
            if analysis.confidence.value in ['high', 'very_high']:
                if analysis.signal.value in ['buy', 'strong_buy', 'sell', 'strong_sell']:
                    order = await execute_signal(analysis)
                    if order:
                        executed_orders.append(order)
                        print(f"✅ {order.side.upper()} {order.quantity:.0f} {order.symbol}")
                        print(f"    📝 {order.reasoning}")
                    else:
                        print(f"❌ Failed to execute signal for {analysis.symbol}")
            else:
                print(f"⏭️  Skipped {analysis.symbol} - Low confidence ({analysis.confidence.value})")
        
        print(f"\n📊 Executed {len(executed_orders)} orders")
        return executed_orders
    
    async def run_pairs_trading(self) -> None:
        """Run pairs trading strategy."""
        print("\n🔄 PAIRS TRADING")
        print("=" * 50)
        
        try:
            # Define symbols for pairs analysis (including some potentially cointegrated pairs)
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'BAC', 'WFC', 'C']
            
            # Get historical price data for all symbols
            print("📈 Fetching historical data for pairs analysis...")
            price_data = {}
            
            for symbol in symbols:
                try:
                    # Get 1 year of historical data for cointegration testing using yfinance directly
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    hist_data = ticker.history(period="1y")  # 1 year
                    
                    if not hist_data.empty:
                        price_data[symbol] = hist_data['Close']
                        print(f"  ✅ {symbol}: {len(hist_data)} days of data")
                    else:
                        print(f"  ❌ {symbol}: No historical data available")
                except Exception as e:
                    print(f"  ❌ {symbol}: Error getting data - {e}")
            
            if len(price_data) < 2:
                print("❌ Insufficient data for pairs analysis")
                return
            
            # Find trading pairs
            print(f"\n🔍 Analyzing {len(price_data)} symbols for cointegrated pairs...")
            pairs = self.pairs_strategy.find_pairs(list(price_data.keys()), price_data)
            
            if not pairs:
                print("❌ No suitable cointegrated pairs found")
                return
            
            print(f"✅ Found {len(pairs)} potential trading pairs")
            
            # Display top pairs
            for i, pair in enumerate(pairs[:5], 1):  # Top 5 pairs
                print(f"\n📊 Pair #{i}: {pair.symbol_a}/{pair.symbol_b}")
                print(f"    Correlation: {pair.correlation:.3f}")
                print(f"    Cointegration p-value: {pair.p_value:.4f}")
                print(f"    Hedge ratio (beta): {pair.beta:.3f}")
                print(f"    Half-life: {pair.half_life:.1f} days")
                print(f"    Quality score: {pair.score:.3f}")
                
                # Calculate current z-score for demonstration
                try:
                    current_spread, current_zscore = self.pairs_strategy.calculate_spread_zscore(
                        pair.symbol_a, pair.symbol_b, pair.beta, price_data
                    )
                    
                    if not np.isnan(current_zscore):
                        # Determine signal based on z-score thresholds
                        if abs(current_zscore) > 2.0:
                            signal_type = "ENTRY" if current_zscore > 2.0 else "ENTRY (Short)"
                            signal_strength = "Strong" if abs(current_zscore) > 2.5 else "Moderate"
                            print(f"    🎯 {signal_type}: Z-score = {current_zscore:.2f} ({signal_strength})")
                            
                            # Show trade direction
                            if current_zscore > 2.0:
                                print(f"       → SELL {pair.symbol_a}, BUY {pair.symbol_b}")
                            else:
                                print(f"       → BUY {pair.symbol_a}, SELL {pair.symbol_b}")
                        else:
                            print(f"    ⏸️  No signal: Z-score = {current_zscore:.2f} (within ±2.0 threshold)")
                    else:
                        print(f"    ⚠️  Cannot calculate current z-score")
                        
                except Exception as signal_error:
                    print(f"    ⚠️  Z-score calculation error: {signal_error}")
            
            print(f"\n📊 Pairs trading analysis complete - {len(pairs)} pairs identified")
            
        except Exception as e:
            logger.error(f"Pairs trading error: {e}")
            print(f"❌ Pairs trading failed: {e}")
    
    async def run_portfolio_management(self) -> None:
        """Run portfolio analysis and management."""
        print("\n💼 PORTFOLIO MANAGEMENT")
        print("=" * 50)
        
        try:
            # Get current portfolio metrics
            metrics = await get_portfolio_metrics()
            
            print(f"💰 Total Value: ${metrics.total_value:,.2f}")
            print(f"💵 Cash: ${metrics.cash:,.2f}")
            print(f"📈 Equity: ${metrics.equity:,.2f}")
            print(f"📊 Day P&L: ${metrics.day_pnl:,.2f}")
            
            print(f"\n📋 Positions ({len(metrics.positions)}):")
            for position in metrics.positions:
                pnl_emoji = "📈" if position.unrealized_pnl >= 0 else "📉"
                print(f"  {position.symbol:6} | {position.quantity:8.0f} shares | "
                      f"${position.market_value:8,.2f} | {pnl_emoji} ${position.unrealized_pnl:6.2f}")
            
            print(f"\n🎯 Risk Metrics:")
            for metric, value in metrics.risk_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
                else:
                    print(f"  {metric.replace('_', ' ').title()}: {value}")
            
            # Suggest rebalancing if needed
            if metrics.risk_metrics.get('concentration_risk', 0) > 0.25:
                print("\n⚠️  High concentration risk detected - consider rebalancing")
            
        except Exception as e:
            logger.error(f"Portfolio management error: {e}")
            print(f"❌ Portfolio analysis failed: {e}")
    
    async def run_performance_summary(self) -> None:
        """Display trading performance summary."""
        print("\n📊 PERFORMANCE SUMMARY")
        print("=" * 50)
        
        try:
            performance = trading_engine.get_performance_summary()
            
            print(f"📈 Total Orders: {performance['total_orders']}")
            print(f"⏳ Pending Orders: {performance['pending_orders']}")
            print(f"✅ Success Rate: {performance['success_rate']:.1%}")
            
            if performance['strategies']:
                print(f"\n🎯 Strategy Breakdown:")
                for strategy, count in performance['strategies'].items():
                    print(f"  {strategy.replace('_', ' ').title()}: {count} orders")
            
            if performance.get('recent_orders'):
                print(f"\n📋 Recent Orders:")
                for order in performance['recent_orders'][-5:]:  # Last 5
                    timestamp = datetime.fromisoformat(order['timestamp']).strftime('%H:%M')
                    print(f"  {timestamp} | {order['side'].upper()} {order['quantity']:.0f} {order['symbol']} | {order['strategy']}")
            
        except Exception as e:
            logger.error(f"Performance summary error: {e}")
            print(f"❌ Performance summary failed: {e}")
    
    async def run_full_cycle(self) -> None:
        """Run a complete trading cycle with all components."""
        print("\n🚀 FULL TRADING CYCLE")
        print("=" * 80)
        
        try:
            # 1. Market Analysis
            analyses = await self.run_market_analysis()
            
            # 2. Trading Execution (with high-confidence signals only)
            high_confidence_analyses = [
                a for a in analyses 
                if a.confidence.value in ['high', 'very_high']
                and a.signal.value != 'hold'
            ]
            
            if high_confidence_analyses:
                await self.run_trading_signals(high_confidence_analyses)
            else:
                print("\n⏸️  No high-confidence trading signals found")
            
            # 3. Pairs Trading
            await self.run_pairs_trading()
            
            # 4. Portfolio Management
            await self.run_portfolio_management()
            
            # 5. Performance Summary
            await self.run_performance_summary()
            
            print("\n✅ Full trading cycle complete!")
            
        except Exception as e:
            logger.error(f"Full cycle error: {e}")
            print(f"❌ Full cycle failed: {e}")
    
    async def interactive_mode(self) -> None:
        """Run interactive trading system."""
        print("\n🎮 INTERACTIVE TRADING SYSTEM")
        print("=" * 50)
        print("Commands:")
        print("  1 - Market Analysis")
        print("  2 - Trading Execution") 
        print("  3 - Pairs Trading")
        print("  4 - Portfolio Management")
        print("  5 - Performance Summary")
        print("  6 - Full Trading Cycle")
        print("  q - Quit")
        print()
        
        while True:
            try:
                choice = input("Enter command: ").strip().lower()
                
                if choice == 'q':
                    break
                elif choice == '1':
                    await self.run_market_analysis()
                elif choice == '2':
                    await self.run_trading_signals()
                elif choice == '3':
                    await self.run_pairs_trading()
                elif choice == '4':
                    await self.run_portfolio_management()
                elif choice == '5':
                    await self.run_performance_summary()
                elif choice == '6':
                    await self.run_full_cycle()
                else:
                    print("Invalid command. Try again.")
                
                print("\n" + "-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Interactive mode error: {e}")
                print(f"❌ Error: {e}")
        
        print("\n👋 Goodbye!")

async def main():
    """Main application entry point."""
    print("🤖 SIMPLIFIED TRADING SYSTEM")
    print("=" * 80)
    print("A unified, easy-to-use trading system with:")
    print("  • Market Intelligence & Analysis")  
    print("  • Automated Trading Execution")
    print("  • Pairs Trading Strategies")
    print("  • Portfolio Management")
    print("  • Performance Monitoring")
    print("=" * 80)
    
    system = SimplifiedTradingSystem()
    
    try:
        # Check if we should run interactively or automatically
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
            await system.interactive_mode()
        else:
            # Run full automated cycle
            await system.run_full_cycle()
    
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
    except Exception as e:
        logger.error(f"Main application error: {e}")
        print(f"❌ Application error: {e}")
    finally:
        # Cleanup
        await market_intelligence.close()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    asyncio.run(main())