#!/usr/bin/env python3
"""
Force Rebalance Script

Forces the trading system to rebalance from current concentrated positions 
to a diversified 25-50 stock portfolio across asset classes.
"""

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def force_rebalance():
    """Force rebalance to diversified portfolio."""
    
    print("🔄 FORCE REBALANCING TO DIVERSIFIED PORTFOLIO")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Import required modules
        from agents.diversified_portfolio import diversified_portfolio_manager
        from tools.alpaca_client import alpaca_client
        from config.settings import settings
        
        print(f"📊 Target Portfolio Size: {settings.target_portfolio_size} stocks")
        print(f"💰 Max Position Size: {settings.max_position_size:.1%}")
        print(f"🏢 Max Sector Allocation: {settings.max_sector_allocation:.1%}")
        
        # Get current portfolio status
        print("\n🔍 ANALYZING CURRENT PORTFOLIO")
        print("-" * 40)
        
        account_info = alpaca_client.get_account_info()
        current_portfolio_value = account_info['equity']
        current_positions = alpaca_client.get_positions()
        
        print(f"Portfolio Value: ${current_portfolio_value:,.2f}")
        print(f"Current Positions: {len(current_positions)}")
        
        if current_positions:
            print("Current Holdings:")
            for symbol, pos in current_positions.items():
                qty = pos.get('qty', 0)
                pnl = pos.get('unrealized_pl', 0)
                side = 'LONG' if qty > 0 else 'SHORT'
                print(f"  {symbol}: {abs(qty)} shares ({side}), P&L: ${pnl:.2f}")
        
        # Construct diversified target portfolio
        print(f"\n🏗️ CONSTRUCTING DIVERSIFIED PORTFOLIO")
        print("-" * 40)
        
        target_portfolio = await diversified_portfolio_manager.construct_diversified_portfolio(
            portfolio_value=current_portfolio_value,
            risk_tolerance="moderate"
        )
        
        print(f"✅ Constructed portfolio with {len(target_portfolio)} positions")
        
        # Calculate diversification metrics
        metrics = diversified_portfolio_manager.calculate_diversification_metrics()
        print(f"📈 Diversification Score: {metrics.diversification_score:.3f}")
        print(f"📊 Max Position Weight: {metrics.max_position_weight:.2%}")
        print(f"⚖️ Risk Concentration: {metrics.risk_concentration:.2%}")
        
        print("\nAsset Class Distribution:")
        for asset_class, allocation in metrics.asset_class_distribution.items():
            print(f"  {asset_class}: {allocation:.1%}")
        
        # Generate rebalancing orders
        print(f"\n📋 GENERATING REBALANCING ORDERS")
        print("-" * 40)
        
        # Convert current positions to expected format
        current_portfolio_dict = {}
        for symbol, pos in current_positions.items():
            current_portfolio_dict[symbol] = {
                'quantity': pos.get('qty', 0),
                'market_value': pos.get('market_value', 0),
                'unrealized_pnl': pos.get('unrealized_pl', 0),
                'sector': 'Unknown'  # Would need to lookup actual sector
            }
        
        rebalancing_orders = await diversified_portfolio_manager.generate_rebalancing_orders(
            current_portfolio_dict, current_portfolio_value
        )
        
        print(f"📝 Generated {len(rebalancing_orders)} rebalancing orders")
        
        if not rebalancing_orders:
            print("⚠️ No rebalancing orders generated. Portfolio may already be optimal.")
            return
        
        # Display rebalancing plan
        print(f"\n📈 REBALANCING PLAN")
        print("-" * 40)
        
        buy_orders = [o for o in rebalancing_orders if o['side'] == 'buy']
        sell_orders = [o for o in rebalancing_orders if o['side'] == 'sell']
        
        print(f"🟢 BUY ORDERS: {len(buy_orders)}")
        for i, order in enumerate(buy_orders[:10]):  # Show first 10
            print(f"  {i+1}. BUY {order['quantity']} {order['symbol']} - {order['reason']}")
        if len(buy_orders) > 10:
            print(f"  ... and {len(buy_orders) - 10} more buy orders")
        
        print(f"\n🔴 SELL ORDERS: {len(sell_orders)}")
        for i, order in enumerate(sell_orders[:10]):  # Show first 10
            print(f"  {i+1}. SELL {order['quantity']} {order['symbol']} - {order['reason']}")
        if len(sell_orders) > 10:
            print(f"  ... and {len(sell_orders) - 10} more sell orders")
        
        # Ask for confirmation
        print(f"\n⚠️ CONFIRMATION REQUIRED")
        print("-" * 40)
        print("This will execute the following changes:")
        print(f"• Close {len([o for o in rebalancing_orders if 'Close position' in o['reason']])} existing positions")
        print(f"• Open {len([o for o in rebalancing_orders if 'New position' in o['reason']])} new positions") 
        print(f"• Rebalance {len([o for o in rebalancing_orders if 'Rebalance' in o['reason']])} existing positions")
        print(f"• Target portfolio size: {len(target_portfolio)} stocks")
        
        confirmation = input("\n🤔 Execute rebalancing orders? (yes/no): ").lower().strip()
        
        if confirmation in ['yes', 'y']:
            print(f"\n🚀 EXECUTING REBALANCING ORDERS")
            print("-" * 40)
            
            successful_orders = 0
            failed_orders = 0
            
            for i, order in enumerate(rebalancing_orders):
                try:
                    print(f"[{i+1}/{len(rebalancing_orders)}] {order['side'].upper()} {order['quantity']} {order['symbol']}")
                    
                    # Execute the order
                    result = alpaca_client.place_order(
                        symbol=order['symbol'],
                        qty=order['quantity'],
                        side=order['side'],
                        type='market',
                        time_in_force='day'
                    )
                    
                    if result:
                        successful_orders += 1
                        print(f"  ✅ Order placed successfully")
                    else:
                        failed_orders += 1
                        print(f"  ❌ Order failed")
                        
                except Exception as e:
                    failed_orders += 1
                    print(f"  ❌ Order error: {e}")
                
                # Small delay between orders
                await asyncio.sleep(0.5)
            
            print(f"\n📊 REBALANCING COMPLETE")
            print("-" * 40)
            print(f"✅ Successful orders: {successful_orders}")
            print(f"❌ Failed orders: {failed_orders}")
            print(f"📈 Success rate: {successful_orders/(successful_orders+failed_orders)*100:.1f}%")
            
            if successful_orders > 0:
                print(f"\n🎉 Portfolio successfully rebalanced to diversified strategy!")
                print(f"📊 New portfolio should have ~{len(target_portfolio)} positions")
                print(f"🎯 Diversification score: {metrics.diversification_score:.3f}")
            
        else:
            print("\n❌ Rebalancing cancelled by user")
            print("💡 No orders were executed")
        
    except Exception as e:
        print(f"\n❌ REBALANCING ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\n" + "=" * 60)
    print("🏁 FORCE REBALANCE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(force_rebalance())