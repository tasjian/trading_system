#!/usr/bin/env python3
"""
Force Rebalance Script

Forces the trading system to rebalance from current concentrated positions 
to a diversified 25-50 stock portfolio across asset classes.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Any

# Add the current directory to Python path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

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
            for pos in current_positions:
                symbol = pos.get('symbol', '')
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
        
        # Generate rebalancing orders with buying power management
        print(f"\n📋 GENERATING REBALANCING ORDERS")
        print("-" * 40)
        
        # Convert current positions to expected format
        current_portfolio_dict = {}
        for pos in current_positions:
            symbol = pos.get('symbol', '')
            current_portfolio_dict[symbol] = {
                'quantity': pos.get('qty', 0),
                'market_value': pos.get('market_value', 0),
                'unrealized_pnl': pos.get('unrealized_pl', 0),
                'sector': 'Unknown'  # Would need to lookup actual sector
            }
        
        # Get available buying power
        available_buying_power = account_info['buying_power']
        print(f"Available Buying Power: ${available_buying_power:,.2f}")
        
        rebalancing_orders = await diversified_portfolio_manager.generate_rebalancing_orders(
            current_portfolio_dict, current_portfolio_value
        )
        
        # Apply buying power constraints to orders
        rebalancing_orders = await _apply_buying_power_constraints(
            rebalancing_orders, available_buying_power, alpaca_client
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
            
            # Execute sell orders first to free up buying power
            sell_orders = [o for o in rebalancing_orders if o['side'] == 'sell']
            buy_orders = [o for o in rebalancing_orders if o['side'] == 'buy']
            
            print(f"Phase 1: Executing {len(sell_orders)} SELL orders to free up buying power...")
            
            for i, order in enumerate(sell_orders):
                try:
                    print(f"[{i+1}/{len(sell_orders)}] SELL {order['quantity']} {order['symbol']}")
                    
                    result = alpaca_client.place_order(
                        symbol=order['symbol'],
                        qty=order['quantity'],
                        side=order['side'],
                        order_type='market',
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
                
                await asyncio.sleep(0.5)
            
            # Wait for sell orders to settle before buying
            if sell_orders:
                print(f"\n⏳ Waiting 5 seconds for sell orders to settle...")
                await asyncio.sleep(5)
            
            print(f"\nPhase 2: Executing {len(buy_orders)} BUY orders...")
            
            for i, order in enumerate(buy_orders):
                try:
                    print(f"[{i+1}/{len(buy_orders)}] BUY {order['quantity']} {order['symbol']}")
                    
                    result = alpaca_client.place_order(
                        symbol=order['symbol'],
                        qty=order['quantity'],
                        side=order['side'],
                        order_type='market',
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


async def _apply_buying_power_constraints(orders: List[Dict], available_buying_power: float, alpaca_client) -> List[Dict]:
    """Apply buying power constraints to rebalancing orders."""
    
    print(f"📊 Applying buying power constraints...")
    
    # Separate buy and sell orders
    buy_orders = [o for o in orders if o['side'] == 'buy']
    sell_orders = [o for o in orders if o['side'] == 'sell']
    
    # Calculate total buy order value needed
    total_buy_value = 0
    for order in buy_orders:
        try:
            # Get current price for the symbol
            market_data = alpaca_client.get_market_data(order['symbol'], limit=1)
            if not market_data.empty:
                current_price = market_data.iloc[-1]['close']
                order_value = order['quantity'] * current_price
                total_buy_value += order_value
                order['estimated_value'] = order_value
            else:
                # If no market data, estimate conservatively
                order['estimated_value'] = order['quantity'] * 100  # Conservative estimate
                total_buy_value += order['estimated_value']
        except Exception as e:
            logger.warning(f"Could not get price for {order['symbol']}: {e}")
            order['estimated_value'] = order['quantity'] * 100
            total_buy_value += order['estimated_value']
    
    print(f"Total buy order value needed: ${total_buy_value:,.2f}")
    print(f"Available buying power: ${available_buying_power:,.2f}")
    
    # If we need more buying power than available, prioritize orders
    if total_buy_value > available_buying_power:
        print(f"⚠️ Insufficient buying power. Prioritizing buy orders...")
        
        # Sort buy orders by importance (could be by sector diversity, position size, etc.)
        buy_orders.sort(key=lambda x: x.get('importance', 0.5), reverse=True)
        
        # Select orders that fit within buying power
        selected_buy_orders = []
        used_buying_power = 0
        
        for order in buy_orders:
            estimated_value = order.get('estimated_value', 0)
            if used_buying_power + estimated_value <= available_buying_power:
                selected_buy_orders.append(order)
                used_buying_power += estimated_value
            else:
                print(f"⏭️ Skipping {order['symbol']} buy order (insufficient buying power)")
        
        print(f"✅ Selected {len(selected_buy_orders)}/{len(buy_orders)} buy orders")
        print(f"💰 Will use ${used_buying_power:,.2f} of ${available_buying_power:,.2f} buying power")
        
        # Return all sell orders plus prioritized buy orders
        return sell_orders + selected_buy_orders
    
    else:
        print(f"✅ Sufficient buying power for all orders")
        return orders


if __name__ == "__main__":
    asyncio.run(force_rebalance())