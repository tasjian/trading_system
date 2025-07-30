#!/usr/bin/env python3
"""
Sell All Positions
Clean slate by selling all current positions before re-running pipeline.
"""

import asyncio
import logging
from tools.alpaca_client import alpaca_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sell_all_positions():
    """Sell all current positions to create a clean slate."""
    
    print("💰 SELLING ALL CURRENT POSITIONS")
    print("=" * 50)
    
    try:
        # Get current positions
        positions = alpaca_client.get_positions()
        
        if not positions:
            print("✅ No positions to sell - portfolio is already clean")
            return True
        
        print(f"📊 Found {len(positions)} positions to sell:")
        
        orders_placed = []
        total_value = 0
        
        # Handle both dict and list formats
        if isinstance(positions, dict):
            position_items = positions.items()
        else:
            # Convert list of position objects to dict format
            position_items = [(pos.symbol, pos) for pos in positions if hasattr(pos, 'symbol')]
        
        for symbol, position in position_items:
            # Handle both dict and object formats
            if hasattr(position, 'qty'):
                qty = float(position.qty)
                market_value = float(position.market_value) if hasattr(position, 'market_value') else 0
            else:
                qty = float(position.get('qty', 0))
                market_value = float(position.get('market_value', 0))
            
            total_value += abs(market_value)
            
            if abs(qty) < 0.01:  # Skip very small positions
                continue
                
            print(f"  {symbol}: {qty} shares, ${market_value:,.2f}")
            
            try:
                # Determine side based on current position
                side = "sell" if qty > 0 else "buy"  # Close long/short positions
                qty_to_trade = abs(qty)
                
                # Place market order to close position
                order_result = alpaca_client.place_order(
                    symbol=symbol,
                    qty=qty_to_trade,
                    side=side,
                    order_type="market",
                    time_in_force="day"
                )
                
                if order_result and 'id' in order_result:
                    orders_placed.append({
                        'symbol': symbol,
                        'order_id': order_result['id'],
                        'side': side,
                        'qty': qty_to_trade,
                        'status': order_result.get('status', 'unknown')
                    })
                    print(f"    ✅ {side.upper()} order placed: {order_result['id']}")
                else:
                    print(f"    ❌ Failed to place order for {symbol}")
                    
            except Exception as e:
                print(f"    ❌ Error selling {symbol}: {e}")
        
        print(f"\n📋 LIQUIDATION SUMMARY:")
        print(f"Total positions liquidated: {len(orders_placed)}")
        print(f"Total portfolio value: ${total_value:,.2f}")
        
        if orders_placed:
            print(f"\n📑 ORDERS PLACED:")
            for order in orders_placed:
                print(f"  {order['symbol']}: {order['side']} {order['qty']} shares")
                print(f"    Order ID: {order['order_id']}")
                print(f"    Status: {order['status']}")
        
        # Wait a moment for orders to process
        print(f"\n⏱️ Waiting 5 seconds for orders to process...")
        import time
        time.sleep(5)
        
        # Check final positions
        final_positions = alpaca_client.get_positions()
        print(f"\n🏁 FINAL CHECK:")
        print(f"Remaining positions: {len(final_positions)}")
        
        if final_positions:
            print("⚠️ Some positions may still be processing:")
            for symbol, position in final_positions.items():
                qty = position.get('qty', 0)
                if abs(float(qty)) > 0.01:  # Ignore tiny fractional shares
                    print(f"  {symbol}: {qty} shares")
        else:
            print("✅ All positions successfully liquidated!")
        
        return len(orders_placed) > 0
        
    except Exception as e:
        logger.error(f"Error selling positions: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting position liquidation...")
    success = sell_all_positions()
    
    if success:
        print(f"\n✅ SUCCESS: All positions have been liquidated!")
        print("Portfolio is now ready for fresh trading pipeline test.")
    else:
        print(f"\n❌ FAILED: Could not liquidate all positions")