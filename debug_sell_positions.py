#!/usr/bin/env python3
"""
Debug and Sell All Positions
Debug position data structure and sell all positions.
"""

import logging
from tools.alpaca_client import alpaca_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_and_sell_positions():
    """Debug position format and sell all positions."""
    
    print("🔍 DEBUGGING POSITIONS AND SELLING ALL")
    print("=" * 50)
    
    try:
        # Get current positions
        positions = alpaca_client.get_positions()
        
        print(f"Position data type: {type(positions)}")
        print(f"Position count: {len(positions) if positions else 0}")
        
        if not positions:
            print("✅ No positions found - portfolio is clean")
            return True
        
        print(f"\n📊 Position structure analysis:")
        
        # Debug first position
        if positions:
            first_pos = positions[0] if isinstance(positions, list) else list(positions.values())[0]
            print(f"First position type: {type(first_pos)}")
            print(f"First position attributes: {dir(first_pos)}")
            if hasattr(first_pos, 'symbol'):
                print(f"Sample: {first_pos.symbol} - {first_pos.qty} shares")
        
        orders_placed = []
        
        # Handle list of position objects
        if isinstance(positions, list):
            print(f"\n💰 Selling {len(positions)} positions:")
            
            for position in positions:
                # Positions are dict objects, not attribute objects
                if not isinstance(position, dict) or 'symbol' not in position or 'qty' not in position:
                    print(f"  ⚠️ Skipping invalid position: {position}")
                    continue
                
                symbol = position['symbol']
                qty = float(position['qty'])
                market_value = float(position.get('market_value', 0))
                
                if abs(qty) < 0.01:  # Skip tiny positions
                    continue
                
                print(f"  📈 {symbol}: {qty} shares, ${market_value:,.2f}")
                
                try:
                    # Determine order side to close position
                    side = "sell" if qty > 0 else "buy"
                    qty_to_trade = abs(qty)
                    
                    # Place closing order
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
                            'qty': qty_to_trade
                        })
                        print(f"    ✅ {side.upper()} order: {order_result['id']}")
                    else:
                        print(f"    ❌ Failed to place {side} order for {symbol}")
                        
                except Exception as e:
                    print(f"    ❌ Error with {symbol}: {e}")
        
        # Handle dict format  
        elif isinstance(positions, dict):
            print(f"\n💰 Selling {len(positions)} positions (dict format):")
            
            for symbol, position in positions.items():
                qty = float(position.get('qty', 0))
                market_value = float(position.get('market_value', 0))
                
                if abs(qty) < 0.01:
                    continue
                
                print(f"  📈 {symbol}: {qty} shares, ${market_value:,.2f}")
                
                try:
                    side = "sell" if qty > 0 else "buy"
                    qty_to_trade = abs(qty)
                    
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
                            'qty': qty_to_trade
                        })
                        print(f"    ✅ {side.upper()} order: {order_result['id']}")
                    else:
                        print(f"    ❌ Failed to place {side} order for {symbol}")
                        
                except Exception as e:
                    print(f"    ❌ Error with {symbol}: {e}")
        
        print(f"\n📋 LIQUIDATION RESULTS:")
        print(f"Orders placed: {len(orders_placed)}")
        
        if orders_placed:
            print(f"✅ Successfully initiated liquidation of {len(orders_placed)} positions")
            for order in orders_placed:
                print(f"  {order['symbol']}: {order['side']} {order['qty']} - {order['order_id']}")
            return True
        else:
            print(f"❌ No orders were placed")
            return False
            
    except Exception as e:
        logger.error(f"Error in debug_and_sell_positions: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Debugging positions and attempting liquidation...")
    success = debug_and_sell_positions()
    
    if success:
        print(f"\n🎉 SUCCESS: Position liquidation initiated!")
    else:
        print(f"\n❌ FAILED: Could not liquidate positions")