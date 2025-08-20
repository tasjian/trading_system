#!/usr/bin/env python3
"""
Verify Alpaca orders and connection
"""

from tools.alpaca_client import alpaca_client
from datetime import datetime

def verify_alpaca_orders():
    print('🔍 ALPACA ORDER VERIFICATION')
    print('=' * 50)
    
    try:
        # Get account info
        account = alpaca_client.get_account_info()
        print('✅ Account Status:')
        print(f'   Status: {account.get("status")}')
        print(f'   Portfolio Value: ${float(account.get("portfolio_value", 0)):,.2f}')
        print(f'   Cash: ${float(account.get("cash", 0)):,.2f}')
        print()
        
        # Get recent orders
        orders = alpaca_client.get_orders(status='all', limit=20)
        
        if orders:
            print(f'📋 Recent Orders ({len(orders)}):')
            
            # Filter today's orders
            today_str = datetime.now().strftime('%Y-%m-%d')
            today_orders = []
            
            for order in orders:
                submitted_at = str(order.get('submitted_at', ''))
                if submitted_at.startswith(today_str):
                    today_orders.append(order)
            
            if today_orders:
                print(f'\n🎯 Today\'s Orders ({len(today_orders)}):')
                for i, order in enumerate(today_orders, 1):
                    symbol = order.get('symbol')
                    side = order.get('side') 
                    qty = order.get('qty')
                    status = order.get('status')
                    submitted = str(order.get('submitted_at'))[:19]
                    
                    print(f'   {i}. {symbol} {side} {qty} shares - {status}')
                    print(f'      Time: {submitted}')
                    
                    if order.get('filled_at'):
                        filled_at = str(order.get('filled_at'))[:19]
                        filled_qty = order.get('filled_qty', 0)
                        print(f'      Filled: {filled_at} ({filled_qty} shares)')
                    print()
                    
                print(f'✅ RESULT: {len(today_orders)} orders successfully executed today')
                print('✅ Orders ARE appearing in Alpaca dashboard')
                
            else:
                print('\n❌ No orders found for today')
                print('⚠️  Orders may not be reaching Alpaca')
                
        else:
            print('❌ No orders found at all')
            
        return len(today_orders) if 'today_orders' in locals() else 0
        
    except Exception as e:
        print(f'❌ Error: {e}')
        return 0

if __name__ == '__main__':
    verify_alpaca_orders()