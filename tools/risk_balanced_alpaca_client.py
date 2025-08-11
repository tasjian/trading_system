"""
Risk-balanced modifications for Alpaca client to enable balanced trading
"""

def create_balanced_pre_trade_checks(original_alpaca_client):
    """Create balanced pre-trade checks that allow appropriate buy/sell/short balance"""
    
    def balanced_pre_trade_checks(symbol: str, qty: float, side: str, notional=None) -> bool:
        """Balanced pre-trade safety checks that enable proper trading balance"""
        try:
            # Check account status
            account = original_alpaca_client.get_account_info()
            if account["account_blocked"] or account["trading_blocked"]:
                print("Trading is blocked on this account")
                return False
            
            portfolio_value = account["portfolio_value"]
            buying_power = account["buying_power"]
            
            # More lenient buying power checks for balanced trading
            if side.lower() == "buy":
                try:
                    if notional is not None:
                        order_value = notional
                    else:
                        market_data = original_alpaca_client.get_market_data(symbol, limit=1)
                        if not market_data.empty:
                            current_price = market_data.iloc[-1]["close"]
                            order_value = qty * current_price
                        else:
                            print(f"No market data for {symbol}, allowing small orders")
                            return qty <= 10  # Allow small orders without market data
                    
                    # More flexible buying power check
                    if order_value > buying_power * 1.5:  # Allow 1.5x buying power for margin
                        print(f"Order too large: ${order_value:.2f} vs ${buying_power:.2f} buying power")
                        return False
                    
                    # For very small orders, be more lenient
                    if order_value < 1000:  # Orders under $1000
                        if order_value <= buying_power or buying_power > 50:
                            return True
                        
                except Exception as e:
                    print(f"Could not verify buying power: {e}")
                    # For small quantities, allow the trade
                    if qty <= 5:
                        return True
            
            # Short selling checks - more lenient for balance
            elif side.lower() == "sell_short":
                try:
                    if notional is not None:
                        order_value = notional
                    else:
                        market_data = original_alpaca_client.get_market_data(symbol, limit=1)
                        if not market_data.empty:
                            current_price = market_data.iloc[-1]["close"]
                            order_value = qty * current_price
                        else:
                            return qty <= 5  # Allow small short orders
                    
                    # For short selling, check if we have adequate buying power for margin requirements
                    required_margin = order_value * 0.5  # 50% margin requirement estimate
                    if required_margin <= buying_power or order_value < 500:  # Small shorts OK
                        return True
                    else:
                        print(f"Insufficient margin for short: ${required_margin:.2f} required vs ${buying_power:.2f} available")
                        return False
                        
                except Exception as e:
                    print(f"Could not verify short selling requirements: {e}")
                    return qty <= 2  # Allow very small shorts
            
            # Position size limits - more reasonable for balanced trading
            if portfolio_value > 0:
                try:
                    if notional is not None:
                        position_value = notional
                    else:
                        market_data = original_alpaca_client.get_market_data(symbol, limit=1)
                        if not market_data.empty:
                            current_price = market_data.iloc[-1]["close"]
                            position_value = qty * current_price
                        else:
                            return qty <= 10
                    
                    position_percent = position_value / portfolio_value
                    
                    # More reasonable position limits
                    max_position = 0.15  # 15% max position size (up from likely 5%)
                    
                    if position_percent > max_position:
                        print(f"Position size too large: {position_percent:.2%} > {max_position:.2%}")
                        return False
                        
                except Exception as e:
                    print(f"Could not verify position size: {e}")
                    return qty <= 10
            
            return True
            
        except Exception as e:
            print(f"Pre-trade checks failed: {e}")
            # For very small orders, be permissive
            return qty <= 3
    
    return balanced_pre_trade_checks

def apply_balanced_risk_management(alpaca_client_instance):
    """Apply balanced risk management to the alpaca client"""
    # Replace the restrictive pre-trade checks with balanced ones
    alpaca_client_instance._pre_trade_checks = create_balanced_pre_trade_checks(alpaca_client_instance)
    print("✅ Applied balanced risk management to trading client")
    return alpaca_client_instance