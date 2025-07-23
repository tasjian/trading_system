#!/usr/bin/env python3
"""
High Frequency Trading Settings

Specialized configuration for high-frequency trading operations that maintain
$25,000 minimum account balance and can execute unlimited day trades.
"""

import os
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class HFTSettings:
    """High Frequency Trading configuration settings."""
    
    # Account Requirements
    minimum_account_balance: float = 25000.0  # PDT rule compliance
    minimum_cash_reserve: float = 5000.0      # Always keep cash available
    max_account_utilization: float = 0.95     # Use max 95% of account
    
    # Trading Frequency
    max_trades_per_minute: int = 10           # Rate limiting
    max_trades_per_hour: int = 200            # Hourly limit
    max_trades_per_day: int = 1000            # Daily limit
    min_time_between_trades: float = 2.0      # Seconds between trades
    
    # Position Management
    max_positions: int = 100                  # Can hold many positions
    min_position_size: float = 100.0          # Minimum $100 position
    max_position_size: float = 0.05           # Max 5% per position
    position_hold_time_min: int = 1           # Minimum 1 minute hold
    position_hold_time_max: int = 60          # Maximum 1 hour hold
    
    # Risk Management
    max_daily_loss: float = 0.02              # 2% max daily loss
    max_drawdown: float = 0.05                # 5% max drawdown
    stop_loss_percent: float = 0.01           # 1% stop loss
    take_profit_percent: float = 0.02         # 2% take profit
    
    # HFT Strategy Parameters
    price_momentum_threshold: float = 0.002   # 0.2% price movement to trigger
    volume_spike_threshold: float = 2.0       # 2x average volume spike
    volatility_threshold: float = 0.01        # 1% volatility trigger
    market_spread_max: float = 0.005          # Max 0.5% bid-ask spread
    
    # Technical Indicators
    rsi_oversold: float = 25.0               # RSI oversold level
    rsi_overbought: float = 75.0             # RSI overbought level
    macd_signal_threshold: float = 0.001      # MACD signal threshold
    bollinger_band_width: float = 2.0        # Bollinger band standard deviations
    
    # Order Execution
    preferred_order_types: List[str] = None   # Limit, market, stop-limit
    max_slippage: float = 0.001               # Max 0.1% slippage
    order_timeout_seconds: int = 30           # Cancel orders after 30 seconds
    
    # Market Data
    tick_data_symbols: List[str] = None       # Symbols for tick-level data
    level_2_data: bool = True                 # Use Level 2 market data if available
    news_sentiment_weight: float = 0.3        # Weight of news sentiment
    
    # Compliance
    wash_sale_prevention: bool = True         # Prevent wash sales
    position_limit_monitoring: bool = True    # Monitor position limits
    risk_limit_monitoring: bool = True        # Monitor risk limits
    
    def __post_init__(self):
        """Initialize default values."""
        if self.preferred_order_types is None:
            self.preferred_order_types = ["limit", "market", "stop_limit"]
        
        if self.tick_data_symbols is None:
            # High-volume, liquid stocks suitable for HFT
            self.tick_data_symbols = [
                # Mega Cap Tech
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
                # Financial Services
                "JPM", "BAC", "WFC", "GS", "MS", "C",
                # Consumer & Retail
                "WMT", "HD", "COST", "MCD", "DIS", "NKE",
                # Healthcare & Pharma
                "JNJ", "UNH", "PFE", "ABBV", "TMO", "DHR",
                # Industrial & Defense
                "CAT", "BA", "HON", "LMT", "RTX", "MMM",
                # ETFs for diversification
                "SPY", "QQQ", "IWM", "EFA", "VTI"
            ]
    
    def get_max_position_value(self, account_balance: float) -> float:
        """Calculate maximum position value based on account balance."""
        usable_balance = min(
            account_balance - self.minimum_cash_reserve,
            account_balance * self.max_account_utilization
        )
        return usable_balance * self.max_position_size
    
    def get_available_trading_capital(self, account_balance: float) -> float:
        """Calculate available capital for trading."""
        return max(0, account_balance - self.minimum_account_balance)
    
    def validate_account_balance(self, current_balance: float) -> bool:
        """Validate account meets minimum balance requirements."""
        return current_balance >= self.minimum_account_balance
    
    def get_position_sizing_rules(self) -> Dict[str, float]:
        """Get position sizing rules for different asset classes."""
        return {
            "large_cap_stocks": self.max_position_size,      # 5% max
            "mid_cap_stocks": self.max_position_size * 0.8,  # 4% max  
            "small_cap_stocks": self.max_position_size * 0.6, # 3% max
            "etfs": self.max_position_size * 1.2,            # 6% max (more liquid)
            "sector_etfs": self.max_position_size * 0.8,     # 4% max
            "international": self.max_position_size * 0.6    # 3% max (less liquid)
        }

# Global HFT settings instance
hft_settings = HFTSettings()

# Environment-specific overrides
if os.getenv("TRADING_ENV") == "production":
    hft_settings.max_trades_per_day = 500  # More conservative in production
    hft_settings.max_daily_loss = 0.015    # 1.5% max loss in production
elif os.getenv("TRADING_ENV") == "development":
    hft_settings.minimum_account_balance = 1000.0  # Lower for testing
    hft_settings.max_trades_per_day = 50           # Fewer trades for testing