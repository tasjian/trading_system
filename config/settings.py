"""Configuration settings for the agentic trading system."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class TradingSettings(BaseSettings):
    """Trading system configuration settings."""
    
    # Alpaca API Configuration
    alpaca_api_key: str = Field(..., env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(..., env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(default="https://paper-api.alpaca.markets/v2", env="ALPACA_BASE_URL")
    
    # LLM API Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Trading Parameters
    trading_mode: str = Field(default="paper", env="TRADING_MODE")
    max_portfolio_risk: float = Field(default=0.01, env="MAX_PORTFOLIO_RISK")
    max_position_size: float = Field(default=0.05, env="MAX_POSITION_SIZE")
    stop_loss_percent: float = Field(default=0.03, env="STOP_LOSS_PERCENT")
    
    # Risk Management
    max_daily_trades: int = Field(default=10, env="MAX_DAILY_TRADES")
    max_daily_loss: float = Field(default=0.05, env="MAX_DAILY_LOSS")  # 5% daily loss limit
    min_cash_reserve: float = Field(default=0.1, env="MIN_CASH_RESERVE")  # 10% cash reserve
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/trading_system.log", env="LOG_FILE")
    
    # Data Sources - Multiple API Keys for Enhanced Data Aggregation
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    finnhub_api_key: Optional[str] = Field(default=None, env="FINNHUB_API_KEY")
    fmp_api_key: Optional[str] = Field(default=None, env="FMP_API_KEY")  # Financial Modeling Prep
    nasdaq_api_key: Optional[str] = Field(default=None, env="NASDAQ_API_KEY")  # Nasdaq Data Link
    news_api_key: Optional[str] = Field(default=None, env="NEWS_API_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = TradingSettings()

# Validation
def validate_settings():
    """Validate critical settings."""
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        raise ValueError("Alpaca API credentials are required")
    
    if not settings.openai_api_key and not settings.anthropic_api_key:
        raise ValueError("At least one LLM API key is required")
    
    if settings.trading_mode not in ["paper", "live"]:
        raise ValueError("Trading mode must be 'paper' or 'live'")
    
    if settings.max_portfolio_risk <= 0 or settings.max_portfolio_risk > 0.1:
        raise ValueError("Max portfolio risk must be between 0 and 0.1 (10%)")

if __name__ == "__main__":
    validate_settings()
    print("Configuration validated successfully")
    print(f"Trading Mode: {settings.trading_mode}")
    print(f"Max Portfolio Risk: {settings.max_portfolio_risk}")
    print(f"Max Position Size: {settings.max_position_size}")