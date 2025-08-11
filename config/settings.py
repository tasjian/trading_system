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
    
    # LLM API Configuration - OpenAI DISABLED, using only FinGPT and Ollama
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")  # PRESENT BUT IGNORED
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # FinGPT Configuration (HuggingFace)
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    fingpt_model: str = Field(default="FinGPT/fingpt-sentiment_llama2-13b_lora", env="FINGPT_MODEL")
    use_fingpt_primary: bool = Field(default=False, env="USE_FINGPT_PRIMARY")  # Disabled - using Ollama
    
    # Llama 3.1 Configuration (via Ollama)
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3:8b", env="OLLAMA_MODEL")  # Optimized for reliability over speed
    use_llama_fallback: bool = Field(default=True, env="USE_LLAMA_FALLBACK")
    
    # Trading Parameters
    trading_mode: str = Field(default="paper", env="TRADING_MODE")
    max_portfolio_risk: float = Field(default=0.05, env="MAX_PORTFOLIO_RISK")
    max_position_size: float = Field(default=0.35, env="MAX_POSITION_SIZE")  # Reduced to 4% for 25-50 stocks (1/25 = 4%)
    min_position_size: float = Field(default=0.01, env="MIN_POSITION_SIZE")  # Minimum 1% position
    stop_loss_percent: float = Field(default=0.08, env="STOP_LOSS_PERCENT")
    rebalance_frequency: int = Field(default=3600, env="REBALANCE_FREQUENCY")
    min_rebalance_threshold: float = Field(default=0.05, env="MIN_REBALANCE_THRESHOLD")
    
    # Monitoring Intervals (seconds)
    balance_check_interval: int = Field(default=30, env="BALANCE_CHECK_INTERVAL")
    risk_check_interval: int = Field(default=60, env="RISK_CHECK_INTERVAL")  
    performance_log_interval: int = Field(default=900, env="PERFORMANCE_LOG_INTERVAL")
    
    # Portfolio Diversification
    target_portfolio_size: int = Field(default=35, env="TARGET_PORTFOLIO_SIZE")  # Target 35 stocks
    min_portfolio_size: int = Field(default=25, env="MIN_PORTFOLIO_SIZE")  # Minimum 25 stocks
    max_portfolio_size: int = Field(default=50, env="MAX_PORTFOLIO_SIZE")  # Maximum 50 stocks
    max_sector_allocation: float = Field(default=0.25, env="MAX_SECTOR_ALLOCATION")  # Max 25% per sector
    max_asset_class_allocation: float = Field(default=0.40, env="MAX_ASSET_CLASS_ALLOCATION")  # Max 40% per asset class
    
    # Symbol Configuration
    # Use dynamic universe filtering instead of hardcoded symbols
    use_dynamic_universe: bool = Field(default=True, env="USE_DYNAMIC_UNIVERSE")
    focus_etfs: str = Field(default="SPY,QQQ,IWM,VXX", env="FOCUS_ETFS")  # Comma-separated ETFs for testing/fallback
    excluded_symbols: str = Field(default="", env="EXCLUDED_SYMBOLS")  # Comma-separated symbols to exclude
    
    # Crypto Trading Configuration
    crypto_enabled: bool = Field(default=True, env="CRYPTO_ENABLED")  # Enable crypto for 24/7 trading
    crypto_pairs: str = Field(default="BTCUSD,ETHUSD,DOGEUSD,LTCUSD,BCHUSD,LINKUSD,UNIUSD,AAVEUSD", env="CRYPTO_PAIRS")  # All supported Alpaca crypto pairs
    crypto_base_currencies: str = Field(default="USD,USDT,USDC", env="CRYPTO_BASE_CURRENCIES")
    crypto_max_position_size: float = Field(default=0.20, env="CRYPTO_MAX_POSITION_SIZE")  # Higher limit for crypto volatility
    crypto_stop_loss_percent: float = Field(default=0.15, env="CRYPTO_STOP_LOSS_PERCENT")  # Wider stops for crypto
    crypto_min_trade_amount: float = Field(default=10.0, env="CRYPTO_MIN_TRADE_AMOUNT")  # Minimum $10 crypto trades
    crypto_portfolio_allocation: float = Field(default=0.15, env="CRYPTO_PORTFOLIO_ALLOCATION")  # Max 15% of total portfolio in crypto
    crypto_max_single_position: float = Field(default=0.05, env="CRYPTO_MAX_SINGLE_POSITION")  # Max 5% per crypto asset
    
    # Risk Management
    max_daily_trades: int = Field(default=10, env="MAX_DAILY_TRADES")
    max_daily_loss: float = Field(default=0.05, env="MAX_DAILY_LOSS")  # 5% daily loss limit
    min_cash_reserve: float = Field(default=0.02, env="MIN_CASH_RESERVE")  # 2% cash reserve
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/trading_system.log", env="LOG_FILE")
    
    # Data Sources - Multiple API Keys for Enhanced Data Aggregation
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    finnhub_api_key: Optional[str] = Field(default=None, env="FINNHUB_API_KEY")
    fmp_api_key: Optional[str] = Field(default=None, env="FMP_API_KEY")  # Financial Modeling Prep
    nasdaq_api_key: Optional[str] = Field(default=None, env="NASDAQ_API_KEY")  # Nasdaq Data Link
    news_api_key: Optional[str] = Field(default=None, env="NEWS_API_KEY")
    
    # Email Notification Settings
    gmail_email: Optional[str] = Field(default=None, env="GMAIL_EMAIL")
    gmail_app_password: Optional[str] = Field(default=None, env="GMAIL_APP_PASSWORD")
    zapier_webhook_url: Optional[str] = Field(default=None, env="ZAPIER_WEBHOOK_URL")
    make_webhook_url: Optional[str] = Field(default=None, env="MAKE_WEBHOOK_URL")
    n8n_webhook_url: Optional[str] = Field(default=None, env="N8N_WEBHOOK_URL")
    
    # Sentiment Analysis - Social Media API Configuration
    reddit_client_id: Optional[str] = Field(default=None, env="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = Field(default=None, env="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(default="TradingBot/1.0", env="REDDIT_USER_AGENT")
    twitter_api_key: Optional[str] = Field(default=None, env="TWITTER_API_KEY")
    twitter_api_secret: Optional[str] = Field(default=None, env="TWITTER_API_SECRET")
    twitter_bearer_token: Optional[str] = Field(default=None, env="TWITTER_BEARER_TOKEN")
    
    # RapidAPI Configuration for Enhanced Twitter Access
    rapidapi_key: Optional[str] = Field(default=None, env="RAPIDAPI_KEY")
    
    # Crypto Data Sources
    cryptocompare_api_key: Optional[str] = Field(default=None, env="CRYPTOCOMPARE_API_KEY")
    binance_api_key: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    binance_secret_key: Optional[str] = Field(default=None, env="BINANCE_SECRET_KEY")
    coinmarketcap_api_key: Optional[str] = Field(default=None, env="COINMARKETCAP_API_KEY")
    
    # Redis Configuration for Inter-Agent Communication
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Sentiment Analysis Configuration
    sentiment_polling_interval: int = Field(default=300, env="SENTIMENT_POLLING_INTERVAL")  # 5 minutes
    sentiment_confidence_threshold: float = Field(default=0.6, env="SENTIMENT_CONFIDENCE_THRESHOLD")
    sentiment_volume_threshold: int = Field(default=5, env="SENTIMENT_VOLUME_THRESHOLD")
    
    # TikTok Scraping Configuration
    tiktok_enabled: bool = Field(default=True, env="TIKTOK_ENABLED")
    tiktok_max_posts: int = Field(default=50, env="TIKTOK_MAX_POSTS")
    tiktok_rate_limit: int = Field(default=5, env="TIKTOK_RATE_LIMIT")  # Max scraping sessions per hour
    tiktok_hashtags: str = Field(default="stocks,investing,trading,finance,stonks,wallstreet", env="TIKTOK_HASHTAGS")
    tiktok_timeout: int = Field(default=15000, env="TIKTOK_TIMEOUT")  # Page load timeout in ms
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = TradingSettings()

# Dynamic symbol management
def get_focus_symbols():
    """Get focus symbols for trading (dynamic or fallback ETFs)."""
    symbols = []
    
    # Add traditional symbols
    if settings.use_dynamic_universe:
        # In production, this would come from the universe filter
        # For now, return ETFs as a safe fallback
        symbols.extend(settings.focus_etfs.split(','))
    else:
        symbols.extend(settings.focus_etfs.split(','))
    
    # Add crypto pairs if enabled
    if settings.crypto_enabled:
        symbols.extend(get_crypto_pairs())
    
    return symbols

def get_excluded_symbols():
    """Get symbols to exclude from trading."""
    if settings.excluded_symbols:
        return settings.excluded_symbols.split(',')
    return []

def get_crypto_pairs():
    """Get enabled crypto trading pairs."""
    if settings.crypto_enabled and settings.crypto_pairs:
        return [pair.strip() for pair in settings.crypto_pairs.split(',')]
    return []

def get_crypto_base_currencies():
    """Get supported crypto base currencies."""
    if settings.crypto_base_currencies:
        return [curr.strip() for curr in settings.crypto_base_currencies.split(',')]
    return ['USD']

def is_crypto_symbol(symbol: str) -> bool:
    """Check if a symbol is a cryptocurrency pair."""
    return symbol.upper() in [pair.upper() for pair in get_crypto_pairs()]

# Validation
def validate_settings():
    """Validate critical settings."""
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        raise ValueError("Alpaca API credentials are required")
    
    if not any([settings.huggingface_api_key, settings.openai_api_key, settings.anthropic_api_key, settings.use_llama_fallback]):
        raise ValueError("At least one LLM provider (FinGPT/HuggingFace, OpenAI, Anthropic, or Llama) must be configured")
    
    if settings.trading_mode not in ["paper", "live"]:
        raise ValueError("Trading mode must be 'paper' or 'live'")
    
    if settings.max_portfolio_risk <= 0 or settings.max_portfolio_risk > 0.1:
        raise ValueError("Max portfolio risk must be between 0 and 0.1 (10%)")
    
    # Crypto-specific validation
    if settings.crypto_enabled:
        if settings.crypto_max_position_size <= 0 or settings.crypto_max_position_size > 0.5:
            raise ValueError("Crypto max position size must be between 0 and 0.5 (50%)")
        
        if settings.crypto_stop_loss_percent <= 0 or settings.crypto_stop_loss_percent > 0.3:
            raise ValueError("Crypto stop loss must be between 0 and 0.3 (30%)")
        
        if settings.crypto_min_trade_amount < 1.0:
            raise ValueError("Crypto minimum trade amount must be at least $1.00")
        
        if settings.crypto_portfolio_allocation <= 0 or settings.crypto_portfolio_allocation > 0.5:
            raise ValueError("Crypto portfolio allocation must be between 0 and 0.5 (50%)")
            
        if settings.crypto_max_single_position <= 0 or settings.crypto_max_single_position > 0.2:
            raise ValueError("Crypto max single position must be between 0 and 0.2 (20%)")
        
        # Validate crypto pairs format
        crypto_pairs = get_crypto_pairs()
        for pair in crypto_pairs:
            if len(pair) < 6 or not pair.isalpha():
                raise ValueError(f"Invalid crypto pair format: {pair}. Expected format: BTCUSD, ETHUSD, etc.")

if __name__ == "__main__":
    validate_settings()
    print("Configuration validated successfully")
    print(f"Trading Mode: {settings.trading_mode}")
    print(f"Max Portfolio Risk: {settings.max_portfolio_risk}")
    print(f"Max Position Size: {settings.max_position_size}")
    
    if settings.crypto_enabled:
        crypto_pairs = get_crypto_pairs()
        print(f"Crypto Trading: Enabled")
        print(f"Crypto Pairs ({len(crypto_pairs)}): {', '.join(crypto_pairs)}")
        print(f"Crypto Portfolio Allocation: {settings.crypto_portfolio_allocation*100:.1f}%")
        print(f"Crypto Max Single Position: {settings.crypto_max_single_position*100:.1f}%")
        print(f"Crypto Max Position: {settings.crypto_max_position_size*100:.1f}%")
        print(f"Crypto Stop Loss: {settings.crypto_stop_loss_percent*100:.1f}%")
    else:
        print(f"Crypto Trading: Disabled")