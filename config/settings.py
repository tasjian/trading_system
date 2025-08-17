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
    
    # LLM API Configuration - GPT-5-nano PRIMARY, Ollama commented out as fallback
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")  # REQUIRED for GPT-5-nano sentiment analysis
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # FinGPT Configuration (HuggingFace)
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    fingpt_model: str = Field(default="FinGPT/fingpt-sentiment_llama2-13b_lora", env="FINGPT_MODEL")
    use_fingpt_primary: bool = Field(default=False, env="USE_FINGPT_PRIMARY")  # Disabled - using Ollama
    
    # Llama 3 Configuration (via Ollama) - FAIL-FAST CONFIGURATION
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2:1b", env="OLLAMA_MODEL")  # Use faster, smaller model
    use_llama_fallback: bool = Field(default=True, env="USE_LLAMA_FALLBACK")
    
    # Optimized Ollama Configuration - FAIL-FAST with short timeouts
    ollama_max_workers: int = Field(default=2, env="OLLAMA_MAX_WORKERS")  # Reduced for stability
    ollama_queue_size: int = Field(default=10, env="OLLAMA_QUEUE_SIZE")  # Small queue for fast failure
    ollama_cache_size: int = Field(default=100, env="OLLAMA_CACHE_SIZE")  # Smaller cache for faster lookups
    ollama_cache_ttl: int = Field(default=300, env="OLLAMA_CACHE_TTL")  # 5 minutes
    ollama_default_timeout: float = Field(default=10.0, env="OLLAMA_DEFAULT_TIMEOUT")  # Short timeout for fail-fast
    ollama_connection_timeout: float = Field(default=3.0, env="OLLAMA_CONNECTION_TIMEOUT")  # Quick connection check
    ollama_retry_attempts: int = Field(default=0, env="OLLAMA_RETRY_ATTEMPTS")  # No retries - fail fast
    
    # Advanced Connection Management  
    ollama_max_connections: int = Field(default=50, env="OLLAMA_MAX_CONNECTIONS")  # Reduced for stability
    ollama_max_connections_per_host: int = Field(default=20, env="OLLAMA_MAX_CONNECTIONS_PER_HOST")  # Reduced for stability
    ollama_enable_http2: bool = Field(default=True, env="OLLAMA_ENABLE_HTTP2")
    ollama_keepalive_timeout: int = Field(default=30, env="OLLAMA_KEEPALIVE_TIMEOUT")
    ollama_dns_cache_ttl: int = Field(default=300, env="OLLAMA_DNS_CACHE_TTL")
    
    # Enhanced Caching Configuration for Resilience
    enable_aggressive_caching: bool = Field(default=True, env="ENABLE_AGGRESSIVE_CACHING")
    market_data_cache_ttl: int = Field(default=300, env="MARKET_DATA_CACHE_TTL")  # 5 minutes for market data
    sentiment_cache_ttl: int = Field(default=1800, env="SENTIMENT_CACHE_TTL")  # 30 minutes for sentiment
    news_cache_ttl: int = Field(default=3600, env="NEWS_CACHE_TTL")  # 1 hour for news
    emergency_cache_ttl: int = Field(default=7200, env="EMERGENCY_CACHE_TTL")  # 2 hours for emergency fallback
    
    # Performance Tuning for Trading System
    enable_ollama_bridge: bool = Field(default=True, env="ENABLE_OLLAMA_BRIDGE")
    sentiment_analysis_timeout: float = Field(default=10.0, env="SENTIMENT_ANALYSIS_TIMEOUT")  # Fail-fast timeout
    parallel_sentiment_requests: int = Field(default=3, env="PARALLEL_SENTIMENT_REQUESTS")  # Reduced to prevent rate limiting
    enable_llm_caching: bool = Field(default=True, env="ENABLE_LLM_CACHING")
    
    # Rate Limiting Configuration for External APIs - YFINANCE REMOVED
    max_news_api_requests_per_minute: int = Field(default=2, env="MAX_NEWS_API_REQUESTS_PER_MINUTE")  # Conservative for stability
    max_alpha_vantage_requests_per_minute: int = Field(default=4, env="MAX_ALPHA_VANTAGE_REQUESTS_PER_MINUTE")
    max_finnhub_requests_per_minute: int = Field(default=20, env="MAX_FINNHUB_REQUESTS_PER_MINUTE")  # Conservative
    
    # MLOps Performance Monitoring
    enable_performance_monitoring: bool = Field(default=True, env="ENABLE_PERFORMANCE_MONITORING")
    metrics_collection_interval: int = Field(default=10, env="METRICS_COLLECTION_INTERVAL")  # seconds
    performance_alert_thresholds: dict = Field(
        default={
            "p95_latency_ms": 5000,
            "error_rate_percent": 5.0,
            "memory_growth_mb": 500,
            "cpu_usage_percent": 80.0,
            "queue_depth": 50,
            "throughput_drop_percent": 50.0
        },
        env="PERFORMANCE_ALERT_THRESHOLDS"
    )
    
    # Health Check Configuration
    enable_health_checks: bool = Field(default=True, env="ENABLE_HEALTH_CHECKS")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")  # seconds
    health_check_timeout: float = Field(default=10.0, env="HEALTH_CHECK_TIMEOUT")
    enable_model_warmup: bool = Field(default=True, env="ENABLE_MODEL_WARMUP")
    
    # Load Balancing Configuration
    enable_load_balancing: bool = Field(default=True, env="ENABLE_LOAD_BALANCING")
    load_balancing_strategy: str = Field(default="health_weighted", env="LOAD_BALANCING_STRATEGY")
    ollama_instances: list = Field(
        default=[
            {"url": "http://localhost:11434", "name": "ollama-primary", "weight": 2.0, "priority": 1}
        ],
        env="OLLAMA_INSTANCES"
    )
    
    # Error Handling Configuration
    enable_advanced_retry: bool = Field(default=True, env="ENABLE_ADVANCED_RETRY")
    retry_max_attempts: int = Field(default=3, env="RETRY_MAX_ATTEMPTS")
    retry_base_delay: float = Field(default=0.1, env="RETRY_BASE_DELAY")
    retry_max_delay: float = Field(default=30.0, env="RETRY_MAX_DELAY")
    retry_exponential_base: float = Field(default=2.0, env="RETRY_EXPONENTIAL_BASE")
    retry_jitter_range: float = Field(default=0.1, env="RETRY_JITTER_RANGE")
    
    # Structured Logging Configuration
    enable_structured_logging: bool = Field(default=True, env="ENABLE_STRUCTURED_LOGGING")
    log_correlation_tracking: bool = Field(default=True, env="LOG_CORRELATION_TRACKING")
    log_performance_traces: bool = Field(default=True, env="LOG_PERFORMANCE_TRACES")
    log_debug_requests: bool = Field(default=False, env="LOG_DEBUG_REQUESTS")  # Disabled in production
    
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
    # ALGO AGENT ENHANCED: Expanded to 12 assets across asset classes for diversification
    focus_etfs: str = Field(default="SPY,QQQ,IWM,XLF,XLK,EFA,EEM,TLT,HYG,GLD,VNQ,DBC", env="FOCUS_ETFS")  # Diversified across equities, bonds, commodities, REITs, international
    excluded_symbols: str = Field(default="", env="EXCLUDED_SYMBOLS")  # Comma-separated symbols to exclude
    
    # Fallback watchlist for when universe filter fails to find symbols - dynamically populated
    fallback_watchlist: str = Field(default="", env="FALLBACK_WATCHLIST")  # Will be populated by universe filter
    enable_fallback_watchlist: bool = Field(default=False, env="ENABLE_FALLBACK_WATCHLIST")  # Disabled by default to use dynamic discovery
    
    # Crypto Trading Configuration
    # CRYPTO TRADING DISABLED - Comment out for later implementation
    # # ALGO AGENT RECOMMENDATION: Enable crypto with 5% allocation
    crypto_enabled: bool = Field(default=False, env="CRYPTO_ENABLED")  # DISABLED - commented out for later implementation
    # crypto_pairs: str = Field(default="BTCUSD,ETHUSD", env="CRYPTO_PAIRS")  # Start with BTC and ETH as recommended
    # crypto_base_currencies: str = Field(default="USD,USDT,USDC", env="CRYPTO_BASE_CURRENCIES")
    # crypto_max_position_size: float = Field(default=0.20, env="CRYPTO_MAX_POSITION_SIZE")  # Higher limit for crypto volatility
    # crypto_stop_loss_percent: float = Field(default=0.15, env="CRYPTO_STOP_LOSS_PERCENT")  # Wider stops for crypto
    # crypto_min_trade_amount: float = Field(default=10.0, env="CRYPTO_MIN_TRADE_AMOUNT")  # Minimum $10 crypto trades
    # # ALGO AGENT RECOMMENDATION: Start with 5% total crypto allocation
    # crypto_portfolio_allocation: float = Field(default=0.05, env="CRYPTO_PORTFOLIO_ALLOCATION")  # Max 5% of total portfolio in crypto (algo agent rec)
    # crypto_max_single_position: float = Field(default=0.025, env="CRYPTO_MAX_SINGLE_POSITION")  # Max 2.5% per crypto asset (half of total)
    
    # Risk Management
    max_daily_trades: int = Field(default=10, env="MAX_DAILY_TRADES")
    max_daily_loss: float = Field(default=0.05, env="MAX_DAILY_LOSS")  # 5% daily loss limit
    min_cash_reserve: float = Field(default=0.02, env="MIN_CASH_RESERVE")  # 2% cash reserve
    
    # Enhanced Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/trading_system.log", env="LOG_FILE")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    log_rotation_size: str = Field(default="100MB", env="LOG_ROTATION_SIZE")
    log_retention_days: int = Field(default=30, env="LOG_RETENTION_DAYS")
    
    # Metrics Export Configuration
    enable_metrics_export: bool = Field(default=True, env="ENABLE_METRICS_EXPORT")
    metrics_export_format: str = Field(default="prometheus", env="METRICS_EXPORT_FORMAT")  # prometheus or json
    metrics_export_interval: int = Field(default=60, env="METRICS_EXPORT_INTERVAL")  # seconds
    
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
    
    # Enhanced Sentiment Analysis Configuration
    sentiment_polling_interval: int = Field(default=300, env="SENTIMENT_POLLING_INTERVAL")  # 5 minutes
    sentiment_confidence_threshold: float = Field(default=0.6, env="SENTIMENT_CONFIDENCE_THRESHOLD")
    sentiment_volume_threshold: int = Field(default=5, env="SENTIMENT_VOLUME_THRESHOLD")
    
    # Sentiment Analysis Performance Settings
    sentiment_batch_size: int = Field(default=10, env="SENTIMENT_BATCH_SIZE")
    sentiment_max_concurrent: int = Field(default=5, env="SENTIMENT_MAX_CONCURRENT")
    sentiment_priority_symbols: str = Field(default="AAPL,GOOGL,MSFT,TSLA,NVDA", env="SENTIMENT_PRIORITY_SYMBOLS")
    enable_sentiment_caching: bool = Field(default=True, env="ENABLE_SENTIMENT_CACHING")
    sentiment_cache_ttl: int = Field(default=300, env="SENTIMENT_CACHE_TTL")  # 5 minutes
    
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

# CRYPTO TRADING DISABLED - Comment out for later implementation
def get_crypto_pairs():
    """Get enabled crypto trading pairs - DISABLED."""
    # if settings.crypto_enabled and settings.crypto_pairs:
    #     return [pair.strip() for pair in settings.crypto_pairs.split(',')]
    return []  # Always return empty list when crypto is disabled

def get_fallback_watchlist():
    """Get fallback watchlist when universe filter fails."""
    # First try the explicit fallback watchlist
    if settings.enable_fallback_watchlist:
        watchlist = settings.fallback_watchlist.strip()
        if watchlist:
            return [symbol.strip() for symbol in watchlist.split(',') if symbol.strip()]
    
    # If no fallback watchlist, use focus_etfs for portfolio bootstrapping
    focus_etfs = settings.focus_etfs.strip()
    if focus_etfs:
        return [symbol.strip() for symbol in focus_etfs.split(',') if symbol.strip()]
    
    return []

# CRYPTO TRADING DISABLED - Comment out for later implementation
def get_crypto_base_currencies():
    """Get supported crypto base currencies - DISABLED."""
    # if settings.crypto_base_currencies:
    #     return [curr.strip() for curr in settings.crypto_base_currencies.split(',')]
    return ['USD']  # Return default when crypto disabled

def is_crypto_symbol(symbol: str) -> bool:
    """Check if a symbol is a cryptocurrency pair - DISABLED."""
    # return symbol.upper() in [pair.upper() for pair in get_crypto_pairs()]
    return False  # Always return False when crypto is disabled

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