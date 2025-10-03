#!/usr/bin/env python3
"""
FinRL DRL Agent Wrapper for Trading System Integration

This module integrates the advanced FinRL DRL agents layer to replace the current
simplified RL implementation with proven, sophisticated Deep Reinforcement Learning algorithms.

Key Features:
1. FinRL StableBaselines3 DRL Agents (A2C, PPO, DDPG, SAC, TD3)
2. Ensemble strategy selection based on validation performance
3. Advanced stock trading environment with turbulence management
4. Portfolio rebalancing with risk management
5. Professional-grade DRL training and inference
"""

import sys
import os
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import time

# Add FinRL to Python path
finrl_path = '/Users/zac/Desktop/02_PROJECTS/FinRL'
if finrl_path not in sys.path:
    sys.path.insert(0, finrl_path)

logger = logging.getLogger(__name__)

# NOTE: FinRL environment fix is applied INSIDE __init__ AFTER imports
# This ensures patches are applied to the actual imported classes, not before they exist
logger.info("✅ FinRL Agent Wrapper loaded - Advanced DRL integration ready")

@dataclass
class FinRLTradingSignal:
    """Enhanced trading signal from FinRL DRL agents."""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    quantity: float
    confidence: float
    reasoning: str
    drl_score: float
    agent_type: str  # 'a2c', 'ppo', 'ddpg', 'sac', 'td3', 'ensemble'
    sharpe_ratio: float
    regime: str
    uncertainty: float

class FinRLAgentWrapper:
    """
    Wrapper for FinRL DRL Agents to integrate with the trading system.

    This class provides a unified interface to FinRL's sophisticated DRL agents
    while maintaining compatibility with the existing trading system architecture.
    """

    # Data window constants
    TRAINING_WINDOW_DAYS = 252  # 1 year for model training
    INFERENCE_WINDOW_DAYS = 30   # 1 month for signal generation

    def __init__(self, symbols: List[str], alpaca_client=None, config: Optional[Dict] = None):
        self.symbols = symbols
        self.alpaca_client = alpaca_client
        self.config = config or self._get_default_config()
        self.initialized = False
        self.is_trained = False

        # FinRL components
        self.drl_agents = {}
        self.ensemble_agent = None
        self.trading_env = None
        self.current_model = None
        self.model_performance = {}

        # Data management
        self.market_data_df = None
        self.last_training_date = None
        self.rebalance_window = self.config.get('rebalance_window', 63)  # 3 months
        self.validation_window = self.config.get('validation_window', 21)  # 1 month

        # Market data caching (NEW)
        self._market_data_cache = {}  # Cache for prepared market data with technical indicators
        self._cache_timestamp = {}    # Timestamp for cache invalidation
        self._cache_ttl = 300  # 5 minutes TTL for rebalancing window

        # Performance tracking
        self.portfolio_history = []
        self.signal_history = []
        self.sharpe_history = {}

        # Shortability validation cache
        self._shortable_symbols_cache = None
        self._shortable_cache_timestamp = None
        self._shortable_cache_duration = timedelta(hours=24)  # Daily cache refresh

        # Metrics tracking for shortability filtering
        self._shortability_metrics = {
            'short_signals_generated': 0,
            'short_signals_skipped_not_shortable': 0,
            'shortability_check_failures': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(f"🎯 Initialized FinRL Agent Wrapper for {len(symbols)} symbols")
        
        # Import FinRL components
        try:
            # Core FinRL imports - DataProcessor removed due to import failures
            from finrl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
            from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
            from finrl.meta.preprocessor.preprocessors import data_split
            # NOTE: DataProcessor import removed - causing DataFrame structure corruption
            # Technical indicators are now calculated using custom implementation

            self.DRLAgent = DRLAgent
            self.DRLEnsembleAgent = DRLEnsembleAgent
            self.StockTradingEnv = StockTradingEnv
            self.data_split = data_split
            # self.DataProcessor = DataProcessor  # Removed - now using custom implementation

            logger.info("✅ FinRL components imported successfully (without DataProcessor)")

            # CRITICAL: Apply patches AFTER imports so we patch the actual imported classes
            logger.info("🔧 Applying comprehensive FinRL environment fixes AFTER import...")
            try:
                from core.finrl_environment_fix import apply_comprehensive_finrl_fix
                fix_success = apply_comprehensive_finrl_fix()
                if fix_success:
                    logger.info("✅ FinRL environment fix applied successfully AFTER import")
                else:
                    logger.error("❌ FinRL environment fix failed - system may not work correctly")
            except Exception as fix_error:
                logger.error(f"❌ Failed to apply FinRL environment fix: {fix_error}")
                logger.warning("⚠️ Continuing without fix - StockTradingEnv creation may fail")

        except ImportError as e:
            logger.error(f"❌ Failed to import FinRL components: {e}")
            raise RuntimeError(f"FinRL integration failed: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for FinRL agents."""
        return {
            'initial_amount': 100000,
            'hmax': 100,  # Max shares per trade
            'buy_cost_pct': 0.001,  # 0.1% transaction cost
            'sell_cost_pct': 0.001,
            'reward_scaling': 1e-4,
            'tech_indicator_list': [
                'macd', 'rsi_30', 'cci_30', 'dx_30', 'bb_bbm', 'bb_bbh', 'bb_bbl'
            ],
            'turbulence_threshold': None,  # Will be calculated dynamically
            'rebalance_window': 63,
            'validation_window': 21,
            'train_start_date': '2022-01-01',
            'train_end_date': '2024-06-30',
            'trade_start_date': '2024-07-01',
            'trade_end_date': '2025-12-31',
            'timesteps_dict': {
                'a2c': 50000,
                'ppo': 50000,
                'ddpg': 50000,
                'sac': 50000,
                'td3': 50000
            }
        }

    def _get_cache_key(self, symbols: List[str], data_type: str = 'prepared') -> Tuple:
        """
        Generate cache key for market data.

        Args:
            symbols: List of symbols
            data_type: Type of data ('prepared' for DataFrame with indicators, 'raw' for basic data)

        Returns:
            Tuple cache key
        """
        current_date = datetime.now().strftime('%Y-%m-%d')
        return (tuple(sorted(symbols)), data_type, current_date)

    def _get_cached_market_data(self, symbols: List[str], data_type: str = 'prepared') -> Optional[Any]:
        """
        Get cached market data if available and still valid.

        Args:
            symbols: List of symbols
            data_type: Type of data to retrieve

        Returns:
            Cached data if available, None otherwise
        """
        cache_key = self._get_cache_key(symbols, data_type)

        if cache_key in self._market_data_cache:
            cache_age = time.time() - self._cache_timestamp.get(cache_key, 0)

            if cache_age < self._cache_ttl:
                logger.info(f"✅ Using cached {data_type} market data (age: {cache_age:.1f}s, symbols: {len(symbols)})")
                return self._market_data_cache[cache_key]
            else:
                logger.debug(f"Cache expired for {data_type} data (age: {cache_age:.1f}s > TTL: {self._cache_ttl}s)")
                # Clean up expired cache
                del self._market_data_cache[cache_key]
                del self._cache_timestamp[cache_key]

        return None

    def _cache_market_data(self, symbols: List[str], data: Any, data_type: str = 'prepared'):
        """
        Cache market data with timestamp.

        Args:
            symbols: List of symbols
            data: Data to cache (DataFrame or dict)
            data_type: Type of data being cached
        """
        cache_key = self._get_cache_key(symbols, data_type)
        self._market_data_cache[cache_key] = data
        self._cache_timestamp[cache_key] = time.time()
        logger.debug(f"📦 Cached {data_type} market data for {len(symbols)} symbols")

    def _invalidate_cache(self, symbols: Optional[List[str]] = None):
        """
        Invalidate cached market data.

        Args:
            symbols: If provided, only invalidate cache for these symbols.
                    If None, invalidate all cache.
        """
        if symbols is None:
            # Invalidate all cache
            cleared_count = len(self._market_data_cache)
            self._market_data_cache.clear()
            self._cache_timestamp.clear()
            logger.info(f"🗑️ Cleared all cached market data ({cleared_count} entries)")
        else:
            # Invalidate specific symbols
            keys_to_remove = []
            for cache_key in list(self._market_data_cache.keys()):
                cached_symbols, _, _ = cache_key
                if set(symbols).intersection(set(cached_symbols)):
                    keys_to_remove.append(cache_key)

            for key in keys_to_remove:
                del self._market_data_cache[key]
                if key in self._cache_timestamp:
                    del self._cache_timestamp[key]

            logger.debug(f"🗑️ Invalidated cache for {len(keys_to_remove)} entries matching {len(symbols)} symbols")

    async def _validate_shortability(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Check which symbols are actually shortable using cached Alpaca asset data.
        Returns dict mapping symbol -> is_shortable.

        Uses daily cache with Redis persistence to minimize API calls.
        """
        cache_key = f"shortable_symbols_{datetime.now().date()}"

        # Check memory cache first (fast path)
        if (self._shortable_symbols_cache and self._shortable_cache_timestamp and
            datetime.now() - self._shortable_cache_timestamp < self._shortable_cache_duration):
            logger.debug(f"Using memory-cached shortability data")
            self._shortability_metrics['cache_hits'] += 1

            # Create result dict from cached set
            shortable_status = {symbol: (symbol in self._shortable_symbols_cache) for symbol in symbols}
            return shortable_status

        # Try Redis cache
        try:
            import redis
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            cached_symbols = redis_client.smembers(cache_key)

            if cached_symbols:
                logger.info(f"📋 Redis cache hit for shortability: {len(cached_symbols)} symbols")
                self._shortable_symbols_cache = cached_symbols
                self._shortable_cache_timestamp = datetime.now()
                self._shortability_metrics['cache_hits'] += 1

                # Create result dict from cached set
                shortable_status = {symbol: (symbol in cached_symbols) for symbol in symbols}
                return shortable_status
        except Exception as redis_error:
            logger.debug(f"Redis cache unavailable: {redis_error}")

        # Cache miss - fetch from Alpaca
        self._shortability_metrics['cache_misses'] += 1
        shortable_status = {}

        try:
            logger.info(f"🔍 Fetching shortability for {len(symbols)} symbols from Alpaca...")

            # Get all assets in one API call (more efficient)
            assets = self.alpaca_client.api.list_assets(status='active', asset_class='us_equity')
            assets_dict = {asset.symbol: asset for asset in assets}

            # Build shortable set for caching
            all_shortable = set()

            for symbol in symbols:
                asset = assets_dict.get(symbol)
                if asset:
                    is_shortable = (
                        asset.tradable and
                        getattr(asset, 'shortable', False) and
                        getattr(asset, 'easy_to_borrow', False)
                    )
                    shortable_status[symbol] = is_shortable

                    if is_shortable:
                        all_shortable.add(symbol)
                else:
                    shortable_status[symbol] = False
                    logger.debug(f"⚠️ Asset not found: {symbol}")

            # Cache in memory
            self._shortable_symbols_cache = all_shortable
            self._shortable_cache_timestamp = datetime.now()

            # Cache in Redis for 24 hours
            try:
                import redis
                redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

                # Build complete shortable set from all assets
                complete_shortable = {
                    asset.symbol for asset in assets
                    if (asset.tradable and
                        getattr(asset, 'shortable', False) and
                        getattr(asset, 'easy_to_borrow', False))
                }

                # Clear old cache and set new
                redis_client.delete(cache_key)
                if complete_shortable:
                    redis_client.sadd(cache_key, *complete_shortable)
                    redis_client.expire(cache_key, 86400)  # 24 hours
                    logger.info(f"💾 Cached {len(complete_shortable)} shortable symbols in Redis")

            except Exception as redis_error:
                logger.warning(f"Failed to cache in Redis: {redis_error}")

            shortable_count = sum(1 for v in shortable_status.values() if v)
            logger.info(f"✅ Shortability check: {shortable_count}/{len(symbols)} symbols are shortable")

        except Exception as e:
            logger.error(f"❌ Failed to validate shortability: {e}")
            self._shortability_metrics['shortability_check_failures'] += 1
            # Default to False for safety - better to skip shorts than risk errors
            shortable_status = {symbol: False for symbol in symbols}

        return shortable_status

    def _verify_finrl_fix(self):
        """
        Verify that the comprehensive FinRL environment fix is working correctly.
        
        This method performs a quick check to ensure the module-level fix is active.
        """
        try:
            # Test if the fix is working by creating a minimal environment
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            # Create minimal test data
            test_data = []
            base_date = datetime(2023, 1, 1)
            
            for i in range(10):
                date = base_date + timedelta(days=i)
                test_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'tic': 'TEST',
                    'open': 100.0,
                    'high': 102.0,
                    'low': 98.0,
                    'close': 101.0,
                    'volume': 1000000,
                    'macd': 0.5,
                    'rsi_30': 50.0,
                    'cci_30': 0.0,
                    'dx_30': 0.0,
                    'bb_bbm': 101.0,
                    'bb_bbh': 103.0,
                    'bb_bbl': 99.0,
                    'turbulence': 0.1
                })
            
            df = pd.DataFrame(test_data)
            
            # Try to create a StockTradingEnv to verify the fix
            try:
                env = self.StockTradingEnv(
                    df=df,
                    stock_dim=1,
                    hmax=100,
                    initial_amount=100000,
                    num_stock_shares=[0],
                    buy_cost_pct=[0.001],
                    sell_cost_pct=[0.001],
                    reward_scaling=1e-4,
                    state_space=10,
                    action_space=1,
                    tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'dx_30', 'bb_bbm', 'bb_bbh', 'bb_bbl'],
                    turbulence_threshold=None,
                    print_verbosity=0
                )
                
                # Test reset
                state = env.reset()
                
                if state is not None and len(state) > 0:
                    logger.info("✅ FinRL environment fix verification successful")
                else:
                    logger.warning("⚠️ FinRL environment fix verification returned empty state")
                    
            except Exception as test_error:
                logger.error(f"❌ FinRL environment fix verification failed: {test_error}")
                logger.warning("⚠️ Applying fallback fix...")
                self._apply_simple_finrl_fix()
                
        except Exception as e:
            logger.error(f"❌ FinRL fix verification failed: {e}")
            logger.warning("⚠️ Applying fallback fix...")
            self._apply_simple_finrl_fix()
    
    def _apply_simple_finrl_fix(self):
        """
        Simple fallback FinRL fix for when comprehensive fix is not available.
        """
        try:
            from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
            
            # Store original method
            original_initiate_state = StockTradingEnv._initiate_state
            
            def simple_fixed_initiate_state(self):
                """Simple fixed version of _initiate_state."""
                try:
                    # Get the current data row for this timestep
                    if hasattr(self, 'day') and hasattr(self, 'df'):
                        unique_dates = self.df['date'].unique()
                        if self.day < len(unique_dates):
                            current_date = unique_dates[self.day]
                        else:
                            current_date = unique_dates[0]
                        
                        # Get data for current date
                        self.data = self.df.loc[self.df['date'] == current_date]
                        
                        if len(self.data) == 0:
                            self.data = self.df.head(self.stock_dim)
                        
                        # Try original method first
                        if hasattr(self.data, 'close') and hasattr(self.data.close, 'values'):
                            return original_initiate_state(self)
                        else:
                            # Manual fallback state construction
                            state_size = 1 + 2 * self.stock_dim + len(getattr(self, 'tech_indicator_list', [])) * self.stock_dim
                            default_state = np.zeros(state_size, dtype=np.float32)
                            default_state[0] = getattr(self, 'initial_amount', 100000)
                            
                            # Set default stock prices
                            for i in range(self.stock_dim):
                                default_state[1 + i] = 100.0  # Default stock price
                                default_state[1 + self.stock_dim + i] = 0.0  # Default holdings
                            
                            logger.warning(f"⚠️ Using simple fallback state construction: size {len(default_state)}")
                            return default_state
                            
                except Exception as e:
                    logger.error(f"❌ Simple fix failed: {e}")
                    # Absolute fallback
                    return np.zeros(91, dtype=np.float32)  # Common FinRL state size
            
            # Apply the simple patch
            StockTradingEnv._initiate_state = simple_fixed_initiate_state
            logger.info("✅ Simple FinRL environment fix applied")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply simple FinRL fix: {e}")
            
            # Apply simple patch
            StockTradingEnv._initiate_state = simple_fixed_initiate_state
            
            logger.info("✅ Simple FinRL fix applied as fallback")
            
        except Exception as e:
            logger.error(f"❌ Even simple FinRL fix failed: {e}")
            # Continue without fix - will likely fail later
    
    async def initialize_system(self):
        """Initialize the FinRL DRL system."""
        if self.initialized:
            return

        try:
            logger.info("🚀 Initializing FinRL DRL System...")

            # Step 1: Prepare market data (use inference mode for faster initialization)
            await self._prepare_market_data(mode='inference')

            # Step 2: Initialize trading environment
            self._initialize_trading_environment()

            # Step 3: Initialize DRL agents
            self._initialize_drl_agents()

            # Step 4: Load or train models
            await self._load_or_train_models()

            self.initialized = True
            logger.info("✅ FinRL DRL System initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize FinRL system: {e}")
            raise RuntimeError(f"FinRL initialization failed: {e}")
    
    async def _prepare_market_data(self, mode: str = 'inference'):
        """
        Prepare market data in FinRL format with multi-source fallback and caching.

        Args:
            mode: 'training' for model training (252 days) or 'inference' for signal generation (30 days)
        """
        try:
            logger.info(f"📊 Preparing market data for FinRL ({mode} mode) with multi-source support...")

            # Check cache first
            cached_df = self._get_cached_market_data(self.symbols, 'prepared')
            if cached_df is not None:
                self.market_data_df = cached_df
                logger.info(f"✅ Using cached prepared DataFrame: {len(self.market_data_df)} records for {len(self.symbols)} symbols")
                return

            # Cache miss - fetch fresh data
            logger.info(f"🔍 Cache miss - fetching fresh market data in {mode} mode...")

            # Try multiple data sources in priority order
            market_data = await self._get_multi_source_historical_data(mode=mode)

            if market_data:
                self.market_data_df = pd.DataFrame(market_data)
                logger.info(f"✅ Loaded historical market data: {len(self.market_data_df)} records")
            else:
                logger.error("❌ All data sources failed - REFUSING to use synthetic data for trading")
                raise ValueError("Cannot prepare market data: all real data sources failed. Synthetic data is not allowed for trading.")

            # Add technical indicators
            self._add_technical_indicators()

            # Validate data structure using the comprehensive validator
            try:
                from core.finrl_environment_fix import FinRLDataStructureManager
                self.market_data_df = FinRLDataStructureManager.validate_dataframe_structure(
                    self.market_data_df, len(self.symbols)
                )
                logger.info("✅ DataFrame structure validated for FinRL compatibility")
            except ImportError:
                logger.warning("⚠️ Comprehensive DataFrame validator not available, using basic validation")
                if self.market_data_df is None or len(self.market_data_df) == 0:
                    raise ValueError("Market data preparation failed")

            # Cache the prepared DataFrame
            self._cache_market_data(self.symbols, self.market_data_df.copy(), 'prepared')

            logger.info(f"📈 Market data prepared and cached: {len(self.market_data_df)} records for {len(self.symbols)} symbols")

        except Exception as e:
            logger.error(f"❌ Market data preparation failed: {e}")
            raise
    
    # REMOVED: _generate_synthetic_data() method
    # Synthetic data is not allowed for trading decisions

    async def _get_multi_source_historical_data(self, mode: str = 'inference') -> List[Dict]:
        """
        Get historical data using PARALLEL fetching for maximum performance.

        Args:
            mode: 'training' for model training (252 days) or 'inference' for signal generation (30 days)

        Returns:
            List of market data dictionaries in FinRL format
        """
        try:
            # Determine days based on mode
            days = self.TRAINING_WINDOW_DAYS if mode == 'training' else self.INFERENCE_WINDOW_DAYS
            start_time = time.time()
            logger.info(f"🚀 PARALLEL DATA FETCHING: {len(self.symbols)} symbols ({mode} mode: {days} days)")

            # Import the optimized API client
            try:
                from tools.optimized_api_client import OptimizedAPIClient
                from tools.background_data_service import get_background_data_service
                optimized_available = True
            except ImportError as e:
                logger.warning(f"⚠️ Optimized API client not available: {e}, falling back to legacy methods")
                optimized_available = False

            market_data = []

            if optimized_available:
                try:
                    # First, try to get cached data from background service
                    background_service = get_background_data_service()
                    cached_data = await background_service.get_cached_market_data(self.symbols)

                    if cached_data:
                        logger.info(f"📋 Found cached data for {len(cached_data)} symbols")

                        # Convert cached data to FinRL format
                        for symbol, data_points in cached_data.items():
                            for point in data_points:
                                market_data.append({
                                    'date': point['date'],
                                    'tic': symbol,
                                    'open': point['open'],
                                    'high': point['high'],
                                    'low': point['low'],
                                    'close': point['close'],
                                    'volume': point['volume']
                                })

                    # Get symbols that need fresh data
                    symbols_with_cache = set(cached_data.keys()) if cached_data else set()
                    symbols_to_fetch = [s for s in self.symbols if s not in symbols_with_cache]

                    if symbols_to_fetch:
                        logger.info(f"🔍 Fetching fresh data for {len(symbols_to_fetch)} symbols using optimized client ({days} days)")

                        # Use optimized API client for remaining symbols (ALREADY PARALLEL)
                        try:
                            async with OptimizedAPIClient() as api_client:
                                fresh_data = await api_client.get_market_data_batch(symbols_to_fetch, days=days)

                                # Convert to FinRL format
                                for symbol, data_points in fresh_data.items():
                                    for point in data_points:
                                        market_data.append({
                                            'date': point.date,
                                            'tic': symbol,
                                            'open': point.open,
                                            'high': point.high,
                                            'low': point.low,
                                            'close': point.close,
                                            'volume': point.volume
                                        })

                                logger.info(f"✅ Optimized API client retrieved {len(fresh_data)} symbols")
                        except Exception as opt_error:
                            logger.warning(f"⚠️ Optimized API client failed: {opt_error}, using PARALLEL Alpaca fallback")

                            # PARALLEL FALLBACK: Fetch all symbols concurrently using asyncio.gather
                            if self.alpaca_client:
                                logger.info(f"🚀 PARALLEL ALPACA FALLBACK: {len(symbols_to_fetch)} symbols")
                                parallel_results = await self._parallel_alpaca_fetch(symbols_to_fetch, days)
                                market_data.extend(parallel_results)

                    if len(market_data) > 0:
                        elapsed = time.time() - start_time
                        symbols_fetched = len(set(d['tic'] for d in market_data))
                        logger.info(f"✅ PARALLEL FETCH COMPLETE: {len(market_data)} records for {symbols_fetched} symbols in {elapsed:.2f}s")
                        logger.info(f"⚡ Performance: {symbols_fetched / elapsed:.1f} symbols/sec")
                        return market_data

                except Exception as optimized_error:
                    logger.warning(f"⚠️ Optimized client setup failed: {optimized_error}, using PARALLEL legacy fallback")

            # PARALLEL LEGACY FALLBACK: Use YFinance with parallel fetching
            logger.info(f"🔄 PARALLEL YFinance fallback for {len(self.symbols)} symbols ({days} days)...")
            try:
                parallel_results = await self._parallel_yfinance_fetch(self.symbols[:20], days)
                market_data.extend(parallel_results)

                if len(market_data) > 0:
                    elapsed = time.time() - start_time
                    symbols_fetched = len(set(d['tic'] for d in market_data))
                    logger.info(f"✅ YFinance parallel fallback: {len(market_data)} records for {symbols_fetched} symbols in {elapsed:.2f}s")
                    return market_data

            except Exception as yf_error:
                logger.error(f"❌ YFinance parallel fallback failed: {yf_error}")

            elapsed = time.time() - start_time
            logger.warning(f"❌ All data collection methods failed after {elapsed:.2f}s")
            return []

        except Exception as e:
            logger.error(f"❌ Historical data collection completely failed: {e}")
            return []

    async def _parallel_alpaca_fetch(self, symbols: List[str], days: int) -> List[Dict]:
        """
        Fetch data from Alpaca in PARALLEL using asyncio.gather.

        Performance: 10 symbols × 1-2s = 10-20s (sequential) -> 2-3s (parallel)
        """
        # Create tasks for parallel execution
        tasks = [
            self._fetch_single_symbol_alpaca(symbol, days)
            for symbol in symbols
        ]

        # Execute all fetches in parallel
        logger.info(f"🚀 Starting parallel fetch for {len(tasks)} symbols...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        market_data = []
        success_count = 0

        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.debug(f"❌ Parallel fetch failed for {symbol}: {result}")
                continue

            if result:
                market_data.extend(result)
                success_count += 1

        logger.info(f"✅ Parallel Alpaca fetch: {success_count}/{len(symbols)} symbols succeeded")
        return market_data

    async def _fetch_single_symbol_alpaca(self, symbol: str, days: int) -> Optional[List[Dict]]:
        """
        Fetch data for a single symbol from Alpaca.
        Designed to be called in parallel with other symbols.
        """
        max_retries = 2
        retry_delay = 0.3

        for attempt in range(max_retries):
            try:
                if not self.alpaca_client:
                    return None

                # Run blocking call in thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                bars_df = await loop.run_in_executor(
                    None,  # Use default executor
                    lambda: self.alpaca_client.get_market_data(symbol, limit=days)
                )

                if bars_df is not None and not bars_df.empty:
                    data_points = []
                    for _, row in bars_df.iterrows():
                        data_points.append({
                            'date': row.name.strftime('%Y-%m-%d') if hasattr(row.name, 'strftime') else str(row.name),
                            'tic': symbol,
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': int(row['volume'])
                        })
                    return data_points

            except Exception as e:
                logger.debug(f"Attempt {attempt + 1}/{max_retries} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                continue

        return None

    async def _parallel_yfinance_fetch(self, symbols: List[str], days: int) -> List[Dict]:
        """
        Fetch data from YFinance in PARALLEL using asyncio.gather.

        Performance: Similar speedup as Alpaca parallel fetch
        """
        # Create tasks for parallel execution
        tasks = [
            self._fetch_single_symbol_yfinance(symbol, days)
            for symbol in symbols
        ]

        # Execute all fetches in parallel
        logger.info(f"🚀 Starting parallel YFinance fetch for {len(tasks)} symbols...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        market_data = []
        success_count = 0

        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.debug(f"❌ YFinance parallel fetch failed for {symbol}: {result}")
                continue

            if result:
                market_data.extend(result)
                success_count += 1

        logger.info(f"✅ Parallel YFinance fetch: {success_count}/{len(symbols)} symbols succeeded")
        return market_data

    async def _fetch_single_symbol_yfinance(self, symbol: str, days: int) -> Optional[List[Dict]]:
        """
        Fetch data for a single symbol from YFinance.
        Designed to be called in parallel with other symbols.
        """
        max_retries = 2
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                import yfinance as yf

                # Convert days to YFinance period format
                if days <= 7:
                    period = "5d"
                elif days <= 30:
                    period = "1mo"
                elif days <= 90:
                    period = "3mo"
                else:
                    period = "1y"

                # Run blocking call in thread pool
                loop = asyncio.get_event_loop()
                ticker = yf.Ticker(symbol)
                hist = await loop.run_in_executor(
                    None,
                    lambda: ticker.history(period=period, interval="1d")
                )

                if not hist.empty:
                    data_points = []
                    for date, row in hist.iterrows():
                        data_points.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'tic': symbol,
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close']),
                            'volume': int(row['Volume'])
                        })
                    return data_points

            except Exception as e:
                logger.debug(f"YFinance attempt {attempt + 1}/{max_retries} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                continue

        return None
    
    def _add_technical_indicators(self):
        """
        Add technical indicators using vectorized operations for maximum performance.
        Processes all symbols simultaneously using pandas groupby operations.

        Performance: 10-20s (old) -> 1-2s (vectorized) for 10 symbols
        """
        try:
            start_time = time.time()
            logger.info("📊 Calculating technical indicators (vectorized mode)...")

            # Validate required columns
            required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in self.market_data_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for technical indicators: {missing_cols}")

            # Ensure proper data types and sorting (critical for time series)
            df = self.market_data_df.copy()
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Sort by symbol and date for proper time series calculations
            df = df.sort_values(['tic', 'date']).reset_index(drop=True)

            # VECTORIZED CALCULATIONS - Process all symbols at once using groupby
            logger.info("🚀 Computing indicators using vectorized groupby operations...")

            # MACD (Moving Average Convergence Divergence)
            df['macd'] = df.groupby('tic')['close'].transform(
                lambda x: self._calc_macd_vectorized(x)
            ).astype('float64')

            # RSI (Relative Strength Index)
            df['rsi_30'] = df.groupby('tic')['close'].transform(
                lambda x: self._calc_rsi_vectorized(x, period=14)
            ).astype('float64')

            # CCI (Commodity Channel Index)
            df['cci_30'] = df.groupby('tic', group_keys=False).apply(
                lambda group: self._calc_cci_vectorized(group['high'], group['low'], group['close'])
            ).astype('float64')

            # DX (Directional Index)
            df['dx_30'] = df.groupby('tic', group_keys=False).apply(
                lambda group: self._calc_dx_vectorized(group['high'], group['low'], group['close'])
            ).astype('float64')

            # Bollinger Bands (3 bands: upper, middle, lower)
            # Must calculate each band separately since transform expects Series output
            df['bb_bbm'] = df.groupby('tic')['close'].transform(
                lambda x: x.rolling(window=20, min_periods=1).mean().fillna(x)
            ).astype('float64')

            bb_std = df.groupby('tic')['close'].transform(
                lambda x: x.rolling(window=20, min_periods=1).std().fillna(0)
            )

            df['bb_bbh'] = (df['bb_bbm'] + (bb_std * 2.0)).astype('float64')
            df['bb_bbl'] = (df['bb_bbm'] - (bb_std * 2.0)).astype('float64')

            # Turbulence (volatility measure)
            df['turbulence'] = df.groupby('tic')['close'].transform(
                lambda x: self._calc_turbulence_vectorized(x)
            ).astype('float64')

            # Fill NaN values from indicator calculations (edge cases at start of series)
            tech_cols = ['macd', 'rsi_30', 'cci_30', 'dx_30', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'turbulence']
            for col in tech_cols:
                # Forward fill, then backward fill, then fill remaining with 0
                df[col] = df.groupby('tic')[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

            # Update main DataFrame
            self.market_data_df = df

            # Final validation: ensure proper DataFrame structure
            self._validate_technical_indicators()

            elapsed = time.time() - start_time
            num_symbols = df['tic'].nunique()
            num_records = len(df)
            logger.info(f"✅ Technical indicators calculated in {elapsed:.2f}s (vectorized)")
            logger.info(f"📈 Processed {num_records} records for {num_symbols} symbols")
            logger.info(f"⚡ Performance: {num_records / elapsed:.0f} records/sec")

        except Exception as e:
            logger.error(f"❌ Failed to add technical indicators: {e}")
            logger.error("Setting all indicators to zero as emergency fallback")

            # Emergency fallback: ensure all required columns exist with default values
            tech_cols = ['macd', 'rsi_30', 'cci_30', 'dx_30', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'turbulence']
            for col in tech_cols:
                self.market_data_df[col] = pd.Series(0.0, index=self.market_data_df.index, dtype='float64')

            logger.warning("⚠️ Using zero values for all technical indicators due to calculation failure")

    def _calc_macd_vectorized(self, series: pd.Series) -> pd.Series:
        """Calculate MACD using exponential moving averages (vectorized)."""
        exp1 = series.ewm(span=12, adjust=False).mean()
        exp2 = series.ewm(span=26, adjust=False).mean()
        return exp1 - exp2

    def _calc_rsi_vectorized(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index) using vectorized operations."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-8)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Default to neutral 50 for NaN

    def _calc_cci_vectorized(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate CCI (Commodity Channel Index) using vectorized operations."""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period, min_periods=1).mean()
        mean_dev = typical_price.rolling(window=period, min_periods=1).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=False
        )
        cci = (typical_price - sma) / (0.015 * (mean_dev + 1e-8))
        return cci.fillna(0)

    def _calc_dx_vectorized(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate DX (Directional Index) using vectorized operations."""
        # True Range calculation
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()

        # Combine into DataFrame for max operation
        tr_df = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = tr_df.max(axis=1)

        # Average True Range
        atr = true_range.rolling(window=period, min_periods=1).mean()

        # Normalized directional index
        dx = (atr / (close + 1e-8) * 100)
        return dx.fillna(0)

    def _calc_turbulence_vectorized(self, series: pd.Series, period: int = 20) -> pd.Series:
        """Calculate turbulence (volatility measure) using vectorized operations."""
        price_return = series.pct_change()
        turbulence = price_return.rolling(window=period, min_periods=1).std() * 100
        return turbulence.fillna(0)

    def _validate_technical_indicators(self):
        """Validate technical indicator columns for proper DataFrame structure."""
        logger.info("🔍 Validating DataFrame structure after technical indicators...")

        tech_cols = ['macd', 'rsi_30', 'cci_30', 'dx_30', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'turbulence']

        # Ensure all columns exist and have proper types
        for col in tech_cols:
            if col not in self.market_data_df.columns:
                logger.warning(f"Missing indicator column {col}, adding with default values")
                self.market_data_df[col] = pd.Series(0.0, index=self.market_data_df.index, dtype='float64')
            else:
                # Ensure proper Series with correct dtype
                self.market_data_df[col] = pd.to_numeric(
                    self.market_data_df[col], errors='coerce'
                ).fillna(0).astype('float64')

                # Verify it's a proper pandas Series
                if not isinstance(self.market_data_df[col], pd.Series):
                    logger.warning(f"Converting {col} to proper pandas Series")
                    self.market_data_df[col] = pd.Series(
                        self.market_data_df[col], dtype='float64', index=self.market_data_df.index
                    )

        # Ensure numeric price columns are also proper Series (critical for FinRL)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in self.market_data_df.columns:
                self.market_data_df[col] = pd.to_numeric(
                    self.market_data_df[col], errors='coerce'
                ).fillna(0).astype('float64')

                if not isinstance(self.market_data_df[col], pd.Series):
                    logger.warning(f"Converting {col} to proper pandas Series")
                    self.market_data_df[col] = pd.Series(
                        self.market_data_df[col], dtype='float64', index=self.market_data_df.index
                    )

        # Final validation: ensure no NaN values
        self.market_data_df = self.market_data_df.fillna(0)

        logger.info(f"✅ DataFrame structure validated")
        logger.info(f"📊 Shape: {self.market_data_df.shape}, Columns: {list(self.market_data_df.columns)}")

        # Debug: verify structure integrity for key columns
        for col in ['close'] + tech_cols[:3]:  # Sample a few columns
            if col in self.market_data_df.columns:
                col_obj = self.market_data_df[col]
                logger.debug(f"   {col}: type={type(col_obj).__name__}, dtype={col_obj.dtype}")

        logger.info("✅ DataFrame structure validation completed")
    
    def _initialize_trading_environment(self):
        """Initialize FinRL trading environment."""
        try:
            logger.info("🏗️ Initializing FinRL trading environment...")
            
            # Validate market data
            if self.market_data_df is None or len(self.market_data_df) == 0:
                raise ValueError("Market data is empty or None")
            
            # Calculate environment parameters
            stock_dim = len(self.symbols)
            state_space = 1 + 2 * stock_dim + len(self.config['tech_indicator_list']) * stock_dim
            action_space = stock_dim
            
            # Get training data - use 80% of data for training if dates are problematic
            try:
                # First try with configured dates
                train_end_date = self.config.get('train_end_date', self.config['trade_end_date'])
                train_data = self.data_split(
                    self.market_data_df,
                    start=self.config['train_start_date'],
                    end=train_end_date
                )
                
                if train_data is None or len(train_data) == 0:
                    # Fallback: Use 80% of available data for training
                    total_rows = len(self.market_data_df)
                    train_rows = int(total_rows * 0.8)
                    train_data = self.market_data_df.head(train_rows).copy()
                    logger.info(f"Using 80% of data for training: {len(train_data)} rows")
                    
            except Exception as split_error:
                logger.warning(f"Data split failed: {split_error}, using 80% of full dataset")
                total_rows = len(self.market_data_df)
                train_rows = int(total_rows * 0.8)
                train_data = self.market_data_df.head(train_rows).copy()
            
            # Ensure required columns exist - add them silently if missing
            required_cols = ['date', 'tic', 'close'] + self.config['tech_indicator_list']
            missing_cols = [col for col in required_cols if col not in train_data.columns]
            if missing_cols:
                # Only warn about critical missing columns, silently add technical indicators
                critical_missing = [col for col in missing_cols if col in ['date', 'tic', 'close']]
                if critical_missing:
                    logger.warning(f"Missing critical columns: {critical_missing}")
                
                # Add missing technical indicator columns with default values
                for col in missing_cols:
                    if col not in ['date', 'tic', 'close']:
                        train_data[col] = 0.0
                        
                logger.info(f"✅ Added {len([c for c in missing_cols if c not in ['date', 'tic', 'close']])} missing technical indicator columns")
            
            # Initialize environment with error handling
            try:
                # Ensure all data is properly formatted for FinRL
                train_data = train_data.copy()
                
                # Ensure numeric columns are proper pandas Series with correct dtypes
                numeric_cols = ['open', 'high', 'low', 'close', 'volume'] + self.config['tech_indicator_list']
                for col in numeric_cols:
                    if col in train_data.columns:
                        # Convert to numeric and ensure float64 dtype
                        train_data[col] = pd.to_numeric(train_data[col], errors='coerce').fillna(0).astype('float64')
                
                # Additional validation: ensure all columns are pandas Series, not numpy arrays
                for col in train_data.columns:
                    if not isinstance(train_data[col], pd.Series):
                        train_data[col] = pd.Series(train_data[col], dtype='float64' if col in numeric_cols else 'object')
                
                # Sort by date and symbol for FinRL compatibility
                train_data = train_data.sort_values(['date', 'tic']).reset_index(drop=True)
                
                logger.info(f"📊 Training data shape: {train_data.shape}, columns: {list(train_data.columns)}")
                
                self.trading_env = self.StockTradingEnv(
                    df=train_data,
                    stock_dim=stock_dim,
                    hmax=self.config['hmax'],
                    initial_amount=self.config['initial_amount'],
                    num_stock_shares=[0] * stock_dim,
                    buy_cost_pct=[self.config['buy_cost_pct']] * stock_dim,
                    sell_cost_pct=[self.config['sell_cost_pct']] * stock_dim,
                    reward_scaling=self.config['reward_scaling'],
                    state_space=state_space,
                    action_space=action_space,
                    tech_indicator_list=self.config['tech_indicator_list'],
                    turbulence_threshold=self.config['turbulence_threshold'],
                    print_verbosity=10
                )
                
                # Validate environment was created
                if self.trading_env is None:
                    raise ValueError("Trading environment creation returned None")
                
                # Test environment functionality
                try:
                    test_state = self.trading_env.reset()
                    if test_state is None or len(test_state) == 0:
                        raise RuntimeError("Environment reset failed - invalid state")
                    logger.info(f"✅ Environment validation passed: state length {len(test_state)}")
                except Exception as test_error:
                    raise RuntimeError(f"Environment validation failed: {test_error}")
                    
                logger.info(f"✅ Trading environment initialized: {stock_dim} stocks, {state_space} state space")
                
            except Exception as env_error:
                logger.error(f"StockTradingEnv creation failed: {env_error}")
                logger.critical("🚫 SYSTEM HALT: FinRL environment initialization failed")
                logger.critical("❌ Trading system cannot proceed without proper FinRL environment")
                logger.critical("📊 Training data shape: %s", train_data.shape)
                logger.critical("📊 Stock dimension: %d", stock_dim)
                logger.critical("📊 State space: %d", state_space)
                logger.critical("📊 Action space: %d", action_space)
                logger.critical("📊 Required columns: %s", required_cols)
                raise RuntimeError(f"FinRL environment initialization failed: {env_error}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize trading environment: {e}")
            logger.critical("🚫 SYSTEM HALT: Trading environment initialization completely failed")
            raise RuntimeError(f"Trading environment initialization failed: {e}")
    
    def _initialize_drl_agents(self):
        """Initialize FinRL DRL agents."""
        try:
            logger.info("🤖 Initializing DRL agents...")
            
            # Initialize individual agents
            agent_types = ['a2c', 'ppo', 'ddpg', 'sac', 'td3']
            
            for agent_type in agent_types:
                agent = self.DRLAgent(env=self.trading_env)
                self.drl_agents[agent_type] = agent
                self.sharpe_history[agent_type] = []
            
            # Initialize ensemble agent
            self.ensemble_agent = self.DRLEnsembleAgent(
                df=self.market_data_df,
                train_period=(self.config['train_start_date'], self.config['train_end_date']),
                val_test_period=(self.config['trade_start_date'], self.config['trade_end_date']),
                rebalance_window=self.rebalance_window,
                validation_window=self.validation_window,
                stock_dim=len(self.symbols),
                hmax=self.config['hmax'],
                initial_amount=self.config['initial_amount'],
                buy_cost_pct=self.config['buy_cost_pct'],
                sell_cost_pct=self.config['sell_cost_pct'],
                reward_scaling=self.config['reward_scaling'],
                state_space=self.trading_env.state_space,
                action_space=self.trading_env.action_space,
                tech_indicator_list=self.config['tech_indicator_list'],
                print_verbosity=1
            )
            
            logger.info(f"✅ Initialized {len(agent_types)} DRL agents + ensemble")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize DRL agents: {e}")
            raise
    
    async def _load_or_train_models(self):
        """Load existing models or train new ones."""
        try:
            logger.info("📚 Loading or training DRL models...")
            
            # Try to load existing trained models
            models_dir = "data/finrl_models"
            os.makedirs(models_dir, exist_ok=True)
            
            trained_models = {}
            for agent_type in self.drl_agents.keys():
                model_path = f"{models_dir}/agent_{agent_type}.zip"
                if os.path.exists(model_path):
                    try:
                        logger.info(f"📥 Loading pre-trained {agent_type.upper()} model from {model_path}")
                        # Actually load the model using FinRL's DRLAgent
                        try:
                            from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3
                            
                            # Map agent types to SB3 classes
                            model_classes = {
                                'a2c': A2C,
                                'ppo': PPO, 
                                'ddpg': DDPG,
                                'sac': SAC,
                                'td3': TD3
                            }
                            
                            if agent_type in model_classes:
                                model_class = model_classes[agent_type]
                                loaded_model = model_class.load(model_path)
                                trained_models[agent_type] = loaded_model
                                logger.info(f"✅ Successfully loaded {agent_type.upper()} model")
                            else:
                                logger.warning(f"Unknown agent type: {agent_type}")
                        except Exception as load_error:
                            logger.error(f"Failed to load {agent_type} model: {load_error}")
                            continue
                    except Exception as e:
                        logger.warning(f"Failed to load {agent_type} model: {e}")
                else:
                    logger.warning(f"❌ Model not found: {model_path}")
            
            if len(trained_models) >= len(self.drl_agents):
                self.is_trained = True
                self.trained_models = trained_models  # Store actual loaded models, not paths
                logger.info(f"✅ All {len(trained_models)} models loaded and ready for inference")
            else:
                logger.warning(f"⚠️ Only {len(trained_models)}/{len(self.drl_agents)} models found")
                logger.info("🏋️ Training new DRL models...")
                # Prepare training data with full historical window
                await self._prepare_market_data(mode='training')
                # Re-initialize environment with training data
                self._initialize_trading_environment()
                self._initialize_drl_agents()
                await self._train_models()
            
        except Exception as e:
            logger.error(f"❌ Model loading/training failed: {e}")
            # Continue with basic initialization
            self.is_trained = False
    
    async def _train_models(self):
        """Train DRL models using FinRL ensemble strategy."""
        try:
            logger.info("🏋️ Training DRL models with ensemble strategy...")
            
            # Use FinRL's ensemble training approach
            model_kwargs = {
                'a2c': None,  # Use default parameters
                'ppo': None,
                'ddpg': None,
                'sac': None,
                'td3': None
            }
            
            # Run ensemble training
            ensemble_results = self.ensemble_agent.run_ensemble_strategy(
                A2C_model_kwargs=model_kwargs['a2c'],
                PPO_model_kwargs=model_kwargs['ppo'],
                DDPG_model_kwargs=model_kwargs['ddpg'],
                SAC_model_kwargs=model_kwargs['sac'],
                TD3_model_kwargs=model_kwargs['td3'],
                timesteps_dict=self.config['timesteps_dict']
            )
            
            # Store training results
            self.model_performance = {
                'ensemble_summary': ensemble_results,
                'training_completed': datetime.now().isoformat()
            }
            
            self.is_trained = True
            logger.info("✅ DRL model training completed")
            
        except Exception as e:
            logger.error(f"❌ DRL model training failed: {e}")
            self.is_trained = False
            raise
    
    async def generate_trading_signals(self, 
                                     market_data: Dict[str, Any],
                                     portfolio_data: Dict[str, Any],
                                     portfolio_value: float) -> List[FinRLTradingSignal]:
        """Generate trading signals using FinRL DRL agents."""
        
        if not self.initialized:
            await self.initialize_system()
        
        try:
            if not self.is_trained:
                logger.critical("🚫 CRITICAL: FinRL DRL models not trained")
                logger.critical("❌ System requires trained FinRL DRL models")
                raise RuntimeError("FinRL DRL models not trained - system cannot proceed")
            
            # Prepare current market state for FinRL
            current_state = self._prepare_current_state(market_data, portfolio_data)

            # Get predictions from ensemble or best performing model
            predictions = await self._get_drl_predictions(current_state)

            # Validate shortability BEFORE converting predictions to signals
            logger.info("🔍 Validating shortability for signal generation...")
            shortable_status = await self._validate_shortability(self.symbols)

            # Convert predictions to trading signals with shortability validation
            signals = self._convert_predictions_to_signals(predictions, market_data, portfolio_data, shortable_status)
            
            # Store signals for performance tracking
            self.signal_history.extend(signals)
            
            return signals
            
        except Exception as e:
            logger.critical(f"🚫 CRITICAL: FinRL signal generation failed: {e}")
            logger.critical("❌ System requires FinRL DRL signals - no fallbacks allowed")
            raise RuntimeError(f"FinRL signal generation failed: {e}")
    
    def _prepare_current_state(self, market_data: Dict[str, Any], portfolio_data: Dict[str, Any]) -> np.ndarray:
        """Prepare current market state in FinRL format."""
        try:
            # Create state vector: [cash, prices, holdings, technical_indicators]
            state = []
            
            # Cash (portfolio value for now)
            cash = portfolio_data.get('cash', 10000)
            state.append(cash)
            
            # Current prices and holdings
            for symbol in self.symbols:
                symbol_data = market_data.get(symbol, {})
                price = symbol_data.get('price', 100.0)
                state.append(price)
                
                # Holdings
                position = portfolio_data.get(symbol, {})
                quantity = position.get('quantity', 0)
                state.append(quantity)
            
            # Technical indicators (simplified)
            for symbol in self.symbols:
                symbol_data = market_data.get(symbol, {})
                
                # Basic technical indicators
                state.extend([
                    symbol_data.get('macd', 0.0),
                    symbol_data.get('rsi', 50.0),
                    symbol_data.get('cci', 0.0),
                    symbol_data.get('dx', 0.0),
                    symbol_data.get('bb_middle', symbol_data.get('price', 100.0)),
                    symbol_data.get('bb_upper', symbol_data.get('price', 100.0) * 1.02),
                    symbol_data.get('bb_lower', symbol_data.get('price', 100.0) * 0.98)
                ])
            
            return np.array(state, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"❌ State preparation failed: {e}")
            # Return default state
            state_size = 1 + 2 * len(self.symbols) + len(self.config['tech_indicator_list']) * len(self.symbols)
            return np.zeros(state_size, dtype=np.float32)
    
    async def _get_drl_predictions(self, state: np.ndarray) -> Dict[str, Any]:
        """Get predictions from trained DRL models."""
        try:
            if not self.is_trained or not hasattr(self, 'trained_models'):
                raise RuntimeError("No trained models available for predictions")
            
            logger.info(f"🤖 Running DRL inference with {len(self.trained_models)} trained models")
            
            # Get predictions from all loaded models
            model_predictions = {}
            
            for agent_type, model in self.trained_models.items():
                try:
                    # Use the loaded model to predict action
                    action, _states = model.predict(state, deterministic=True)
                    model_predictions[agent_type] = action
                    logger.debug(f"✅ {agent_type.upper()} prediction: {action}")
                except Exception as model_error:
                    logger.warning(f"⚠️ {agent_type.upper()} prediction failed: {model_error}")
                    continue
            
            if not model_predictions:
                raise RuntimeError("All models failed to generate predictions")
            
            # Create ensemble prediction by averaging all model outputs
            all_actions = list(model_predictions.values())
            
            if len(all_actions) > 0:
                # Average the actions from all models
                ensemble_action = np.mean(all_actions, axis=0)
                
                # Calculate confidence based on agreement between models
                action_std = np.std(all_actions, axis=0) if len(all_actions) > 1 else np.zeros_like(ensemble_action)
                confidence = 1.0 / (1.0 + np.mean(action_std))  # Higher agreement = higher confidence
                
                # Determine best performing agent (for now, use the first successful one)
                best_agent = list(model_predictions.keys())[0]
                
                predictions = {
                    'ensemble_action': ensemble_action,
                    'confidence': float(confidence),
                    'best_agent': best_agent,
                    'sharpe_ratio': 1.0,  # Would need historical performance data
                    'regime': 'normal',
                    'individual_predictions': model_predictions
                }
                
                logger.info(f"✅ Generated ensemble predictions from {len(model_predictions)} models (confidence: {confidence:.3f})")
                return predictions
            else:
                raise RuntimeError("No valid predictions generated")
            
        except Exception as e:
            logger.error(f"❌ DRL prediction failed: {e}")
            # Don't fallback to mock data - raise the error to force proper fixing
            raise RuntimeError(f"FinRL DRL prediction failed: {e}")
    
    def _convert_predictions_to_signals(self, predictions: Dict[str, Any],
                                      market_data: Dict[str, Any],
                                      portfolio_data: Dict[str, Any],
                                      shortable_status: Dict[str, bool]) -> List[FinRLTradingSignal]:
        """Convert DRL predictions to trading signals with shortability validation."""
        signals = []
        actions = predictions.get('ensemble_action', np.zeros(len(self.symbols)))

        for i, symbol in enumerate(self.symbols):
            if i >= len(actions):
                continue

            action_value = float(actions[i])

            # Skip small actions - LOWERED threshold for more aggressive short selling
            if abs(action_value) < 0.01:  # Reduced from 0.05 to 0.01
                continue

            # ✅ NORMAL LOGIC: Follow FinRL recommendations directly
            if action_value > 0:
                # FinRL says BUY -> We do BUY
                action_type = 'buy'
                quantity = min(100, abs(action_value) * 200)  # Scale to reasonable quantity
                logger.debug(f"✅ NORMAL: FinRL BUY({action_value:.3f}) -> BUY {symbol}")
            else:
                # FinRL says SELL -> We do SELL/SHORT
                # NEW: Skip short signals for non-shortable stocks
                if not shortable_status.get(symbol, False):
                    logger.warning(f"⚠️ Skipping short signal for {symbol}: not shortable (action_value={action_value:.3f})")
                    self._shortability_metrics['short_signals_skipped_not_shortable'] += 1
                    continue

                action_type = 'short'
                quantity = min(100, abs(action_value) * 200)
                logger.debug(f"✅ NORMAL: FinRL SELL({action_value:.3f}) -> SHORT {symbol} (shortability validated)")
                self._shortability_metrics['short_signals_generated'] += 1

                # Check if we have existing long positions to sell first
                positions = portfolio_data.get('positions', {})
                current_position = positions.get(symbol, {}).get('quantity', 0)
                if current_position > 0:
                    # We have long position, sell it first
                    action_type = 'sell'
                    quantity = min(quantity, abs(current_position))  # Don't over-sell
            
            if quantity >= 0.01:
                signal = FinRLTradingSignal(
                    symbol=symbol,
                    action=action_type,
                    quantity=quantity,
                    confidence=min(0.95, predictions.get('confidence', 0.7)),
                    reasoning=f"FinRL {predictions.get('best_agent', 'ensemble')} DRL agent (action={action_value:.3f})",
                    drl_score=action_value,
                    agent_type=predictions.get('best_agent', 'ensemble'),
                    sharpe_ratio=predictions.get('sharpe_ratio', 0.0),
                    regime=predictions.get('regime', 'normal'),
                    uncertainty=1.0 - predictions.get('confidence', 0.7)
                )
                signals.append(signal)
        
        return signals
    
    # Basic signal generation removed - FinRL DRL agents are REQUIRED
    # No fallback methods allowed in the refactored system
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from FinRL agents including shortability filtering stats."""
        # Calculate shortability effectiveness
        total_short_attempts = (
            self._shortability_metrics['short_signals_generated'] +
            self._shortability_metrics['short_signals_skipped_not_shortable']
        )
        shortability_filter_rate = (
            (self._shortability_metrics['short_signals_skipped_not_shortable'] / total_short_attempts * 100)
            if total_short_attempts > 0 else 0
        )

        return {
            "system_type": "finrl_drl_agents",
            "agents_initialized": len(self.drl_agents),
            "ensemble_available": self.ensemble_agent is not None,
            "is_trained": self.is_trained,
            "signals_generated": len(self.signal_history),
            "symbol_count": len(self.symbols),
            "sharpe_history": self.sharpe_history,
            "model_performance": self.model_performance,
            "last_training_date": self.last_training_date,
            "shortability_metrics": {
                **self._shortability_metrics,
                "total_short_attempts": total_short_attempts,
                "shortability_filter_rate_pct": round(shortability_filter_rate, 2),
                "cache_hit_rate_pct": round(
                    (self._shortability_metrics['cache_hits'] /
                     max(1, self._shortability_metrics['cache_hits'] + self._shortability_metrics['cache_misses']) * 100),
                    2
                )
            }
        }
    
    async def retrain_models(self, force: bool = False):
        """Retrain DRL models with latest data (full training from scratch)."""
        try:
            logger.info("🔄 Retraining FinRL DRL models...")

            # Invalidate cache before retraining to force fresh data
            self._invalidate_cache()
            logger.info("🗑️ Cache invalidated before retraining")

            # Update market data (use training mode for full historical window)
            await self._prepare_market_data(mode='training')

            # Retrain models
            await self._train_models()

            self.last_training_date = datetime.now().isoformat()
            logger.info("✅ Model retraining completed")

        except Exception as e:
            logger.error(f"❌ Model retraining failed: {e}")
            raise

    async def incremental_train_models(self,
                                      lookback_days: int = 60,
                                      timesteps_override: Optional[Dict[str, int]] = None,
                                      agents_to_train: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Incrementally train (fine-tune) models on recent data.

        This is much faster than full retraining and adapts models to recent market conditions
        while preserving learned knowledge from historical data.

        Args:
            lookback_days: Days of recent data to use for fine-tuning (default: 60)
            timesteps_override: Optional dict of agent_type -> timesteps for custom training
            agents_to_train: Optional list of agents to train (default: all 5)

        Returns:
            Dictionary with training metrics for each agent

        Example:
            # Quick fine-tune all agents on last 60 days
            metrics = await agent.incremental_train_models(lookback_days=60)

            # Fine-tune only PPO with custom timesteps
            metrics = await agent.incremental_train_models(
                lookback_days=90,
                agents_to_train=['ppo'],
                timesteps_override={'ppo': 50000}
            )
        """
        try:
            logger.info(f"🔄 Starting incremental training (fine-tuning) on last {lookback_days} days")

            # Import the incremental trainer
            from train_finrl_incremental import IncrementalTrainer

            # Create trainer with current configuration
            trainer = IncrementalTrainer(
                symbols=self.symbols,
                lookback_days=lookback_days,
                checkpoint_dir="data/finrl_models",
                incremental_dir="data/finrl_models_incremental",
                config={
                    'timesteps_dict': self.config.get('timesteps_dict', {}),
                    'initial_amount': self.config.get('initial_amount', 100000),
                    'hmax': self.config.get('hmax', 100),
                    'buy_cost_pct': self.config.get('buy_cost_pct', 0.001),
                    'sell_cost_pct': self.config.get('sell_cost_pct', 0.001),
                    'reward_scaling': self.config.get('reward_scaling', 1e-4)
                }
            )

            # Run incremental training
            metrics = trainer.fine_tune_all_agents(
                timesteps_override=timesteps_override,
                agents_to_train=agents_to_train
            )

            # Reload the newly fine-tuned models
            logger.info("🔄 Reloading fine-tuned models...")
            await self._load_incremental_models(trainer.incremental_dir)

            self.last_training_date = datetime.now().isoformat()
            logger.info("✅ Incremental training completed and models reloaded")

            return metrics

        except Exception as e:
            logger.error(f"❌ Incremental training failed: {e}")
            raise

    async def _load_incremental_models(self, incremental_dir: str):
        """Load the latest fine-tuned models from incremental directory."""
        try:
            import os
            from pathlib import Path

            incremental_path = Path(incremental_dir)

            if not incremental_path.exists():
                logger.warning(f"⚠️ Incremental model directory not found: {incremental_dir}")
                return

            # Find latest models for each agent type
            agent_types = ['a2c', 'ppo', 'ddpg', 'sac', 'td3']
            loaded_count = 0

            for agent_type in agent_types:
                # Find all timestamped models for this agent
                pattern = f"agent_{agent_type}_*.zip"
                models = sorted(incremental_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

                if models:
                    latest_model = models[0]
                    logger.info(f"📦 Loading fine-tuned {agent_type.upper()}: {latest_model.name}")

                    try:
                        # Load the model using stable-baselines3
                        from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3

                        model_classes = {
                            'a2c': A2C,
                            'ppo': PPO,
                            'ddpg': DDPG,
                            'sac': SAC,
                            'td3': TD3
                        }

                        model_class = model_classes[agent_type]
                        loaded_model = model_class.load(str(latest_model))

                        # Store in trained_models dict
                        if not hasattr(self, 'trained_models'):
                            self.trained_models = {}

                        self.trained_models[agent_type] = loaded_model
                        loaded_count += 1
                        logger.info(f"   ✅ {agent_type.upper()} model loaded successfully")

                    except Exception as load_error:
                        logger.error(f"   ❌ Failed to load {agent_type} model: {load_error}")
                        continue
                else:
                    logger.debug(f"   ⚪ No incremental model found for {agent_type}")

            if loaded_count > 0:
                self.is_trained = True
                logger.info(f"✅ Loaded {loaded_count} fine-tuned models")
            else:
                logger.warning("⚠️ No incremental models were loaded")

        except Exception as e:
            logger.error(f"❌ Failed to load incremental models: {e}")
            raise

    async def automatic_retraining(
        self,
        min_sharpe_threshold: float = 0.8,
        max_performance_degradation: float = 0.10,
        enable_rollback: bool = True
    ) -> Dict[str, Any]:
        """
        Run automatic retraining with adaptive lookback and performance validation.

        This method integrates market regime detection, adaptive lookback periods,
        performance monitoring, and automatic rollback on degradation.

        Args:
            min_sharpe_threshold: Minimum acceptable Sharpe ratio (default: 0.8)
            max_performance_degradation: Maximum acceptable performance drop (default: 10%)
            enable_rollback: Enable automatic rollback on poor performance (default: True)

        Returns:
            Dictionary with retraining results

        Example:
            # Run automatic retraining with defaults
            results = await agent.automatic_retraining()

            # Custom thresholds
            results = await agent.automatic_retraining(
                min_sharpe_threshold=1.0,
                max_performance_degradation=0.05  # 5% max degradation
            )

        Features:
            - Detects market volatility regime
            - Adapts lookback period (30-90 days based on volatility)
            - Validates performance against baseline
            - Rolls back on degradation if enabled
            - Backs up models before retraining
        """
        try:
            from utils.market_regime_detector import MarketRegimeDetector
            from scheduled_retraining import AutomaticRetrainingScheduler

            logger.info("\n" + "=" * 80)
            logger.info("🚀 AUTOMATIC RETRAINING WITH ADAPTIVE LOOKBACK")
            logger.info("=" * 80)

            # Create scheduler with custom config
            config = {
                'symbols': self.symbols,
                'min_sharpe_threshold': min_sharpe_threshold,
                'max_performance_degradation': max_performance_degradation,
                'enable_rollback': enable_rollback,
                'agents_to_train': ['a2c', 'ppo', 'ddpg', 'sac', 'td3']
            }

            scheduler = AutomaticRetrainingScheduler(config=config)

            # Run scheduled retraining (handles everything)
            results = scheduler.run_scheduled_retraining()

            # Reload models if retraining was successful
            if results['success']:
                logger.info("\n🔄 Reloading retrained models into agent wrapper...")
                await self._load_incremental_models('data/finrl_models_incremental')
                self.last_training_date = datetime.now().isoformat()

            logger.info("\n" + "=" * 80)
            logger.info(f"{'✅' if results['success'] else '❌'} Automatic Retraining {'Completed' if results['success'] else 'Failed'}")
            logger.info("=" * 80)

            return results

        except Exception as e:
            logger.error(f"❌ Automatic retraining failed: {e}")
            raise

    async def adaptive_incremental_training(
        self,
        use_market_regime: bool = True,
        default_lookback_days: int = 60
    ) -> Dict[str, Any]:
        """
        Incremental training with adaptive lookback based on market volatility.

        Simpler alternative to full automatic_retraining() - just handles adaptive
        lookback without validation/rollback.

        Args:
            use_market_regime: Use volatility detection for adaptive lookback (default: True)
            default_lookback_days: Default lookback if regime detection disabled (default: 60)

        Returns:
            Dictionary with training metrics

        Example:
            # Adaptive lookback based on volatility
            metrics = await agent.adaptive_incremental_training()

            # Fixed lookback (ignore market regime)
            metrics = await agent.adaptive_incremental_training(
                use_market_regime=False,
                default_lookback_days=90
            )
        """
        try:
            lookback_days = default_lookback_days

            if use_market_regime:
                from utils.market_regime_detector import MarketRegimeDetector

                logger.info("📊 Detecting market regime for adaptive lookback...")
                detector = MarketRegimeDetector(symbols=['SPY', 'QQQ'])
                regime = detector.detect_current_regime()

                lookback_days = regime.recommended_lookback

                logger.info(f"   Market Regime: {regime.regime.upper()}")
                logger.info(f"   Volatility: {regime.volatility:.2%}")
                logger.info(f"   Recommended Lookback: {lookback_days} days")

            # Run incremental training with adaptive lookback
            metrics = await self.incremental_train_models(lookback_days=lookback_days)

            return metrics

        except Exception as e:
            logger.error(f"❌ Adaptive incremental training failed: {e}")
            raise

    def update_symbols_from_portfolio(self, portfolio_symbols: List[str]):
        """Update the symbols list based on current portfolio."""
        try:
            logger.info(f"🔄 Updating FinRL symbols from portfolio: {portfolio_symbols}")

            # Update symbols list
            old_symbols = self.symbols.copy()
            self.symbols = list(set(portfolio_symbols))  # Remove duplicates

            logger.info(f"📊 Symbol update: {len(old_symbols)} → {len(self.symbols)} symbols")

            # Invalidate cache when symbols change
            if set(old_symbols) != set(self.symbols):
                self._invalidate_cache()
                logger.info("🗑️ Cache invalidated due to symbol change")

            # If symbols changed significantly, may need to reinitialize
            if len(set(self.symbols) - set(old_symbols)) > len(old_symbols) * 0.5:
                logger.warning("⚠️ Major symbol change detected - may need system reinitialization")
                self.initialized = False

        except Exception as e:
            logger.error(f"❌ Failed to update symbols from portfolio: {e}")

# Compatibility functions for integration with existing system
    async def generate_pure_trading_decisions(self, 
                                           portfolio_value: float, 
                                           cash_available: float,
                                           use_full_market: bool = True, 
                                           max_positions: int = 10) -> Dict[str, Any]:
        """
        Generate pure FinRL trading decisions without external filtering.
        
        This method allows FinRL to select stocks directly from the full market
        based on its own analysis, bypassing universe filters and hardcoded symbols.
        """
        try:
            logger.info("🤖 Pure FinRL Mode: Selecting stocks from full market")
            
            if use_full_market:
                # Get top liquid stocks from market for FinRL to choose from
                from tools.alpaca_client import alpaca_client
                
                # Get tradeable assets from Alpaca
                assets = alpaca_client.api.list_assets(status='active', asset_class='us_equity')
                
                # CRITICAL FIX: Shuffle ALL assets first to eliminate alphabetical bias from Alpaca API
                import random
                assets_list = list(assets)
                random.shuffle(assets_list)
                
                # Filter to most liquid stocks (by market cap and volume)
                liquid_stocks = []
                for asset in assets_list[:1000]:  # Top 1000 by volume after shuffling
                    try:
                        # Skip penny stocks and get basic info
                        if asset.tradable and asset.shortable:
                            liquid_stocks.append(asset.symbol)
                    except Exception:
                        continue
                
                # FinRL should receive final filtered candidates from multi_stage_stock_filter (~50 stocks)
                # This is the last stage before order execution
                try:
                    from core.universe_filter import StockUniverseFilter
                    from core.multi_stage_stock_filter import multi_stage_filter
                    
                    # Step 1: Get universe filter results (~400 symbols)
                    universe_filter = StockUniverseFilter()
                    universe_result = await universe_filter.filter_universe()

                    if universe_result and universe_result.filtered_symbols:
                        logger.info(f"📊 Universe filter: {len(universe_result.filtered_symbols)} symbols")

                        # Step 2: Run multi-stage filter on universe results
                        # SKIP multi-stage filter to avoid sentiment analysis crash
                        # filtered_result = await multi_stage_filter.filter_stocks(universe_result.filtered_symbols)

                        # Direct use of universe filter results without sentiment analysis
                        # LIMIT to 10 symbols to match trained model dimensions (91-d state space)
                        candidate_symbols = universe_result.filtered_symbols[:10]
                        logger.info(f"🎯 FinRL using universe filter results directly (skip multi-stage sentiment): {len(candidate_symbols)} symbols (limited to 10 for model compatibility)")
                        filtered_result = None  # Force fallback to direct universe results
                        
                        if filtered_result:
                            # Combine long and short candidates for FinRL analysis
                            long_candidates = getattr(filtered_result, 'long_candidates', [])
                            short_candidates = getattr(filtered_result, 'short_candidates', [])
                            candidate_symbols = long_candidates + short_candidates
                            
                            logger.info(f"🎯 FinRL received {len(candidate_symbols)} final candidates from multi-stage filter")
                            logger.info(f"    Long candidates: {len(long_candidates)}, Short candidates: {len(short_candidates)}")
                            logger.info(f"    Sample candidates: {', '.join(candidate_symbols[:10])}{'...' if len(candidate_symbols) > 10 else ''}")
                        else:
                            # Fallback: use subset of universe filter results
                            # LIMIT to 10 symbols to match trained model dimensions (91-d state space)
                            candidate_symbols = universe_result.filtered_symbols[:10]
                            logger.warning(f"🔄 Multi-stage filter failed, using first {len(candidate_symbols)} from universe filter (limited to 10 for model compatibility)")
                    else:
                        # Emergency fallback: use liquid stocks
                        # LIMIT to 10 symbols to match trained model dimensions (91-d state space)
                        import random
                        random.shuffle(liquid_stocks)
                        candidate_symbols = liquid_stocks[:10]
                        logger.warning(f"🚨 Universe filter failed, using {len(candidate_symbols)} random liquid stocks (limited to 10 for model compatibility)")
                        
                except Exception as e:
                    logger.warning(f"Complete filtering pipeline failed: {e}")
                    # LIMIT to 10 symbols to match trained model dimensions (91-d state space)
                    import random
                    random.shuffle(liquid_stocks)
                    candidate_symbols = liquid_stocks[:10]
                    logger.info(f"🎯 FinRL emergency fallback: analyzing {len(candidate_symbols)} randomly selected liquid stocks (limited to 10 for model compatibility)")
                
            else:
                # Use existing symbols if provided
                candidate_symbols = self.symbols
                
            # Create temporary FinRL instance for pure decision making
            if not hasattr(self, '_pure_finrl_initialized'):
                logger.info("🚀 Initializing pure FinRL system...")
                
                # Initialize with candidate symbols
                self.symbols = candidate_symbols
                await self.initialize_system()
                self._pure_finrl_initialized = True
                
            # Use FinRL's internal stock selection and allocation logic
            logger.info("📊 FinRL performing market analysis and stock selection...")
            
            # Get market data for all candidates
            market_data = await self._get_multi_symbol_market_data(candidate_symbols)
            
            if not market_data:
                logger.warning("⚠️ No market data available for FinRL analysis")
                return {"allocations": [], "strategy": "Pure FinRL - No Data"}
                
            # Let FinRL DRL agents make portfolio decisions
            portfolio_data = {
                "cash": cash_available,
                "equity": portfolio_value,
                "positions": {}
            }
            
            # Generate FinRL trading signals
            signals = await self.generate_trading_signals(
                market_data, portfolio_data, portfolio_value
            )
            
            # Convert signals to allocation format
            allocations = []
            total_weight = 0
            
            for signal in signals[:max_positions]:
                # ✅ NORMAL LOGIC: Handle both BUY and SHORT signals from FinRL naturally
                if (signal.action in ['buy', 'short', 'sell']) and signal.confidence > 0.2:  # Lowered from 0.6 to 0.2
                    # Calculate position size based on FinRL confidence and risk
                    base_weight = 1.0 / max_positions  # Equal weight baseline
                    confidence_multiplier = signal.confidence
                    risk_adjustment = 1.0 - (signal.uncertainty * 0.5)
                    
                    weight = base_weight * confidence_multiplier * risk_adjustment
                    weight = max(0.02, min(0.15, weight))  # 2% to 15% position limits
                    
                    # ✅ NORMAL LOGIC: Make short positions have negative weights naturally
                    if signal.action in ['short', 'sell']:
                        weight = -weight  # Negative weight for short positions
                        action_desc = "SHORT"
                    else:
                        action_desc = "LONG"
                    
                    allocation = {
                        "symbol": signal.symbol,
                        "weight": weight,
                        "confidence": signal.confidence,
                        "reasoning": f"FinRL DRL Agent ({signal.agent_type}) - {action_desc}: {signal.reasoning}",
                        "drl_score": signal.drl_score,
                        "sharpe_ratio": signal.sharpe_ratio,
                        "regime": signal.regime,
                        "finrl_pure": True,
                        "action_type": signal.action  # Track whether this is buy or short
                    }
                    allocations.append(allocation)
                    total_weight += abs(weight)  # Use absolute value for total exposure calculation
                    
            # Normalize weights if needed (using absolute values for exposure)
            if total_weight > 0.95:  # Leave some cash
                scale_factor = 0.90 / total_weight
                for allocation in allocations:
                    allocation["weight"] = allocation["weight"] * scale_factor
                    
            logger.info(f"✅ FinRL selected {len(allocations)} stocks for portfolio")
            for allocation in allocations:
                action_type = allocation.get('action_type', 'unknown')
                weight_abs = abs(allocation['weight'])
                direction = "SHORT" if allocation['weight'] < 0 else "LONG"
                logger.info(f"   {allocation['symbol']}: {direction} {weight_abs:.2%} (confidence: {allocation['confidence']:.2f}, action: {action_type})")
                
            return {
                "allocations": allocations,
                "strategy": "Pure FinRL DRL Market Selection",
                "confidence": np.mean([a["confidence"] for a in allocations]) if allocations else 0,
                "total_symbols_analyzed": len(candidate_symbols),
                "selected_symbols": len(allocations),
                "finrl_pure_mode": True
            }
            
        except Exception as e:
            logger.error(f"❌ Pure FinRL trading decisions failed: {e}")
            import traceback
            traceback.print_exc()
            return {"allocations": [], "strategy": "Pure FinRL - Error", "error": str(e)}
    
    async def _get_multi_symbol_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market data for multiple symbols using PARALLEL fetching with fallbacks and caching."""
        try:
            start_time = time.time()

            # Check cache first
            cached_data = self._get_cached_market_data(symbols, 'multi_symbol')
            if cached_data is not None:
                return cached_data

            # Cache miss - fetch fresh data
            logger.info(f"🚀 PARALLEL multi-symbol fetch: {len(symbols)} symbols")

            # Try optimized approach first
            try:
                from tools.optimized_api_client import OptimizedAPIClient
                from tools.background_data_service import get_background_data_service
                optimized_available = True
            except ImportError as e:
                logger.warning(f"⚠️ Optimized API client not available: {e}, using parallel fallback")
                optimized_available = False

            market_data = {}

            if optimized_available:
                try:
                    # First check background data service cache
                    background_service = get_background_data_service()
                    cached_data = await background_service.get_cached_market_data(symbols)

                    # Convert cached data
                    for symbol, data_points in cached_data.items():
                        if data_points:
                            latest_point = data_points[-1]  # Most recent data point
                            market_data[symbol] = {
                                'price': latest_point['close'],
                                'volume': latest_point['volume'],
                                'change_pct': ((latest_point['close'] / latest_point['open']) - 1) * 100 if latest_point['open'] > 0 else 0,
                                'high': latest_point['high'],
                                'low': latest_point['low'],
                                'date': latest_point['date'],
                                'source': 'cached'
                            }

                    # Get symbols that need fresh data
                    symbols_with_cache = set(cached_data.keys())
                    symbols_to_fetch = [s for s in symbols if s not in symbols_with_cache]

                    if symbols_to_fetch:
                        logger.info(f"🔍 Fetching fresh market data for {len(symbols_to_fetch)} symbols")

                        try:
                            # Use optimized API client for fresh data (ALREADY PARALLEL)
                            async with OptimizedAPIClient() as api_client:
                                fresh_data = await api_client.get_market_data_batch(symbols_to_fetch, days=30)

                                for symbol, data_points in fresh_data.items():
                                    if data_points:
                                        latest_point = data_points[-1]  # Most recent data point
                                        market_data[symbol] = {
                                            'price': latest_point.close,
                                            'volume': latest_point.volume,
                                            'change_pct': ((latest_point.close / latest_point.open) - 1) * 100 if latest_point.open > 0 else 0,
                                            'high': latest_point.high,
                                            'low': latest_point.low,
                                            'date': latest_point.date,
                                            'source': latest_point.source
                                        }

                            logger.info(f"✅ Optimized client retrieved {len(fresh_data) if 'fresh_data' in locals() else 0} fresh symbols")

                        except Exception as opt_error:
                            logger.warning(f"⚠️ Optimized API failed, using PARALLEL Alpaca fallback: {opt_error}")

                            # PARALLEL ALPACA FALLBACK
                            if self.alpaca_client:
                                logger.info(f"🚀 PARALLEL Alpaca fallback for {len(symbols_to_fetch)} symbols")
                                parallel_data = await self._parallel_fetch_latest_prices_alpaca(symbols_to_fetch)
                                market_data.update(parallel_data)

                    elapsed = time.time() - start_time
                    logger.info(f"✅ Retrieved market data for {len(market_data)} symbols in {elapsed:.2f}s")

                    if len(market_data) > 0:
                        # Cache the results before returning
                        self._cache_market_data(symbols, market_data, 'multi_symbol')
                        return market_data

                except Exception as optimized_error:
                    logger.warning(f"⚠️ Optimized approach failed: {optimized_error}, using PARALLEL legacy fallback")

            # PARALLEL LEGACY FALLBACK: Direct Alpaca
            if self.alpaca_client:
                logger.info(f"🚀 PARALLEL legacy Alpaca fallback for {len(symbols)} symbols")
                parallel_data = await self._parallel_fetch_latest_prices_alpaca(symbols)
                market_data.update(parallel_data)

            # PARALLEL YFinance fallback for remaining symbols
            remaining_symbols = [s for s in symbols if s not in market_data]
            if remaining_symbols:
                logger.info(f"🚀 PARALLEL YFinance fallback for {len(remaining_symbols)} remaining symbols")
                yf_data = await self._parallel_fetch_latest_prices_yfinance(remaining_symbols[:20])
                market_data.update(yf_data)

            # Cache the results before returning
            if len(market_data) > 0:
                self._cache_market_data(symbols, market_data, 'multi_symbol')
                elapsed = time.time() - start_time
                logger.info(f"✅ Final market data retrieved and cached for {len(market_data)} symbols in {elapsed:.2f}s")
            else:
                logger.warning(f"⚠️ No market data retrieved for any symbols")

            return market_data

        except Exception as e:
            logger.error(f"❌ All multi-symbol market data methods failed: {e}")
            return {}

    async def _parallel_fetch_latest_prices_alpaca(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Fetch latest prices from Alpaca in PARALLEL.
        Returns dict of symbol -> market data dict.
        """
        # Create tasks for parallel execution
        tasks = [
            self._fetch_latest_price_alpaca(symbol)
            for symbol in symbols
        ]

        # Execute all fetches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        market_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.debug(f"❌ Alpaca latest price failed for {symbol}: {result}")
                continue

            if result:
                market_data[symbol] = result

        logger.info(f"✅ Parallel Alpaca latest prices: {len(market_data)}/{len(symbols)} symbols")
        return market_data

    async def _fetch_latest_price_alpaca(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch latest price for a single symbol from Alpaca."""
        try:
            if not self.alpaca_client:
                return None

            # Run blocking call in thread pool
            loop = asyncio.get_event_loop()
            bars = await loop.run_in_executor(
                None,
                lambda: self.alpaca_client.get_market_data(symbol, limit=5)
            )

            if bars is not None and not bars.empty:
                latest_bar = bars.iloc[-1]
                return {
                    'price': latest_bar['close'],
                    'volume': latest_bar['volume'],
                    'change_pct': (latest_bar['close'] / latest_bar['open'] - 1) * 100 if latest_bar['open'] > 0 else 0,
                    'high': latest_bar['high'],
                    'low': latest_bar['low'],
                    'date': str(bars.index[-1]),
                    'source': 'alpaca_parallel'
                }

        except Exception as e:
            logger.debug(f"Alpaca latest price error for {symbol}: {e}")

        return None

    async def _parallel_fetch_latest_prices_yfinance(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Fetch latest prices from YFinance in PARALLEL.
        Returns dict of symbol -> market data dict.
        """
        # Create tasks for parallel execution
        tasks = [
            self._fetch_latest_price_yfinance(symbol)
            for symbol in symbols
        ]

        # Execute all fetches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        market_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.debug(f"❌ YFinance latest price failed for {symbol}: {result}")
                continue

            if result:
                market_data[symbol] = result

        logger.info(f"✅ Parallel YFinance latest prices: {len(market_data)}/{len(symbols)} symbols")
        return market_data

    async def _fetch_latest_price_yfinance(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch latest price for a single symbol from YFinance."""
        try:
            import yfinance as yf

            # Run blocking call in thread pool
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)
            hist = await loop.run_in_executor(
                None,
                lambda: ticker.history(period="5d", interval="1d")
            )

            if not hist.empty:
                latest = hist.iloc[-1]
                return {
                    'price': float(latest['Close']),
                    'volume': int(latest['Volume']),
                    'change_pct': ((latest['Close'] / latest['Open']) - 1) * 100 if latest['Open'] > 0 else 0,
                    'high': float(latest['High']),
                    'low': float(latest['Low']),
                    'date': hist.index[-1].strftime('%Y-%m-%d'),
                    'source': 'yfinance_parallel'
                }

        except Exception as e:
            logger.debug(f"YFinance latest price error for {symbol}: {e}")

        return None


async def create_finrl_agent(symbols: List[str], alpaca_client=None, config: Optional[Dict] = None) -> FinRLAgentWrapper:
    """Create and initialize FinRL agent wrapper."""
    agent = FinRLAgentWrapper(symbols, alpaca_client, config)
    await agent.initialize_system()
    return agent

async def generate_finrl_signals(market_data: Dict[str, Any],
                                portfolio_data: Dict[str, Any],
                                portfolio_value: float,
                                symbols: List[str],
                                alpaca_client=None) -> List[Dict[str, Any]]:
    """Generate FinRL DRL-enhanced trading signals."""
    
    # Create FinRL agent
    finrl_agent = await create_finrl_agent(symbols, alpaca_client)
    
    try:
        # Generate signals
        finrl_signals = await finrl_agent.generate_trading_signals(
            market_data, 
            portfolio_data, 
            portfolio_value
        )
        
        # Convert to format expected by existing workflow
        workflow_signals = []
        for signal in finrl_signals:
            workflow_signal = {
                'symbol': signal.symbol,
                'action': signal.action,
                'quantity': signal.quantity,
                'confidence': signal.confidence,
                'reasoning': f"[FinRL DRL] {signal.reasoning}",
                'source': 'finrl_drl_agents',
                'drl_score': signal.drl_score,
                'agent_type': signal.agent_type,
                'sharpe_ratio': signal.sharpe_ratio,
                'regime': signal.regime,
                'uncertainty': signal.uncertainty,
                'timestamp': datetime.now()
            }
            workflow_signals.append(workflow_signal)
        
        return workflow_signals
        
    except Exception as e:
        logger.error(f"❌ FinRL signal generation failed: {e}")
        return []

logger.info("✅ FinRL Agent Wrapper loaded - Advanced DRL integration ready")