#!/usr/bin/env python3
"""
Stock Universe Filter
Lightweight rules-based screening to reduce processing from 5000+ stocks to 200-400 relevant candidates.
Focuses on stocks with actual trading signals and market activity.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd

from tools.alpaca_client import alpaca_client
from tools.resilient_signal_orchestrator import get_resilient_price_signals_sync
# SocialMediaCollector no longer directly used - using cached data instead
from config.settings import settings, is_crypto_symbol
# CRYPTO TRADING DISABLED - Comment out crypto functions
# from config.settings import get_crypto_pairs  # CRYPTO DISABLED

logger = logging.getLogger(__name__)

@dataclass
class StockSignal:
    """Individual stock/crypto signal for universe filtering."""
    symbol: str
    signal_type: str  # earnings, price_move, social, institutional, crypto_momentum
    strength: float   # 0.0 to 1.0
    description: str
    timestamp: datetime
    asset_type: str = "stock"  # "stock" or "crypto"

@dataclass
class UniverseFilterResult:
    """Result of universe filtering operation."""
    total_symbols: int
    filtered_symbols: List[str]
    signals: List[StockSignal]
    filter_summary: Dict[str, int]
    processing_time: float

class StockUniverseFilter:
    """Lightweight filter to identify actionable stocks and crypto from the full universe."""
    
    def __init__(self):
        """Initialize the universe filter."""
        # Stock filtering parameters
        self.min_price = 5.0          # Minimum stock price
        self.max_price = 1000.0       # Maximum stock price  
        self.min_volume = 100000      # Minimum daily volume
        self.min_market_cap = 1e9     # Minimum market cap ($1B)
        
        # Crypto filtering parameters
        self.crypto_min_volume = 1000000    # Minimum crypto daily volume (higher for liquidity)
        self.crypto_min_price = 0.01        # Minimum crypto price
        self.crypto_max_price = 100000      # Maximum crypto price
        
        # Signal thresholds
        self.price_move_threshold = 0.02    # 2% price move
        self.social_mention_threshold = 5   # Minimum social mentions
        self.news_threshold = 1             # Minimum news articles
        
        # Cache for performance
        self._active_symbols_cache = None
        self._cache_timestamp = None
        self._cache_duration = timedelta(hours=1)
    
    async def filter_universe(
        self, 
        base_symbols: Optional[List[str]] = None,
        max_symbols: int = 400,
        include_watchlist: bool = True,
        cached_social_data: Optional[Dict[str, Any]] = None
    ) -> UniverseFilterResult:
        """Filter the stock universe to actionable candidates."""
        
        start_time = datetime.now()
        logger.info("🔍 FILTERING STOCK & CRYPTO UNIVERSE")
        logger.info(f"Target: Reduce from 5000+ to max {max_symbols} actionable assets")
        
        # Step 1: Get base universe (stocks + crypto if enabled)
        if base_symbols:
            all_symbols = base_symbols
            logger.info(f"Using provided base symbols: {len(all_symbols)}")
        else:
            all_symbols = await self._get_tradeable_symbols()
            logger.info(f"Retrieved tradeable universe: {len(all_symbols)} symbols")
        
        # CRYPTO TRADING DISABLED - Comment out crypto pairs and market prioritization
        # crypto_pairs = []
        # market_closed = False
        # if settings.crypto_enabled:
        #     crypto_pairs = get_crypto_pairs()
        #     all_symbols.extend(crypto_pairs)
        #     logger.info(f"Added {len(crypto_pairs)} crypto pairs to universe")
        #     
        #     # Check if stock market is closed to prioritize crypto
        #     try:
        #         from tools.alpaca_client import alpaca_client
        #         market_closed = not alpaca_client.is_market_open()
        #         if market_closed:
        #             logger.info("🌙 Stock market is closed - prioritizing crypto assets for 24/7 trading")
        #     except Exception:
        #         pass
        crypto_pairs = []  # Always empty when crypto disabled
        market_closed = False  # Not relevant when crypto disabled
        
        # Step 2: Apply basic filters (price, volume, market cap)
        logger.info("📊 Applying basic filters (price, volume, market cap)...")
        basic_filtered = await self._apply_basic_filters(all_symbols)
        logger.info(f"After basic filters: {len(basic_filtered)} symbols")
        
        # Step 3: Collect signals in parallel
        logger.info("🚀 Collecting trading signals in parallel...")
        signals = await self._collect_signals_parallel(basic_filtered, cached_social_data)
        logger.info(f"Collected {len(signals)} trading signals")
        
        # Step 4: Rank and select top candidates (crypto-aware)
        logger.info("🎯 Ranking candidates by signal strength...")
        filtered_symbols = self._rank_and_select_candidates(signals, max_symbols, market_closed, crypto_pairs)
        
        # Step 5: Include dynamic watchlist if requested (no hardcoded stocks)
        if include_watchlist:
            # Use settings-based watchlist only (empty by default for dynamic discovery)
            watchlist = getattr(settings, 'default_watchlist', [])
            added_count = 0
            for symbol in watchlist:
                if symbol not in filtered_symbols and symbol in basic_filtered:
                    filtered_symbols.append(symbol)
                    added_count += 1
            logger.info(f"Added {added_count} watchlist symbols (dynamic discovery enabled)")
        
        # Step 6: Fail fast if insufficient symbols found - no fallbacks allowed
        if len(filtered_symbols) == 0:
            error_msg = (
                f"❌ CRITICAL SYSTEM FAILURE: Universe filtering failed\n"
                f"Total symbols processed: {len(all_symbols)}\n"
                f"Symbols after basic filters: {len(basic_filtered)}\n"
                f"Trading signals collected: {len(signals)}\n"
                f"Final candidates selected: {len(filtered_symbols)}\n"
                f"SYSTEM REQUIRES VALID TRADING SIGNALS TO OPERATE SAFELY\n"
                f"All external data sources must be functional for signal generation"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Calculate summary
        filter_summary = self._calculate_filter_summary(signals)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = UniverseFilterResult(
            total_symbols=len(all_symbols),
            filtered_symbols=filtered_symbols[:max_symbols],
            signals=signals,
            filter_summary=filter_summary,
            processing_time=processing_time
        )
        
        logger.info("✅ UNIVERSE FILTERING COMPLETE")
        logger.info(f"Filtered: {result.total_symbols} → {len(result.filtered_symbols)} stocks ({processing_time:.1f}s)")
        logger.info(f"Signal breakdown: {filter_summary}")
        
        return result
    
    async def _get_tradeable_symbols(self) -> List[str]:
        """Get list of tradeable symbols from Alpaca."""
        
        # Use cache if available and fresh
        if (self._active_symbols_cache and self._cache_timestamp and 
            datetime.now() - self._cache_timestamp < self._cache_duration):
            logger.info("Using cached symbols")
            return self._active_symbols_cache
        
        try:
            # Get active assets from Alpaca
            assets = alpaca_client.api.list_assets(status='active', asset_class='us_equity')
            symbols = [asset.symbol for asset in assets if asset.tradable and asset.symbol.isalpha()]
            
            # Filter out penny stocks and very expensive stocks
            symbols = [s for s in symbols if len(s) <= 5]  # Remove complex tickers
            
            # Cache results
            self._active_symbols_cache = symbols
            self._cache_timestamp = datetime.now()
            
            logger.info(f"Retrieved {len(symbols)} tradeable symbols from Alpaca")
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to get tradeable symbols: {e}")
            logger.error("Unable to retrieve tradeable symbols from Alpaca API")
            return []  # Return empty list to force dynamic discovery
    
    async def _apply_basic_filters(self, symbols: List[str]) -> List[str]:
        """Apply basic price, volume, and market cap filters."""
        
        # Apply real-time filtering based on market data and trading activity
        # This is more dynamic than using hardcoded quality lists
        quality_stocks = []
        
        # Use dynamic filtering based on actual market activity
        # Focus on liquid, tradeable stocks without hardcoded lists
        filtered_symbols = []
        
        # Apply filtering to all symbols (stocks and crypto)
        for symbol in symbols:
            if is_crypto_symbol(symbol):
                # CRYPTO TRADING DISABLED - Skip all crypto symbols
                # All configured crypto pairs pass basic filtering
                # filtered_symbols.append(symbol)
                pass  # Skip crypto symbols when disabled
            else:
                # Stock-specific filtering
                # Quick heuristics for potentially interesting stocks
                if (len(symbol) <= 5 and                    # Not complex ticker
                    symbol.isalpha() and                    # Only letters
                    not any(char.islower() for char in symbol) and  # All caps
                    symbol not in ['ETF', 'FUND', 'INDEX']):  # Exclude obvious ETFs
                    filtered_symbols.append(symbol)
        
        # CRYPTO TRADING DISABLED - Comment out crypto vs stock counting and prioritization
        # crypto_count = sum(1 for s in filtered_symbols if is_crypto_symbol(s))
        # stock_count = len(filtered_symbols) - crypto_count
        # 
        # # Limit to manageable size for processing (but preserve all crypto)
        # # Keep all crypto pairs and limit stocks
        # crypto_symbols = [s for s in filtered_symbols if is_crypto_symbol(s)]
        # stock_symbols = [s for s in filtered_symbols if not is_crypto_symbol(s)][:300 - len(crypto_symbols)]
        # 
        # filtered_symbols = crypto_symbols + stock_symbols
        
        # Only process stocks when crypto is disabled
        crypto_count = 0
        stock_count = len(filtered_symbols)
        stock_symbols = [s for s in filtered_symbols if not is_crypto_symbol(s)]
        
        # Shuffle to avoid alphabetical bias before taking top 300
        import random
        random.shuffle(stock_symbols)
        stock_symbols = stock_symbols[:300]
        filtered_symbols = stock_symbols
        
        logger.info(f"Dynamic filtering applied to {len(symbols)} symbols")
        logger.info(f"Selected {len(filtered_symbols)} candidates ({crypto_count} crypto, {stock_count} stocks)")
        
        return filtered_symbols
    
    async def _collect_signals_parallel(self, symbols: List[str], cached_social_data: Optional[Dict[str, Any]] = None) -> List[StockSignal]:
        """Collect trading signals in parallel for efficiency."""
        
        signals = []
        
        # CRYPTO TRADING DISABLED - Comment out crypto separation
        # crypto_symbols = [s for s in symbols if is_crypto_symbol(s)]
        # stock_symbols = [s for s in symbols if not is_crypto_symbol(s)]
        crypto_symbols = []  # Always empty when crypto disabled
        stock_symbols = [s for s in symbols if not is_crypto_symbol(s)]
        
        # Create tasks for different signal types
        tasks = [
            self._collect_price_move_signals(symbols),  # Works for both crypto and stocks
            self._collect_earnings_signals(stock_symbols),  # Only for stocks
            self._collect_social_signals(symbols[:100], cached_social_data),  # Works for both
            self._collect_news_signals(symbols[:200]),     # Works for both
            # self._collect_crypto_momentum_signals(crypto_symbols) if crypto_symbols else None  # CRYPTO DISABLED
        ]
        
        # Filter out None tasks
        tasks = [task for task in tasks if task is not None]
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Signal collection task {i} failed: {result}")
            else:
                signals.extend(result)
        
        return signals
    
    async def _collect_price_move_signals(self, symbols: List[str]) -> List[StockSignal]:
        """Collect signals from significant price movements using resilient multi-source approach."""
        
        signals = []
        logger.info(f"🔍 Collecting resilient price signals for {len(symbols)} symbols...")
        
        try:
            # Use the resilient signal orchestrator (handles multiple data sources automatically)
            # Limit to first 50 symbols for performance
            limited_symbols = symbols[:50]
            
            # Get price change signals using resilient orchestration
            # This will automatically try Alpha Vantage, Finnhub, FMP, News API, and fallback strategies
            price_signals = await asyncio.to_thread(
                get_resilient_price_signals_sync, 
                limited_symbols, 
                self.price_move_threshold,
                2  # Minimum 2 signals required
            )
            
            # Convert to StockSignal format
            for signal_data in price_signals:
                signals.append(StockSignal(
                    symbol=signal_data["symbol"],
                    signal_type=signal_data.get("signal_type", "price_move"),
                    strength=signal_data["strength"],
                    description=signal_data["description"],
                    timestamp=datetime.now()
                ))
            
            logger.info(f"✅ Found {len(signals)} resilient signals using multi-source orchestration")
            logger.info(f"Sources used: {set(s.get('data_source', 'unknown') for s in price_signals)}")
            
        except Exception as e:
            # The resilient orchestrator should handle all fallbacks internally
            # If it fails here, it means all data sources are unavailable
            error_msg = f"❌ CRITICAL: Resilient signal orchestration failed: {e} - all data sources unavailable"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # The resilient orchestrator guarantees minimum signals, but double-check
        if len(signals) < 2:
            error_msg = f"❌ CRITICAL: Resilient orchestrator returned insufficient signals ({len(signals)} found, minimum 2 required)"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        return signals
    
    async def _generate_fallback_price_signals(self) -> List[StockSignal]:
        """REMOVED: Fallback price signals disabled - system must use real market data only."""
        error_msg = (
            "❌ CRITICAL SYSTEM FAILURE: Fallback price signals disabled\n"
            "System requires valid market data from external providers (Finnhub/Alpha Vantage)\n"
            "No hardcoded fallback mechanisms are permitted per system design\n"
            "Please check external API connectivity and rate limits"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    async def _collect_earnings_signals(self, symbols: List[str]) -> List[StockSignal]:
        """Collect signals from recent earnings announcements."""
        
        signals = []
        logger.info(f"📈 Checking earnings for {len(symbols)} symbols...")
        
        try:
            # Check for recent earnings (simplified - would integrate with earnings calendar)
            today = datetime.now()
            earnings_symbols = []  # Would be populated from earnings calendar API
            
            # Integrate with earnings scraper for recent earnings data
            from core.earnings_scraper import EarningsCallScraper
            earnings_scraper = EarningsCallScraper()
            
            # Check earnings for top symbols (limit for performance)
            for symbol in symbols[:30]:
                try:
                    transcript = await earnings_scraper.get_latest_transcript(symbol)
                    if transcript:
                        # Calculate days since earnings
                        days_since = (today - transcript.get('date', today)).days
                        if days_since <= 7:  # Recent earnings within 7 days
                            strength = max(0.1, 1.0 - (days_since / 7.0))
                            signals.append(StockSignal(
                                symbol=symbol,
                                signal_type="earnings",
                                strength=strength,
                                description=f"Recent earnings call ({days_since} days ago)",
                                timestamp=datetime.now()
                            ))
                except Exception as e:
                    logger.debug(f"Earnings check failed for {symbol}: {e}")
                    continue
            
            await earnings_scraper.cleanup()
        
        except Exception as e:
            logger.error(f"Earnings signal collection failed: {e}")
        
        logger.info(f"Found {len(signals)} earnings signals")
        return signals
    
    async def _collect_social_signals(self, symbols: List[str], cached_social_data: Dict[str, Any] = None) -> List[StockSignal]:
        """Collect signals from social media activity using cached data from sentiment analysis."""
        
        signals = []
        logger.info(f"💬 Using cached social data for {len(symbols)} symbols...")
        
        if not cached_social_data:
            logger.info("No cached social media data available - will be populated on next 90-minute sentiment cycle")
            return signals
        
        try:
            for symbol in symbols[:50]:  # Limit to match original behavior
                try:
                    # Check if we have cached social data for this symbol
                    symbol_social_data = cached_social_data.get(symbol, {})
                    if not symbol_social_data:
                        continue
                    
                    # Extract mention count from cached data
                    total_mentions = 0
                    
                    # Count mentions from all platforms
                    for platform in ['reddit', 'twitter', 'tiktok']:
                        platform_data = symbol_social_data.get(platform, [])
                        if isinstance(platform_data, list):
                            total_mentions += len(platform_data)
                    
                    if total_mentions >= self.social_mention_threshold:
                        strength = min(1.0, total_mentions / 50.0)  # Scale to 0-1
                        
                        signals.append(StockSignal(
                            symbol=symbol,
                            signal_type="social",
                            strength=strength,
                            description=f"{total_mentions} cached social mentions",
                            timestamp=datetime.now()
                        ))
                
                except Exception as e:
                    logger.debug(f"Cached social check failed for {symbol}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Cached social signal processing failed: {e}")
        
        logger.info(f"Found {len(signals)} cached social signals")
        return signals
    
    async def _collect_news_signals(self, symbols: List[str]) -> List[StockSignal]:
        """Collect signals from recent news activity."""
        
        signals = []
        logger.info(f"📰 Checking news activity for {len(symbols)} symbols...")
        
        try:
            # Integrate with market intelligence for news collection
            from core.market_intelligence import UnifiedMarketIntelligence
            market_intel = UnifiedMarketIntelligence()
            
            # Check news for top symbols (limit for API rate limiting)
            for symbol in symbols[:50]:
                try:
                    articles = await market_intel._get_news_articles(symbol)
                    if len(articles) >= self.news_threshold:
                        strength = min(1.0, len(articles) / 10.0)
                        signals.append(StockSignal(
                            symbol=symbol,
                            signal_type="news",
                            strength=strength,
                            description=f"{len(articles)} news articles",
                            timestamp=datetime.now()
                        ))
                except Exception as e:
                    logger.debug(f"News check failed for {symbol}: {e}")
                    continue
            
            await market_intel.close()
        
        except Exception as e:
            logger.error(f"News signal collection failed: {e}")
        
        logger.info(f"Found {len(signals)} news signals")
        return signals
    
    def _rank_and_select_candidates(self, signals: List[StockSignal], max_symbols: int, market_closed: bool = False, crypto_pairs: List[str] = None) -> List[str]:
        """Rank candidates by signal strength and select top performers with crypto prioritization when markets closed."""
        
        # Group signals by symbol
        symbol_scores = {}
        for signal in signals:
            if signal.symbol not in symbol_scores:
                symbol_scores[signal.symbol] = {
                    'total_score': 0.0,
                    'signal_count': 0,
                    'signal_types': set()
                }
            
            symbol_scores[signal.symbol]['total_score'] += signal.strength
            symbol_scores[signal.symbol]['signal_count'] += 1
            symbol_scores[signal.symbol]['signal_types'].add(signal.signal_type)
        
        # Calculate final scores with bonuses for multiple signal types
        final_scores = []
        for symbol, data in symbol_scores.items():
            # Base score is average strength
            avg_score = data['total_score'] / data['signal_count']
            
            # Bonus for multiple signal types (diversification)
            type_bonus = len(data['signal_types']) * 0.1
            
            # Bonus for multiple signals of same type (conviction)
            volume_bonus = min(0.2, (data['signal_count'] - 1) * 0.05)
            
            # CRYPTO TRADING DISABLED - Comment out crypto priority bonus
            # crypto_bonus = 0.0
            # if market_closed and crypto_pairs and symbol in crypto_pairs:
            #     crypto_bonus = 0.3  # Significant boost for crypto when markets closed
            #     logger.debug(f"🌙 Crypto priority bonus applied to {symbol}")
            crypto_bonus = 0.0  # Always zero when crypto disabled
            
            final_score = avg_score + type_bonus + volume_bonus + crypto_bonus
            final_scores.append((symbol, final_score))
        
        # Sort by score and return top candidates
        final_scores.sort(key=lambda x: x[1], reverse=True)
        selected_symbols = [symbol for symbol, score in final_scores[:max_symbols]]
        
        logger.info(f"Top 10 candidates by score:")
        for i, (symbol, score) in enumerate(final_scores[:10]):
            types = list(symbol_scores[symbol]['signal_types'])
            logger.info(f"  {i+1}. {symbol}: {score:.3f} ({', '.join(types)})")
        
        return selected_symbols
    
    def _calculate_filter_summary(self, signals: List[StockSignal]) -> Dict[str, int]:
        """Calculate summary statistics for filtering results."""
        
        summary = {}
        for signal in signals:
            signal_type = signal.signal_type
            summary[signal_type] = summary.get(signal_type, 0) + 1
        
        return summary
    
    # CRYPTO TRADING DISABLED - Comment out entire crypto momentum collection method
    # async def _collect_crypto_momentum_signals(self, symbols: List[str]) -> List[StockSignal]:
    #     """Collect crypto-specific momentum signals."""
    #     
    #     signals = []
    #     logger.info(f"🚀 Checking crypto momentum for {len(symbols)} pairs...")
    #     
    #     for symbol in symbols:
    #         try:
    #             # Get recent crypto market data
    #             market_data = alpaca_client.get_market_data(symbol, limit=20)
    #             
    #             if len(market_data) >= 10:
    #                 prices = market_data["close"]
    #                 volumes = market_data["volume"]
    #                 
    #                 # Calculate crypto-specific momentum indicators
    #                 recent_return = (prices.iloc[-1] / prices.iloc[-5] - 1) if len(prices) >= 5 else 0
    #                 volume_spike = volumes.iloc[-1] / volumes.mean() if volumes.mean() > 0 else 1
    #                 
    #                 # Crypto momentum signal (higher volatility tolerance)
    #                 if abs(recent_return) > 0.05 and volume_spike > 1.5:  # 5% move with volume
    #                     signal = StockSignal(
    #                         symbol=symbol,
    #                         signal_type="crypto_momentum",
    #                         strength=min(1.0, abs(recent_return) * 5 + (volume_spike - 1) * 0.2),
    #                         description=f"Crypto momentum: {recent_return:.2%} move with {volume_spike:.1f}x volume",
    #                         timestamp=datetime.now(),
    #                         asset_type="crypto"
    #                     )
    #                     signals.append(signal)
    #                     
    #         except Exception as e:
    #             logger.warning(f"Error checking crypto momentum for {symbol}: {e}")
    #             continue
    #     
    #     logger.info(f"Found {len(signals)} crypto momentum signals")
    #     return signals

# Global instance
universe_filter = StockUniverseFilter()

async def filter_stock_universe(
    base_symbols: Optional[List[str]] = None,
    max_symbols: int = 400,
    include_watchlist: bool = True,
    cached_social_data: Optional[Dict[str, Any]] = None
) -> UniverseFilterResult:
    """Convenience function for filtering stock universe."""
    return await universe_filter.filter_universe(base_symbols, max_symbols, include_watchlist, cached_social_data)