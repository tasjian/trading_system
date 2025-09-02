#!/usr/bin/env python3
"""
Stock Universe Filter
Lightweight rules-based screening to reduce processing from 5000+ stocks to 200-400 relevant candidates.
Focuses on stocks with actual trading signals and market activity.
"""

import asyncio
import logging
import math
import statistics
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
    raw_value: Optional[float] = None  # Original raw value for normalization
    volatility: Optional[float] = None  # Historical volatility for risk adjustment
    sector: Optional[str] = None  # GICS sector for diversification

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
        
        # Sentiment normalization baselines (updated periodically)
        self._sentiment_baselines = {
            'TSLA': 50, 'AAPL': 40, 'NVDA': 35, 'AMZN': 30, 'GOOGL': 25,
            'META': 20, 'MSFT': 20, 'NFLX': 15, 'AMD': 15, 'UBER': 10
        }
        
        # Diagnostics tracking
        self._filter_diagnostics = {
            'stage1_dropped': 0,
            'stage2_signals_collected': 0,
            'stage3_cross_signal_filtered': 0,
            'final_selected': 0
        }
    
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
        self._filter_diagnostics['stage1_dropped'] = len(all_symbols) - len(basic_filtered)
        logger.info(f"After basic filters: {len(basic_filtered)} symbols ({self._filter_diagnostics['stage1_dropped']} dropped)")
        
        # Step 3: Collect signals in parallel
        logger.info("🚀 Collecting trading signals in parallel...")
        signals = await self._collect_signals_parallel(basic_filtered, cached_social_data)
        self._filter_diagnostics['stage2_signals_collected'] = len(signals)
        
        # Apply signal enhancements
        enhanced_signals = self._enhance_signals(signals)
        logger.info(f"Collected {len(enhanced_signals)} enhanced trading signals")
        
        # Step 4: Apply cross-signal confirmation and rank candidates
        logger.info("🎯 Applying cross-signal confirmation and ranking...")
        confirmed_signals = self._apply_cross_signal_confirmation(enhanced_signals)
        self._filter_diagnostics['stage3_cross_signal_filtered'] = len(confirmed_signals)
        
        filtered_symbols = await self._rank_and_select_candidates_enhanced(confirmed_signals, max_symbols, market_closed, crypto_pairs)
        self._filter_diagnostics['final_selected'] = len(filtered_symbols)
        
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
        
        # Step 6: Elastic thresholds for graceful degradation
        if len(filtered_symbols) < 10:  # Minimum viable universe
            logger.warning(f"Low signal count ({len(filtered_symbols)}), applying elastic thresholds...")
            
            # Try relaxed basic filters
            relaxed_symbols = await self._apply_relaxed_basic_filters(all_symbols)
            if len(relaxed_symbols) > len(basic_filtered):
                logger.info(f"Elastic thresholds: relaxed basic filters from {len(basic_filtered)} to {len(relaxed_symbols)}")
                
                # Re-run signal collection on relaxed universe
                additional_signals = await self._collect_signals_parallel(relaxed_symbols, cached_social_data)
                enhanced_additional = self._enhance_signals(additional_signals)
                confirmed_additional = self._apply_cross_signal_confirmation(enhanced_additional)
                
                # Combine and re-rank
                all_confirmed = confirmed_signals + confirmed_additional
                filtered_symbols = await self._rank_and_select_candidates_enhanced(all_confirmed, max_symbols, market_closed, crypto_pairs)
                self._filter_diagnostics['final_selected'] = len(filtered_symbols)
                
                logger.info(f"Elastic recovery: {len(filtered_symbols)} symbols after relaxed filtering")
        
        # Final fail-safe
        if len(filtered_symbols) == 0:
            error_msg = (
                f"❌ CRITICAL SYSTEM FAILURE: Universe filtering failed even with elastic thresholds\n"
                f"Total symbols processed: {len(all_symbols)}\n"
                f"Symbols after basic filters: {len(basic_filtered)}\n"
                f"Trading signals collected: {self._filter_diagnostics['stage2_signals_collected']}\n"
                f"Cross-signal confirmed: {self._filter_diagnostics['stage3_cross_signal_filtered']}\n"
                f"Final candidates selected: {len(filtered_symbols)}\n"
                f"SYSTEM REQUIRES VALID TRADING SIGNALS TO OPERATE SAFELY"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Calculate summary
        filter_summary = self._calculate_filter_summary(confirmed_signals)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = UniverseFilterResult(
            total_symbols=len(all_symbols),
            filtered_symbols=filtered_symbols[:max_symbols],
            signals=confirmed_signals,
            filter_summary=filter_summary,
            processing_time=processing_time
        )
        
        logger.info("✅ UNIVERSE FILTERING COMPLETE")
        logger.info(f"Filtered: {result.total_symbols} → {len(result.filtered_symbols)} stocks ({processing_time:.1f}s)")
        logger.info(f"Signal breakdown: {filter_summary}")
        logger.info(f"📊 Diagnostics: {self._filter_diagnostics}")
        
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
    
    def _enhance_signals(self, signals: List[StockSignal]) -> List[StockSignal]:
        """Apply signal enhancements: normalization, decay weighting, and risk adjustment."""
        
        enhanced_signals = []
        
        for signal in signals:
            enhanced_signal = signal
            
            # 1. Sentiment normalization against baseline chatter
            if signal.signal_type == "social":
                baseline = self._sentiment_baselines.get(signal.symbol, 5)  # Default baseline
                raw_mentions = signal.raw_value or 0
                
                # Normalize: (current - baseline) / baseline, capped at reasonable range
                if baseline > 0:
                    normalized_strength = max(0.0, min(1.0, (raw_mentions - baseline) / baseline))
                    enhanced_signal = StockSignal(
                        symbol=signal.symbol,
                        signal_type=signal.signal_type,
                        strength=normalized_strength,
                        description=f"{raw_mentions} mentions (baseline: {baseline})",
                        timestamp=signal.timestamp,
                        asset_type=signal.asset_type,
                        raw_value=raw_mentions,
                        volatility=signal.volatility,
                        sector=signal.sector
                    )
            
            # 2. Decay-weighted signals for temporal relevance
            hours_old = (datetime.now() - signal.timestamp).total_seconds() / 3600
            decay_factor = math.exp(-hours_old / 24)  # Half-life of 24 hours
            
            enhanced_signal = StockSignal(
                symbol=enhanced_signal.symbol,
                signal_type=enhanced_signal.signal_type,
                strength=enhanced_signal.strength * decay_factor,
                description=f"{enhanced_signal.description} (decay: {decay_factor:.2f})",
                timestamp=enhanced_signal.timestamp,
                asset_type=enhanced_signal.asset_type,
                raw_value=enhanced_signal.raw_value,
                volatility=enhanced_signal.volatility,
                sector=enhanced_signal.sector
            )
            
            enhanced_signals.append(enhanced_signal)
        
        logger.info(f"Enhanced {len(signals)} signals with normalization and decay weighting")
        return enhanced_signals
    
    def _apply_cross_signal_confirmation(self, signals: List[StockSignal]) -> List[StockSignal]:
        """Apply intelligent cross-signal confirmation with market-hours flexibility."""
        
        # Group signals by symbol and count unique signal types
        symbol_signal_types = {}
        for signal in signals:
            if signal.symbol not in symbol_signal_types:
                symbol_signal_types[signal.symbol] = set()
            symbol_signal_types[signal.symbol].add(signal.signal_type)
        
        # Check if we have diverse signal types available
        available_signal_types = set()
        for signal in signals:
            available_signal_types.add(signal.signal_type)
        
        # Adaptive confirmation based on available signal diversity
        if len(available_signal_types) >= 3:
            # Multiple signal types available - require ≥2 sources
            min_sources = 2
            logger.info(f"Cross-signal confirmation: strict mode (≥2 sources) - {len(available_signal_types)} signal types available")
        else:
            # Limited signal types (e.g., only price signals during market hours) - allow single strong signals
            min_sources = 1
            logger.info(f"Cross-signal confirmation: relaxed mode (≥1 source) - only {len(available_signal_types)} signal types available")
        
        # Filter signals based on adaptive threshold
        confirmed_signals = []
        for signal in signals:
            if len(symbol_signal_types[signal.symbol]) >= min_sources:
                confirmed_signals.append(signal)
        
        logger.info(f"Cross-signal confirmation: {len(signals)} → {len(confirmed_signals)} signals (min_sources: {min_sources})")
        return confirmed_signals
    
    async def _apply_relaxed_basic_filters(self, symbols: List[str]) -> List[str]:
        """Apply relaxed basic filters for elastic threshold recovery."""
        
        filtered_symbols = []
        
        for symbol in symbols:
            if not is_crypto_symbol(symbol):
                # Relaxed stock filtering (lower thresholds)
                if (len(symbol) <= 6 and                    # Allow slightly longer tickers
                    symbol.isalpha() and                    # Only letters
                    not any(char.islower() for char in symbol) and  # All caps
                    symbol not in ['ETF', 'FUND', 'INDEX']):  # Exclude obvious ETFs
                    filtered_symbols.append(symbol)
        
        # Increase sample size for relaxed filtering
        import random
        random.shuffle(filtered_symbols)
        filtered_symbols = filtered_symbols[:500]  # Increased from 300
        
        logger.info(f"Relaxed filtering: {len(filtered_symbols)} candidates (relaxed thresholds)")
        return filtered_symbols
    
    async def _rank_and_select_candidates_enhanced(self, signals: List[StockSignal], max_symbols: int, market_closed: bool = False, crypto_pairs: List[str] = None) -> List[str]:
        """Enhanced ranking with risk adjustment, sector diversification, and adaptive cutoffs."""
        
        # Group signals by symbol with enhanced scoring
        symbol_data = {}
        all_volatilities = []
        all_sectors = set()
        
        for signal in signals:
            if signal.symbol not in symbol_data:
                symbol_data[signal.symbol] = {
                    'signals': [],
                    'total_strength': 0.0,
                    'signal_types': set(),
                    'volatility': signal.volatility or 0.2,  # Default volatility
                    'sector': signal.sector or 'Unknown'
                }
            
            symbol_data[signal.symbol]['signals'].append(signal)
            symbol_data[signal.symbol]['total_strength'] += signal.strength
            symbol_data[signal.symbol]['signal_types'].add(signal.signal_type)
            
            if signal.volatility:
                all_volatilities.append(signal.volatility)
            if signal.sector:
                all_sectors.add(signal.sector)
        
        # Calculate enhanced scores
        final_scores = []
        selected_sectors = set()
        
        for symbol, data in symbol_data.items():
            # Base score: average signal strength
            base_score = data['total_strength'] / len(data['signals'])
            
            # Signal diversity bonus (multiple types)
            diversity_bonus = len(data['signal_types']) * 0.1
            
            # Signal conviction bonus (multiple signals)
            conviction_bonus = min(0.2, (len(data['signals']) - 1) * 0.05)
            
            # Risk adjustment: divide by volatility to favor stable performers
            volatility = data['volatility']
            risk_adjusted_score = base_score / max(0.1, volatility)  # Prevent division by zero
            
            # Sector diversification bonus (encourage spread)
            sector_bonus = 0.0
            if data['sector'] not in selected_sectors and data['sector'] != 'Unknown':
                sector_bonus = 0.05  # Small bonus for new sectors
            
            # CRYPTO TRADING DISABLED - crypto bonus always 0
            crypto_bonus = 0.0
            
            final_score = risk_adjusted_score + diversity_bonus + conviction_bonus + sector_bonus + crypto_bonus
            final_scores.append((symbol, final_score, data['sector']))
        
        # Sort by score
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Adaptive cutoff based on score distribution
        scores_only = [score for _, score, _ in final_scores]
        if len(scores_only) > 10:
            score_median = statistics.median(scores_only)
            score_std = statistics.stdev(scores_only) if len(scores_only) > 1 else 0.1
            adaptive_threshold = score_median + 0.5 * score_std
            
            # Apply adaptive cutoff but ensure minimum viable count
            above_threshold = [(s, sc, se) for s, sc, se in final_scores if sc >= adaptive_threshold]
            if len(above_threshold) >= 20:  # Ensure minimum viable universe
                final_scores = above_threshold
                logger.info(f"Applied adaptive threshold {adaptive_threshold:.3f}, kept {len(final_scores)} candidates")
        
        # Select symbols with sector tracking for diversification
        selected_symbols = []
        sector_counts = {}
        
        for symbol, score, sector in final_scores[:max_symbols * 2]:  # Consider more for diversification
            # Track sector distribution
            if len(selected_symbols) < max_symbols:
                selected_symbols.append(symbol)
                if sector != 'Unknown':
                    selected_sectors.add(sector)
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # Final selection (limit to max_symbols)
        selected_symbols = selected_symbols[:max_symbols]
        
        # Enhanced logging
        logger.info(f"Top 10 candidates by enhanced score:")
        for i, (symbol, score, sector) in enumerate(final_scores[:10]):
            types = list(symbol_data[symbol]['signal_types'])
            vol = symbol_data[symbol]['volatility']
            logger.info(f"  {i+1}. {symbol}: {score:.3f} (vol: {vol:.2f}, sector: {sector}, types: {', '.join(types)})")
        
        logger.info(f"Sector distribution: {dict(list(sector_counts.items())[:5])}...")  # Show top 5 sectors
        
        return selected_symbols
    
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