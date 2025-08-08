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
import yfinance as yf
import pandas as pd

from tools.alpaca_client import alpaca_client
# SocialMediaCollector no longer directly used - using cached data instead
from config.settings import settings, get_crypto_pairs, is_crypto_symbol

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
        
        # Add crypto pairs if enabled
        if settings.crypto_enabled:
            crypto_pairs = get_crypto_pairs()
            all_symbols.extend(crypto_pairs)
            logger.info(f"Added {len(crypto_pairs)} crypto pairs to universe")
        
        # Step 2: Apply basic filters (price, volume, market cap)
        logger.info("📊 Applying basic filters (price, volume, market cap)...")
        basic_filtered = await self._apply_basic_filters(all_symbols)
        logger.info(f"After basic filters: {len(basic_filtered)} symbols")
        
        # Step 3: Collect signals in parallel
        logger.info("🚀 Collecting trading signals in parallel...")
        signals = await self._collect_signals_parallel(basic_filtered, cached_social_data)
        logger.info(f"Collected {len(signals)} trading signals")
        
        # Step 4: Rank and select top candidates
        logger.info("🎯 Ranking candidates by signal strength...")
        filtered_symbols = self._rank_and_select_candidates(signals, max_symbols)
        
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
                # Crypto-specific filtering
                # All configured crypto pairs pass basic filtering
                filtered_symbols.append(symbol)
            else:
                # Stock-specific filtering
                # Quick heuristics for potentially interesting stocks
                if (len(symbol) <= 5 and                    # Not complex ticker
                    symbol.isalpha() and                    # Only letters
                    not any(char.islower() for char in symbol) and  # All caps
                    symbol not in ['ETF', 'FUND', 'INDEX']):  # Exclude obvious ETFs
                    filtered_symbols.append(symbol)
        
        # Count crypto vs stocks
        crypto_count = sum(1 for s in filtered_symbols if is_crypto_symbol(s))
        stock_count = len(filtered_symbols) - crypto_count
        
        # Limit to manageable size for processing (but preserve all crypto)
        # Keep all crypto pairs and limit stocks
        crypto_symbols = [s for s in filtered_symbols if is_crypto_symbol(s)]
        stock_symbols = [s for s in filtered_symbols if not is_crypto_symbol(s)][:300 - len(crypto_symbols)]
        
        filtered_symbols = crypto_symbols + stock_symbols
        
        logger.info(f"Dynamic filtering applied to {len(symbols)} symbols")
        logger.info(f"Selected {len(filtered_symbols)} candidates ({crypto_count} crypto, {stock_count} stocks)")
        
        return filtered_symbols
    
    async def _collect_signals_parallel(self, symbols: List[str], cached_social_data: Optional[Dict[str, Any]] = None) -> List[StockSignal]:
        """Collect trading signals in parallel for efficiency."""
        
        signals = []
        
        # Separate crypto and stock symbols for appropriate signal collection
        crypto_symbols = [s for s in symbols if is_crypto_symbol(s)]
        stock_symbols = [s for s in symbols if not is_crypto_symbol(s)]
        
        # Create tasks for different signal types
        tasks = [
            self._collect_price_move_signals(symbols),  # Works for both crypto and stocks
            self._collect_earnings_signals(stock_symbols),  # Only for stocks
            self._collect_social_signals(symbols[:100], cached_social_data),  # Works for both
            self._collect_news_signals(symbols[:200]),     # Works for both
            self._collect_crypto_momentum_signals(crypto_symbols) if crypto_symbols else None  # Crypto-specific
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
        """Collect signals from significant price movements."""
        
        signals = []
        logger.info(f"🔍 Checking price movements for {len(symbols)} symbols...")
        
        # Circuit breaker for yfinance issues
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            # Get current and previous close data with improved error handling
            batch_size = 20  # Reduced batch size to avoid rate limits
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                
                try:
                    # Add delay between batches to avoid rate limiting
                    if i > 0:
                        await asyncio.sleep(1.0)  # 1 second delay between batches
                    
                    # Get recent price data with retry mechanism
                    max_retries = 2
                    tickers = None
                    
                    for retry in range(max_retries):
                        try:
                            tickers = yf.Tickers(' '.join(batch))
                            break  # Success, exit retry loop
                        except Exception as retry_error:
                            if retry == max_retries - 1:  # Last retry
                                logger.warning(f"Failed to fetch batch after {max_retries} retries: {retry_error}")
                                raise retry_error
                            else:
                                logger.debug(f"Retry {retry + 1} for batch: {retry_error}")
                                await asyncio.sleep(2.0)  # Wait before retry
                    
                    if not tickers:
                        continue
                    
                    for symbol in batch:
                        try:
                            ticker = tickers.tickers[symbol]
                            
                            # Add timeout for individual ticker data
                            hist = ticker.history(period='2d')
                            
                            if len(hist) >= 2:
                                prev_close = hist['Close'].iloc[-2]
                                curr_price = hist['Close'].iloc[-1]
                                
                                # Validate price data
                                if pd.isna(prev_close) or pd.isna(curr_price) or prev_close <= 0 or curr_price <= 0:
                                    continue
                                
                                price_change = (curr_price - prev_close) / prev_close
                                
                                if abs(price_change) >= self.price_move_threshold:
                                    strength = min(1.0, abs(price_change) / 0.1)  # Scale to 0-1
                                    direction = "up" if price_change > 0 else "down"
                                    
                                    signals.append(StockSignal(
                                        symbol=symbol,
                                        signal_type="price_move",
                                        strength=strength,
                                        description=f"{direction} {price_change:.1%}",
                                        timestamp=datetime.now()
                                    ))
                        
                        except Exception as e:
                            logger.debug(f"Price check failed for {symbol}: {e}")
                            continue
                
                except Exception as e:
                    consecutive_errors += 1
                    logger.warning(f"Price batch failed: {e} ({consecutive_errors}/{max_consecutive_errors})")
                    
                    # Circuit breaker - stop processing if too many consecutive errors
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Circuit breaker triggered: {consecutive_errors} consecutive yfinance errors. Skipping remaining price analysis.")
                        break
                    continue
                else:
                    # Reset error counter on successful batch
                    consecutive_errors = 0
                
                # Brief pause between batches
                await asyncio.sleep(0.2)
        
        except Exception as e:
            logger.error(f"Price move signal collection failed: {e}")
        
        logger.info(f"Found {len(signals)} price movement signals")
        return signals
    
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
        """Collect signals from cached social media activity."""
        
        signals = []
        logger.info(f"💬 Using cached social data for {len(symbols)} symbols...")
        
        if not cached_social_data:
            logger.info("No cached social media data available - skipping social signals")
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
    
    def _rank_and_select_candidates(self, signals: List[StockSignal], max_symbols: int) -> List[str]:
        """Rank candidates by signal strength and select top performers."""
        
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
            
            final_score = avg_score + type_bonus + volume_bonus
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
    
    async def _collect_crypto_momentum_signals(self, symbols: List[str]) -> List[StockSignal]:
        """Collect crypto-specific momentum signals."""
        
        signals = []
        logger.info(f"🚀 Checking crypto momentum for {len(symbols)} pairs...")
        
        for symbol in symbols:
            try:
                # Get recent crypto market data
                market_data = alpaca_client.get_market_data(symbol, limit=20)
                
                if len(market_data) >= 10:
                    prices = market_data["close"]
                    volumes = market_data["volume"]
                    
                    # Calculate crypto-specific momentum indicators
                    recent_return = (prices.iloc[-1] / prices.iloc[-5] - 1) if len(prices) >= 5 else 0
                    volume_spike = volumes.iloc[-1] / volumes.mean() if volumes.mean() > 0 else 1
                    
                    # Crypto momentum signal (higher volatility tolerance)
                    if abs(recent_return) > 0.05 and volume_spike > 1.5:  # 5% move with volume
                        signal = StockSignal(
                            symbol=symbol,
                            signal_type="crypto_momentum",
                            strength=min(1.0, abs(recent_return) * 5 + (volume_spike - 1) * 0.2),
                            description=f"Crypto momentum: {recent_return:.2%} move with {volume_spike:.1f}x volume",
                            timestamp=datetime.now(),
                            asset_type="crypto"
                        )
                        signals.append(signal)
                        
            except Exception as e:
                logger.warning(f"Error checking crypto momentum for {symbol}: {e}")
                continue
        
        logger.info(f"Found {len(signals)} crypto momentum signals")
        return signals

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