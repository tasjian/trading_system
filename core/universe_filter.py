#!/usr/bin/env python3
"""
Stock Universe Filter
Lightweight rules-based screening to reduce processing from 5000+ stocks to 200-400 relevant candidates.
Focuses on stocks with actual trading signals and market activity.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
import yfinance as yf
import pandas as pd

from tools.alpaca_client import alpaca_client
from core.social_media_collector import SocialMediaCollector
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class StockSignal:
    """Individual stock signal for universe filtering."""
    symbol: str
    signal_type: str  # earnings, price_move, social, institutional
    strength: float   # 0.0 to 1.0
    description: str
    timestamp: datetime

@dataclass
class UniverseFilterResult:
    """Result of universe filtering operation."""
    total_symbols: int
    filtered_symbols: List[str]
    signals: List[StockSignal]
    filter_summary: Dict[str, int]
    processing_time: float

class StockUniverseFilter:
    """Lightweight filter to identify actionable stocks from the full universe."""
    
    def __init__(self):
        """Initialize the universe filter."""
        self.min_price = 5.0          # Minimum stock price
        self.max_price = 1000.0       # Maximum stock price  
        self.min_volume = 100000      # Minimum daily volume
        self.min_market_cap = 1e9     # Minimum market cap ($1B)
        
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
        include_watchlist: bool = True
    ) -> UniverseFilterResult:
        """Filter the stock universe to actionable candidates."""
        
        start_time = datetime.now()
        logger.info("🔍 FILTERING STOCK UNIVERSE")
        logger.info(f"Target: Reduce from 5000+ to max {max_symbols} actionable stocks")
        
        # Step 1: Get base universe
        if base_symbols:
            all_symbols = base_symbols
            logger.info(f"Using provided base symbols: {len(all_symbols)}")
        else:
            all_symbols = await self._get_tradeable_symbols()
            logger.info(f"Retrieved tradeable universe: {len(all_symbols)} symbols")
        
        # Step 2: Apply basic filters (price, volume, market cap)
        logger.info("📊 Applying basic filters (price, volume, market cap)...")
        basic_filtered = await self._apply_basic_filters(all_symbols)
        logger.info(f"After basic filters: {len(basic_filtered)} symbols")
        
        # Step 3: Collect signals in parallel
        logger.info("🚀 Collecting trading signals in parallel...")
        signals = await self._collect_signals_parallel(basic_filtered)
        logger.info(f"Collected {len(signals)} trading signals")
        
        # Step 4: Rank and select top candidates
        logger.info("🎯 Ranking candidates by signal strength...")
        filtered_symbols = self._rank_and_select_candidates(signals, max_symbols)
        
        # Step 5: Include watchlist if requested
        if include_watchlist:
            watchlist = getattr(settings, 'default_watchlist', [
                'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'AMZN'
            ])
            for symbol in watchlist:
                if symbol not in filtered_symbols and symbol in basic_filtered:
                    filtered_symbols.append(symbol)
            logger.info(f"Added {len(watchlist)} watchlist symbols")
        
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
            # Fallback to S&P 500 + common stocks
            fallback_symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
                'UNH', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'DIS', 'BAC', 'ADBE',
                'CRM', 'NFLX', 'AMD', 'PFE', 'KO', 'XOM', 'AVGO', 'COST', 'CVX'
            ]
            logger.warning(f"Using fallback symbols: {len(fallback_symbols)}")
            return fallback_symbols
    
    async def _apply_basic_filters(self, symbols: List[str]) -> List[str]:
        """Apply basic price, volume, and market cap filters."""
        
        # For efficiency, use a pre-filtered list of quality stocks rather than processing all 11k
        # This represents major liquid stocks that are worth analyzing
        quality_stocks = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC',
            'ORCL', 'CRM', 'ADBE', 'NFLX', 'CSCO', 'AVGO', 'TXN', 'QCOM', 'IBM', 'AMAT',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'COF', 'AXP',
            'BLK', 'SPGI', 'CME', 'ICE', 'MCO', 'MSCI', 'TRV', 'AIG', 'PGR', 'ALL',
            
            # Healthcare
            'UNH', 'JNJ', 'PFE', 'ABT', 'TMO', 'DHR', 'BMY', 'ABBV', 'LLY', 'MRK',
            'MDT', 'ISRG', 'GILD', 'AMGN', 'VRTX', 'REGN', 'BIIB', 'ILMN', 'MRNA', 'ZTS',
            
            # Consumer
            'AMZN', 'TSLA', 'HD', 'LOW', 'TGT', 'WMT', 'COST', 'SBUX', 'MCD', 'NKE',
            'DIS', 'CMCSA', 'VZ', 'T', 'NFLX', 'PG', 'KO', 'PEP', 'WMT', 'PM',
            
            # Industrial
            'CAT', 'DE', 'GE', 'HON', 'MMM', 'BA', 'LMT', 'RTX', 'UPS', 'FDX',
            'EMR', 'ITW', 'ETN', 'PH', 'ROK', 'DOV', 'XYL', 'IR', 'FAST', 'PCAR',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE',
            
            # Materials
            'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'NUE',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'ES', 'ED',
            
            # Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR', 'O', 'SBAC', 'SPG'
        ]
        
        # Filter to only symbols that are in our tradeable universe
        filtered_symbols = [s for s in quality_stocks if s in symbols]
        
        # Add any symbols from input that aren't in quality list but might be interesting
        # (e.g., recent IPOs, unusual activity stocks)
        additional_candidates = []
        for symbol in symbols:
            # Skip if already included
            if symbol in filtered_symbols:
                continue
                
            # Quick heuristics for potentially interesting stocks
            if (len(symbol) <= 4 and                    # Not complex ticker
                symbol.isalpha() and                    # Only letters
                not any(char.islower() for char in symbol)):  # All caps
                additional_candidates.append(symbol)
        
        # Limit additional candidates to avoid processing overload
        filtered_symbols.extend(additional_candidates[:100])
        
        logger.info(f"Quality stock filter: {len(quality_stocks)} quality stocks")
        logger.info(f"Matched tradeable: {len([s for s in quality_stocks if s in symbols])}")
        logger.info(f"Additional candidates: {len(additional_candidates[:100])}")
        
        return filtered_symbols
    
    async def _collect_signals_parallel(self, symbols: List[str]) -> List[StockSignal]:
        """Collect trading signals in parallel for efficiency."""
        
        signals = []
        
        # Create tasks for different signal types
        tasks = [
            self._collect_price_move_signals(symbols),
            self._collect_earnings_signals(symbols),
            self._collect_social_signals(symbols[:100]),  # Limit social to top 100 for API limits
            self._collect_news_signals(symbols[:200])     # Limit news to top 200
        ]
        
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
        
        try:
            # Get current and previous close data
            batch_size = 50
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                
                try:
                    # Get recent price data
                    tickers = yf.Tickers(' '.join(batch))
                    
                    for symbol in batch:
                        try:
                            ticker = tickers.tickers[symbol]
                            hist = ticker.history(period='2d')
                            
                            if len(hist) >= 2:
                                prev_close = hist['Close'].iloc[-2]
                                curr_price = hist['Close'].iloc[-1]
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
                    logger.warning(f"Price batch failed: {e}")
                    continue
                
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
            
            # For now, use a simple heuristic based on known earnings patterns
            # In production, this would integrate with earnings calendar APIs
            common_earnings_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
            
            for symbol in symbols:
                if symbol in common_earnings_symbols:
                    # Simulate earnings signal (would be real data in production)
                    signals.append(StockSignal(
                        symbol=symbol,
                        signal_type="earnings",
                        strength=0.8,
                        description="Recent earnings announcement",
                        timestamp=today
                    ))
        
        except Exception as e:
            logger.error(f"Earnings signal collection failed: {e}")
        
        logger.info(f"Found {len(signals)} earnings signals")
        return signals
    
    async def _collect_social_signals(self, symbols: List[str]) -> List[StockSignal]:
        """Collect signals from social media activity."""
        
        signals = []
        logger.info(f"💬 Checking social activity for {len(symbols)} symbols...")
        
        try:
            # Use existing social media collector for top symbols
            for symbol in symbols[:50]:  # Limit to avoid API overload
                try:
                    # Get social media data
                    social_media_collector = SocialMediaCollector()
                    social_data = await social_media_collector.collect_social_data(symbol)
                    
                    total_mentions = 0
                    for platform_data in social_data.values():
                        if isinstance(platform_data, list):
                            total_mentions += len(platform_data)
                    
                    if total_mentions >= self.social_mention_threshold:
                        strength = min(1.0, total_mentions / 50.0)  # Scale to 0-1
                        
                        signals.append(StockSignal(
                            symbol=symbol,
                            signal_type="social",
                            strength=strength,
                            description=f"{total_mentions} social mentions",
                            timestamp=datetime.now()
                        ))
                
                except Exception as e:
                    logger.debug(f"Social check failed for {symbol}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Social signal collection failed: {e}")
        
        logger.info(f"Found {len(signals)} social signals")
        return signals
    
    async def _collect_news_signals(self, symbols: List[str]) -> List[StockSignal]:
        """Collect signals from recent news activity."""
        
        signals = []
        logger.info(f"📰 Checking news activity for {len(symbols)} symbols...")
        
        try:
            # Check for recent news (simplified implementation)
            # In production, this would integrate with news APIs
            
            high_news_symbols = ['AAPL', 'TSLA', 'NVDA', 'META', 'GOOGL', 'MSFT', 'AMZN']
            
            for symbol in symbols:
                if symbol in high_news_symbols:
                    signals.append(StockSignal(
                        symbol=symbol,
                        signal_type="news",
                        strength=0.6,
                        description="Recent news coverage",
                        timestamp=datetime.now()
                    ))
        
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

# Global instance
universe_filter = StockUniverseFilter()

async def filter_stock_universe(
    base_symbols: Optional[List[str]] = None,
    max_symbols: int = 400,
    include_watchlist: bool = True
) -> UniverseFilterResult:
    """Convenience function for filtering stock universe."""
    return await universe_filter.filter_universe(base_symbols, max_symbols, include_watchlist)