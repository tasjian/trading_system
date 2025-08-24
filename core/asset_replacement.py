#!/usr/bin/env python3
"""
Asset Replacement Engine

A sophisticated asset replacement system for tax-loss harvesting that identifies
suitable replacement securities to maintain portfolio exposure while avoiding
wash sale violations. Uses correlation analysis, sector matching, and risk
profiling to find optimal substitutes.

Key Features:
- Correlation-based replacement identification
- Sector and industry matching for similar exposure
- Risk profile analysis and matching
- ETF-to-ETF and stock-to-ETF replacement strategies
- Real-time market data integration for correlation calculation
- Wash sale avoidance through substantially different securities
- Portfolio impact analysis and optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Literal, Union
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import json
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from scipy.stats import pearsonr
import yfinance as yf

from tools.alpaca_client import alpaca_client
from config.settings import settings

logger = logging.getLogger(__name__)

class ReplacementStrategy(Enum):
    """Asset replacement strategies."""
    HIGH_CORRELATION = "high_correlation"      # Find highly correlated assets
    SECTOR_MATCH = "sector_match"             # Match by sector/industry
    ETF_SUBSTITUTE = "etf_substitute"         # Replace individual stock with sector ETF
    INDEX_TRACKING = "index_tracking"         # Replace with index tracking alternatives
    INVERSE_CORRELATION = "inverse_correlation" # For hedging strategies

class AssetType(Enum):
    """Types of assets for replacement analysis."""
    STOCK = "stock"
    ETF = "etf"
    INDEX = "index"
    SECTOR_ETF = "sector_etf"
    COMMODITY_ETF = "commodity_etf"
    BOND_ETF = "bond_etf"

@dataclass
class ReplacementCandidate:
    """Represents a potential replacement asset."""
    symbol: str
    name: str
    asset_type: AssetType
    
    # Similarity metrics
    correlation: float
    sector_match_score: float
    risk_similarity_score: float
    liquidity_score: float
    
    # Replacement strategy details
    replacement_strategy: ReplacementStrategy
    confidence_score: float  # Overall confidence in this replacement
    
    # Market characteristics
    market_cap: Optional[Decimal] = None
    avg_volume: Optional[int] = None
    beta: Optional[float] = None
    expense_ratio: Optional[float] = None  # For ETFs
    
    # Risk metrics
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    
    # Practical considerations
    current_price: Optional[Decimal] = None
    min_position_size: Optional[Decimal] = None
    supports_fractional_shares: bool = True
    
    # Reasoning and metadata
    replacement_reasoning: str = ""
    data_quality_score: float = 1.0  # 0-1, quality of underlying data
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ReplacementAnalysis:
    """Comprehensive analysis of replacement options for an asset."""
    original_symbol: str
    analysis_date: datetime
    
    # Candidates by strategy
    high_correlation_candidates: List[ReplacementCandidate]
    sector_match_candidates: List[ReplacementCandidate]
    etf_substitute_candidates: List[ReplacementCandidate]
    index_tracking_candidates: List[ReplacementCandidate]
    
    # Best overall recommendations
    top_recommendations: List[ReplacementCandidate]
    
    # Analysis metadata
    correlation_lookback_days: int
    data_sources_used: List[str]
    analysis_quality_score: float
    
    # Portfolio impact
    estimated_tracking_error: float
    portfolio_risk_change: float
    liquidity_impact: str

class ReplacementDatabase:
    """SQLite database for caching replacement analysis and market data."""
    
    def __init__(self, db_path: str = "data/asset_replacement.db"):
        """Initialize database connection and create tables."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Asset metadata table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS asset_metadata (
                        symbol TEXT PRIMARY KEY,
                        name TEXT,
                        asset_type TEXT,
                        sector TEXT,
                        industry TEXT,
                        market_cap DECIMAL(20,2),
                        avg_volume INTEGER,
                        beta DECIMAL(5,2),
                        expense_ratio DECIMAL(5,4),
                        current_price DECIMAL(10,2),
                        supports_fractional BOOLEAN DEFAULT TRUE,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for asset_metadata table
                conn.execute('CREATE INDEX IF NOT EXISTS idx_asset_metadata_sector ON asset_metadata(sector)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_asset_metadata_industry ON asset_metadata(industry)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_asset_metadata_type ON asset_metadata(asset_type)')
                
                # Correlation cache table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS correlation_cache (
                        symbol_pair TEXT PRIMARY KEY,
                        symbol1 TEXT NOT NULL,
                        symbol2 TEXT NOT NULL,
                        correlation DECIMAL(5,4),
                        lookback_days INTEGER,
                        calculation_date TIMESTAMP,
                        data_quality_score DECIMAL(3,2),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for correlation_cache table
                conn.execute('CREATE INDEX IF NOT EXISTS idx_correlation_cache_symbol1 ON correlation_cache(symbol1)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_correlation_cache_symbol2 ON correlation_cache(symbol2)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_correlation_cache_date ON correlation_cache(calculation_date)')
                
                # Replacement candidates cache
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS replacement_candidates (
                        cache_id TEXT PRIMARY KEY,
                        original_symbol TEXT NOT NULL,
                        candidate_symbol TEXT NOT NULL,
                        replacement_strategy TEXT NOT NULL,
                        correlation DECIMAL(5,4),
                        sector_match_score DECIMAL(3,2),
                        risk_similarity_score DECIMAL(3,2),
                        liquidity_score DECIMAL(3,2),
                        confidence_score DECIMAL(3,2),
                        replacement_reasoning TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                ''')
                
                # Create indexes for replacement_candidates table
                conn.execute('CREATE INDEX IF NOT EXISTS idx_replacement_candidates_orig ON replacement_candidates(original_symbol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_replacement_candidates_cand ON replacement_candidates(candidate_symbol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_replacement_candidates_expires ON replacement_candidates(expires_at)')
                
                # Market data cache table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS market_data_cache (
                        symbol TEXT,
                        date DATE,
                        open_price DECIMAL(10,2),
                        high_price DECIMAL(10,2),
                        low_price DECIMAL(10,2),
                        close_price DECIMAL(10,2),
                        volume INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (symbol, date)
                    )
                ''')
                
                # Create indexes for market_data_cache table
                conn.execute('CREATE INDEX IF NOT EXISTS idx_market_data_cache_symbol ON market_data_cache(symbol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_market_data_cache_date ON market_data_cache(date)')
                
                conn.commit()
                logger.info("Asset replacement database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing replacement database: {e}")
            raise

class AssetReplacementEngine:
    """
    Comprehensive asset replacement engine for tax-loss harvesting.
    
    This engine identifies suitable replacement securities that maintain similar
    market exposure while avoiding wash sale violations. It uses multiple
    strategies including correlation analysis, sector matching, and ETF substitution
    to find optimal replacement candidates.
    """
    
    def __init__(self):
        """Initialize the asset replacement engine."""
        self.db = ReplacementDatabase()
        
        # Predefined replacement universes by category
        self.sector_etfs = self._initialize_sector_etfs()
        self.index_etfs = self._initialize_index_etfs()
        self.broad_market_etfs = self._initialize_broad_market_etfs()
        
        # Analysis parameters
        self.correlation_lookback_days = 252  # 1 year of trading days
        self.min_correlation_threshold = 0.70
        self.min_liquidity_volume = 100000  # Minimum daily volume
        
        # Caching parameters
        self.cache_expiry_hours = 24
        self.market_data_cache_days = 30
        
        logger.info("Asset Replacement Engine initialized")
        logger.info(f"Correlation lookback: {self.correlation_lookback_days} days")
        logger.info(f"Min correlation threshold: {self.min_correlation_threshold}")
    
    async def find_replacement_candidates(self, 
                                        original_symbol: str,
                                        max_candidates: int = 10,
                                        strategies: Optional[List[ReplacementStrategy]] = None) -> List[ReplacementCandidate]:
        """
        Find replacement candidates for an asset using multiple strategies.
        
        Args:
            original_symbol: Symbol to find replacements for
            max_candidates: Maximum number of candidates to return
            strategies: Specific strategies to use (defaults to all)
            
        Returns:
            List of ReplacementCandidate objects sorted by confidence score
        """
        try:
            logger.info(f"🔍 Finding replacement candidates for {original_symbol}")
            
            # Default to all strategies if none specified
            if strategies is None:
                strategies = list(ReplacementStrategy)
            
            # Get asset metadata for the original symbol
            original_metadata = await self._get_asset_metadata(original_symbol)
            if not original_metadata:
                logger.warning(f"Could not get metadata for {original_symbol}")
                return []
            
            all_candidates = []
            
            # Apply each requested strategy
            for strategy in strategies:
                try:
                    strategy_candidates = await self._apply_replacement_strategy(
                        original_symbol, original_metadata, strategy
                    )
                    all_candidates.extend(strategy_candidates)
                    logger.debug(f"Strategy {strategy.value}: {len(strategy_candidates)} candidates found")
                except Exception as e:
                    logger.error(f"Error in strategy {strategy.value}: {e}")
                    continue
            
            # Remove duplicates and filter by quality
            unique_candidates = self._deduplicate_candidates(all_candidates)
            quality_filtered = [c for c in unique_candidates if c.confidence_score >= 0.5]
            
            # Sort by confidence score and return top candidates
            quality_filtered.sort(key=lambda x: x.confidence_score, reverse=True)
            top_candidates = quality_filtered[:max_candidates]
            
            # Enhance candidates with real-time data
            enhanced_candidates = await self._enhance_candidates_with_realtime_data(top_candidates)
            
            logger.info(f"✅ Found {len(enhanced_candidates)} replacement candidates for {original_symbol}")
            for i, candidate in enumerate(enhanced_candidates[:5], 1):
                logger.info(f"   {i}. {candidate.symbol}: {candidate.replacement_strategy.value} "
                          f"(correlation: {candidate.correlation:.3f}, confidence: {candidate.confidence_score:.3f})")
            
            return enhanced_candidates
            
        except Exception as e:
            logger.error(f"Error finding replacement candidates for {original_symbol}: {e}")
            return []
    
    async def analyze_replacement_impact(self, 
                                       original_symbol: str,
                                       replacement_symbol: str,
                                       position_size: Decimal) -> Dict[str, Union[float, str, Decimal]]:
        """
        Analyze the impact of replacing one asset with another.
        
        Args:
            original_symbol: Original asset symbol
            replacement_symbol: Replacement asset symbol
            position_size: Size of position in dollars
            
        Returns:
            Dict with impact analysis
        """
        try:
            logger.info(f"📊 Analyzing replacement impact: {original_symbol} → {replacement_symbol}")
            
            # Get correlation between assets
            correlation = await self._calculate_correlation(original_symbol, replacement_symbol)
            
            # Get volatility data
            original_vol = await self._calculate_volatility(original_symbol)
            replacement_vol = await self._calculate_volatility(replacement_symbol)
            
            # Get beta data
            original_beta = await self._get_beta(original_symbol)
            replacement_beta = await self._get_beta(replacement_symbol)
            
            # Calculate tracking error
            tracking_error = await self._calculate_tracking_error(original_symbol, replacement_symbol)
            
            # Get liquidity metrics
            original_liquidity = await self._get_liquidity_score(original_symbol)
            replacement_liquidity = await self._get_liquidity_score(replacement_symbol)
            
            # Calculate costs
            estimated_transaction_costs = await self._estimate_transaction_costs(
                replacement_symbol, position_size
            )
            
            impact_analysis = {
                "correlation": correlation,
                "tracking_error_annualized": tracking_error,
                "volatility_change": replacement_vol - original_vol if original_vol and replacement_vol else None,
                "beta_change": replacement_beta - original_beta if original_beta and replacement_beta else None,
                "liquidity_score_change": replacement_liquidity - original_liquidity,
                "estimated_transaction_costs": estimated_transaction_costs,
                "position_size_analyzed": position_size,
                "risk_profile_match": "high" if abs(replacement_vol - original_vol) < 0.05 else "medium" if abs(replacement_vol - original_vol) < 0.10 else "low",
                "liquidity_assessment": "high" if replacement_liquidity > 0.8 else "medium" if replacement_liquidity > 0.6 else "low",
                "overall_suitability": self._calculate_overall_suitability(correlation, tracking_error, replacement_liquidity)
            }
            
            return impact_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing replacement impact: {e}")
            return {"error": str(e)}
    
    async def generate_replacement_report(self, original_symbol: str) -> Dict[str, Union[str, List, Dict]]:
        """
        Generate comprehensive replacement analysis report for an asset.
        
        Args:
            original_symbol: Symbol to analyze
            
        Returns:
            Comprehensive replacement report
        """
        try:
            logger.info(f"📋 Generating replacement report for {original_symbol}")
            
            # Find all replacement candidates
            candidates = await self.find_replacement_candidates(original_symbol, max_candidates=20)
            
            if not candidates:
                return {
                    "original_symbol": original_symbol,
                    "analysis_date": datetime.now().isoformat(),
                    "candidates_found": 0,
                    "error": "No suitable replacement candidates found"
                }
            
            # Group candidates by strategy
            by_strategy = {}
            for candidate in candidates:
                strategy = candidate.replacement_strategy.value
                if strategy not in by_strategy:
                    by_strategy[strategy] = []
                by_strategy[strategy].append(asdict(candidate))
            
            # Analyze top 5 candidates in detail
            detailed_analysis = []
            for candidate in candidates[:5]:
                impact = await self.analyze_replacement_impact(
                    original_symbol, candidate.symbol, Decimal("10000")  # $10K position analysis
                )
                detailed_analysis.append({
                    "candidate": asdict(candidate),
                    "impact_analysis": impact
                })
            
            # Get original asset metadata
            original_metadata = await self._get_asset_metadata(original_symbol)
            
            # Generate recommendations
            recommendations = self._generate_replacement_recommendations(candidates)
            
            replacement_report = {
                "original_symbol": original_symbol,
                "original_metadata": original_metadata,
                "analysis_date": datetime.now().isoformat(),
                "candidates_found": len(candidates),
                "candidates_by_strategy": by_strategy,
                "top_candidates": [asdict(c) for c in candidates[:10]],
                "detailed_analysis": detailed_analysis,
                "recommendations": recommendations,
                "analysis_parameters": {
                    "correlation_lookback_days": self.correlation_lookback_days,
                    "min_correlation_threshold": self.min_correlation_threshold,
                    "min_liquidity_volume": self.min_liquidity_volume
                }
            }
            
            logger.info(f"✅ Replacement report generated for {original_symbol}: {len(candidates)} candidates")
            return replacement_report
            
        except Exception as e:
            logger.error(f"Error generating replacement report for {original_symbol}: {e}")
            return {"error": str(e)}
    
    async def _apply_replacement_strategy(self, 
                                        original_symbol: str,
                                        original_metadata: Dict,
                                        strategy: ReplacementStrategy) -> List[ReplacementCandidate]:
        """Apply a specific replacement strategy."""
        try:
            if strategy == ReplacementStrategy.HIGH_CORRELATION:
                return await self._find_high_correlation_replacements(original_symbol)
            
            elif strategy == ReplacementStrategy.SECTOR_MATCH:
                return await self._find_sector_match_replacements(original_symbol, original_metadata)
            
            elif strategy == ReplacementStrategy.ETF_SUBSTITUTE:
                return await self._find_etf_substitute_replacements(original_symbol, original_metadata)
            
            elif strategy == ReplacementStrategy.INDEX_TRACKING:
                return await self._find_index_tracking_replacements(original_symbol)
            
            elif strategy == ReplacementStrategy.INVERSE_CORRELATION:
                return await self._find_inverse_correlation_replacements(original_symbol)
            
            else:
                logger.warning(f"Unknown replacement strategy: {strategy}")
                return []
                
        except Exception as e:
            logger.error(f"Error applying strategy {strategy.value}: {e}")
            return []
    
    async def _find_high_correlation_replacements(self, original_symbol: str) -> List[ReplacementCandidate]:
        """Find replacements based on high correlation."""
        candidates = []
        
        try:
            # Get list of potential replacement symbols (broad universe)
            potential_symbols = await self._get_replacement_universe(original_symbol)
            
            # Calculate correlations with original symbol
            correlations = []
            for symbol in potential_symbols[:50]:  # Limit to prevent excessive API calls
                try:
                    correlation = await self._calculate_correlation(original_symbol, symbol)
                    if correlation >= self.min_correlation_threshold:
                        correlations.append((symbol, correlation))
                except Exception as e:
                    logger.debug(f"Correlation calculation failed for {symbol}: {e}")
                    continue
            
            # Sort by correlation and create candidates
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            for symbol, correlation in correlations[:15]:  # Top 15 correlated assets
                try:
                    metadata = await self._get_asset_metadata(symbol)
                    if not metadata:
                        continue
                    
                    # Calculate other scores
                    liquidity_score = await self._get_liquidity_score(symbol)
                    risk_similarity = await self._calculate_risk_similarity(original_symbol, symbol)
                    
                    confidence_score = (correlation * 0.5 + liquidity_score * 0.3 + risk_similarity * 0.2)
                    
                    candidate = ReplacementCandidate(
                        symbol=symbol,
                        name=metadata.get("name", symbol),
                        asset_type=AssetType(metadata.get("asset_type", "stock")),
                        correlation=correlation,
                        sector_match_score=0.0,  # Not applicable for this strategy
                        risk_similarity_score=risk_similarity,
                        liquidity_score=liquidity_score,
                        replacement_strategy=ReplacementStrategy.HIGH_CORRELATION,
                        confidence_score=confidence_score,
                        replacement_reasoning=f"High correlation ({correlation:.3f}) with {original_symbol}"
                    )
                    
                    candidates.append(candidate)
                    
                except Exception as e:
                    logger.debug(f"Error creating candidate for {symbol}: {e}")
                    continue
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error finding high correlation replacements: {e}")
            return []
    
    async def _find_sector_match_replacements(self, original_symbol: str, original_metadata: Dict) -> List[ReplacementCandidate]:
        """Find replacements based on sector/industry matching."""
        candidates = []
        
        try:
            original_sector = original_metadata.get("sector")
            original_industry = original_metadata.get("industry")
            
            if not original_sector:
                return candidates
            
            # Find sector ETFs that match
            for etf_symbol, etf_data in self.sector_etfs.items():
                if etf_data.get("sector") == original_sector:
                    # Calculate correlation to verify match
                    correlation = await self._calculate_correlation(original_symbol, etf_symbol)
                    
                    if correlation >= 0.4:  # Lower threshold for sector ETFs
                        liquidity_score = await self._get_liquidity_score(etf_symbol)
                        
                        sector_match_score = 1.0 if etf_data.get("sector") == original_sector else 0.7
                        if etf_data.get("industry") == original_industry:
                            sector_match_score = 1.0
                        
                        confidence_score = (correlation * 0.3 + sector_match_score * 0.4 + liquidity_score * 0.3)
                        
                        candidate = ReplacementCandidate(
                            symbol=etf_symbol,
                            name=etf_data.get("name", etf_symbol),
                            asset_type=AssetType.SECTOR_ETF,
                            correlation=correlation,
                            sector_match_score=sector_match_score,
                            risk_similarity_score=0.8,  # ETFs generally have lower individual stock risk
                            liquidity_score=liquidity_score,
                            replacement_strategy=ReplacementStrategy.SECTOR_MATCH,
                            confidence_score=confidence_score,
                            replacement_reasoning=f"Sector ETF tracking {original_sector} sector"
                        )
                        
                        candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error finding sector match replacements: {e}")
            return []
    
    async def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two assets."""
        try:
            # Check cache first
            cache_key = f"{min(symbol1, symbol2)}_{max(symbol1, symbol2)}"
            cached_correlation = await self._get_cached_correlation(cache_key)
            
            if cached_correlation is not None:
                return cached_correlation
            
            # Fetch price data for both symbols
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.correlation_lookback_days + 30)  # Extra buffer
            
            try:
                # Use yfinance for correlation calculation
                data1 = yf.download(symbol1, start=start_date, end=end_date, progress=False)
                data2 = yf.download(symbol2, start=start_date, end=end_date, progress=False)
                
                if data1.empty or data2.empty:
                    return 0.0
                
                # Get close prices and align dates
                prices1 = data1['Close'].dropna()
                prices2 = data2['Close'].dropna()
                
                # Align the data by date
                aligned_data = pd.concat([prices1, prices2], axis=1, join='inner').dropna()
                
                if len(aligned_data) < 50:  # Need at least 50 data points
                    return 0.0
                
                # Calculate correlation
                correlation, _ = pearsonr(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
                
                if np.isnan(correlation):
                    correlation = 0.0
                
                # Cache the result
                await self._cache_correlation(cache_key, correlation, len(aligned_data))
                
                return float(correlation)
                
            except Exception as e:
                logger.debug(f"Error calculating correlation between {symbol1} and {symbol2}: {e}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in correlation calculation: {e}")
            return 0.0
    
    def _initialize_sector_etfs(self) -> Dict[str, Dict]:
        """Initialize database of sector ETFs."""
        return {
            "XLK": {"name": "Technology Select Sector SPDR Fund", "sector": "Technology", "expense_ratio": 0.0012},
            "XLF": {"name": "Financial Select Sector SPDR Fund", "sector": "Financial Services", "expense_ratio": 0.0012},
            "XLE": {"name": "Energy Select Sector SPDR Fund", "sector": "Energy", "expense_ratio": 0.0012},
            "XLV": {"name": "Health Care Select Sector SPDR Fund", "sector": "Healthcare", "expense_ratio": 0.0012},
            "XLI": {"name": "Industrial Select Sector SPDR Fund", "sector": "Industrials", "expense_ratio": 0.0012},
            "XLP": {"name": "Consumer Staples Select Sector SPDR Fund", "sector": "Consumer Defensive", "expense_ratio": 0.0012},
            "XLY": {"name": "Consumer Discretionary Select Sector SPDR Fund", "sector": "Consumer Cyclical", "expense_ratio": 0.0012},
            "XLU": {"name": "Utilities Select Sector SPDR Fund", "sector": "Utilities", "expense_ratio": 0.0012},
            "XLB": {"name": "Materials Select Sector SPDR Fund", "sector": "Basic Materials", "expense_ratio": 0.0012},
            "XLRE": {"name": "Real Estate Select Sector SPDR Fund", "sector": "Real Estate", "expense_ratio": 0.0012},
            "VGT": {"name": "Vanguard Information Technology ETF", "sector": "Technology", "expense_ratio": 0.0010},
            "VFH": {"name": "Vanguard Financials ETF", "sector": "Financial Services", "expense_ratio": 0.0010},
            "VDE": {"name": "Vanguard Energy ETF", "sector": "Energy", "expense_ratio": 0.0010},
            "VHT": {"name": "Vanguard Health Care ETF", "sector": "Healthcare", "expense_ratio": 0.0010},
            "VIS": {"name": "Vanguard Industrials ETF", "sector": "Industrials", "expense_ratio": 0.0010},
        }
    
    def _initialize_index_etfs(self) -> Dict[str, Dict]:
        """Initialize database of broad market index ETFs."""
        return {
            "SPY": {"name": "SPDR S&P 500 ETF Trust", "index": "S&P 500", "expense_ratio": 0.0945},
            "IVV": {"name": "iShares Core S&P 500 ETF", "index": "S&P 500", "expense_ratio": 0.0003},
            "VOO": {"name": "Vanguard S&P 500 ETF", "index": "S&P 500", "expense_ratio": 0.0003},
            "QQQ": {"name": "Invesco QQQ Trust", "index": "NASDAQ-100", "expense_ratio": 0.0020},
            "IWM": {"name": "iShares Russell 2000 ETF", "index": "Russell 2000", "expense_ratio": 0.0019},
            "VTI": {"name": "Vanguard Total Stock Market ETF", "index": "Total Stock Market", "expense_ratio": 0.0003},
            "ITOT": {"name": "iShares Core S&P Total US Stock Market ETF", "index": "S&P TMI", "expense_ratio": 0.0003},
        }
    
    def _initialize_broad_market_etfs(self) -> Dict[str, Dict]:
        """Initialize database of broad market ETFs for general replacements."""
        return {**self._initialize_index_etfs(), **self._initialize_sector_etfs()}

    # Additional helper methods would be implemented here...
    # (truncated for space, but would include all necessary implementation details)

# Global instance
asset_replacement_engine = AssetReplacementEngine()

# Convenience functions
async def find_replacement_candidates(original_symbol: str, **kwargs) -> List[ReplacementCandidate]:
    """Find replacement candidates for an asset."""
    return await asset_replacement_engine.find_replacement_candidates(original_symbol, **kwargs)

async def analyze_replacement_impact(original_symbol: str, replacement_symbol: str, 
                                   position_size: Decimal) -> Dict:
    """Analyze impact of replacing one asset with another."""
    return await asset_replacement_engine.analyze_replacement_impact(original_symbol, replacement_symbol, position_size)

__all__ = [
    'AssetReplacementEngine', 'ReplacementCandidate', 'ReplacementAnalysis',
    'ReplacementStrategy', 'AssetType', 'asset_replacement_engine',
    'find_replacement_candidates', 'analyze_replacement_impact'
]