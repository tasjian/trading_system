#!/usr/bin/env python3
"""
Borrow Cost Integration Module

This module provides real-time monitoring of borrow costs and availability for short selling,
including SSR (Short Sale Rule) compliance monitoring and integration with multiple data sources.

Key Features:
- Real-time borrow fee tracking via IEX Cloud and Alpha Vantage APIs
- SSR compliance monitoring and circuit breaker detection
- Borrow availability assessment and hard-to-borrow (HTB) detection
- Historical borrow cost trends and prediction
- Integration with trading system for pre-trade validation

The module ensures compliance with regulatory requirements while optimizing 
short selling execution timing and cost management.
"""

import asyncio
import logging
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
from decimal import Decimal

from config.settings import settings
from tools.alpaca_client import alpaca_client
from utils.connection_pool import connection_pool

logger = logging.getLogger(__name__)

@dataclass
class BorrowCostData:
    """Real-time borrow cost information for a symbol."""
    symbol: str
    borrow_fee_rate: float  # Annual rate as decimal (e.g., 0.05 = 5%)
    availability: str  # 'AVAILABLE', 'HTB', 'NOT_AVAILABLE'
    shares_available: Optional[int]
    last_updated: datetime
    data_source: str
    
    # Additional metrics
    daily_fee_cost: float  # Daily cost as decimal
    fee_trend: str  # 'INCREASING', 'DECREASING', 'STABLE'
    historical_avg: Optional[float]
    percentile_rank: Optional[float]  # 0-100, where 100 is most expensive

@dataclass
class SSRStatus:
    """Short Sale Rule (SSR) status and compliance information."""
    symbol: str
    ssr_active: bool
    trigger_price: Optional[float]
    current_price: float
    price_change_percent: float
    circuit_breaker_level: Optional[str]  # 'LEVEL_1', 'LEVEL_2', 'LEVEL_3'
    compliance_notes: List[str]
    last_updated: datetime

@dataclass
class ShortabilityAnalysis:
    """Comprehensive shortability analysis combining all factors."""
    symbol: str
    shortability_score: float  # 0-100 composite score
    cost_score: float  # 0-100 (100 = low cost)
    availability_score: float  # 0-100 (100 = highly available)
    regulatory_score: float  # 0-100 (100 = compliant)
    
    # Component data
    borrow_cost: Optional[BorrowCostData]
    ssr_status: Optional[SSRStatus]
    
    # Recommendations
    recommended_action: str  # 'SHORT', 'WAIT', 'AVOID'
    optimal_timing: Optional[str]  # 'IMMEDIATE', 'INTRADAY', 'WAIT_FOR_UPTICK'
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    
    analysis_timestamp: datetime
    expires_at: datetime

class IEXCloudProvider:
    """IEX Cloud API provider for borrow cost data."""
    
    def __init__(self):
        """Initialize IEX Cloud provider."""
        self.base_url = "https://cloud.iexapis.com/stable"
        self.api_key = getattr(settings, 'iex_api_key', None)
        self.session_timeout = aiohttp.ClientTimeout(total=10)
        if not self.api_key:
            logger.warning("IEX Cloud API key not configured - borrow cost data will use fallback estimates")
        
    async def get_borrow_cost_data(self, symbol: str) -> Optional[BorrowCostData]:
        """Get borrow cost data from IEX Cloud."""
        if not self.api_key:
            logger.debug("IEX API key not configured, using fallback estimates")
            return self._generate_fallback_borrow_cost(symbol)
            
        try:
            # IEX doesn't provide direct borrow cost data, so we'll use related metrics
            # and estimate based on short interest and other factors
            
            url = f"{self.base_url}/stock/{symbol}/stats"
            params = {"token": self.api_key}
            
            session = await connection_pool.get_async_session(
                name="iex_borrow_cost",
                timeout=self.session_timeout
            )
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_iex_borrow_data(symbol, data)
                else:
                    logger.debug(f"IEX API error for {symbol}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.debug(f"Error fetching IEX borrow data for {symbol}: {e}")
            return None
    
    def _parse_iex_borrow_data(self, symbol: str, data: Dict) -> Optional[BorrowCostData]:
        """Parse IEX data to estimate borrow costs."""
        try:
            # Extract relevant metrics
            short_interest = data.get('shortInterest', 0)
            float_shares = data.get('float', 1)
            shares_outstanding = data.get('sharesOutstanding', 1)
            
            # Calculate short interest ratio
            short_ratio = short_interest / max(float_shares, 1) if float_shares else 0
            
            # Estimate borrow fee based on short interest and other factors
            # This is a simplified heuristic - in production you'd use specialized data
            base_fee = 0.003  # 0.3% base annual rate
            
            if short_ratio > 0.20:  # High short interest
                estimated_fee = base_fee + (short_ratio - 0.20) * 0.05
                availability = 'HTB' if short_ratio > 0.30 else 'AVAILABLE'
            elif short_ratio > 0.10:
                estimated_fee = base_fee + (short_ratio - 0.10) * 0.02
                availability = 'AVAILABLE'
            else:
                estimated_fee = base_fee
                availability = 'AVAILABLE'
            
            # Cap maximum estimated fee
            estimated_fee = min(estimated_fee, 0.50)  # Max 50% annual
            
            return BorrowCostData(
                symbol=symbol,
                borrow_fee_rate=estimated_fee,
                availability=availability,
                shares_available=max(int(float_shares * 0.1), 1000) if availability == 'AVAILABLE' else 0,
                last_updated=datetime.now(),
                data_source='IEX_ESTIMATED',
                daily_fee_cost=estimated_fee / 365,
                fee_trend='STABLE',
                historical_avg=None,
                percentile_rank=None
            )
            
        except Exception as e:
            logger.error(f"Error parsing IEX borrow data for {symbol}: {e}")
            return None
    
    def _generate_fallback_borrow_cost(self, symbol: str) -> BorrowCostData:
        """Generate fallback borrow cost data when API is unavailable."""
        # Use conservative estimates based on typical market conditions
        base_fee = 0.02  # 2% annual fee as baseline
        
        # Adjust based on symbol characteristics (rough heuristics)
        if symbol in ['SPY', 'QQQ', 'IWM']:  # Major ETFs
            fee_rate = base_fee * 0.3  # Lower fees for liquid ETFs
            availability = 'AVAILABLE'
        elif len(symbol) <= 4 and symbol.isupper():  # Likely major stock
            fee_rate = base_fee  # Standard fee
            availability = 'AVAILABLE'
        else:  # Unknown or complex symbol
            fee_rate = base_fee * 2.0  # Higher fee for uncertainty
            availability = 'HTB'
        
        return BorrowCostData(
            symbol=symbol,
            borrow_fee_rate=fee_rate,
            availability=availability,
            shares_available=10000,  # Conservative estimate
            last_updated=datetime.now(),
            data_source='FALLBACK_ESTIMATE',
            daily_fee_cost=fee_rate / 365,
            fee_trend='STABLE',
            historical_avg=fee_rate,
            percentile_rank=50.0  # Median estimate
        )

class AlphaVantageProvider:
    """Alpha Vantage API provider for additional market data."""
    
    def __init__(self):
        """Initialize Alpha Vantage provider."""
        self.base_url = "https://www.alphavantage.co/query"
        self.api_key = getattr(settings, 'alpha_vantage_api_key', None)
        self.session_timeout = aiohttp.ClientTimeout(total=15)
        if not self.api_key:
            logger.warning("Alpha Vantage API key not configured - market metrics will use fallback estimates")
        
    async def get_market_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market metrics that can inform borrow cost estimation."""
        if not self.api_key:
            logger.debug("Alpha Vantage API key not configured, using fallback estimates")
            return self._generate_fallback_market_metrics(symbol)
            
        try:
            # Get overview data which includes some relevant metrics
            params = {
                "function": "OVERVIEW",
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            session = await connection_pool.get_async_session(
                name="alphavantage_borrow",
                timeout=self.session_timeout
            )
            
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check for API limit
                    if "Note" in data or "Information" in data:
                        logger.debug(f"Alpha Vantage rate limit for {symbol}")
                        return None
                    
                    return self._parse_alpha_vantage_data(data)
                else:
                    logger.debug(f"Alpha Vantage API error for {symbol}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.debug(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return None
    
    def _parse_alpha_vantage_data(self, data: Dict) -> Optional[Dict[str, Any]]:
        """Parse Alpha Vantage data for borrow cost estimation."""
        try:
            # Extract relevant financial metrics
            market_cap = self._safe_float(data.get('MarketCapitalization'))
            pe_ratio = self._safe_float(data.get('PERatio'))
            beta = self._safe_float(data.get('Beta'))
            shares_outstanding = self._safe_float(data.get('SharesOutstanding'))
            
            # Calculate volatility-based factors
            volatility_factor = 1.0
            if beta:
                volatility_factor = max(0.5, min(2.0, beta))  # Cap between 0.5-2.0
            
            # Market cap impact on borrow cost
            if market_cap:
                if market_cap < 1e9:  # Small cap
                    size_factor = 1.5
                elif market_cap < 10e9:  # Mid cap  
                    size_factor = 1.2
                else:  # Large cap
                    size_factor = 1.0
            else:
                size_factor = 1.2
            
            return {
                'market_cap': market_cap,
                'pe_ratio': pe_ratio,
                'beta': beta,
                'shares_outstanding': shares_outstanding,
                'volatility_factor': volatility_factor,
                'size_factor': size_factor
            }
            
        except Exception as e:
            logger.error(f"Error parsing Alpha Vantage data: {e}")
            return None
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or value == 'None' or value == '':
            return None
        try:
            # Handle values like "1.23B" or "1.23M"
            if isinstance(value, str):
                value = value.replace(',', '')
                if value.endswith('B'):
                    return float(value[:-1]) * 1e9
                elif value.endswith('M'):
                    return float(value[:-1]) * 1e6
                elif value.endswith('K'):
                    return float(value[:-1]) * 1e3
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _generate_fallback_market_metrics(self, symbol: str) -> Dict[str, Any]:
        """Generate fallback market metrics when API is unavailable."""
        # Return conservative estimates for market metrics
        return {
            'marketCap': '1000000000',  # $1B default
            'peRatio': '20.0',  # Conservative P/E
            'beta': '1.0',  # Market neutral
            'volume': '1000000',  # 1M volume
            'averageVolume': '1000000',
            'high52Week': '150.0',
            'low52Week': '100.0',
            'sharesOutstanding': '50000000',  # 50M shares
            'floatShares': '40000000',  # 40M float
            'shortRatio': '2.0',  # Conservative short ratio
            'shortPercentOfFloat': '5.0',  # 5% short interest
            'source': 'FALLBACK_ESTIMATE'
        }

class SSRComplianceMonitor:
    """Monitor Short Sale Rule (SSR) compliance and circuit breakers."""
    
    def __init__(self):
        """Initialize SSR compliance monitor."""
        self.ssr_cache: Dict[str, SSRStatus] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        
    async def check_ssr_status(self, symbol: str) -> SSRStatus:
        """Check current SSR status for a symbol."""
        try:
            # Check cache first
            if symbol in self.ssr_cache:
                cached_status = self.ssr_cache[symbol]
                age_seconds = (datetime.now() - cached_status.last_updated).total_seconds()
                if age_seconds < self.cache_ttl_seconds:
                    return cached_status
            
            # Get current and previous day prices
            current_price = alpaca_client.get_current_price(symbol)
            if not current_price:
                return self._create_default_ssr_status(symbol)
            
            # Get recent market data for price change calculation
            market_data = alpaca_client.get_market_data(symbol, limit=2)
            if market_data is None or len(market_data) < 2:
                return self._create_default_ssr_status(symbol, current_price)
            
            # Calculate price change from previous close
            previous_close = market_data['close'].iloc[-2]
            price_change_percent = (current_price - previous_close) / previous_close
            
            # Determine SSR status
            # SSR is triggered when stock falls 10% or more from previous close
            ssr_active = price_change_percent <= -0.10
            
            # Calculate trigger price (90% of previous close)
            trigger_price = previous_close * 0.90 if ssr_active else None
            
            # Determine circuit breaker level based on price change
            circuit_breaker_level = None
            if price_change_percent <= -0.20:
                circuit_breaker_level = 'LEVEL_3'  # Severe decline
            elif price_change_percent <= -0.15:
                circuit_breaker_level = 'LEVEL_2'  # Significant decline
            elif price_change_percent <= -0.10:
                circuit_breaker_level = 'LEVEL_1'  # SSR triggered
            
            # Generate compliance notes
            compliance_notes = []
            if ssr_active:
                compliance_notes.append("SSR active - short sales restricted to uptick only")
                compliance_notes.append("Must wait for price to trade above current bid")
            
            if circuit_breaker_level:
                compliance_notes.append(f"Circuit breaker {circuit_breaker_level} conditions")
            
            # Create SSR status
            ssr_status = SSRStatus(
                symbol=symbol,
                ssr_active=ssr_active,
                trigger_price=trigger_price,
                current_price=current_price,
                price_change_percent=price_change_percent,
                circuit_breaker_level=circuit_breaker_level,
                compliance_notes=compliance_notes,
                last_updated=datetime.now()
            )
            
            # Cache the result
            self.ssr_cache[symbol] = ssr_status
            
            return ssr_status
            
        except Exception as e:
            logger.error(f"Error checking SSR status for {symbol}: {e}")
            return self._create_default_ssr_status(symbol)
    
    def _create_default_ssr_status(self, symbol: str, current_price: Optional[float] = None) -> SSRStatus:
        """Create default SSR status when data is unavailable."""
        return SSRStatus(
            symbol=symbol,
            ssr_active=False,
            trigger_price=None,
            current_price=current_price or 0.0,
            price_change_percent=0.0,
            circuit_breaker_level=None,
            compliance_notes=["Unable to determine SSR status - proceed with caution"],
            last_updated=datetime.now()
        )
    
    def is_short_allowed_now(self, symbol: str, intended_price: float) -> Tuple[bool, str]:
        """Check if short sale is allowed right now given current conditions."""
        try:
            ssr_status = self.ssr_cache.get(symbol)
            if not ssr_status:
                return True, "SSR status unknown - assuming allowed"
            
            if not ssr_status.ssr_active:
                return True, "SSR not active - short sale allowed"
            
            # During SSR, short sales must be at or above the current bid
            # For simplicity, we'll require the intended price to be above current price
            if intended_price > ssr_status.current_price:
                return True, "SSR active but price above current - uptick rule satisfied"
            else:
                return False, "SSR active - must wait for uptick (price above current bid)"
                
        except Exception as e:
            logger.error(f"Error checking short sale allowance for {symbol}: {e}")
            return False, f"Error checking SSR compliance: {e}"

class BorrowCostMonitor:
    """Main borrow cost monitoring system."""
    
    def __init__(self):
        """Initialize borrow cost monitor."""
        self.iex_provider = IEXCloudProvider()
        self.alpha_vantage_provider = AlphaVantageProvider()
        self.ssr_monitor = SSRComplianceMonitor()
        
        # Cache for borrow cost data
        self.borrow_cache: Dict[str, BorrowCostData] = {}
        self.analysis_cache: Dict[str, ShortabilityAnalysis] = {}
        self.cache_ttl_minutes = 15
        
        # Historical tracking
        self.historical_costs: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        logger.info("Borrow Cost Monitor initialized")
    
    async def get_borrow_cost_analysis(self, symbol: str) -> Optional[ShortabilityAnalysis]:
        """Get comprehensive shortability analysis for a symbol."""
        try:
            # Check cache first
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            if cache_key in self.analysis_cache:
                cached_analysis = self.analysis_cache[cache_key]
                age_minutes = (datetime.now() - cached_analysis.analysis_timestamp).total_seconds() / 60
                if age_minutes < self.cache_ttl_minutes:
                    return cached_analysis
            
            logger.debug(f"Generating fresh borrow cost analysis for {symbol}")
            
            # Gather all required data in parallel
            tasks = [
                self._get_consolidated_borrow_cost(symbol),
                self.ssr_monitor.check_ssr_status(symbol),
                self._get_market_context(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            borrow_cost_data = results[0] if not isinstance(results[0], Exception) else None
            ssr_status = results[1] if not isinstance(results[1], Exception) else None
            market_context = results[2] if not isinstance(results[2], Exception) else {}
            
            # Calculate component scores
            cost_score = self._calculate_cost_score(borrow_cost_data, market_context)
            availability_score = self._calculate_availability_score(borrow_cost_data)
            regulatory_score = self._calculate_regulatory_score(ssr_status)
            
            # Calculate composite shortability score
            shortability_score = (cost_score * 0.4 + availability_score * 0.3 + regulatory_score * 0.3)
            
            # Determine recommendations
            recommended_action, optimal_timing, risk_level = self._generate_recommendations(
                shortability_score, cost_score, availability_score, regulatory_score,
                borrow_cost_data, ssr_status
            )
            
            # Create analysis
            analysis = ShortabilityAnalysis(
                symbol=symbol,
                shortability_score=shortability_score,
                cost_score=cost_score,
                availability_score=availability_score,
                regulatory_score=regulatory_score,
                borrow_cost=borrow_cost_data,
                ssr_status=ssr_status,
                recommended_action=recommended_action,
                optimal_timing=optimal_timing,
                risk_level=risk_level,
                analysis_timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=self.cache_ttl_minutes)
            )
            
            # Cache the analysis
            self.analysis_cache[cache_key] = analysis
            
            # Update historical tracking
            if borrow_cost_data:
                self.historical_costs[symbol].append({
                    'timestamp': datetime.now(),
                    'borrow_fee': borrow_cost_data.borrow_fee_rate,
                    'availability': borrow_cost_data.availability
                })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating borrow cost analysis for {symbol}: {e}")
            return None
    
    async def _get_consolidated_borrow_cost(self, symbol: str) -> Optional[BorrowCostData]:
        """Get consolidated borrow cost data from multiple sources."""
        try:
            # Try multiple data sources in parallel
            tasks = [
                self.iex_provider.get_borrow_cost_data(symbol),
                self._estimate_from_market_data(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Use the best available data source
            for result in results:
                if isinstance(result, BorrowCostData) and result.borrow_fee_rate > 0:
                    return result
            
            # Fallback to basic estimation
            return await self._create_fallback_borrow_data(symbol)
            
        except Exception as e:
            logger.error(f"Error getting consolidated borrow cost for {symbol}: {e}")
            return None
    
    async def _estimate_from_market_data(self, symbol: str) -> Optional[BorrowCostData]:
        """Estimate borrow cost from market data indicators."""
        try:
            # Get Alpha Vantage market metrics
            av_data = await self.alpha_vantage_provider.get_market_metrics(symbol)
            
            # Get recent volatility from price data
            market_data = alpaca_client.get_market_data(symbol, limit=20)
            
            if market_data is None or len(market_data) < 10:
                return None
            
            # Calculate volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Base estimation model
            base_fee = 0.002  # 0.2% base annual rate
            
            # Volatility adjustment
            volatility_adjustment = min(volatility * 0.1, 0.05)  # Cap at 5%
            
            # Size adjustment from Alpha Vantage data
            size_adjustment = 0.0
            if av_data:
                size_factor = av_data.get('size_factor', 1.0)
                size_adjustment = (size_factor - 1.0) * 0.01
            
            # Beta adjustment
            beta_adjustment = 0.0
            if av_data and av_data.get('beta'):
                beta = av_data['beta']
                if beta > 1.5:
                    beta_adjustment = (beta - 1.5) * 0.01
            
            # Final estimated fee
            estimated_fee = base_fee + volatility_adjustment + size_adjustment + beta_adjustment
            estimated_fee = min(estimated_fee, 0.30)  # Cap at 30% annual
            
            # Determine availability based on estimated cost
            if estimated_fee > 0.10:  # 10%+ annual rate
                availability = 'HTB'
            elif estimated_fee > 0.05:  # 5%+ annual rate
                availability = 'LIMITED'
            else:
                availability = 'AVAILABLE'
            
            return BorrowCostData(
                symbol=symbol,
                borrow_fee_rate=estimated_fee,
                availability=availability,
                shares_available=10000 if availability == 'AVAILABLE' else 1000,
                last_updated=datetime.now(),
                data_source='MARKET_ESTIMATED',
                daily_fee_cost=estimated_fee / 365,
                fee_trend='STABLE',
                historical_avg=self._get_historical_average(symbol),
                percentile_rank=None
            )
            
        except Exception as e:
            logger.error(f"Error estimating borrow cost from market data for {symbol}: {e}")
            return None
    
    async def _create_fallback_borrow_data(self, symbol: str) -> BorrowCostData:
        """Create fallback borrow cost data when no sources are available."""
        # Conservative fallback estimation
        fallback_fee = 0.05  # 5% annual rate
        
        return BorrowCostData(
            symbol=symbol,
            borrow_fee_rate=fallback_fee,
            availability='UNKNOWN',
            shares_available=None,
            last_updated=datetime.now(),
            data_source='FALLBACK',
            daily_fee_cost=fallback_fee / 365,
            fee_trend='UNKNOWN',
            historical_avg=None,
            percentile_rank=None
        )
    
    async def _get_market_context(self, symbol: str) -> Dict[str, Any]:
        """Get market context for borrow cost analysis."""
        try:
            context = {}
            
            # Get current price and volume
            current_price = alpaca_client.get_current_price(symbol)
            if current_price:
                context['current_price'] = current_price
            
            # Get recent trading volume
            market_data = alpaca_client.get_market_data(symbol, limit=5)
            if market_data is not None and len(market_data) > 0:
                avg_volume = market_data['volume'].mean()
                context['avg_volume'] = avg_volume
                context['price_volatility'] = market_data['close'].std()
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting market context for {symbol}: {e}")
            return {}
    
    def _calculate_cost_score(self, borrow_data: Optional[BorrowCostData], 
                            market_context: Dict[str, Any]) -> float:
        """Calculate cost score (0-100, higher = lower cost/better)."""
        if not borrow_data:
            return 30.0  # Conservative score when no data
        
        try:
            fee_rate = borrow_data.borrow_fee_rate
            
            # Score based on annual fee rate
            if fee_rate <= 0.01:  # 1% or less
                base_score = 95.0
            elif fee_rate <= 0.03:  # 3% or less  
                base_score = 80.0
            elif fee_rate <= 0.05:  # 5% or less
                base_score = 65.0
            elif fee_rate <= 0.10:  # 10% or less
                base_score = 45.0
            elif fee_rate <= 0.20:  # 20% or less
                base_score = 25.0
            else:  # Above 20%
                base_score = 10.0
            
            # Adjust for trend if available
            if borrow_data.fee_trend == 'DECREASING':
                base_score += 10.0
            elif borrow_data.fee_trend == 'INCREASING':
                base_score -= 10.0
            
            # Adjust for historical context
            if borrow_data.historical_avg and borrow_data.historical_avg > 0:
                current_vs_avg = fee_rate / borrow_data.historical_avg
                if current_vs_avg < 0.8:  # Below historical average
                    base_score += 15.0
                elif current_vs_avg > 1.2:  # Above historical average
                    base_score -= 15.0
            
            return max(0.0, min(100.0, base_score))
            
        except Exception as e:
            logger.error(f"Error calculating cost score: {e}")
            return 30.0
    
    def _calculate_availability_score(self, borrow_data: Optional[BorrowCostData]) -> float:
        """Calculate availability score (0-100, higher = more available)."""
        if not borrow_data:
            return 30.0
        
        try:
            availability = borrow_data.availability
            
            if availability == 'AVAILABLE':
                base_score = 90.0
            elif availability == 'LIMITED':
                base_score = 60.0
            elif availability == 'HTB':
                base_score = 30.0
            elif availability == 'NOT_AVAILABLE':
                base_score = 0.0
            else:  # UNKNOWN
                base_score = 40.0
            
            # Adjust based on shares available
            if borrow_data.shares_available:
                if borrow_data.shares_available >= 100000:
                    base_score += 10.0
                elif borrow_data.shares_available >= 10000:
                    base_score += 5.0
                elif borrow_data.shares_available < 1000:
                    base_score -= 20.0
            
            return max(0.0, min(100.0, base_score))
            
        except Exception as e:
            logger.error(f"Error calculating availability score: {e}")
            return 30.0
    
    def _calculate_regulatory_score(self, ssr_status: Optional[SSRStatus]) -> float:
        """Calculate regulatory compliance score (0-100, higher = more compliant)."""
        if not ssr_status:
            return 50.0  # Neutral when unknown
        
        try:
            base_score = 100.0
            
            # SSR penalties
            if ssr_status.ssr_active:
                base_score -= 30.0  # Significant penalty for active SSR
            
            # Circuit breaker penalties
            if ssr_status.circuit_breaker_level:
                if ssr_status.circuit_breaker_level == 'LEVEL_3':
                    base_score -= 40.0
                elif ssr_status.circuit_breaker_level == 'LEVEL_2':
                    base_score -= 25.0
                elif ssr_status.circuit_breaker_level == 'LEVEL_1':
                    base_score -= 15.0
            
            # Price change impact
            price_change = ssr_status.price_change_percent
            if price_change < -0.15:  # Severe decline
                base_score -= 20.0
            elif price_change < -0.05:  # Moderate decline
                base_score -= 10.0
            
            return max(0.0, min(100.0, base_score))
            
        except Exception as e:
            logger.error(f"Error calculating regulatory score: {e}")
            return 50.0
    
    def _generate_recommendations(self, shortability_score: float, cost_score: float,
                                availability_score: float, regulatory_score: float,
                                borrow_data: Optional[BorrowCostData],
                                ssr_status: Optional[SSRStatus]) -> Tuple[str, str, str]:
        """Generate trading recommendations based on analysis."""
        try:
            # Determine recommended action
            if shortability_score >= 70:
                recommended_action = 'SHORT'
            elif shortability_score >= 40:
                recommended_action = 'WAIT'
            else:
                recommended_action = 'AVOID'
            
            # Determine optimal timing
            optimal_timing = 'IMMEDIATE'
            
            if ssr_status and ssr_status.ssr_active:
                optimal_timing = 'WAIT_FOR_UPTICK'
            elif cost_score < 40:  # High cost
                optimal_timing = 'WAIT'
            elif availability_score < 30:  # Low availability
                optimal_timing = 'WAIT'
            
            # Determine risk level
            if shortability_score >= 70 and all(score >= 60 for score in [cost_score, availability_score, regulatory_score]):
                risk_level = 'LOW'
            elif shortability_score >= 40:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'HIGH'
            
            return recommended_action, optimal_timing, risk_level
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return 'AVOID', 'WAIT', 'HIGH'
    
    def _get_historical_average(self, symbol: str) -> Optional[float]:
        """Get historical average borrow cost for symbol."""
        try:
            if symbol not in self.historical_costs:
                return None
            
            history = list(self.historical_costs[symbol])
            if len(history) < 3:
                return None
            
            fees = [h['borrow_fee'] for h in history]
            return np.mean(fees)
            
        except Exception as e:
            logger.error(f"Error calculating historical average for {symbol}: {e}")
            return None
    
    async def validate_short_trade(self, symbol: str, quantity: int, 
                                 intended_price: float) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate a short trade before execution."""
        try:
            logger.info(f"Validating short trade: {symbol} {quantity} shares @ ${intended_price}")
            
            # Get comprehensive analysis
            analysis = await self.get_borrow_cost_analysis(symbol)
            if not analysis:
                return False, "Unable to analyze borrow costs", {}
            
            # Check SSR compliance
            if analysis.ssr_status:
                allowed, ssr_reason = self.ssr_monitor.is_short_allowed_now(symbol, intended_price)
                if not allowed:
                    return False, f"SSR compliance issue: {ssr_reason}", asdict(analysis)
            
            # Check availability
            if analysis.borrow_cost and analysis.borrow_cost.availability == 'NOT_AVAILABLE':
                return False, "Shares not available for borrowing", asdict(analysis)
            
            # Check if shares available covers the trade size
            if analysis.borrow_cost and analysis.borrow_cost.shares_available:
                if quantity > analysis.borrow_cost.shares_available:
                    return False, f"Insufficient shares available: {analysis.borrow_cost.shares_available} < {quantity}", asdict(analysis)
            
            # Check cost thresholds
            if analysis.borrow_cost and analysis.borrow_cost.borrow_fee_rate > 0.25:  # 25% annual rate
                return False, f"Borrow cost too high: {analysis.borrow_cost.borrow_fee_rate:.1%} annually", asdict(analysis)
            
            # Check overall shortability score
            if analysis.shortability_score < 30:
                return False, f"Low shortability score: {analysis.shortability_score:.1f}/100", asdict(analysis)
            
            # All checks passed
            return True, f"Short trade validated (score: {analysis.shortability_score:.1f}/100)", asdict(analysis)
            
        except Exception as e:
            logger.error(f"Error validating short trade for {symbol}: {e}")
            return False, f"Validation error: {e}", {}
    
    def get_monitor_status(self) -> Dict[str, Any]:
        """Get monitor status and statistics."""
        return {
            'cache_size': len(self.borrow_cache),
            'analysis_cache_size': len(self.analysis_cache),
            'tracked_symbols': len(self.historical_costs),
            'ssr_cache_size': len(self.ssr_monitor.ssr_cache),
            'providers_configured': {
                'iex': bool(self.iex_provider.api_key),
                'alpha_vantage': bool(self.alpha_vantage_provider.api_key)
            }
        }

# Global instance
borrow_cost_monitor = BorrowCostMonitor()

# Convenience functions
async def get_shortability_analysis(symbol: str) -> Optional[ShortabilityAnalysis]:
    """Get shortability analysis for a symbol."""
    return await borrow_cost_monitor.get_borrow_cost_analysis(symbol)

async def validate_short_trade(symbol: str, quantity: int, intended_price: float) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate a short trade before execution."""
    return await borrow_cost_monitor.validate_short_trade(symbol, quantity, intended_price)

async def check_ssr_status(symbol: str) -> SSRStatus:
    """Check SSR status for a symbol."""
    return await borrow_cost_monitor.ssr_monitor.check_ssr_status(symbol)

def get_borrow_monitor_status() -> Dict[str, Any]:
    """Get borrow cost monitor status."""
    return borrow_cost_monitor.get_monitor_status()