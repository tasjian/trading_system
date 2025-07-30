#!/usr/bin/env python3
"""
Dynamic Market Screener with Industry Classification

Provides proper diversification across industries using dynamic stock discovery
instead of hardcoded lists. Uses real-time market data and trading activity
to classify stocks by industry and create balanced portfolios.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import yfinance as yf
from dataclasses import dataclass
from tools.alpaca_client import alpaca_client
from core.market_intelligence import market_intelligence

logger = logging.getLogger(__name__)

@dataclass
class IndustryAllocation:
    """Industry allocation definition."""
    name: str
    target_percentage: float
    max_percentage: float
    min_stocks: int
    max_stocks: int

class DynamicMarketScreener:
    """Dynamic market screener with industry classification and diversification."""
    
    def __init__(self):
        """Initialize dynamic screener with industry detection."""
        
        # Price limit for better diversification
        self.max_stock_price = 1000.0
        self.min_stock_price = 5.0
        self.min_market_cap = 1e9  # $1B minimum market cap
        
        # Industry allocation targets for balanced diversification
        self.industry_allocations = {
            'Technology': IndustryAllocation('Technology', 0.20, 0.25, 4, 8),
            'Healthcare': IndustryAllocation('Healthcare', 0.15, 0.20, 3, 6),
            'Financial Services': IndustryAllocation('Financial', 0.12, 0.18, 3, 6),
            'Consumer Cyclical': IndustryAllocation('Consumer Discretionary', 0.10, 0.15, 2, 5),
            'Consumer Defensive': IndustryAllocation('Consumer Staples', 0.08, 0.12, 2, 4),
            'Industrials': IndustryAllocation('Industrials', 0.10, 0.15, 2, 5),
            'Energy': IndustryAllocation('Energy', 0.07, 0.10, 2, 4),
            'Utilities': IndustryAllocation('Utilities', 0.05, 0.08, 1, 3),
            'Communication Services': IndustryAllocation('Communications', 0.05, 0.08, 1, 3),
            'Real Estate': IndustryAllocation('Real Estate', 0.03, 0.06, 1, 2),
            'Basic Materials': IndustryAllocation('Materials', 0.03, 0.05, 1, 2),
            'Biotechnology': IndustryAllocation('Biotech', 0.02, 0.05, 1, 2)
        }
        
        # Cache for industry classifications
        self.industry_cache = {}
        self.cache_duration = 3600  # 1 hour
        
    async def get_dynamic_stock_universe(self, target_size: int = 500) -> List[str]:
        """Get a dynamic universe of stocks from market data."""
        try:
            # Get tradeable assets from Alpaca
            assets = alpaca_client.api.list_assets(status='active', asset_class='us_equity')
            
            # Filter for liquid, tradeable stocks
            candidates = []
            for asset in assets:
                if (asset.tradable and 
                    asset.symbol.isalpha() and 
                    len(asset.symbol) <= 5 and
                    not any(char.islower() for char in asset.symbol)):
                    candidates.append(asset.symbol)
            
            logger.info(f"Found {len(candidates)} tradeable stock candidates")
            
            # Randomly sample for diversity if too many
            if len(candidates) > target_size:
                import random
                candidates = random.sample(candidates, target_size)
            
            return candidates[:target_size]
            
        except Exception as e:
            logger.error(f"Error getting dynamic stock universe: {e}")
            return []
    
    async def classify_stock_by_industry(self, symbol: str) -> str:
        """Classify a stock into an industry using market data."""
        cache_key = f"industry_{symbol}"
        
        # Check cache
        if cache_key in self.industry_cache:
            data, timestamp = self.industry_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_duration:
                return data
        
        try:
            # Get stock info using yfinance as fallback
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Map yfinance sectors to our industry classifications
            yf_sector = info.get('sector', 'Unknown')
            yf_industry = info.get('industry', 'Unknown')
            
            # Map to our standardized industries
            industry = self._map_to_standard_industry(yf_sector, yf_industry)
            
            # Cache result
            self.industry_cache[cache_key] = (industry, datetime.now())
            
            return industry
            
        except Exception as e:
            logger.debug(f"Error classifying {symbol}: {e}")
            return "Unknown"
    
    def _map_to_standard_industry(self, sector: str, industry: str) -> str:
        """Map yfinance sector/industry to our standard classifications."""
        sector_lower = sector.lower()
        industry_lower = industry.lower()
        
        # Technology
        if ('technology' in sector_lower or 'software' in industry_lower or 
            'semiconductor' in industry_lower or 'electronic' in industry_lower):
            return 'Technology'
        
        # Healthcare & Biotech
        elif ('healthcare' in sector_lower or 'pharmaceutical' in industry_lower or 
              'biotechnology' in industry_lower or 'medical' in industry_lower):
            if 'biotechnology' in industry_lower:
                return 'Biotechnology'
            return 'Healthcare'
        
        # Financial Services
        elif ('financial' in sector_lower or 'bank' in industry_lower or 
              'insurance' in industry_lower or 'asset management' in industry_lower):
            return 'Financial Services'
        
        # Consumer
        elif 'consumer' in sector_lower:
            if 'defensive' in sector_lower or 'staples' in sector_lower:
                return 'Consumer Defensive'
            else:
                return 'Consumer Cyclical'
        
        # Industrials
        elif ('industrial' in sector_lower or 'aerospace' in industry_lower or 
              'defense' in industry_lower or 'machinery' in industry_lower):
            return 'Industrials'
        
        # Energy
        elif 'energy' in sector_lower or 'oil' in industry_lower:
            return 'Energy'
        
        # Utilities
        elif 'utilities' in sector_lower:
            return 'Utilities'
        
        # Communication Services
        elif ('communication' in sector_lower or 'media' in industry_lower or 
              'telecommunications' in industry_lower):
            return 'Communication Services'
        
        # Real Estate
        elif 'real estate' in sector_lower:
            return 'Real Estate'
        
        # Basic Materials
        elif ('materials' in sector_lower or 'mining' in industry_lower or 
              'chemicals' in industry_lower):
            return 'Basic Materials'
        
        else:
            return 'Unknown'
    
    async def get_diversified_stock_selection(self, target_size: int) -> Dict[str, List[str]]:
        """Select stocks for cross-industry diversification using dynamic discovery."""
        
        logger.info(f"Dynamically selecting {target_size} stocks across industries")
        
        # Get dynamic stock universe
        stock_universe = await self.get_dynamic_stock_universe(target_size * 3)  # Get 3x for filtering
        
        if not stock_universe:
            logger.error("No stocks found in dynamic universe")
            return {}
        
        # Classify stocks by industry and filter by basic criteria
        industry_stocks = {}
        qualified_stocks = []
        
        # Process in batches to avoid rate limits
        batch_size = 20
        for i in range(0, len(stock_universe), batch_size):
            batch = stock_universe[i:i + batch_size]
            
            # Get stock data and classify by industry
            tasks = []
            for symbol in batch:
                tasks.append(self._get_stock_with_industry(symbol))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, results):
                if isinstance(result, Exception):
                    continue
                
                if result is None:
                    continue
                
                stock_data, industry = result
                
                # Apply basic filters
                if (stock_data['price'] >= self.min_stock_price and 
                    stock_data['price'] <= self.max_stock_price and
                    stock_data.get('market_cap', 0) >= self.min_market_cap):
                    
                    if industry not in industry_stocks:
                        industry_stocks[industry] = []
                    
                    industry_stocks[industry].append(symbol)
                    qualified_stocks.append((symbol, stock_data, industry))
            
            # Small delay between batches
            await asyncio.sleep(0.5)
        
        logger.info(f"Classified {len(qualified_stocks)} qualified stocks across {len(industry_stocks)} industries")
        
        # Select stocks based on industry allocation targets
        selection = {}
        total_selected = 0
        
        for industry, allocation in self.industry_allocations.items():
            available_stocks = industry_stocks.get(industry, [])
            
            if not available_stocks:
                continue
            
            # Calculate target count for this industry
            target_count = max(
                allocation.min_stocks,
                min(allocation.max_stocks, int(target_size * allocation.target_percentage))
            )
            
            # Select top stocks from this industry (could add scoring here)
            selected_stocks = available_stocks[:target_count]
            
            if selected_stocks:
                selection[industry] = selected_stocks
                total_selected += len(selected_stocks)
                
                logger.info(f"Industry {allocation.name}: {len(selected_stocks)} stocks "
                          f"({len(selected_stocks)/target_size*100:.1f}% of portfolio)")
        
        # Fill remaining slots with best available stocks
        remaining = target_size - total_selected
        if remaining > 0:
            # Get unselected stocks
            already_selected = set()
            for stocks in selection.values():
                already_selected.update(stocks)
            
            unselected = [(s, d, i) for s, d, i in qualified_stocks if s not in already_selected]
            
            # Sort by market cap and select top remaining
            unselected.sort(key=lambda x: x[1].get('market_cap', 0), reverse=True)
            
            for symbol, stock_data, industry in unselected[:remaining]:
                if industry not in selection:
                    selection[industry] = []
                selection[industry].append(symbol)
                total_selected += 1
        
        logger.info(f"Final dynamic selection: {total_selected} stocks across {len(selection)} industries")
        
        # Log industry breakdown
        for industry, stocks in selection.items():
            percentage = len(stocks) / total_selected * 100
            logger.info(f"  {industry}: {len(stocks)} stocks ({percentage:.1f}%)")
        
        return selection
    
    async def _get_stock_with_industry(self, symbol: str) -> Optional[Tuple[Dict, str]]:
        """Get stock data with industry classification."""
        try:
            # Get basic stock data
            market_data = await market_intelligence.get_market_data(symbol)
            if market_data.price <= 0:
                return None
            
            # Get additional info using yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            stock_data = {
                'symbol': symbol,
                'price': float(market_data.price),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'sector': info.get('sector', 'Unknown')
            }
            
            # Classify by industry
            industry = await self.classify_stock_by_industry(symbol)
            
            return stock_data, industry
            
        except Exception as e:
            logger.debug(f"Error processing {symbol}: {e}")
            return None
    
    def get_industry_for_symbol(self, symbol: str) -> str:
        """Get cached industry classification for a symbol."""
        cache_key = f"industry_{symbol}"
        if cache_key in self.industry_cache:
            data, timestamp = self.industry_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_duration:
                return data
        return "Unknown"
    
    def calculate_industry_diversification_score(self, positions: Dict[str, float]) -> Dict[str, float]:
        """Calculate diversification score across industries."""
        
        # Group positions by industry
        industry_weights = {}
        for symbol, weight in positions.items():
            industry = self.get_industry_for_symbol(symbol)
            industry_weights[industry] = industry_weights.get(industry, 0) + weight
        
        # Calculate diversification metrics
        total_industries = len(industry_weights)
        max_industry_weight = max(industry_weights.values()) if industry_weights else 0
        
        # Ideal diversification would be equal weights across industries
        ideal_weight = 1.0 / max(1, total_industries)
        
        # Calculate concentration penalty
        concentration_penalty = sum(
            abs(weight - ideal_weight) for weight in industry_weights.values()
        ) / 2.0  # Normalize to 0-1 range
        
        diversification_score = max(0, 1.0 - concentration_penalty)
        
        return {
            'diversification_score': diversification_score,
            'total_industries': total_industries,
            'max_industry_weight': max_industry_weight,
            'industry_weights': industry_weights,
            'concentration_penalty': concentration_penalty
        }

# Global instance
dynamic_screener = DynamicMarketScreener()

# For backward compatibility
class EnhancedMarketScreener(DynamicMarketScreener):
    """Backward compatible wrapper - now fully dynamic."""
    pass

enhanced_screener = dynamic_screener  # Redirect to dynamic version