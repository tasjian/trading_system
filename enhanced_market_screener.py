"""
Enhanced Market Screener
Provides enhanced stock screening capabilities for universe filtering fallback.
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EnhancedMarketScreener:
    """Enhanced market screener with multiple filtering criteria."""
    
    def __init__(self):
        """Initialize the enhanced market screener."""
        self.last_refresh = None
        self.cached_stocks = []
        self.cache_duration_hours = 24
        
    def get_all_stocks(self) -> List[str]:
        """Get all screened stocks using enhanced criteria."""
        
        try:
            # Check cache first
            if self._is_cache_valid():
                logger.info(f"Using cached enhanced stock universe: {len(self.cached_stocks)} stocks")
                return self.cached_stocks
            
            logger.info("Enhanced market screener: Generating stock universe...")
            
            # Get tradeable universe from Alpaca
            from tools.alpaca_client import alpaca_client
            
            # Get all tradeable assets using native Alpaca API
            all_assets = alpaca_client.api.list_assets(status='active', asset_class='us_equity')
            
            if not all_assets:
                logger.warning("No assets returned from Alpaca, using fallback list")
                return self._get_fallback_stocks()
            
            # Enhanced filtering criteria
            screened_stocks = []
            
            for asset in all_assets:
                try:
                    # Handle Alpaca asset objects
                    symbol = asset.symbol if hasattr(asset, 'symbol') else asset.get('symbol', '')
                    
                    # Basic filters
                    if not symbol or len(symbol) > 5:
                        continue
                        
                    if any(char in symbol for char in ['.', '-', '/']):
                        continue
                    
                    # Enhanced criteria
                    if self._meets_enhanced_criteria(asset):
                        screened_stocks.append(symbol)
                        
                        # Limit to reasonable size
                        if len(screened_stocks) >= 500:
                            break
                            
                except Exception as e:
                    logger.debug(f"Error processing asset {asset}: {e}")
                    continue
            
            # Update cache
            self.cached_stocks = screened_stocks
            self.last_refresh = datetime.now()
            
            logger.info(f"Enhanced screener generated {len(screened_stocks)} stocks")
            return screened_stocks
            
        except Exception as e:
            logger.error(f"Enhanced market screener error: {e}")
            return self._get_fallback_stocks()
    
    def _meets_enhanced_criteria(self, asset) -> bool:
        """Check if asset meets enhanced screening criteria."""
        
        try:
            # Handle both object attributes and dictionary keys
            symbol = getattr(asset, 'symbol', '') or asset.get('symbol', '')
            asset_class = getattr(asset, 'asset_class', '') or asset.get('class', '')
            tradable = getattr(asset, 'tradable', False) or asset.get('tradable', False)
            marginable = getattr(asset, 'marginable', False) or asset.get('marginable', False)
            shortable = getattr(asset, 'shortable', False) or asset.get('shortable', False)
            exchange = getattr(asset, 'exchange', '') or asset.get('exchange', '')
            name = getattr(asset, 'name', '') or asset.get('name', '')
            
            # Must be a stock
            if asset_class != 'us_equity':
                return False
                
            # Must be tradeable
            if not tradable:
                return False
                
            # Must be marginable for enhanced strategies
            if not marginable:
                return False
                
            # Must be shortable for short strategies
            if not shortable:
                return False
                
            # Exchange filters (prefer major exchanges)
            if exchange.upper() not in ['NYSE', 'NASDAQ', 'NYSEARCA']:
                return False
            
            # Sector diversity (basic check)
            if any(term in name.upper() for term in ['ETF', 'FUND', 'TRUST', 'NOTE']):
                return False
                
            return True
            
        except Exception as e:
            logger.debug(f"Error in enhanced criteria check: {e}")
            return False
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        
        if not self.last_refresh or not self.cached_stocks:
            return False
            
        age_hours = (datetime.now() - self.last_refresh).total_seconds() / 3600
        return age_hours < self.cache_duration_hours
    
    def _get_fallback_stocks(self) -> List[str]:
        """Get fallback stock list when screening fails."""
        
        # Curated list of liquid, well-known stocks
        fallback_stocks = [
            # Tech giants
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK.B', 'V', 'MA',
            
            # Industrial
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UNP', 'FDX', 'UPS',
            
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'AMGN', 'ABT', 'CVS',
            
            # Consumer
            'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'DIS', 'NKE',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO',
            
            # Telecom/Utilities
            'VZ', 'T', 'TMUS', 'NEE', 'SO', 'D', 'EXC', 'AEP',
            
            # Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'AVB',
            
            # Materials
            'LIN', 'APD', 'SHW', 'NEM', 'FCX', 'AA', 'CF', 'MOS'
        ]
        
        logger.info(f"Using fallback stock universe: {len(fallback_stocks)} stocks")
        return fallback_stocks
    
    def get_stocks_by_sector(self, sector: str) -> List[str]:
        """Get stocks filtered by sector."""
        
        # This is a simplified implementation
        # In production, you'd integrate with fundamental data providers
        
        sector_maps = {
            'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX'],
            'financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK.B', 'V', 'MA'],
            'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'AMGN', 'ABT', 'CVS'],
            'energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO']
        }
        
        return sector_maps.get(sector.lower(), [])
    
    def refresh_cache(self):
        """Force refresh of cached data."""
        self.last_refresh = None
        self.cached_stocks = []
        logger.info("Enhanced market screener cache refreshed")