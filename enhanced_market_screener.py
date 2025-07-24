#!/usr/bin/env python3
"""
Enhanced Market Screener with True Cross-Industry Diversification

Provides proper diversification across defense, consumer, healthcare, financial, 
industrial, technology, and other major industry sectors.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import yfinance as yf
from dataclasses import dataclass
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

class EnhancedMarketScreener:
    """Enhanced market screener with true cross-industry diversification."""
    
    def __init__(self):
        """Initialize enhanced screener with comprehensive industry coverage."""
        
        # Comprehensive stock universe organized by industry sectors
        self.industry_universe = {
            'defense_aerospace': {
                'description': 'Defense contractors and aerospace companies',
                'stocks': [
                    'LMT', 'RTX', 'NOC', 'GD', 'LHX', 'HII', 'TXT', 'HON',
                    'BA', 'LDOS', 'KTOS', 'AJRD', 'WWD', 'TDG', 'CW', 'HEI'
                ]
            },
            'healthcare_pharma': {
                'description': 'Healthcare services, pharmaceuticals, and medical devices',
                'stocks': [
                    'UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'DHR', 'BMY', 'MDT',
                    'AMGN', 'GILD', 'CVS', 'CI', 'HUM', 'REGN', 'VRTX', 'ISRG',
                    'ZBH', 'SYK', 'BSX', 'EW', 'HOLX', 'DXCM', 'ILMN', 'IQV'
                ]
            },
            'financial_banking': {
                'description': 'Banks, insurance, and financial services',
                'stocks': [
                    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'USB',
                    'PNC', 'TFC', 'COF', 'SCHW', 'BK', 'STT', 'BLK', 'SPGI',
                    'ICE', 'CME', 'MCO', 'AON', 'MMC', 'AJG', 'BRO', 'CB'
                ]
            },
            'consumer_retail': {
                'description': 'Consumer goods, retail, and restaurants',
                'stocks': [
                    'WMT', 'HD', 'COST', 'TGT', 'LOW', 'MCD', 'SBUX', 'NKE',
                    'DIS', 'AMZN', 'TSLA', 'F', 'GM', 'TJX', 'DG', 'DLTR',
                    'YUM', 'CMG', 'QSR', 'DPZ', 'BKNG', 'MAR', 'HLT', 'MGM'
                ]
            },
            'consumer_staples': {
                'description': 'Food, beverages, and household products',
                'stocks': [
                    'PG', 'KO', 'PEP', 'WMT', 'KMB', 'CL', 'GIS', 'K',
                    'CPB', 'HSY', 'SJM', 'CLX', 'CHD', 'KHC', 'CAG', 'HRL',
                    'TSN', 'TAP', 'STZ', 'BF.B', 'MKC', 'SYY', 'COST', 'KR'
                ]
            },
            'industrial_manufacturing': {
                'description': 'Industrial equipment, manufacturing, and transportation',
                'stocks': [
                    'CAT', 'DE', 'HON', 'UPS', 'FDX', 'UNP', 'CSX', 'NSC',
                    'MMM', 'GE', 'EMR', 'ITW', 'PH', 'ROK', 'DOV', 'XYL',
                    'FTV', 'AME', 'ROP', 'IEX', 'FAST', 'PCAR', 'CHRW', 'EXPD'
                ]
            },
            'technology_software': {
                'description': 'Technology hardware, software, and services',
                'stocks': [
                    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AVGO', 'ORCL',
                    'CRM', 'ADBE', 'QCOM', 'AMD', 'INTC', 'IBM', 'CSCO', 'TXN',
                    'AMAT', 'ADI', 'LRCX', 'KLAC', 'MCHP', 'CDNS', 'SNPS', 'FTNT'
                ]
            },
            'telecommunications': {
                'description': 'Telecommunications and media',
                'stocks': [
                    'T', 'VZ', 'TMUS', 'CHTR', 'CMCSA', 'DIS', 'NFLX', 'PARA',
                    'WBD', 'FOX', 'FOXA', 'DISH', 'SIRI', 'AMC', 'ROKU', 'SPOT'
                ]
            },
            'energy_utilities': {
                'description': 'Energy production and utilities',
                'stocks': [
                    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX',
                    'DVN', 'PXD', 'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC',
                    'XEL', 'SRE', 'ES', 'PEG', 'FE', 'ETR', 'WEC', 'DTE'
                ]
            },
            'real_estate': {
                'description': 'Real estate investment trusts and property',
                'stocks': [
                    'AMT', 'CCI', 'EQIX', 'PLD', 'PSA', 'EQR', 'AVB', 'O',
                    'WELL', 'SPG', 'DLR', 'EXR', 'VTR', 'ESS', 'MAA', 'UDR',
                    'BXP', 'KIM', 'REG', 'FRT', 'AIV', 'CPT', 'ELS', 'SUI'
                ]
            },
            'materials_chemicals': {
                'description': 'Materials, chemicals, and mining',
                'stocks': [
                    'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'STLD',
                    'VMC', 'MLM', 'DOW', 'DD', 'PPG', 'RPM', 'ALB', 'CE',
                    'FMC', 'IFF', 'EMN', 'CF', 'MOS', 'LYB', 'WLK', 'OLN'
                ]
            },
            'biotech_emerging': {
                'description': 'Biotechnology and emerging growth companies',
                'stocks': [
                    'SNOW', 'CRWD', 'DDOG', 'OKTA', 'TEAM', 'TWLO', 'ZS', 'BILL',
                    'PLTR', 'U', 'RBLX', 'COIN', 'HOOD', 'SOFI', 'SQ', 'PYPL',
                    'MRNA', 'BNTX', 'GILD', 'BIIB', 'CELG', 'BMRN', 'SGEN', 'ALXN'
                ]
            }
        }
        
        # Industry allocation targets for balanced diversification
        self.industry_allocations = {
            'technology_software': IndustryAllocation('Technology', 0.20, 0.25, 4, 8),
            'healthcare_pharma': IndustryAllocation('Healthcare', 0.15, 0.20, 3, 6),
            'financial_banking': IndustryAllocation('Financial', 0.12, 0.18, 3, 6),
            'consumer_retail': IndustryAllocation('Consumer Discretionary', 0.10, 0.15, 2, 5),
            'consumer_staples': IndustryAllocation('Consumer Staples', 0.08, 0.12, 2, 4),
            'industrial_manufacturing': IndustryAllocation('Industrials', 0.10, 0.15, 2, 5),
            'defense_aerospace': IndustryAllocation('Defense/Aerospace', 0.08, 0.12, 2, 4),
            'energy_utilities': IndustryAllocation('Energy/Utilities', 0.07, 0.10, 2, 4),
            'telecommunications': IndustryAllocation('Telecommunications', 0.05, 0.08, 1, 3),
            'real_estate': IndustryAllocation('Real Estate', 0.03, 0.06, 1, 2),
            'materials_chemicals': IndustryAllocation('Materials', 0.02, 0.05, 1, 2),
            'biotech_emerging': IndustryAllocation('Biotech/Emerging', 0.05, 0.10, 1, 3)
        }
        
        # Sector to industry mapping for compatibility
        self.sector_mapping = {}
        for industry, info in self.industry_universe.items():
            for stock in info['stocks']:
                self.sector_mapping[stock] = industry
    
    def get_all_stocks(self) -> List[str]:
        """Get all stocks from the universe."""
        all_stocks = []
        for industry_info in self.industry_universe.values():
            all_stocks.extend(industry_info['stocks'])
        return list(set(all_stocks))  # Remove duplicates
    
    def get_industry_for_symbol(self, symbol: str) -> str:
        """Get industry classification for a symbol."""
        return self.sector_mapping.get(symbol, 'unknown')
    
    def get_industry_description(self, industry: str) -> str:
        """Get description for an industry."""
        return self.industry_universe.get(industry, {}).get('description', 'Unknown Industry')
    
    def get_diversified_stock_selection(self, target_size: int) -> Dict[str, List[str]]:
        """Select stocks for true cross-industry diversification."""
        
        logger.info(f"Selecting {target_size} stocks across industries for maximum diversification")
        
        selection = {}
        total_selected = 0
        
        # Calculate allocation for each industry
        for industry, allocation in self.industry_allocations.items():
            # Calculate target number of stocks for this industry
            target_count = max(
                allocation.min_stocks,
                min(allocation.max_stocks, int(target_size * allocation.target_percentage))
            )
            
            # Get available stocks from this industry
            available_stocks = self.industry_universe[industry]['stocks']
            
            # Select stocks (for now, take first N stocks - could be enhanced with scoring)
            selected_stocks = available_stocks[:target_count]
            
            if selected_stocks:
                selection[industry] = selected_stocks
                total_selected += len(selected_stocks)
                
                logger.info(f"Industry {allocation.name}: {len(selected_stocks)} stocks "
                          f"({len(selected_stocks)/target_size*100:.1f}% of portfolio)")
        
        # If we haven't reached target size, add more from largest categories
        remaining = target_size - total_selected
        if remaining > 0:
            # Add more from technology and healthcare (largest allocations)
            large_categories = ['technology_software', 'healthcare_pharma', 'financial_banking']
            
            for category in large_categories:
                if remaining <= 0:
                    break
                    
                current_selection = selection.get(category, [])
                available_stocks = self.industry_universe[category]['stocks']
                
                # Add more stocks from this category
                additional_stocks = [s for s in available_stocks if s not in current_selection][:remaining]
                
                if additional_stocks:
                    selection[category] = current_selection + additional_stocks
                    remaining -= len(additional_stocks)
                    total_selected += len(additional_stocks)
        
        logger.info(f"Final selection: {total_selected} stocks across {len(selection)} industries")
        
        # Log industry breakdown
        for industry, stocks in selection.items():
            allocation = self.industry_allocations[industry]
            percentage = len(stocks) / total_selected * 100
            logger.info(f"  {allocation.name}: {len(stocks)} stocks ({percentage:.1f}%)")
        
        return selection
    
    async def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get basic stock data with industry classification using reliable fallback system."""
        try:
            # Get current price using unified market intelligence
            market_data = await market_intelligence.get_market_data(symbol)
            if market_data.price <= 0:
                return None
            current_price = market_data.price
            
            # Get historical data for change calculation (fallback to yfinance)
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if hist.empty:
                    prev_price = current_price
                else:
                    prev_price = hist['Close'].iloc[0] if len(hist) > 1 else current_price
            except:
                prev_price = current_price
            
            # Get basic info (fallback to yfinance)
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                market_cap = info.get('marketCap', 0)
                pe_ratio = info.get('trailingPE', 0)
                sector = info.get('sector', 'Unknown')
            except:
                market_cap = 0
                pe_ratio = 0
                sector = 'Unknown'
            
            # Calculate change
            change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
            
            return {
                'symbol': symbol,
                'price': float(current_price),
                'change_pct': float(change_pct),
                'market_cap': market_cap,
                'pe_ratio': pe_ratio,
                'sector': sector,
                'industry': self.get_industry_for_symbol(symbol),
                'industry_description': self.get_industry_description(self.get_industry_for_symbol(symbol)),
                'volume': float(hist['Volume'].iloc[-1]) if hist is not None and 'Volume' in hist.columns and not hist.empty else 0
            }
            
        except Exception as e:
            logger.warning(f"Failed to get data for {symbol}: {e}")
            return None
    
    async def get_bulk_stock_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get stock data for multiple symbols."""
        stock_data = {}
        
        # Process in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Get data for batch
            tasks = [self.get_stock_data(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for symbol, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.warning(f"Error getting data for {symbol}: {result}")
                elif result is not None:
                    stock_data[symbol] = result
            
            # Small delay between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(0.5)
        
        logger.info(f"Retrieved data for {len(stock_data)}/{len(symbols)} symbols")
        return stock_data
    
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
    
    def get_portfolio_industry_summary(self, positions: Dict[str, float]) -> str:
        """Get a summary of portfolio industry allocation."""
        
        metrics = self.calculate_industry_diversification_score(positions)
        industry_weights = metrics['industry_weights']
        
        summary = f"Portfolio Industry Diversification:\n"
        summary += f"Industries: {metrics['total_industries']}\n"
        summary += f"Diversification Score: {metrics['diversification_score']:.2f}\n"
        summary += f"Max Industry Weight: {metrics['max_industry_weight']:.1%}\n\n"
        
        summary += "Industry Breakdown:\n"
        for industry, weight in sorted(industry_weights.items(), key=lambda x: x[1], reverse=True):
            industry_name = self.industry_allocations.get(industry, type('', (), {'name': industry})).name
            summary += f"  {industry_name}: {weight:.1%}\n"
        
        return summary

# Global instance
enhanced_screener = EnhancedMarketScreener()

# For backward compatibility
class SimpleMarketScreener(EnhancedMarketScreener):
    """Backward compatible wrapper."""
    pass