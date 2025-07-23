#!/usr/bin/env python3
"""
Simplified Market Screener for AI Trading System

A streamlined version that provides stock recommendations without complex AI analysis.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import yfinance as yf

logger = logging.getLogger(__name__)

class SimpleMarketScreener:
    """Simplified market screener for stock selection."""
    
    def __init__(self):
        """Initialize the simple market screener."""
        
        # Expanded stock universe by asset class and sector for better diversification
        self.stock_universe = {
            'large_cap_tech': [
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',
                'AVGO', 'ORCL', 'CRM', 'ADBE', 'NFLX', 'QCOM', 'AMD', 'INTC'
            ],
            'large_cap_healthcare': [
                'UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'DHR', 'BMY', 'MDT',
                'AMGN', 'GILD', 'CVS', 'CI', 'ANTM', 'HUM', 'REGN', 'VRTX'
            ],
            'large_cap_financials': [
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'USB',
                'PNC', 'TFC', 'COF', 'SCHW', 'BK', 'STT', 'BLK', 'SPGI'
            ],
            'large_cap_consumer': [
                'V', 'MA', 'HD', 'PG', 'KO', 'PEP', 'COST', 'WMT',
                'DIS', 'MCD', 'SBUX', 'NKE', 'LOW', 'TGT', 'CL', 'KMB'
            ],
            'large_cap_industrials': [
                'CAT', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'MMM', 'GE',
                'UBER', 'FDX', 'DE', 'NOC', 'ITW', 'CSX', 'NSC', 'UNP'
            ],
            'energy_materials': [
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX',
                'FCG', 'DVN', 'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM'
            ],
            'utilities_reits': [
                'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE',
                'AMT', 'CCI', 'EQIX', 'PLD', 'PSA', 'EQR', 'AVB', 'O'
            ],
            'mid_cap_growth': [
                'SNOW', 'CRWD', 'DDOG', 'OKTA', 'TEAM', 'TWLO', 'ZS', 'BILL',
                'PLTR', 'U', 'RBLX', 'COIN', 'HOOD', 'SOFI', 'SQ', 'PYPL'
            ],
            'mid_cap_value': [
                'F', 'GM', 'T', 'VZ', 'KHC', 'INTC', 'IBM', 'GIS',
                'K', 'CAG', 'CPB', 'HSY', 'SJM', 'CLX', 'CHD', 'PG'
            ],
            'small_cap_growth': [
                'ROKU', 'ZM', 'SHOP', 'SQ', 'PYPL', 'PINS', 'SNAP', 'SPOT',
                'UBER', 'LYFT', 'DASH', 'ABNB', 'AI', 'SMCI', 'ARM', 'RIVN'
            ]
        }
        
        # Asset class mapping for diversification
        self.asset_classes = {
            'large_cap_growth': ['large_cap_tech', 'mid_cap_growth'],
            'large_cap_value': ['large_cap_financials', 'large_cap_industrials', 'mid_cap_value'],
            'healthcare': ['large_cap_healthcare'],
            'consumer': ['large_cap_consumer'],
            'energy_materials': ['energy_materials'],
            'utilities_reits': ['utilities_reits'],
            'small_mid_cap': ['mid_cap_growth', 'mid_cap_value', 'small_cap_growth']
        }
        
        # Sector mapping for sector-based diversification
        self.sectors = {
            'Technology': ['large_cap_tech', 'mid_cap_growth'],
            'Healthcare': ['large_cap_healthcare'],
            'Financials': ['large_cap_financials'],
            'Consumer': ['large_cap_consumer'],
            'Industrials': ['large_cap_industrials'],
            'Energy': ['energy_materials'],
            'Utilities': ['utilities_reits'],
            'Communication': ['large_cap_tech'],
            'Materials': ['energy_materials']
        }
    
    async def get_stock_data(self, symbol: str) -> Dict:
        """Get basic stock data."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current price and basic info
            hist = ticker.history(period="5d")
            if hist.empty:
                return None
                
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) >= 2 else current_price
            change_pct = ((current_price - prev_price) / prev_price) * 100
            
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': current_price,
                'change_pct': change_pct,
                'volume': hist['Volume'].iloc[-1],
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'sector': info.get('sector', 'Unknown'),
                'beta': info.get('beta', 1.0)
            }
        except Exception as e:
            logger.warning(f"Failed to get data for {symbol}: {e}")
            return None
    
    def get_all_stocks(self) -> List[str]:
        """Get all stocks from the universe."""
        all_stocks = set()
        for category, stocks in self.stock_universe.items():
            all_stocks.update(stocks)
        return list(all_stocks)
    
    def get_diversified_stock_selection(self, target_size: int = 35) -> Dict[str, List[str]]:
        """Select stocks for a diversified portfolio across asset classes and sectors."""
        from config.settings import settings
        
        selection = {}
        total_selected = 0
        
        # Define allocation targets for each asset class (percentages)
        asset_class_targets = {
            'large_cap_growth': 0.30,  # 30%
            'large_cap_value': 0.25,   # 25%
            'healthcare': 0.15,        # 15%
            'consumer': 0.10,          # 10%
            'energy_materials': 0.08,  # 8%
            'utilities_reits': 0.07,   # 7%
            'small_mid_cap': 0.05      # 5%
        }
        
        # Calculate number of stocks per asset class
        for asset_class, target_pct in asset_class_targets.items():
            num_stocks = max(1, int(target_size * target_pct))
            
            # Get stocks from categories in this asset class
            available_stocks = []
            for category in self.asset_classes[asset_class]:
                available_stocks.extend(self.stock_universe[category])
            
            # Remove duplicates and select top stocks
            available_stocks = list(set(available_stocks))
            selected_stocks = available_stocks[:num_stocks]
            
            selection[asset_class] = selected_stocks
            total_selected += len(selected_stocks)
        
        logger.info(f"Selected {total_selected} stocks across {len(selection)} asset classes")
        return selection
    
    def calculate_simple_score(self, data: Dict, risk_level: str) -> float:
        """Calculate a simple scoring system for stocks."""
        score = 50.0  # Base score
        
        try:
            # Price momentum (recent performance)
            if data['change_pct'] > 2:
                score += 15
            elif data['change_pct'] > 0:
                score += 10
            elif data['change_pct'] > -2:
                score += 5
            else:
                score -= 10
            
            # Market cap consideration
            market_cap = data.get('market_cap', 0)
            if market_cap > 100e9:  # Large cap
                score += 10
            elif market_cap > 10e9:  # Mid cap
                score += 5
            
            # P/E ratio (if available)
            pe_ratio = data.get('pe_ratio')
            if pe_ratio:
                if pe_ratio < 15:
                    score += 10
                elif pe_ratio < 25:
                    score += 5
                elif pe_ratio > 50:
                    score -= 10
            
            # Beta consideration based on risk level
            beta = data.get('beta', 1.0)
            if risk_level == 'conservative':
                if beta < 1.0:
                    score += 10
                elif beta > 1.5:
                    score -= 15
            elif risk_level == 'aggressive':
                if beta > 1.2:
                    score += 10
                elif beta < 0.8:
                    score -= 5
            
            # Sector bonuses (simple heuristics)
            sector = data.get('sector', '')
            if 'Technology' in sector:
                if risk_level in ['moderate', 'aggressive']:
                    score += 10
            elif 'Healthcare' in sector:
                score += 5  # Generally stable
            elif 'Financial' in sector:
                if risk_level == 'conservative':
                    score += 5
                    
        except Exception as e:
            logger.warning(f"Error calculating score: {e}")
        
        return min(max(score, 0), 100)  # Clamp between 0-100
    
    async def get_recommended_stocks(self, 
                                   max_recommendations: int = 20,
                                   min_score: float = 60.0,
                                   risk_level: str = "moderate") -> Tuple[List[str], Dict]:
        """Get recommended stocks with simple analysis."""
        
        logger.info(f"🧠 Starting simple stock screening for {risk_level} risk level...")
        
        # Get appropriate stock universe
        candidate_stocks = self.stock_universe.get(risk_level, self.stock_universe['moderate'])
        
        # Get data for all candidates
        logger.info(f"Analyzing {len(candidate_stocks)} candidate stocks...")
        
        stock_scores = []
        
        # Process in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(candidate_stocks), batch_size):
            batch = candidate_stocks[i:i + batch_size]
            
            # Get data for batch
            for symbol in batch:
                data = await self.get_stock_data(symbol)
                if data:
                    score = self.calculate_simple_score(data, risk_level)
                    
                    stock_scores.append({
                        'symbol': symbol,
                        'score': score,
                        'price': data['price'],
                        'change_pct': data['change_pct'],
                        'market_cap': data['market_cap'],
                        'sector': data['sector'],
                        'data': data
                    })
            
            # Small delay between batches
            await asyncio.sleep(0.5)
        
        # Sort by score and filter
        stock_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Filter by minimum score
        recommended = [
            stock for stock in stock_scores 
            if stock['score'] >= min_score
        ][:max_recommendations]
        
        # Extract symbols
        recommended_symbols = [stock['symbol'] for stock in recommended]
        
        # Create analysis summary
        analysis_summary = {
            'total_screened': len(candidate_stocks),
            'qualified_after_screening': len(candidate_stocks),  # All passed basic screening
            'analyzed_with_ai': len(stock_scores),
            'final_recommendations': len(recommended_symbols),
            'avg_score': np.mean([s['score'] for s in recommended]) if recommended else 0,
            'top_stocks': [
                {
                    'symbol': stock['symbol'],
                    'composite_score': stock['score'],
                    'action': 'buy' if stock['score'] >= 70 else 'hold',
                    'confidence': stock['score'] / 100,
                    'reasoning': f"Score: {stock['score']:.1f}, Price: ${stock['price']:.2f}, Change: {stock['change_pct']:+.1f}%"
                }
                for stock in recommended[:10]
            ],
            'risk_level': risk_level,
            'screening_criteria': {
                'min_score': min_score,
                'risk_level': risk_level,
                'stock_universe': len(candidate_stocks)
            },
            'timestamp': datetime.now()
        }
        
        logger.info(f"✅ Recommended {len(recommended_symbols)} stocks with avg score {analysis_summary['avg_score']:.1f}")
        
        return recommended_symbols, analysis_summary

# Global instance
simple_market_screener = SimpleMarketScreener()

# Convenience function for easy import
async def get_ai_recommended_stocks(risk_level: str = "moderate", max_stocks: int = 20) -> Tuple[List[str], Dict]:
    """Get AI-recommended stocks for portfolio construction."""
    return await simple_market_screener.get_recommended_stocks(
        max_recommendations=max_stocks,
        risk_level=risk_level
    )