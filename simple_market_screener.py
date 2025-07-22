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
        
        # Curated stock universe by category
        self.stock_universe = {
            'conservative': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'UNH', 'JNJ', 'V', 'PG', 'XOM',
                'HD', 'CVX', 'MA', 'BAC', 'ABBV', 'PFE', 'COST', 'DIS', 'TMO', 'KO'
            ],
            'moderate': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO',
                'CRM', 'ADBE', 'NFLX', 'PYPL', 'SNOW', 'AMD', 'ORCL', 'QCOM',
                'SBUX', 'ROKU', 'ZM', 'SHOP'
            ],
            'aggressive': [
                'TSLA', 'NVDA', 'META', 'SNOW', 'PLTR', 'COIN', 'RBLX', 'U',
                'CRWD', 'DDOG', 'OKTA', 'TEAM', 'TWLO', 'ZS', 'BILL', 'DKNG',
                'HOOD', 'SOFI', 'RIVN', 'AMD', 'ROKU', 'ZM', 'PYPL', 'SQ'
            ]
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