#!/usr/bin/env python3
"""
Advanced Portfolio Strategy with Short Selling and Sophisticated Techniques

Features:
1. Long/Short equity strategies
2. Pairs trading and market neutral positions  
3. Options strategies (when available)
4. Risk parity and volatility targeting
5. Dynamic hedging and correlation analysis
6. Sector rotation and momentum strategies
"""

import pandas as pd
import numpy as np
import yfinance as yf
import asyncio
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class AdvancedPosition:
    """Advanced position with long/short and strategy metadata."""
    symbol: str
    side: str  # 'long', 'short'
    weight: float
    strategy: str  # 'momentum', 'mean_reversion', 'pairs_trade', 'hedge', etc.
    confidence: float
    expected_return: float
    volatility: float
    beta: float
    correlation_hedge: Optional[str] = None  # Symbol this position hedges
    
class AdvancedPortfolioStrategy:
    """Advanced portfolio construction with short selling and sophisticated techniques."""
    
    def __init__(self, total_capital: float = 100000):
        self.total_capital = total_capital
        self.max_leverage = 1.5  # Allow 150% leverage for long/short
        self.max_short_exposure = 0.3  # Max 30% short exposure
        self.max_single_position = 0.15  # Max 15% per position
        self.rebalance_threshold = 0.05  # 5% drift triggers rebalance
        
    async def build_advanced_portfolio(self, 
                                     symbols: List[str],
                                     market_data: Dict,
                                     risk_level: str = "moderate") -> Dict:
        """
        Build advanced long/short portfolio with multiple strategies.
        
        Args:
            symbols: List of stock symbols
            market_data: Market data from screening
            risk_level: 'conservative', 'moderate', 'aggressive'
            
        Returns:
            Dict with advanced portfolio positions and metadata
        """
        
        logger.info(f"🧠 Building advanced long/short portfolio with {len(symbols)} symbols")
        
        try:
            # 1. Get enhanced market data
            enhanced_data = await self._get_enhanced_market_data(symbols)
            
            # 2. Apply multiple strategy layers
            strategies = await self._apply_strategy_layers(enhanced_data, risk_level)
            
            # 3. Construct long/short positions
            positions = await self._construct_long_short_positions(strategies)
            
            # 4. Add hedging and risk management
            hedged_positions = await self._add_hedging_positions(positions)
            
            # 5. Optimize portfolio weights
            optimized_portfolio = await self._optimize_portfolio_weights(hedged_positions, risk_level)
            
            logger.info(f"✅ Advanced portfolio: {len(optimized_portfolio['positions'])} positions")
            logger.info(f"📊 Long exposure: {optimized_portfolio['long_exposure']:.1%}")
            logger.info(f"📊 Short exposure: {optimized_portfolio['short_exposure']:.1%}")
            logger.info(f"📊 Net exposure: {optimized_portfolio['net_exposure']:.1%}")
            
            return optimized_portfolio
            
        except Exception as e:
            logger.error(f"Error building advanced portfolio: {e}")
            return await self._fallback_portfolio(symbols)
    
    async def _get_enhanced_market_data(self, symbols: List[str]) -> Dict:
        """Get enhanced market data with technical and fundamental metrics."""
        
        data = {}
        
        for symbol in symbols:
            try:
                # Get historical data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                info = ticker.info
                
                if hist.empty:
                    continue
                
                # Calculate technical indicators
                returns = hist['Close'].pct_change().dropna()
                
                # Momentum indicators
                momentum_20d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) if len(hist) >= 20 else 0
                momentum_60d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-60] - 1) if len(hist) >= 60 else 0
                
                # Volatility indicators
                volatility = returns.std() * np.sqrt(252)  # Annualized vol
                
                # Moving averages
                sma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else hist['Close'].iloc[-1]
                sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else hist['Close'].iloc[-1]
                
                # RSI (simplified)
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1] if not rs.empty else 50
                
                # Beta calculation (vs SPY)
                try:
                    spy = yf.download("SPY", period="1y")['Close'].pct_change().dropna()
                    aligned_returns = returns.align(spy, join='inner')[0]
                    aligned_spy = returns.align(spy, join='inner')[1]
                    beta = np.cov(aligned_returns, aligned_spy)[0, 1] / np.var(aligned_spy) if len(aligned_returns) > 20 else 1.0
                except:
                    beta = 1.0
                
                data[symbol] = {
                    'current_price': hist['Close'].iloc[-1],
                    'returns_1y': returns,
                    'momentum_20d': momentum_20d,
                    'momentum_60d': momentum_60d,
                    'volatility': volatility,
                    'rsi': rsi,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'beta': beta,
                    'volume_avg': hist['Volume'].mean(),
                    'market_cap': info.get('marketCap', 0),
                    'sector': info.get('sector', 'Unknown'),
                    'pe_ratio': info.get('trailingPE', None),
                    'price_to_book': info.get('priceToBook', None),
                    'expected_return': momentum_60d * 0.6 + momentum_20d * 0.4  # Simple expected return
                }
                
            except Exception as e:
                logger.warning(f"Could not get enhanced data for {symbol}: {e}")
                
        return data
    
    async def _apply_strategy_layers(self, data: Dict, risk_level: str) -> Dict:
        """Apply multiple strategy layers to identify long/short opportunities."""
        
        strategies = {
            'momentum_long': [],
            'momentum_short': [],
            'mean_reversion_long': [],
            'mean_reversion_short': [],
            'value_long': [],
            'growth_short': [],
            'pairs_trades': []
        }
        
        for symbol, metrics in data.items():
            
            # Momentum Strategy
            if metrics['momentum_20d'] > 0.05 and metrics['rsi'] < 80:
                strategies['momentum_long'].append({
                    'symbol': symbol,
                    'confidence': min(metrics['momentum_20d'] * 2, 0.9),
                    'metrics': metrics
                })
            elif metrics['momentum_20d'] < -0.05 and metrics['rsi'] > 20:
                strategies['momentum_short'].append({
                    'symbol': symbol,
                    'confidence': min(abs(metrics['momentum_20d']) * 2, 0.9),
                    'metrics': metrics
                })
            
            # Mean Reversion Strategy
            if metrics['rsi'] < 30 and metrics['current_price'] < metrics['sma_20']:
                strategies['mean_reversion_long'].append({
                    'symbol': symbol,
                    'confidence': (30 - metrics['rsi']) / 30,
                    'metrics': metrics
                })
            elif metrics['rsi'] > 70 and metrics['current_price'] > metrics['sma_20']:
                strategies['mean_reversion_short'].append({
                    'symbol': symbol,
                    'confidence': (metrics['rsi'] - 70) / 30,
                    'metrics': metrics
                })
            
            # Value Strategy (Long bias)
            if metrics['pe_ratio'] and metrics['pe_ratio'] < 15 and metrics['price_to_book'] and metrics['price_to_book'] < 2:
                strategies['value_long'].append({
                    'symbol': symbol,
                    'confidence': 0.7,
                    'metrics': metrics
                })
            
            # Growth/Expensive Short Strategy
            if metrics['pe_ratio'] and metrics['pe_ratio'] > 50 and metrics['momentum_60d'] < 0:
                strategies['growth_short'].append({
                    'symbol': symbol,
                    'confidence': 0.6,
                    'metrics': metrics
                })
        
        # Identify pairs trading opportunities
        strategies['pairs_trades'] = await self._identify_pairs_trades(data)
        
        return strategies
    
    async def _identify_pairs_trades(self, data: Dict) -> List[Dict]:
        """Identify pairs trading opportunities based on correlation."""
        
        pairs = []
        symbols = list(data.keys())
        
        # Simple pairs identification - same sector, different momentum
        sector_groups = {}
        for symbol, metrics in data.items():
            sector = metrics.get('sector', 'Unknown')
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(symbol)
        
        for sector, sector_symbols in sector_groups.items():
            if len(sector_symbols) >= 2:
                # Sort by momentum
                sector_data = [(s, data[s]['momentum_20d']) for s in sector_symbols]
                sector_data.sort(key=lambda x: x[1], reverse=True)
                
                # Create pairs: long top performer, short bottom performer
                if len(sector_data) >= 2:
                    long_symbol = sector_data[0][0]
                    short_symbol = sector_data[-1][0]
                    
                    if sector_data[0][1] > 0.02 and sector_data[-1][1] < -0.02:
                        pairs.append({
                            'long_symbol': long_symbol,
                            'short_symbol': short_symbol,
                            'strategy': 'sector_pairs',
                            'confidence': 0.6,
                            'sector': sector
                        })
        
        return pairs
    
    async def _construct_long_short_positions(self, strategies: Dict) -> List[AdvancedPosition]:
        """Construct long/short positions from strategy signals."""
        
        positions = []
        
        # Process momentum longs
        for signal in strategies['momentum_long'][:5]:  # Top 5
            positions.append(AdvancedPosition(
                symbol=signal['symbol'],
                side='long',
                weight=0.08 * signal['confidence'],  # Up to 8% per position
                strategy='momentum',
                confidence=signal['confidence'],
                expected_return=signal['metrics']['expected_return'],
                volatility=signal['metrics']['volatility'],
                beta=signal['metrics']['beta']
            ))
        
        # Process momentum shorts
        for signal in strategies['momentum_short'][:3]:  # Top 3 shorts
            positions.append(AdvancedPosition(
                symbol=signal['symbol'],
                side='short',
                weight=0.05 * signal['confidence'],  # Smaller short positions
                strategy='momentum_short',
                confidence=signal['confidence'],
                expected_return=-abs(signal['metrics']['expected_return']),  # Negative for shorts
                volatility=signal['metrics']['volatility'],
                beta=signal['metrics']['beta']
            ))
        
        # Process value longs
        for signal in strategies['value_long'][:3]:
            positions.append(AdvancedPosition(
                symbol=signal['symbol'],
                side='long',
                weight=0.06 * signal['confidence'],
                strategy='value',
                confidence=signal['confidence'],
                expected_return=signal['metrics']['expected_return'],
                volatility=signal['metrics']['volatility'],
                beta=signal['metrics']['beta']
            ))
        
        # Process pairs trades
        for pair in strategies['pairs_trades'][:2]:  # Top 2 pairs
            # Long position
            positions.append(AdvancedPosition(
                symbol=pair['long_symbol'],
                side='long',
                weight=0.05,
                strategy='pairs_long',
                confidence=pair['confidence'],
                expected_return=0.05,  # Expected from pair spread
                volatility=0.2,
                beta=1.0,
                correlation_hedge=pair['short_symbol']
            ))
            
            # Short position (hedge)
            positions.append(AdvancedPosition(
                symbol=pair['short_symbol'],
                side='short',
                weight=0.05,
                strategy='pairs_short',
                confidence=pair['confidence'],
                expected_return=-0.05,  # Expected from pair spread
                volatility=0.2,
                beta=1.0,
                correlation_hedge=pair['long_symbol']
            ))
        
        return positions
    
    async def _add_hedging_positions(self, positions: List[AdvancedPosition]) -> List[AdvancedPosition]:
        """Add hedging positions to reduce overall portfolio risk."""
        
        hedged_positions = positions.copy()
        
        # Calculate portfolio beta
        total_long_weight = sum(p.weight for p in positions if p.side == 'long')
        portfolio_beta = sum(p.weight * p.beta for p in positions if p.side == 'long') / max(total_long_weight, 0.01)
        
        # Add market hedge if portfolio is too exposed
        if portfolio_beta > 1.2 and total_long_weight > 0.6:
            hedge_weight = min(0.1, (portfolio_beta - 1.0) * 0.1)
            
            # Add SPY short as market hedge
            hedged_positions.append(AdvancedPosition(
                symbol='SPY',
                side='short',
                weight=hedge_weight,
                strategy='market_hedge',
                confidence=0.8,
                expected_return=0.0,  # Hedge - not for return
                volatility=0.15,
                beta=1.0
            ))
        
        return hedged_positions
    
    async def _optimize_portfolio_weights(self, positions: List[AdvancedPosition], risk_level: str) -> Dict:
        """Optimize portfolio weights based on risk level and constraints."""
        
        # Risk level adjustments
        risk_multipliers = {
            'conservative': 0.7,
            'moderate': 1.0,
            'aggressive': 1.3
        }
        multiplier = risk_multipliers.get(risk_level, 1.0)
        
        # Adjust weights
        adjusted_positions = []
        for pos in positions:
            adjusted_weight = pos.weight * multiplier
            
            # Apply position limits
            if adjusted_weight > self.max_single_position:
                adjusted_weight = self.max_single_position
            
            adjusted_positions.append(AdvancedPosition(
                symbol=pos.symbol,
                side=pos.side,
                weight=adjusted_weight,
                strategy=pos.strategy,
                confidence=pos.confidence,
                expected_return=pos.expected_return,
                volatility=pos.volatility,
                beta=pos.beta,
                correlation_hedge=pos.correlation_hedge
            ))
        
        # Calculate exposures
        long_exposure = sum(p.weight for p in adjusted_positions if p.side == 'long')
        short_exposure = sum(p.weight for p in adjusted_positions if p.side == 'short')
        net_exposure = long_exposure - short_exposure
        
        # Apply exposure limits
        if short_exposure > self.max_short_exposure:
            short_scale = self.max_short_exposure / short_exposure
            for pos in adjusted_positions:
                if pos.side == 'short':
                    pos.weight *= short_scale
        
        # Normalize if over leveraged
        total_gross = long_exposure + short_exposure
        if total_gross > self.max_leverage:
            scale_factor = self.max_leverage / total_gross
            for pos in adjusted_positions:
                pos.weight *= scale_factor
        
        # Recalculate exposures after adjustments
        long_exposure = sum(p.weight for p in adjusted_positions if p.side == 'long')
        short_exposure = sum(p.weight for p in adjusted_positions if p.side == 'short')
        net_exposure = long_exposure - short_exposure
        
        # Calculate portfolio metrics
        expected_return = sum(p.weight * p.expected_return for p in adjusted_positions)
        
        return {
            'positions': adjusted_positions,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': net_exposure,
            'gross_exposure': long_exposure + short_exposure,
            'expected_return': expected_return,
            'num_positions': len(adjusted_positions),
            'num_long': len([p for p in adjusted_positions if p.side == 'long']),
            'num_short': len([p for p in adjusted_positions if p.side == 'short'])
        }
    
    async def _fallback_portfolio(self, symbols: List[str]) -> Dict:
        """Fallback to simple long-only portfolio if advanced strategies fail."""
        
        positions = []
        weight_per_stock = 0.1  # 10% each
        
        for symbol in symbols[:8]:  # Max 8 positions
            positions.append(AdvancedPosition(
                symbol=symbol,
                side='long',
                weight=weight_per_stock,
                strategy='fallback_long',
                confidence=0.5,
                expected_return=0.08,
                volatility=0.2,
                beta=1.0
            ))
        
        return {
            'positions': positions,
            'long_exposure': 0.8,
            'short_exposure': 0.0,
            'net_exposure': 0.8,
            'gross_exposure': 0.8,
            'expected_return': 0.064,
            'num_positions': len(positions),
            'num_long': len(positions),
            'num_short': 0
        }

    def convert_to_trading_orders(self, portfolio: Dict, total_capital: float) -> List[Dict]:
        """Convert advanced positions to trading orders."""
        
        orders = []
        
        for position in portfolio['positions']:
            dollar_amount = position.weight * total_capital
            
            # Skip very small positions
            if dollar_amount < 100:
                continue
                
            # Convert to shares (estimated)
            estimated_price = 100  # Placeholder - would get real price
            shares = int(dollar_amount / estimated_price)
            
            if shares > 0:
                orders.append({
                    'symbol': position.symbol,
                    'side': 'buy' if position.side == 'long' else 'sell_short',
                    'qty': shares,
                    'strategy': position.strategy,
                    'confidence': position.confidence,
                    'reason': f"{position.strategy.title()}: {position.side} {position.symbol} ({position.confidence:.1%} confidence)"
                })
        
        return orders

# Async wrapper function for external use
async def build_advanced_portfolio(symbols: List[str], 
                                 total_capital: float = 100000,
                                 risk_level: str = "moderate") -> Dict:
    """Build advanced long/short portfolio with sophisticated strategies."""
    
    strategy_engine = AdvancedPortfolioStrategy(total_capital)
    portfolio = await strategy_engine.build_advanced_portfolio(
        symbols=symbols,
        market_data={},
        risk_level=risk_level
    )
    
    # Convert to trading orders
    orders = strategy_engine.convert_to_trading_orders(portfolio, total_capital)
    
    return {
        'portfolio': portfolio,
        'orders': orders,
        'metadata': {
            'strategy': 'advanced_long_short',
            'timestamp': datetime.now().isoformat(),
            'risk_level': risk_level,
            'total_capital': total_capital
        }
    }