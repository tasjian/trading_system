#!/usr/bin/env python3
"""
Diversified Portfolio Management System

Implements advanced portfolio construction with 25-50 stocks across multiple asset classes.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from simple_market_screener import SimpleMarketScreener
from tools.llm_client import llm_client

logger = logging.getLogger(__name__)

@dataclass
class PortfolioPosition:
    """Individual portfolio position."""
    symbol: str
    asset_class: str
    sector: str
    target_weight: float
    current_weight: float
    quantity: int
    market_value: float
    unrealized_pnl: float
    confidence_score: float

@dataclass
class DiversificationMetrics:
    """Portfolio diversification metrics."""
    total_positions: int
    asset_class_distribution: Dict[str, float]
    sector_distribution: Dict[str, float]
    max_position_weight: float
    diversification_score: float
    risk_concentration: float

class DiversifiedPortfolioManager:
    """Advanced portfolio manager supporting 25-50 diversified positions."""
    
    def __init__(self):
        """Initialize the diversified portfolio manager."""
        self.screener = SimpleMarketScreener()
        self.current_positions: Dict[str, PortfolioPosition] = {}
        self.target_positions: Dict[str, PortfolioPosition] = {}
        
        # Portfolio parameters from settings
        self.target_size = settings.target_portfolio_size
        self.min_size = settings.min_portfolio_size
        self.max_size = settings.max_portfolio_size
        self.max_position_size = settings.max_position_size
        self.max_sector_allocation = settings.max_sector_allocation
        self.max_asset_class_allocation = settings.max_asset_class_allocation
        
    async def construct_diversified_portfolio(self, 
                                            portfolio_value: float,
                                            risk_tolerance: str = "moderate") -> Dict[str, PortfolioPosition]:
        """Construct a diversified portfolio with 25-50 positions."""
        
        logger.info(f"Constructing diversified portfolio with target size: {self.target_size}")
        
        # Step 1: Get diversified stock selection
        stock_selection = self.screener.get_diversified_stock_selection(self.target_size)
        
        # Step 2: Get market data for all selected stocks
        all_stocks = []
        for asset_class, stocks in stock_selection.items():
            all_stocks.extend(stocks)
        
        stock_data = await self._get_bulk_stock_data(all_stocks)
        
        # Step 3: Calculate optimal weights using Modern Portfolio Theory principles
        optimal_weights = await self._calculate_optimal_weights(stock_data, stock_selection, risk_tolerance)
        
        # Step 4: Create target positions
        target_positions = {}
        for symbol, weight in optimal_weights.items():
            if weight > settings.min_position_size:  # Only include meaningful positions
                position_value = portfolio_value * weight
                stock_info = stock_data.get(symbol, {})
                
                target_positions[symbol] = PortfolioPosition(
                    symbol=symbol,
                    asset_class=self._get_asset_class_for_symbol(symbol, stock_selection),
                    sector=stock_info.get('sector', 'Unknown'),
                    target_weight=weight,
                    current_weight=0.0,
                    quantity=int(position_value / stock_info.get('price', 1)),
                    market_value=position_value,
                    unrealized_pnl=0.0,
                    confidence_score=stock_info.get('score', 50.0)
                )
        
        self.target_positions = target_positions
        logger.info(f"Constructed portfolio with {len(target_positions)} positions")
        
        return target_positions
    
    async def _get_bulk_stock_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get market data for multiple stocks efficiently."""
        stock_data = {}
        
        # Process in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            tasks = [self.screener.get_stock_data(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, results):
                if isinstance(result, dict) and result:
                    # Add scoring
                    result['score'] = self.screener.calculate_simple_score(result, "moderate")
                    stock_data[symbol] = result
                else:
                    logger.warning(f"Failed to get data for {symbol}")
        
        logger.info(f"Successfully retrieved data for {len(stock_data)}/{len(symbols)} stocks")
        return stock_data
    
    async def _calculate_optimal_weights(self, 
                                       stock_data: Dict[str, Dict],
                                       stock_selection: Dict[str, List[str]],
                                       risk_tolerance: str) -> Dict[str, float]:
        """Calculate optimal portfolio weights using LLM-enhanced analysis."""
        
        # Create LLM prompt for portfolio optimization
        prompt = self._create_portfolio_optimization_prompt(stock_data, stock_selection, risk_tolerance)
        
        try:
            response = await llm_client.get_completion(
                prompt=prompt,
                system_message="You are an expert portfolio manager specializing in diversified equity portfolios."
            )
            
            # Parse LLM response to extract weights
            weights = self._parse_weight_response(response.content, stock_data)
            
        except Exception as e:
            logger.warning(f"LLM optimization failed, using fallback method: {e}")
            weights = self._fallback_weight_calculation(stock_data, stock_selection)
        
        # Apply diversification constraints
        weights = self._apply_diversification_constraints(weights, stock_selection)
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {symbol: weight / total_weight for symbol, weight in weights.items()}
        
        return weights
    
    def _create_portfolio_optimization_prompt(self, 
                                            stock_data: Dict[str, Dict],
                                            stock_selection: Dict[str, List[str]],
                                            risk_tolerance: str) -> str:
        """Create prompt for LLM-based portfolio optimization."""
        
        prompt = f"""
        Create an optimal portfolio allocation for {self.target_size} stocks with the following constraints:
        
        PORTFOLIO REQUIREMENTS:
        - Target portfolio size: {self.target_size} positions
        - Risk tolerance: {risk_tolerance}
        - Maximum position size: {self.max_position_size:.1%}
        - Maximum sector allocation: {self.max_sector_allocation:.1%}
        - Maximum asset class allocation: {self.max_asset_class_allocation:.1%}
        
        STOCK DATA:
        """
        
        # Add top stocks from each asset class
        for asset_class, stocks in stock_selection.items():
            prompt += f"\n{asset_class.upper()}:\n"
            for symbol in stocks[:5]:  # Top 5 per class for brevity
                data = stock_data.get(symbol, {})
                prompt += f"- {symbol}: Price=${data.get('price', 0):.2f}, Score={data.get('score', 0):.1f}, Sector={data.get('sector', 'Unknown')}\n"
        
        prompt += f"""
        
        INSTRUCTIONS:
        1. Allocate weights across ALL {len([s for stocks in stock_selection.values() for s in stocks])} stocks
        2. Ensure diversification across asset classes and sectors
        3. Consider risk-adjusted returns and market conditions
        4. Provide weights as percentages (e.g., AAPL: 2.5%)
        5. Focus on quality companies with strong fundamentals
        
        Respond with ONLY the allocation in this format:
        SYMBOL: WEIGHT%
        """
        
        return prompt
    
    def _parse_weight_response(self, response: str, stock_data: Dict[str, Dict]) -> Dict[str, float]:
        """Parse LLM response to extract portfolio weights."""
        weights = {}
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line and '%' in line:
                try:
                    parts = line.split(':')
                    symbol = parts[0].strip().upper()
                    weight_str = parts[1].strip().replace('%', '')
                    weight = float(weight_str) / 100.0
                    
                    if symbol in stock_data:
                        weights[symbol] = weight
                except ValueError:
                    continue
        
        return weights
    
    def _fallback_weight_calculation(self, 
                                   stock_data: Dict[str, Dict],
                                   stock_selection: Dict[str, List[str]]) -> Dict[str, float]:
        """Fallback weight calculation using inverse volatility weighting."""
        weights = {}
        
        # Calculate base equal weight
        equal_weight = 1.0 / len(stock_data)
        
        for symbol, data in stock_data.items():
            # Adjust by score and inverse volatility
            score_factor = data.get('score', 50) / 100.0
            beta = data.get('beta', 1.0)
            volatility_adjustment = 1.0 / max(beta, 0.5)  # Inverse volatility
            
            base_weight = equal_weight * score_factor * volatility_adjustment
            weights[symbol] = min(base_weight, self.max_position_size)
        
        return weights
    
    def _apply_diversification_constraints(self, 
                                         weights: Dict[str, float],
                                         stock_selection: Dict[str, List[str]]) -> Dict[str, float]:
        """Apply diversification constraints to portfolio weights."""
        
        # Track allocation by asset class and sector
        asset_class_allocation = {}
        sector_allocation = {}
        
        for symbol, weight in weights.items():
            asset_class = self._get_asset_class_for_symbol(symbol, stock_selection)
            asset_class_allocation[asset_class] = asset_class_allocation.get(asset_class, 0) + weight
        
        # Adjust weights if constraints are violated
        adjusted_weights = {}
        for symbol, weight in weights.items():
            asset_class = self._get_asset_class_for_symbol(symbol, stock_selection)
            
            # Apply position size constraint
            adjusted_weight = min(weight, self.max_position_size)
            
            # Apply asset class constraint
            if asset_class_allocation[asset_class] > self.max_asset_class_allocation:
                scaling_factor = self.max_asset_class_allocation / asset_class_allocation[asset_class]
                adjusted_weight *= scaling_factor
            
            adjusted_weights[symbol] = adjusted_weight
        
        return adjusted_weights
    
    def _get_asset_class_for_symbol(self, symbol: str, stock_selection: Dict[str, List[str]]) -> str:
        """Get asset class for a symbol."""
        for asset_class, stocks in stock_selection.items():
            if symbol in stocks:
                return asset_class
        return 'unknown'
    
    def calculate_diversification_metrics(self) -> DiversificationMetrics:
        """Calculate comprehensive diversification metrics."""
        
        if not self.target_positions:
            return DiversificationMetrics(0, {}, {}, 0, 0, 1.0)
        
        # Asset class distribution
        asset_class_dist = {}
        sector_dist = {}
        max_weight = 0
        
        for position in self.target_positions.values():
            # Asset class distribution
            asset_class_dist[position.asset_class] = asset_class_dist.get(position.asset_class, 0) + position.target_weight
            
            # Sector distribution
            sector_dist[position.sector] = sector_dist.get(position.sector, 0) + position.target_weight
            
            # Max position weight
            max_weight = max(max_weight, position.target_weight)
        
        # Calculate diversification score (Herfindahl-Hirschman Index)
        hhi = sum(weight**2 for weight in asset_class_dist.values())
        diversification_score = 1 - hhi
        
        # Risk concentration (maximum allocation)
        risk_concentration = max(asset_class_dist.values()) if asset_class_dist else 1.0
        
        return DiversificationMetrics(
            total_positions=len(self.target_positions),
            asset_class_distribution=asset_class_dist,
            sector_distribution=sector_dist,
            max_position_weight=max_weight,
            diversification_score=diversification_score,
            risk_concentration=risk_concentration
        )
    
    async def generate_rebalancing_orders(self, 
                                        current_portfolio: Dict,
                                        portfolio_value: float) -> List[Dict]:
        """Generate orders to rebalance portfolio toward target allocation."""
        
        orders = []
        
        # Update current positions
        self._update_current_positions(current_portfolio, portfolio_value)
        
        # Compare target vs current and generate orders
        for symbol, target_position in self.target_positions.items():
            current_position = self.current_positions.get(symbol)
            
            if not current_position:
                # New position - buy
                if target_position.quantity > 0:
                    orders.append({
                        'symbol': symbol,
                        'side': 'buy',
                        'quantity': target_position.quantity,
                        'order_type': 'market',
                        'reason': f'New position - target weight {target_position.target_weight:.2%}'
                    })
            else:
                # Existing position - rebalance
                quantity_diff = target_position.quantity - current_position.quantity
                
                if abs(quantity_diff) > 0:  # Only rebalance if meaningful difference
                    side = 'buy' if quantity_diff > 0 else 'sell'
                    orders.append({
                        'symbol': symbol,
                        'side': side,
                        'quantity': abs(quantity_diff),
                        'order_type': 'market',
                        'reason': f'Rebalance - current {current_position.current_weight:.2%} to target {target_position.target_weight:.2%}'
                    })
        
        # Check for positions to close (not in target)
        for symbol, current_position in self.current_positions.items():
            if symbol not in self.target_positions and current_position.quantity > 0:
                orders.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': current_position.quantity,
                    'order_type': 'market',
                    'reason': 'Close position - not in target portfolio'
                })
        
        logger.info(f"Generated {len(orders)} rebalancing orders")
        return orders
    
    def _update_current_positions(self, current_portfolio: Dict, portfolio_value: float):
        """Update current positions from portfolio data."""
        self.current_positions = {}
        
        for symbol, data in current_portfolio.items():
            self.current_positions[symbol] = PortfolioPosition(
                symbol=symbol,
                asset_class=data.get('asset_class', 'unknown'),
                sector=data.get('sector', 'Unknown'),
                target_weight=0.0,
                current_weight=data.get('market_value', 0) / portfolio_value,
                quantity=data.get('quantity', 0),
                market_value=data.get('market_value', 0),
                unrealized_pnl=data.get('unrealized_pnl', 0),
                confidence_score=50.0
            )

# Global instance
diversified_portfolio_manager = DiversifiedPortfolioManager()

if __name__ == "__main__":
    # Test the diversified portfolio construction
    async def test_portfolio_construction():
        manager = DiversifiedPortfolioManager()
        
        print("🔄 Testing diversified portfolio construction...")
        portfolio = await manager.construct_diversified_portfolio(100000.0, "moderate")
        
        print(f"\n📊 Constructed portfolio with {len(portfolio)} positions:")
        for symbol, position in list(portfolio.items())[:10]:  # Show first 10
            print(f"  {symbol}: {position.target_weight:.2%} - {position.asset_class}")
        
        metrics = manager.calculate_diversification_metrics()
        print(f"\n📈 Diversification Metrics:")
        print(f"  Positions: {metrics.total_positions}")
        print(f"  Max Position: {metrics.max_position_weight:.2%}")
        print(f"  Diversification Score: {metrics.diversification_score:.3f}")
        print(f"  Asset Class Distribution:")
        for asset_class, allocation in metrics.asset_class_distribution.items():
            print(f"    {asset_class}: {allocation:.2%}")
    
    asyncio.run(test_portfolio_construction())