"""
Core trading system components.

This package contains the essential, consolidated components of the trading system:
- market_intelligence: Unified market data and analysis
- trading_engine: Unified trading execution and portfolio management
"""

from .market_intelligence import (
    market_intelligence,
    get_market_data,
    analyze_stock,
    analyze_portfolio,
    MarketData,
    AnalysisResult,
    SignalType,
    ConfidenceLevel
)

from .trading_engine import (
    trading_engine,
    execute_signal,
    execute_pairs_trade,
    rebalance_portfolio,
    get_portfolio_metrics,
    TradingOrder,
    Position,
    PortfolioMetrics,
    OrderStatus,
    StrategyType
)

__all__ = [
    # Market Intelligence
    'market_intelligence',
    'get_market_data', 
    'analyze_stock',
    'analyze_portfolio',
    'MarketData',
    'AnalysisResult', 
    'SignalType',
    'ConfidenceLevel',
    # Trading Engine
    'trading_engine',
    'execute_signal',
    'execute_pairs_trade', 
    'rebalance_portfolio',
    'get_portfolio_metrics',
    'TradingOrder',
    'Position',
    'PortfolioMetrics',
    'OrderStatus',
    'StrategyType'
]