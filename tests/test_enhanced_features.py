"""Tests for enhanced market analysis and quantitative trading features."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

def test_market_analysis_agents():
    """Test market analysis agents initialization."""
    from agents.market_analysis import (
        TechnicalAnalysisAgent, FundamentalAnalysisAgent, 
        QuantitativeAnalysisAgent, market_analysis_factory
    )
    
    # Test agent initialization
    tech_agent = TechnicalAnalysisAgent()
    assert tech_agent.name == "Technical Analysis Agent"
    assert tech_agent.lookback_period > 0
    
    fund_agent = FundamentalAnalysisAgent()
    assert fund_agent.name == "Fundamental Analysis Agent"
    
    quant_agent = QuantitativeAnalysisAgent()
    assert quant_agent.name == "Quantitative Analysis Agent"
    
    # Test factory
    assert market_analysis_factory is not None
    assert hasattr(market_analysis_factory, 'analyze_symbols')

def test_technical_indicators():
    """Test technical indicator calculations."""
    from agents.market_analysis import TechnicalAnalysisAgent
    
    agent = TechnicalAnalysisAgent()
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = np.random.randn(100).cumsum() + 100
    volumes = np.random.randint(1000, 10000, 100)
    
    df = pd.DataFrame({
        'close': prices,
        'high': prices + np.random.rand(100),
        'low': prices - np.random.rand(100),
        'volume': volumes
    }, index=dates)
    
    # Test indicator calculation
    indicators = agent._calculate_indicators(df)
    
    # Verify indicators exist
    assert 'current_price' in indicators
    assert 'sma_20' in indicators
    assert 'rsi' in indicators
    assert 'volatility_20' in indicators
    assert 'bb_upper' in indicators
    assert 'bb_lower' in indicators

def test_trading_strategies():
    """Test quantitative trading strategies."""
    from tools.advanced_trading import MomentumStrategy, MeanReversionStrategy
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = np.random.randn(100).cumsum() + 100
    
    df = pd.DataFrame({
        'close': prices,
        'high': prices + np.random.rand(100),
        'low': prices - np.random.rand(100),
    }, index=dates)
    
    # Test momentum strategy
    momentum_strategy = MomentumStrategy(lookback=20, threshold=0.02)
    momentum_signals = momentum_strategy.generate_signals(df)
    
    assert len(momentum_signals) == len(df)
    assert momentum_signals.min() >= -1
    assert momentum_signals.max() <= 1
    
    # Test mean reversion strategy
    mean_rev_strategy = MeanReversionStrategy(window=20, std_threshold=2.0)
    mean_rev_signals = mean_rev_strategy.generate_signals(df)
    
    assert len(mean_rev_signals) == len(df)
    assert mean_rev_signals.min() >= -1
    assert mean_rev_signals.max() <= 1

def test_portfolio_optimization():
    """Test portfolio optimization functionality."""
    from tools.advanced_trading import PortfolioOptimizer
    
    optimizer = PortfolioOptimizer()
    
    # Create sample return data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    returns_data = {
        'AAPL': pd.Series(np.random.randn(252) * 0.02, index=dates),
        'MSFT': pd.Series(np.random.randn(252) * 0.02, index=dates),
        'SPY': pd.Series(np.random.randn(252) * 0.015, index=dates)
    }
    
    # Test optimization
    optimal_weights = optimizer.optimize_portfolio(returns_data)
    
    assert isinstance(optimal_weights, dict)
    if optimal_weights:
        assert len(optimal_weights) <= len(returns_data)
        # Weights should be positive and sum to approximately 1
        total_weight = sum(optimal_weights.values())
        assert 0.8 <= total_weight <= 1.2  # Allow some tolerance

def test_risk_management():
    """Test advanced risk management features."""
    from tools.advanced_trading import RiskManagementEngine
    
    risk_manager = RiskManagementEngine()
    
    # Test position sizing
    position_size = risk_manager.calculate_position_size(
        symbol="AAPL",
        entry_price=150.0,
        stop_loss=147.0,  # 2% stop loss
        portfolio_value=100000.0,
        signal_strength=0.8
    )
    
    assert position_size >= 0
    assert position_size < 1000  # Reasonable position size
    
    # Test portfolio heat calculation
    sample_positions = [
        {"market_value": 10000},
        {"market_value": 15000},
        {"market_value": 8000}
    ]
    
    heat = risk_manager.calculate_portfolio_heat(sample_positions)
    assert 0 <= heat <= 1  # Should be between 0 and 100%

def test_backtesting_engine():
    """Test backtesting functionality."""
    from tools.advanced_trading import AdvancedBacktester, MomentumStrategy
    
    backtester = AdvancedBacktester()
    strategy = MomentumStrategy()
    
    # Mock the alpaca client for backtesting
    with patch('tools.advanced_trading.alpaca_client') as mock_client:
        # Create mock historical data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        mock_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        mock_client.get_market_data.return_value = mock_data
        
        # Run backtest
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 3, 31)
        
        result = backtester.backtest_strategy(strategy, "AAPL", start_date, end_date)
        
        # Verify backtest result structure
        assert hasattr(result, 'strategy_name')
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'max_drawdown')
        assert result.strategy_name == strategy.name

def test_signal_strength_enum():
    """Test signal strength enumeration."""
    from agents.market_analysis import SignalStrength, AnalysisResult
    
    # Test enum values
    assert SignalStrength.VERY_WEAK.value == 0.2
    assert SignalStrength.WEAK.value == 0.4
    assert SignalStrength.MODERATE.value == 0.6
    assert SignalStrength.STRONG.value == 0.8
    assert SignalStrength.VERY_STRONG.value == 1.0
    
    # Test analysis result creation
    result = AnalysisResult(
        symbol="AAPL",
        signal_type="momentum",
        action="buy",
        confidence=0.7,
        strength=SignalStrength.STRONG
    )
    
    assert result.symbol == "AAPL"
    assert result.action == "buy"
    assert result.confidence == 0.7
    assert result.strength == SignalStrength.STRONG

def test_enhanced_workflow_integration():
    """Test integration of enhanced features with workflow."""
    from agents.workflow import TradingWorkflow
    
    # Test workflow creation with enhanced features
    workflow = TradingWorkflow()
    
    assert workflow is not None
    assert hasattr(workflow, '_calculate_position_size')
    
    # Test position size calculation method exists
    assert callable(getattr(workflow, '_calculate_position_size', None))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])