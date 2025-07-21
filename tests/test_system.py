"""Basic tests for the trading system components."""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

# Test configuration
@pytest.mark.asyncio
async def test_config_validation():
    """Test configuration validation."""
    from config.settings import validate_settings
    
    # Should not raise exception with valid config
    try:
        validate_settings()
        assert True
    except Exception as e:
        # Expected if API keys not set
        assert "API" in str(e)

def test_alpaca_client_initialization():
    """Test Alpaca client initialization."""
    with patch('tools.alpaca_client.REST') as mock_rest:
        mock_rest.return_value.get_account.return_value = Mock(id="test_account")
        
        from tools.alpaca_client import AlpacaClient
        client = AlpacaClient()
        
        # Should initialize without error
        assert client is not None

def test_trading_state_creation():
    """Test trading state initialization."""
    from agents.state import create_initial_state
    
    state = create_initial_state("test_session")
    
    assert state["session_id"] == "test_session"
    assert "portfolio" in state
    assert "signals" in state
    assert "risk_limits" in state
    assert len(state["watchlist"]) > 0

def test_risk_monitor_initialization():
    """Test risk monitor initialization."""
    from tools.risk_controls import RiskMonitor
    
    monitor = RiskMonitor()
    
    assert monitor is not None
    assert "daily_loss_limit" in monitor.circuit_breakers
    assert monitor.risk_thresholds["max_position_size"] > 0

@pytest.mark.asyncio
async def test_workflow_creation():
    """Test LangGraph workflow creation."""
    from agents.workflow import TradingWorkflow
    
    workflow = TradingWorkflow()
    
    assert workflow is not None
    assert workflow.graph is not None
    assert workflow.app is not None

def test_trading_tools_import():
    """Test trading tools can be imported."""
    from tools.trading_tools import trading_tools
    
    assert len(trading_tools) > 0
    
    # Check key tools are present
    tool_names = [tool.name for tool in trading_tools]
    assert "get_account_info" in tool_names
    assert "place_market_order" in tool_names
    assert "calculate_portfolio_metrics" in tool_names

def test_signal_creation():
    """Test trading signal creation."""
    from agents.state import TradingSignal
    
    signal = TradingSignal(
        symbol="AAPL",
        action="buy",
        confidence=0.8,
        reasoning="Test signal"
    )
    
    assert signal.symbol == "AAPL"
    assert signal.action == "buy"
    assert signal.confidence == 0.8
    assert signal.signal_id is not None

def test_risk_alert_creation():
    """Test risk alert creation."""
    from tools.risk_controls import RiskAlert, RiskLevel
    
    alert = RiskAlert(
        level=RiskLevel.HIGH,
        message="Test alert",
        metric="test_metric",
        value=0.1,
        threshold=0.05,
        timestamp=datetime.now()
    )
    
    assert alert.level == RiskLevel.HIGH
    assert alert.message == "Test alert"
    assert not alert.resolved

if __name__ == "__main__":
    pytest.main([__file__, "-v"])