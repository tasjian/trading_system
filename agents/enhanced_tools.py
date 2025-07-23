"""Enhanced tool system for trading agents using ReAct patterns."""

from typing import Dict, List, Optional, Any, Literal
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class ToolContext:
    """Rich context for tool execution."""
    agent_name: str
    reasoning_step: str
    confidence: float
    retry_count: int = 0
    max_retries: int = 3
    
@dataclass
class ToolResult:
    """Enhanced tool result with metadata."""
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    context: Optional[ToolContext] = None
    metadata: Dict[str, Any] = None

class TradingToolRegistry:
    """Registry for trading-specific tools with ReAct integration."""
    
    def __init__(self):
        self.tools = {}
        self.tool_usage_stats = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default trading tools."""
        
        @tool
        def analyze_market_conditions(
            symbols: List[str],
            timeframe: Literal["1min", "5min", "15min", "1hour", "1day"] = "1day",
            indicators: List[str] = ["rsi", "macd", "sma", "volume"],
            reasoning: str = "Market analysis for trading decisions"
        ) -> Dict[str, Any]:
            """Analyze market conditions for given symbols with technical indicators."""
            try:
                from tools.alpaca_client import alpaca_client
                
                analysis_results = {}
                for symbol in symbols[:10]:  # Limit to prevent overload
                    try:
                        # Get market data
                        market_data = alpaca_client.get_market_data(symbol, timeframe, limit=100)
                        
                        if market_data.empty:
                            continue
                        
                        # Basic technical analysis
                        latest = market_data.iloc[-1]
                        prev = market_data.iloc[-2] if len(market_data) > 1 else latest
                        
                        # Calculate indicators
                        indicator_data = {}
                        if "rsi" in indicators:
                            # Simple RSI approximation
                            price_changes = market_data['close'].diff()
                            gains = price_changes.where(price_changes > 0, 0)
                            losses = -price_changes.where(price_changes < 0, 0)
                            avg_gain = gains.rolling(14).mean().iloc[-1]
                            avg_loss = losses.rolling(14).mean().iloc[-1]
                            rs = avg_gain / (avg_loss if avg_loss != 0 else 0.01)
                            indicator_data["rsi"] = 100 - (100 / (1 + rs))
                        
                        if "volume" in indicators:
                            indicator_data["volume_ratio"] = latest['volume'] / market_data['volume'].mean()
                        
                        if "sma" in indicators:
                            indicator_data["sma_20"] = market_data['close'].rolling(20).mean().iloc[-1]
                            indicator_data["price_vs_sma"] = latest['close'] / indicator_data["sma_20"]
                        
                        analysis_results[symbol] = {
                            "current_price": float(latest['close']),
                            "price_change": float(latest['close'] - prev['close']),
                            "price_change_pct": float((latest['close'] - prev['close']) / prev['close'] * 100),
                            "volume": float(latest['volume']),
                            "indicators": indicator_data,
                            "timestamp": datetime.now(),
                            "analysis_reasoning": reasoning
                        }
                        
                    except Exception as e:
                        logger.warning(f"Failed to analyze {symbol}: {e}")
                        continue
                
                return {
                    "status": "success",
                    "analysis": analysis_results,
                    "symbols_analyzed": len(analysis_results),
                    "timestamp": datetime.now()
                }
                
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now()
                }
        
        @tool
        def assess_portfolio_risk(
            positions: Dict[str, Dict[str, Any]],
            portfolio_value: float,
            risk_tolerance: Literal["conservative", "moderate", "aggressive"] = "moderate",
            reasoning: str = "Portfolio risk assessment"
        ) -> Dict[str, Any]:
            """Assess current portfolio risk metrics and exposure."""
            try:
                if not positions or portfolio_value <= 0:
                    return {"status": "error", "error": "Invalid portfolio data"}
                
                # Calculate concentration risk
                total_exposure = sum(abs(pos.get("market_value", 0)) for pos in positions.values())
                concentration_risk = {}
                sector_exposure = {}
                
                for symbol, position in positions.items():
                    market_value = abs(position.get("market_value", 0))
                    exposure_pct = market_value / portfolio_value if portfolio_value > 0 else 0
                    concentration_risk[symbol] = exposure_pct
                    
                    # Sector exposure (simplified)
                    sector = position.get("sector", "Unknown")
                    sector_exposure[sector] = sector_exposure.get(sector, 0) + exposure_pct
                
                # Risk thresholds based on tolerance
                risk_thresholds = {
                    "conservative": {"max_position": 0.05, "max_sector": 0.20},
                    "moderate": {"max_position": 0.08, "max_sector": 0.25},
                    "aggressive": {"max_position": 0.15, "max_sector": 0.35}
                }
                
                thresholds = risk_thresholds[risk_tolerance]
                
                # Identify risk violations
                risk_violations = []
                for symbol, exposure in concentration_risk.items():
                    if exposure > thresholds["max_position"]:
                        risk_violations.append(f"Position {symbol}: {exposure:.1%} exceeds {thresholds['max_position']:.1%}")
                
                for sector, exposure in sector_exposure.items():
                    if exposure > thresholds["max_sector"]:
                        risk_violations.append(f"Sector {sector}: {exposure:.1%} exceeds {thresholds['max_sector']:.1%}")
                
                # Calculate diversification score
                num_positions = len(positions)
                ideal_positions = 35  # Target diversification
                diversification_score = min(1.0, num_positions / ideal_positions)
                
                # Overall risk level
                risk_score = len(risk_violations) / max(1, len(positions)) * 100
                risk_level = "low" if risk_score < 10 else "medium" if risk_score < 25 else "high"
                
                return {
                    "status": "success",
                    "risk_assessment": {
                        "overall_risk_level": risk_level,
                        "risk_score": risk_score,
                        "diversification_score": diversification_score,
                        "concentration_risk": concentration_risk,
                        "sector_exposure": sector_exposure,
                        "violations": risk_violations,
                        "total_positions": num_positions,
                        "total_exposure": total_exposure,
                        "cash_ratio": (portfolio_value - total_exposure) / portfolio_value if portfolio_value > 0 else 0
                    },
                    "recommendations": self._generate_risk_recommendations(risk_violations, diversification_score),
                    "timestamp": datetime.now(),
                    "reasoning": reasoning
                }
                
            except Exception as e:
                return {"status": "error", "error": str(e)}
        
        @tool
        def execute_trading_decision(
            action: Literal["buy", "sell", "hold", "rebalance"],
            symbol: str,
            quantity: Optional[float] = None,
            price_limit: Optional[float] = None,
            reasoning: str = "Trading decision execution",
            risk_check: bool = True
        ) -> Dict[str, Any]:
            """Execute a trading decision with risk checks and validation."""
            try:
                from tools.alpaca_client import alpaca_client
                
                # Pre-execution validation
                if risk_check:
                    # Check market status
                    if not alpaca_client.is_market_open():
                        return {
                            "status": "rejected",
                            "reason": "Market closed",
                            "timestamp": datetime.now()
                        }
                    
                    # Get account info for risk check
                    account_info = alpaca_client.get_account_info()
                    if account_info.get("trading_blocked", False):
                        return {
                            "status": "rejected",
                            "reason": "Trading blocked on account",
                            "timestamp": datetime.now()
                        }
                
                # Handle different actions
                if action == "hold":
                    return {
                        "status": "success",
                        "action": "hold",
                        "symbol": symbol,
                        "reasoning": reasoning,
                        "timestamp": datetime.now()
                    }
                
                if action in ["buy", "sell"] and quantity:
                    # Execute trade
                    order_result = alpaca_client.place_order(
                        symbol=symbol,
                        qty=quantity,
                        side=action,
                        order_type="limit" if price_limit else "market",
                        limit_price=price_limit,
                        time_in_force="day"
                    )
                    
                    return {
                        "status": "success",
                        "action": action,
                        "symbol": symbol,
                        "quantity": quantity,
                        "order_id": order_result.get("id"),
                        "order_status": order_result.get("status"),
                        "reasoning": reasoning,
                        "timestamp": datetime.now()
                    }
                
                elif action == "rebalance":
                    # Implement rebalancing logic
                    return {
                        "status": "pending",
                        "action": "rebalance",
                        "message": "Rebalancing logic to be implemented",
                        "timestamp": datetime.now()
                    }
                
                else:
                    return {
                        "status": "error",
                        "error": f"Invalid action or missing parameters: {action}",
                        "timestamp": datetime.now()
                    }
                    
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now()
                }
        
        @tool 
        def research_market_sentiment(
            symbols: List[str],
            sources: List[Literal["news", "social", "analyst"]] = ["news", "social"],
            lookback_hours: int = 24,
            reasoning: str = "Market sentiment research"
        ) -> Dict[str, Any]:
            """Research market sentiment from multiple sources."""
            try:
                # This would integrate with sentiment analysis
                from agents.sentiment_agent import sentiment_agent
                
                sentiment_results = {}
                
                # Run sentiment analysis for symbols
                if len(symbols) > 0:
                    sentiment_outputs = asyncio.run(
                        sentiment_agent.run_sentiment_cycle(
                            tickers=symbols[:5],  # Limit to prevent rate limits
                            platforms=['reddit'] if 'social' in sources else []
                        )
                    )
                    
                    for output in sentiment_outputs:
                        sentiment_results[output.ticker] = {
                            "sentiment": output.overall_sentiment.value,
                            "confidence": output.confidence,
                            "volume": output.volume,
                            "trend": output.trend_signal.value,
                            "key_insights": output.key_insights,
                            "timestamp": output.timestamp
                        }
                
                return {
                    "status": "success",
                    "sentiment_analysis": sentiment_results,
                    "sources_used": sources,
                    "lookback_period": lookback_hours,
                    "reasoning": reasoning,
                    "timestamp": datetime.now()
                }
                
            except Exception as e:
                return {
                    "status": "error", 
                    "error": str(e),
                    "timestamp": datetime.now()
                }
        
        # Register tools
        self.tools = {
            "analyze_market_conditions": analyze_market_conditions,
            "assess_portfolio_risk": assess_portfolio_risk,
            "execute_trading_decision": execute_trading_decision,
            "research_market_sentiment": research_market_sentiment,
        }
    
    def _generate_risk_recommendations(self, violations: List[str], diversification_score: float) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if violations:
            recommendations.append("Consider reducing overweight positions to improve risk profile")
        
        if diversification_score < 0.7:
            recommendations.append(f"Portfolio diversification below target (score: {diversification_score:.2f})")
            recommendations.append("Consider adding positions across different sectors/asset classes")
        
        if len(violations) > 3:
            recommendations.append("Multiple risk violations detected - consider portfolio rebalancing")
        
        return recommendations
    
    def get_tool(self, name: str):
        """Get tool by name."""
        return self.tools.get(name)
    
    def get_all_tools(self) -> List:
        """Get all registered tools."""
        return list(self.tools.values())
    
    def execute_tool_with_context(self, tool_name: str, context: ToolContext, **kwargs) -> ToolResult:
        """Execute tool with enhanced context and error handling."""
        start_time = datetime.now()
        
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool {tool_name} not found",
                context=context
            )
        
        try:
            # Update usage stats
            if tool_name not in self.tool_usage_stats:
                self.tool_usage_stats[tool_name] = {"calls": 0, "successes": 0, "failures": 0}
            
            self.tool_usage_stats[tool_name]["calls"] += 1
            
            # Execute tool
            tool = self.tools[tool_name]
            result = tool.invoke(kwargs)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Check if result indicates success
            success = True
            error = None
            
            if isinstance(result, dict):
                if result.get("status") == "error":
                    success = False
                    error = result.get("error", "Unknown error")
                elif result.get("status") == "rejected":
                    success = False
                    error = result.get("reason", "Request rejected")
            
            # Update stats
            if success:
                self.tool_usage_stats[tool_name]["successes"] += 1
            else:
                self.tool_usage_stats[tool_name]["failures"] += 1
            
            return ToolResult(
                success=success,
                data=result,
                error=error,
                execution_time=execution_time,
                context=context,
                metadata={"tool_name": tool_name, "usage_stats": self.tool_usage_stats[tool_name]}
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.tool_usage_stats[tool_name]["failures"] += 1
            
            logger.error(f"Tool {tool_name} execution failed: {e}")
            
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time,
                context=context
            )

# Global tool registry
trading_tool_registry = TradingToolRegistry()