#!/usr/bin/env python3
"""
Strategy Router
===============

Central routing hub that receives high-level strategy intents and delegates
execution to the appropriate trading agents (equities, options, crypto).

This acts as the orchestration layer between:
- RL/Sentiment Analysis (generates intents)  
- Trading Agents (execute specific trades)
- Audit/Logging (tracks all decisions)

Key Features:
- Multi-asset class support (equity, options, crypto)
- Strategy intent translation
- Comprehensive audit logging
- Risk validation routing
- Performance tracking across agents
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class AssetClass(Enum):
    """Supported asset classes for trading."""
    EQUITY = "equity"
    OPTION = "option" 
    CRYPTO = "crypto"
    FUTURES = "futures"  # For future extension

class Intent(Enum):
    """High-level trading intents that can be routed."""
    BUY = "buy"
    SELL = "sell"
    HEDGE = "hedge"
    YIELD = "yield"
    SPECULATE = "speculate"
    VOLATILITY = "volatility"
    SPREAD = "spread"
    REBALANCE = "rebalance"

@dataclass
class TradingIntent:
    """
    High-level trading intent that gets routed to appropriate agents.
    
    This abstracts away the complexity of different asset classes and
    provides a unified interface for the RL/sentiment system.
    """
    asset_class: AssetClass
    intent: Intent
    symbol: str
    quantity: Union[int, float]
    confidence: float  # 0.0 to 1.0
    rationale: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Optional fields for specific strategies
    strike: Optional[float] = None
    expiry: Optional[str] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Risk parameters
    max_risk: Optional[float] = None
    expected_return: Optional[float] = None
    
    # Metadata
    source_system: str = "RL_Agent"
    priority: int = 1  # 1=low, 5=high

@dataclass
class RoutingResult:
    """Result of routing and executing a trading intent."""
    intent: TradingIntent
    routing_decision: str
    agent_used: str
    execution_result: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

class StrategyRouter:
    """
    Central strategy routing system that orchestrates all trading agents.
    
    This is the main interface between the RL/sentiment layer and the
    execution layer. It handles:
    - Intent classification and routing
    - Agent coordination
    - Risk validation
    - Audit logging
    - Performance tracking
    """
    
    def __init__(self, equities_agent=None, options_agent=None, crypto_agent=None):
        """Initialize router with available trading agents."""
        self.equities_agent = equities_agent
        self.options_agent = options_agent
        self.crypto_agent = crypto_agent
        
        # Tracking and audit
        self.routing_history: List[RoutingResult] = []
        self.agent_performance: Dict[str, Dict] = {}
        self.intent_statistics: Dict[str, int] = {}
        
        # Configuration
        self.audit_log_path = Path("/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system/logs/strategy_router_audit.json")
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Strategy Router initialized")
        logger.info(f"   Equities Agent: {'✅' if equities_agent else '❌'}")
        logger.info(f"   Options Agent: {'✅' if options_agent else '❌'}")  
        logger.info(f"   Crypto Agent: {'✅' if crypto_agent else '❌'}")

    async def route_intent(self, intent: TradingIntent) -> RoutingResult:
        """
        Route a high-level trading intent to the appropriate agent.
        
        Args:
            intent: TradingIntent object describing what to do
            
        Returns:
            RoutingResult with execution details
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"🎯 Routing intent: {intent.intent.value} {intent.asset_class.value}")
            logger.info(f"   Symbol: {intent.symbol}, Qty: {intent.quantity}")
            logger.info(f"   Confidence: {intent.confidence:.2f}, Rationale: {intent.rationale}")
            
            # Update statistics
            intent_key = f"{intent.asset_class.value}_{intent.intent.value}"
            self.intent_statistics[intent_key] = self.intent_statistics.get(intent_key, 0) + 1
            
            # Route based on asset class
            if intent.asset_class == AssetClass.EQUITY:
                result = await self._route_equity_intent(intent)
                agent_used = "equities"
                
            elif intent.asset_class == AssetClass.OPTION:
                result = await self._route_options_intent(intent)
                agent_used = "options"
                
            elif intent.asset_class == AssetClass.CRYPTO:
                result = await self._route_crypto_intent(intent)
                agent_used = "crypto"
                
            else:
                raise ValueError(f"Unsupported asset class: {intent.asset_class}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create routing result
            routing_result = RoutingResult(
                intent=intent,
                routing_decision=f"Routed to {agent_used} agent",
                agent_used=agent_used,
                execution_result=result,
                success=result.get('error') is None,
                execution_time=execution_time
            )
            
            # Update agent performance tracking
            self._update_agent_performance(agent_used, routing_result)
            
            # Record in history
            self.routing_history.append(routing_result)
            
            # Audit logging
            await self._log_routing_audit(routing_result)
            
            logger.info(f"✅ Intent routed successfully in {execution_time:.3f}s")
            return routing_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            error_result = RoutingResult(
                intent=intent,
                routing_decision="ROUTING_FAILED",
                agent_used="none",
                execution_result={"error": str(e)},
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
            
            self.routing_history.append(error_result)
            await self._log_routing_audit(error_result)
            
            logger.error(f"❌ Intent routing failed: {e}")
            return error_result

    async def _route_equity_intent(self, intent: TradingIntent) -> Dict[str, Any]:
        """Route intent to equities trading agent."""
        if not self.equities_agent:
            raise RuntimeError("Equities agent not configured")
        
        try:
            # Convert intent to equities agent format
            if intent.intent == Intent.BUY:
                return await self._execute_equity_trade(intent, "buy")
            elif intent.intent == Intent.SELL:
                return await self._execute_equity_trade(intent, "sell")
            elif intent.intent == Intent.REBALANCE:
                return await self._execute_equity_rebalance(intent)
            else:
                raise ValueError(f"Unsupported equity intent: {intent.intent}")
                
        except Exception as e:
            return {"error": f"Equity routing failed: {str(e)}"}

    async def _route_options_intent(self, intent: TradingIntent) -> Dict[str, Any]:
        """Route intent to options trading agent."""
        if not self.options_agent:
            raise RuntimeError("Options agent not configured")
        
        try:
            # Validate required options parameters
            if not intent.strike or not intent.expiry:
                raise ValueError("Options trades require strike and expiry parameters")
            
            # Route based on specific options intent
            if intent.intent == Intent.HEDGE:
                return await self.options_agent.hedge_with_put(
                    stock=intent.symbol,
                    expiry=intent.expiry,
                    strike=intent.strike,
                    contracts=int(intent.quantity),
                    rationale=intent.rationale
                )
                
            elif intent.intent == Intent.YIELD:
                return await self.options_agent.write_covered_call(
                    stock=intent.symbol,
                    expiry=intent.expiry,
                    strike=intent.strike,
                    contracts=int(intent.quantity),
                    rationale=intent.rationale
                )
                
            elif intent.intent == Intent.SPECULATE:
                return await self.options_agent.speculate_with_call(
                    stock=intent.symbol,
                    expiry=intent.expiry,
                    strike=intent.strike,
                    contracts=int(intent.quantity),
                    rationale=intent.rationale
                )
                
            elif intent.intent == Intent.VOLATILITY:
                return await self.options_agent.volatility_straddle(
                    stock=intent.symbol,
                    expiry=intent.expiry,
                    strike=intent.strike,
                    contracts=int(intent.quantity),
                    rationale=intent.rationale
                )
                
            else:
                raise ValueError(f"Unsupported options intent: {intent.intent}")
                
        except Exception as e:
            return {"error": f"Options routing failed: {str(e)}"}

    async def _route_crypto_intent(self, intent: TradingIntent) -> Dict[str, Any]:
        """Route intent to crypto trading agent."""
        if not self.crypto_agent:
            # For now, return placeholder - crypto agent not implemented yet
            return {
                "status": "placeholder",
                "message": "Crypto agent not yet implemented",
                "intent": intent.intent.value,
                "symbol": intent.symbol,
                "quantity": intent.quantity
            }
        
        # When crypto agent is implemented:
        # return await self.crypto_agent.execute_trade(intent)

    async def _execute_equity_trade(self, intent: TradingIntent, side: str) -> Dict[str, Any]:
        """Execute equity trade through existing trading infrastructure."""
        try:
            # Use existing trading engine
            from core.trading_engine import trading_engine
            from core.market_intelligence import AnalysisResult, SignalType, ConfidenceLevel
            
            # Create analysis result for existing system
            signal_type = SignalType.BUY if side == "buy" else SignalType.SELL
            confidence_level = ConfidenceLevel.HIGH if intent.confidence > 0.8 else ConfidenceLevel.MEDIUM
            
            analysis = AnalysisResult(
                symbol=intent.symbol,
                signal=signal_type,
                confidence=confidence_level,
                score=intent.confidence,
                reasoning=intent.rationale,
                timestamp=intent.timestamp
            )
            
            # Execute through trading engine
            order_result = await trading_engine.execute_analysis_signal(analysis)
            
            return {
                "agent": "equities",
                "symbol": intent.symbol,
                "side": side,
                "quantity": intent.quantity,
                "order_result": order_result.__dict__ if order_result else None,
                "rationale": intent.rationale
            }
            
        except Exception as e:
            return {"error": f"Equity execution failed: {str(e)}"}

    async def _execute_equity_rebalance(self, intent: TradingIntent) -> Dict[str, Any]:
        """Execute portfolio rebalancing."""
        try:
            # Use existing portfolio balancer
            from core.portfolio_balancer import IntelligentPortfolioBalancer
            
            balancer = IntelligentPortfolioBalancer()
            # This would integrate with the existing rebalancing logic
            
            return {
                "agent": "equities", 
                "action": "rebalance",
                "symbol": intent.symbol,
                "rationale": intent.rationale,
                "status": "rebalance_initiated"
            }
            
        except Exception as e:
            return {"error": f"Rebalancing failed: {str(e)}"}

    def _update_agent_performance(self, agent: str, result: RoutingResult):
        """Update performance tracking for an agent."""
        if agent not in self.agent_performance:
            self.agent_performance[agent] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_execution_time": 0.0,
                "last_used": None
            }
        
        perf = self.agent_performance[agent]
        perf["total_requests"] += 1
        perf["last_used"] = result.timestamp.isoformat()
        
        if result.success:
            perf["successful_requests"] += 1
        else:
            perf["failed_requests"] += 1
        
        # Update average execution time
        if result.execution_time:
            current_avg = perf["average_execution_time"]
            total_requests = perf["total_requests"]
            perf["average_execution_time"] = (
                (current_avg * (total_requests - 1) + result.execution_time) / total_requests
            )

    async def _log_routing_audit(self, result: RoutingResult):
        """Log routing decision to audit trail."""
        try:
            audit_entry = {
                "timestamp": result.timestamp.isoformat(),
                "intent": {
                    "asset_class": result.intent.asset_class.value,
                    "intent_type": result.intent.intent.value,
                    "symbol": result.intent.symbol,
                    "quantity": result.intent.quantity,
                    "confidence": result.intent.confidence,
                    "rationale": result.intent.rationale,
                    "source_system": result.intent.source_system
                },
                "routing": {
                    "decision": result.routing_decision,
                    "agent_used": result.agent_used,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "error": result.error_message
                },
                "execution_result": result.execution_result
            }
            
            # Append to audit log file
            audit_entries = []
            if self.audit_log_path.exists():
                with open(self.audit_log_path, 'r') as f:
                    try:
                        audit_entries = json.load(f)
                    except json.JSONDecodeError:
                        audit_entries = []
            
            audit_entries.append(audit_entry)
            
            # Keep only last 1000 entries
            if len(audit_entries) > 1000:
                audit_entries = audit_entries[-1000:]
            
            with open(self.audit_log_path, 'w') as f:
                json.dump(audit_entries, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "routing_statistics": {
                "total_intents_processed": len(self.routing_history),
                "successful_routes": sum(1 for r in self.routing_history if r.success),
                "failed_routes": sum(1 for r in self.routing_history if not r.success),
                "success_rate": (
                    sum(1 for r in self.routing_history if r.success) / len(self.routing_history)
                    if self.routing_history else 0.0
                )
            },
            "intent_breakdown": self.intent_statistics,
            "agent_performance": self.agent_performance,
            "recent_routes": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "asset_class": r.intent.asset_class.value,
                    "intent": r.intent.intent.value,
                    "symbol": r.intent.symbol,
                    "agent": r.agent_used,
                    "success": r.success,
                    "execution_time": r.execution_time
                }
                for r in self.routing_history[-10:]  # Last 10 routes
            ]
        }

# Integration helpers for existing system

async def create_options_intent(symbol: str, strategy: str, strike: float, expiry: str,
                              contracts: int, confidence: float, rationale: str) -> TradingIntent:
    """Helper to create options trading intents."""
    intent_mapping = {
        "hedge": Intent.HEDGE,
        "yield": Intent.YIELD,
        "speculate": Intent.SPECULATE,
        "volatility": Intent.VOLATILITY
    }
    
    return TradingIntent(
        asset_class=AssetClass.OPTION,
        intent=intent_mapping.get(strategy, Intent.SPECULATE),
        symbol=symbol,
        quantity=contracts,
        confidence=confidence,
        rationale=rationale,
        strike=strike,
        expiry=expiry,
        source_system="RL_Sentiment_Analysis"
    )

async def create_equity_intent(symbol: str, action: str, quantity: int,
                             confidence: float, rationale: str) -> TradingIntent:
    """Helper to create equity trading intents."""
    intent_mapping = {
        "buy": Intent.BUY,
        "sell": Intent.SELL,
        "rebalance": Intent.REBALANCE
    }
    
    return TradingIntent(
        asset_class=AssetClass.EQUITY,
        intent=intent_mapping.get(action, Intent.BUY),
        symbol=symbol,
        quantity=quantity,
        confidence=confidence,
        rationale=rationale,
        source_system="RL_Sentiment_Analysis"
    )

# Example integration with existing system
async def example_routing_integration():
    """Example of how to integrate the strategy router."""
    
    # Initialize router with agents
    from agents.options_agent import options_agent
    
    router = StrategyRouter(
        equities_agent=None,  # Will use existing trading engine
        options_agent=options_agent,
        crypto_agent=None
    )
    
    # Example: RL system determines bearish sentiment on AAPL
    hedge_intent = await create_options_intent(
        symbol="AAPL",
        strategy="hedge",
        strike=150.0,
        expiry="250117",
        contracts=1,
        confidence=0.85,
        rationale="RL detected bearish sentiment with 85% confidence, hedging long AAPL position"
    )
    
    # Route and execute
    result = await router.route_intent(hedge_intent)
    print(f"Routing result: {result.success}")
    print(f"Agent used: {result.agent_used}")
    print(f"Execution time: {result.execution_time:.3f}s")
    
    # Example: Generate income on MSFT
    income_intent = await create_options_intent(
        symbol="MSFT",
        strategy="yield",
        strike=420.0,
        expiry="241220",
        contracts=2,
        confidence=0.7,
        rationale="High IV environment detected, generating income on MSFT holdings"
    )
    
    income_result = await router.route_intent(income_intent)
    print(f"Income strategy result: {income_result.success}")
    
    # Get performance report
    report = router.get_performance_report()
    print(f"Router performance: {report['routing_statistics']['success_rate']:.1%}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_routing_integration())