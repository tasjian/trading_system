"""Enhanced trading workflow using advanced LangGraph patterns from AAI course."""

import asyncio
import logging
from typing import Dict, Any, Literal, Optional, List
from datetime import datetime, timedelta

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, AIMessage

from agents.enhanced_state import (
    EnhancedTradingState, create_enhanced_initial_state, 
    add_agent_communication, update_agent_context,
    prioritize_signals, should_interrupt_for_human, create_human_interrupt
)
from agents.enhanced_tools import trading_tool_registry, ToolContext
from agents.agent_communication import trading_agent_team
from config.settings import settings

logger = logging.getLogger(__name__)

class EnhancedTradingWorkflow:
    """Enhanced trading workflow with advanced LangGraph patterns."""
    
    def __init__(self):
        """Initialize enhanced workflow with checkpointing and advanced features."""
        self.memory = MemorySaver()
        self.graph = self._build_enhanced_graph()
        self.app = self.graph.compile(
            checkpointer=self.memory,
            interrupt_before=["human_intervention"],  # Allow human intervention
            interrupt_after=["risk_assessment"],      # Pause after risk assessment
        )
    
    def _build_enhanced_graph(self) -> StateGraph:
        """Build enhanced workflow graph with advanced routing and agent communication."""
        
        workflow = StateGraph(EnhancedTradingState)
        
        # Add enhanced agent nodes
        workflow.add_node("initialization", self.initialization_agent)
        workflow.add_node("market_intelligence", self.market_intelligence_agent)
        workflow.add_node("sentiment_analysis", self.sentiment_analysis_agent)
        workflow.add_node("risk_assessment", self.risk_assessment_agent)
        workflow.add_node("portfolio_analysis", self.portfolio_analysis_agent)
        workflow.add_node("signal_generation", self.signal_generation_agent)
        workflow.add_node("agent_coordination", self.agent_coordination_node)
        workflow.add_node("strategy_synthesis", self.strategy_synthesis_agent)
        workflow.add_node("execution_planning", self.execution_planning_agent)
        workflow.add_node("order_execution", self.order_execution_agent)
        workflow.add_node("portfolio_monitoring", self.portfolio_monitoring_agent)
        workflow.add_node("performance_review", self.performance_review_agent)
        workflow.add_node("human_intervention", self.human_intervention_node)
        workflow.add_node("emergency_response", self.emergency_response_agent)
        
        # Define workflow routing with advanced conditional logic
        workflow.add_edge(START, "initialization")
        workflow.add_edge("initialization", "market_intelligence")
        
        # Parallel analysis phase
        workflow.add_conditional_edges(
            "market_intelligence",
            self.analysis_routing_logic,
            {
                "parallel_analysis": "sentiment_analysis",
                "skip_sentiment": "risk_assessment",
                "emergency": "emergency_response"
            }
        )
        
        workflow.add_edge("sentiment_analysis", "risk_assessment")
        
        # Risk-based routing
        workflow.add_conditional_edges(
            "risk_assessment",
            self.risk_routing_logic,
            {
                "proceed": "portfolio_analysis",
                "human_review": "human_intervention",
                "emergency": "emergency_response",
                "halt": END
            }
        )
        
        workflow.add_edge("portfolio_analysis", "signal_generation")
        workflow.add_edge("signal_generation", "agent_coordination")
        
        # Agent coordination and decision synthesis
        workflow.add_conditional_edges(
            "agent_coordination", 
            self.coordination_routing_logic,
            {
                "consensus": "strategy_synthesis",
                "need_more_input": "market_intelligence",
                "conflict": "human_intervention",
                "emergency": "emergency_response"
            }
        )
        
        workflow.add_edge("strategy_synthesis", "execution_planning")
        
        # Execution routing
        workflow.add_conditional_edges(
            "execution_planning",
            self.execution_routing_logic,
            {
                "execute": "order_execution",
                "hold": "portfolio_monitoring", 
                "review": "human_intervention",
                "emergency": "emergency_response"
            }
        )
        
        workflow.add_edge("order_execution", "portfolio_monitoring")
        workflow.add_edge("portfolio_monitoring", "performance_review")
        
        # Performance review routing
        workflow.add_conditional_edges(
            "performance_review",
            self.performance_routing_logic,
            {
                "continue": END,
                "rebalance": "signal_generation",
                "investigate": "human_intervention"
            }
        )
        
        # Human intervention and emergency handling
        workflow.add_conditional_edges(
            "human_intervention",
            self.human_decision_routing,
            {
                "continue": "strategy_synthesis",
                "restart": "initialization",
                "halt": END,
                "emergency": "emergency_response"
            }
        )
        
        workflow.add_edge("emergency_response", END)
        
        return workflow
    
    async def initialization_agent(self, state: EnhancedTradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize trading session with enhanced setup."""
        try:
            logger.info("Enhanced Initialization Agent: Starting trading session")
            
            # Update workflow stage
            state["workflow_stage"] = "init"
            state["cycle_count"] = state.get("cycle_count", 0) + 1
            
            # Initialize agent contexts
            for agent_name in state["agent_context"].keys():
                update_agent_context(state, agent_name, {
                    "trading_session": state["session_id"],
                    "cycle_number": state["cycle_count"]
                })
            
            # Get account information
            from tools.alpaca_client import alpaca_client
            account_info = alpaca_client.get_account_info()
            
            # Update portfolio state
            state["portfolio"].update({
                "cash": account_info["cash"],
                "equity": account_info["equity"],
                "buying_power": account_info["buying_power"]
            })
            
            # Initialize performance tracking if needed
            if not state["performance_metrics"].get("session_start_value"):
                state["performance_metrics"]["session_start_value"] = account_info["equity"]
            
            # Set up team communication
            await trading_agent_team.broadcast_message(
                from_agent="initialization",
                message=f"Trading session {state['cycle_count']} initialized",
                data={
                    "portfolio_value": account_info["equity"],
                    "session_id": state["session_id"]
                }
            )
            
            state["messages"].append(
                AIMessage(content=f"Trading session initialized - Cycle {state['cycle_count']}")
            )
            
            state["current_agent"] = "initialization"
            state["next_agent"] = "market_intelligence"
            state["last_update"] = datetime.now()
            
            return state
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            state["error_log"].append(f"Initialization failed: {e}")
            return state
    
    async def market_intelligence_agent(self, state: EnhancedTradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced market intelligence with tool integration."""
        try:
            logger.info("Market Intelligence Agent: Gathering market intelligence")
            
            state["workflow_stage"] = "analysis"
            state["current_agent"] = "market_intelligence"
            
            # Use enhanced tools for market analysis
            tool_context = ToolContext(
                agent_name="market_intelligence",
                reasoning_step="Comprehensive market analysis for trading decisions",
                confidence=0.8
            )
            
            # Get watchlist from state or use default
            watchlist = state.get("trading_config", {}).get("watchlist", ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"])
            
            # Execute market analysis tool with error handling
            try:
                analysis_result = trading_tool_registry.execute_tool_with_context(
                    "analyze_market_conditions",
                    tool_context,
                    symbols=watchlist[:10],
                    timeframe="1day",
                    indicators=["rsi", "sma", "volume"],
                    reasoning="Market intelligence gathering for trading cycle"
                )
            except Exception as tool_error:
                logger.warning(f"Market analysis tool failed: {tool_error}")
                # Create a mock successful result for testing
                analysis_result = type('MockResult', (), {
                    'success': False,
                    'error': str(tool_error),
                    'execution_time': 0.0,
                    'data': {}
                })()
            
            if analysis_result.success:
                # Store analysis in state
                state["market_analysis"] = analysis_result.data
                
                # Update agent context with findings
                market_data = analysis_result.data.get("analysis", {})
                if market_data:
                    # Calculate overall market sentiment
                    positive_signals = sum(1 for symbol, data in market_data.items() 
                                         if data.get("price_change_pct", 0) > 0)
                    market_sentiment = positive_signals / len(market_data) if market_data else 0.5
                    
                    update_agent_context(state, "market_intelligence", {
                        "market_regime": "bullish" if market_sentiment > 0.6 else "bearish" if market_sentiment < 0.4 else "neutral",
                        "confidence": {"market_analysis": analysis_result.execution_time < 5.0}  # Fast execution = good confidence
                    })
                
                # Communicate findings to team
                await trading_agent_team.broadcast_message(
                    from_agent="market_intelligence",
                    message=f"Market analysis complete - {len(market_data)} symbols analyzed",
                    data={
                        "market_sentiment": market_sentiment,
                        "symbols_analyzed": len(market_data),
                        "execution_time": analysis_result.execution_time
                    }
                )
                
                state["messages"].append(
                    AIMessage(content=f"Market intelligence: {len(market_data)} symbols analyzed")
                )
            else:
                # Handle analysis failure
                error_msg = f"Market analysis failed: {analysis_result.error}"
                logger.warning(error_msg)
                state["warnings"].append(error_msg)
                
                # Alert team about analysis failure
                await trading_agent_team.escalate_alert(
                    from_agent="market_intelligence",
                    alert_type="analysis_failure",
                    severity="medium",
                    details={"error": analysis_result.error, "execution_time": analysis_result.execution_time}
                )
            
            state["last_update"] = datetime.now()
            return state
            
        except Exception as e:
            logger.error(f"Market Intelligence Agent error: {e}")
            state["error_log"].append(f"Market intelligence failed: {e}")
            return state
    
    async def sentiment_analysis_agent(self, state: EnhancedTradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced sentiment analysis with team coordination."""
        try:
            logger.info("Sentiment Analysis Agent: Analyzing market sentiment")
            
            state["current_agent"] = "sentiment_analysis"
            
            # Get symbols for sentiment analysis
            market_analysis = state.get("market_analysis", {})
            symbols_to_analyze = []
            
            if market_analysis and market_analysis.get("analysis"):
                # Focus on symbols with significant price movements
                for symbol, data in market_analysis["analysis"].items():
                    price_change = abs(data.get("price_change_pct", 0))
                    if price_change > 2.0:  # >2% price change
                        symbols_to_analyze.append(symbol)
            
            # Fallback to watchlist if no significant movements
            if not symbols_to_analyze:
                symbols_to_analyze = state.get("trading_config", {}).get("watchlist", ["SPY", "QQQ", "AAPL"])[:5]
            
            # Execute sentiment analysis tool
            tool_context = ToolContext(
                agent_name="sentiment_analysis",
                reasoning_step="Sentiment analysis for position management",
                confidence=0.7
            )
            
            try:
                sentiment_result = trading_tool_registry.execute_tool_with_context(
                    "research_market_sentiment",
                    tool_context,
                    symbols=symbols_to_analyze,
                    sources=["social"],
                    lookback_hours=24,
                    reasoning="Social sentiment analysis for trading decisions"
                )
            except Exception as tool_error:
                logger.warning(f"Sentiment analysis tool failed: {tool_error}")
                # Create a mock result for testing
                sentiment_result = type('MockResult', (), {
                    'success': False,
                    'error': str(tool_error),
                    'execution_time': 0.0,
                    'data': {}
                })()
            
            if sentiment_result.success:
                state["sentiment_data"] = sentiment_result.data
                
                # Update agent context
                sentiment_analysis = sentiment_result.data.get("sentiment_analysis", {})
                avg_sentiment = 0.0
                if sentiment_analysis:
                    sentiment_scores = []
                    for symbol, data in sentiment_analysis.items():
                        # Convert sentiment to numeric score
                        sentiment_map = {"very_negative": -1, "negative": -0.5, "neutral": 0, "positive": 0.5, "very_positive": 1}
                        score = sentiment_map.get(data.get("sentiment", "neutral"), 0)
                        sentiment_scores.append(score)
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
                
                update_agent_context(state, "sentiment_analysis", {
                    "sentiment_score": avg_sentiment,
                    "confidence": {"sentiment_analysis": sentiment_result.execution_time < 10.0}
                })
                
                # Communicate sentiment findings
                await trading_agent_team.direct_message(
                    from_agent="sentiment_analysis",
                    to_agent="risk_manager",
                    message=f"Sentiment analysis complete - Average sentiment: {avg_sentiment:.2f}",
                    data={
                        "average_sentiment": avg_sentiment,
                        "symbols_analyzed": len(sentiment_analysis),
                        "high_confidence_signals": len([s for s in sentiment_analysis.values() if s.get("confidence", 0) > 0.7])
                    }
                )
                
                state["messages"].append(
                    AIMessage(content=f"Sentiment analysis: {len(sentiment_analysis)} symbols, avg sentiment: {avg_sentiment:.2f}")
                )
            else:
                logger.warning(f"Sentiment analysis failed: {sentiment_result.error}")
                state["warnings"].append(f"Sentiment analysis failed: {sentiment_result.error}")
            
            state["last_update"] = datetime.now()
            return state
            
        except Exception as e:
            logger.error(f"Sentiment Analysis Agent error: {e}")
            state["error_log"].append(f"Sentiment analysis failed: {e}")
            return state
    
    async def risk_assessment_agent(self, state: EnhancedTradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced risk assessment with tool integration."""
        try:
            logger.info("Risk Assessment Agent: Evaluating portfolio risk")
            
            state["current_agent"] = "risk_assessment"
            
            # Get current positions
            from tools.alpaca_client import alpaca_client
            positions = alpaca_client.get_positions()
            portfolio_value = state["portfolio"].get("equity", 0)
            
            # Convert positions to format expected by risk tool
            position_dict = {}
            for pos in positions:
                position_dict[pos["symbol"]] = {
                    "market_value": pos.get("market_value", 0),
                    "sector": "Technology"  # Simplified - would need sector lookup
                }
            
            # Execute risk assessment tool
            tool_context = ToolContext(
                agent_name="risk_assessment",
                reasoning_step="Portfolio risk evaluation for trading decisions",
                confidence=0.9
            )
            
            risk_result = trading_tool_registry.execute_tool_with_context(
                "assess_portfolio_risk",
                tool_context,
                positions=position_dict,
                portfolio_value=portfolio_value,
                risk_tolerance=state["trading_config"].get("risk_tolerance", "moderate"),
                reasoning="Portfolio risk assessment for cycle"
            )
            
            if risk_result.success:
                risk_data = risk_result.data.get("risk_assessment", {})
                
                # Update risk metrics in state
                state["risk_metrics"].update({
                    "overall_risk_level": risk_data.get("overall_risk_level", "medium"),
                    "diversification_score": risk_data.get("diversification_score", 0.5),
                    "concentration_risk": risk_data.get("concentration_risk", {}),
                    "total_positions": risk_data.get("total_positions", 0)
                })
                
                # Check for risk violations
                violations = risk_data.get("violations", [])
                if violations:
                    state["risk_alerts"].extend(violations)
                    
                    # Escalate high-risk situations
                    if len(violations) > 3 or risk_data.get("overall_risk_level") == "high":
                        await trading_agent_team.escalate_alert(
                            from_agent="risk_assessment",
                            alert_type="risk_violation",
                            severity="high",
                            details={
                                "violations": violations,
                                "risk_level": risk_data.get("overall_risk_level"),
                                "diversification_score": risk_data.get("diversification_score")
                            }
                        )
                
                # Update agent context
                update_agent_context(state, "risk_assessment", {
                    "risk_level": risk_data.get("overall_risk_level", "medium"),
                    "confidence": {"risk_assessment": len(violations) == 0}  # No violations = high confidence
                })
                
                state["messages"].append(
                    AIMessage(content=f"Risk assessment: {risk_data.get('overall_risk_level', 'unknown')} risk, "
                                   f"{risk_data.get('total_positions', 0)} positions")
                )
            else:
                logger.warning(f"Risk assessment failed: {risk_result.error}")
                state["risk_alerts"].append(f"Risk assessment failed: {risk_result.error}")
            
            state["last_update"] = datetime.now()
            return state
            
        except Exception as e:
            logger.error(f"Risk Assessment Agent error: {e}")
            state["error_log"].append(f"Risk assessment failed: {e}")
            return state
    
    async def agent_coordination_node(self, state: EnhancedTradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agent communication and decision synthesis."""
        try:
            logger.info("Agent Coordination: Processing team communications")
            
            state["current_agent"] = "agent_coordination"
            
            # Process message queue
            processed_messages = await trading_agent_team.process_message_queue(state, max_messages=15)
            
            # Request team decision on current market conditions
            decision_context = {
                "portfolio_value": state["portfolio"].get("equity", 0),
                "risk_level": state["risk_metrics"].get("overall_risk_level", "medium"),
                "market_regime": state["agent_context"].get("market_intelligence", {}).get("market_regime", "neutral"),
                "diversification_score": state["risk_metrics"].get("diversification_score", 0.5)
            }
            
            team_decision = await trading_agent_team.coordinate_team_decision(
                decision_topic="Portfolio Management Strategy",
                context=decision_context,
                required_agents=["market_intelligence", "risk_manager", "portfolio_manager"]
            )
            
            # Store team decision in state
            state["debug_info"]["latest_team_decision"] = team_decision
            
            # Update agent contexts based on team decision
            consensus_level = team_decision.get("consensus_level", "medium")
            team_recommendation = team_decision.get("team_recommendation", "hold")
            
            for agent_name in state["agent_context"].keys():
                update_agent_context(state, agent_name, {
                    "reasoning": f"Team decision: {team_recommendation} (consensus: {consensus_level})",
                    "confidence": {"team_coordination": team_decision.get("confidence", 0.5)}
                })
            
            state["messages"].append(
                AIMessage(content=f"Agent coordination: {len(processed_messages)} messages processed, "
                               f"team decision: {team_recommendation}")
            )
            
            state["last_update"] = datetime.now()
            return state
            
        except Exception as e:
            logger.error(f"Agent Coordination error: {e}")
            state["error_log"].append(f"Agent coordination failed: {e}")
            return state
    
    # Routing logic functions
    def analysis_routing_logic(self, state: EnhancedTradingState) -> Literal["parallel_analysis", "skip_sentiment", "emergency"]:
        """Route based on market analysis results."""
        if any(state["circuit_breakers"].values()):
            return "emergency"
        
        market_analysis = state.get("market_analysis", {})
        if market_analysis and market_analysis.get("analysis"):
            return "parallel_analysis"
        else:
            return "skip_sentiment"
    
    def risk_routing_logic(self, state: EnhancedTradingState) -> Literal["proceed", "human_review", "emergency", "halt"]:
        """Route based on risk assessment."""
        if any(state["circuit_breakers"].values()):
            return "emergency"
        
        risk_level = state["risk_metrics"].get("overall_risk_level", "medium")
        violations = len(state.get("risk_alerts", []))
        
        if risk_level == "critical" or violations > 5:
            return "emergency"
        elif risk_level == "high" or violations > 2:
            return "human_review" if should_interrupt_for_human(state) else "proceed"
        elif not state["market_data"].get("market_open", True):
            return "halt"
        else:
            return "proceed"
    
    def coordination_routing_logic(self, state: EnhancedTradingState) -> Literal["consensus", "need_more_input", "conflict", "emergency"]:
        """Route based on agent coordination results."""
        if any(state["circuit_breakers"].values()):
            return "emergency"
        
        team_decision = state["debug_info"].get("latest_team_decision", {})
        consensus_level = team_decision.get("consensus_level", "medium")
        
        if consensus_level == "high":
            return "consensus"
        elif consensus_level == "low":
            return "conflict"
        else:
            return "consensus"  # Default to consensus for medium
    
    def execution_routing_logic(self, state: EnhancedTradingState) -> Literal["execute", "hold", "review", "emergency"]:
        """Route based on execution planning."""
        if any(state["circuit_breakers"].values()):
            return "emergency"
        
        signals = state.get("signal_queue", [])
        if not signals:
            return "hold"
        
        high_confidence_signals = [s for s in signals if s.confidence > 0.7]
        if high_confidence_signals:
            return "execute"
        elif should_interrupt_for_human(state):
            return "review"
        else:
            return "hold"
    
    def performance_routing_logic(self, state: EnhancedTradingState) -> Literal["continue", "rebalance", "investigate"]:
        """Route based on performance review."""
        portfolio_size = len(state.get("positions", {}))
        target_size = state["trading_config"].get("target_positions", 35)
        
        if abs(portfolio_size - target_size) > 10:  # Significant deviation
            return "rebalance"
        elif len(state.get("error_log", [])) > 5:  # Many errors
            return "investigate"
        else:
            return "continue"
    
    def human_decision_routing(self, state: EnhancedTradingState) -> Literal["continue", "restart", "halt", "emergency"]:
        """Route based on human intervention decision."""
        # This would be determined by actual human input
        # For now, default to continue
        return "continue"
    
    # Additional agent implementations (simplified for brevity)
    async def portfolio_analysis_agent(self, state: EnhancedTradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        state["current_agent"] = "portfolio_analysis"
        state["messages"].append(AIMessage(content="Portfolio analysis completed"))
        return state
    
    async def signal_generation_agent(self, state: EnhancedTradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        state["current_agent"] = "signal_generation" 
        # Generate signals and add to queue
        prioritize_signals(state)
        state["messages"].append(AIMessage(content=f"Signals generated: {len(state['signal_queue'])}"))
        return state
    
    async def strategy_synthesis_agent(self, state: EnhancedTradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        state["current_agent"] = "strategy_synthesis"
        state["messages"].append(AIMessage(content="Strategy synthesis completed"))
        return state
    
    async def execution_planning_agent(self, state: EnhancedTradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        state["current_agent"] = "execution_planning"
        state["messages"].append(AIMessage(content="Execution planning completed"))
        return state
    
    async def order_execution_agent(self, state: EnhancedTradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        state["current_agent"] = "order_execution"
        state["messages"].append(AIMessage(content="Orders executed"))
        return state
    
    async def portfolio_monitoring_agent(self, state: EnhancedTradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        state["current_agent"] = "portfolio_monitoring"
        state["messages"].append(AIMessage(content="Portfolio monitoring completed"))
        return state
    
    async def performance_review_agent(self, state: EnhancedTradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        state["current_agent"] = "performance_review"
        state["messages"].append(AIMessage(content="Performance review completed"))
        return state
    
    async def human_intervention_node(self, state: EnhancedTradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle human intervention with interrupt."""
        state["current_agent"] = "human_intervention"
        
        # Create human interrupt
        reason = "Risk assessment requires human review"
        interrupt_cmd = create_human_interrupt(state, reason)
        
        # For now, just log and continue (in real implementation, would pause for human input)
        logger.info(f"Human intervention requested: {reason}")
        state["messages"].append(AIMessage(content="Human intervention completed"))
        return state
    
    async def emergency_response_agent(self, state: EnhancedTradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        state["current_agent"] = "emergency_response"
        state["workflow_stage"] = "halt"
        state["messages"].append(AIMessage(content="Emergency response activated"))
        return state
    
    async def run_enhanced_cycle(self, session_id: str, input_message: str = "") -> Dict[str, Any]:
        """Run enhanced trading cycle with advanced features."""
        try:
            # Create or get existing state
            initial_state = create_enhanced_initial_state(session_id)
            
            if input_message:
                initial_state["messages"].append(HumanMessage(content=input_message))
            
            # Run the enhanced workflow
            config = {"configurable": {"thread_id": session_id}}
            
            result = await self.app.ainvoke(initial_state, config)
            
            return {
                "status": "success",
                "session_id": session_id,
                "final_state": result,
                "portfolio_value": result["portfolio"].get("equity", 0),
                "active_positions": len(result["positions"]),
                "signals_generated": len(result["signal_queue"]),
                "workflow_stage": result["workflow_stage"],
                "cycle_count": result["cycle_count"],
                "team_performance": trading_agent_team.get_agent_performance_summary()
            }
            
        except Exception as e:
            logger.error(f"Enhanced trading cycle error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id
            }

# Global enhanced workflow instance
enhanced_trading_workflow = EnhancedTradingWorkflow()