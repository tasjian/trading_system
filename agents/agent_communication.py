"""Multi-agent communication system inspired by CrewAI patterns."""

from typing import Dict, List, Optional, Any, Literal, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from agents.enhanced_state import EnhancedTradingState, TradingContext, AgentMemory
import uuid
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentRole:
    """Define agent role with capabilities and responsibilities."""
    name: str
    role: str
    goal: str
    backstory: str
    capabilities: List[str]
    tools: List[str] = field(default_factory=list)
    memory_enabled: bool = True
    collaboration_style: Literal["cooperative", "competitive", "consultative"] = "cooperative"

@dataclass
class AgentMessage:
    """Message between agents with rich metadata."""
    from_agent: str
    to_agent: str
    message_type: Literal["info", "request", "response", "alert", "broadcast"]
    content: str
    data: Optional[Dict[str, Any]] = None
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    requires_response: bool = False
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
class AgentProtocol(Protocol):
    """Protocol for trading agents."""
    
    async def process_message(self, message: AgentMessage, state: EnhancedTradingState) -> Optional[AgentMessage]:
        """Process incoming message and optionally return response."""
        ...
    
    async def execute_task(self, task: Dict[str, Any], state: EnhancedTradingState) -> Dict[str, Any]:
        """Execute assigned task."""
        ...
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        ...

class TradingAgentTeam:
    """Coordinate team of trading agents with CrewAI-inspired patterns."""
    
    def __init__(self):
        self.agents: Dict[str, AgentRole] = {}
        self.message_queue: List[AgentMessage] = []
        self.conversation_history: Dict[str, List[AgentMessage]] = {}
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        self._define_agent_roles()
    
    def _define_agent_roles(self):
        """Define specialized agent roles for trading system."""
        
        # Market Intelligence Agent
        self.agents["market_intelligence"] = AgentRole(
            name="market_intelligence",
            role="Market Intelligence Specialist",
            goal="Provide comprehensive market analysis and identify trading opportunities",
            backstory=(
                "Expert market analyst with deep understanding of technical analysis, "
                "market microstructure, and macroeconomic factors. Specializes in "
                "real-time data processing and pattern recognition."
            ),
            capabilities=[
                "technical_analysis", "market_data_processing", "pattern_recognition",
                "economic_indicator_analysis", "sector_rotation_analysis"
            ],
            tools=["analyze_market_conditions", "research_market_sentiment"],
            collaboration_style="consultative"
        )
        
        # Risk Management Agent
        self.agents["risk_manager"] = AgentRole(
            name="risk_manager",
            role="Risk Management Officer",
            goal="Ensure portfolio risk stays within acceptable limits and protect capital",
            backstory=(
                "Seasoned risk management professional with expertise in portfolio theory, "
                "VaR modeling, and regulatory compliance. Takes a conservative approach "
                "to protect capital while enabling growth opportunities."
            ),
            capabilities=[
                "risk_assessment", "portfolio_optimization", "stress_testing",
                "compliance_monitoring", "position_sizing"
            ],
            tools=["assess_portfolio_risk"],
            collaboration_style="consultative"
        )
        
        # Strategy Coordinator Agent
        self.agents["strategy_coordinator"] = AgentRole(
            name="strategy_coordinator",
            role="Strategy Coordination Manager",
            goal="Coordinate trading strategies and optimize execution timing",
            backstory=(
                "Strategic trading expert who synthesizes input from various specialists "
                "to create coherent trading plans. Focuses on execution timing, "
                "resource allocation, and strategy optimization."
            ),
            capabilities=[
                "strategy_synthesis", "execution_timing", "resource_allocation",
                "multi_timeframe_analysis", "strategy_optimization"
            ],
            tools=["execute_trading_decision"],
            collaboration_style="cooperative"
        )
        
        # Portfolio Manager Agent
        self.agents["portfolio_manager"] = AgentRole(
            name="portfolio_manager",
            role="Portfolio Manager",
            goal="Maintain optimal portfolio construction and diversification",
            backstory=(
                "Experienced portfolio manager specializing in diversified equity portfolios. "
                "Expert in Modern Portfolio Theory, asset allocation, and rebalancing strategies. "
                "Focused on achieving target portfolio size of 25-50 positions across asset classes."
            ),
            capabilities=[
                "portfolio_construction", "diversification_analysis", "rebalancing",
                "asset_allocation", "performance_attribution"
            ],
            tools=["assess_portfolio_risk", "execute_trading_decision"],
            collaboration_style="cooperative"
        )
        
        # Compliance Monitor Agent  
        self.agents["compliance_monitor"] = AgentRole(
            name="compliance_monitor",
            role="Compliance and Monitoring Specialist",
            goal="Ensure all trading activities comply with regulations and internal policies",
            backstory=(
                "Compliance expert with deep knowledge of securities regulations, "
                "internal risk policies, and best execution practices. Maintains "
                "audit trail and ensures regulatory adherence."
            ),
            capabilities=[
                "regulatory_compliance", "audit_trail_management", "policy_enforcement",
                "transaction_monitoring", "reporting"
            ],
            tools=[],
            collaboration_style="consultative"
        )
    
    async def broadcast_message(self, from_agent: str, message: str, 
                               data: Optional[Dict[str, Any]] = None,
                               priority: Literal["low", "medium", "high", "urgent"] = "medium") -> List[AgentMessage]:
        """Broadcast message to all other agents."""
        
        messages = []
        for agent_name in self.agents.keys():
            if agent_name != from_agent:
                msg = AgentMessage(
                    from_agent=from_agent,
                    to_agent=agent_name,
                    message_type="broadcast",
                    content=message,
                    data=data,
                    priority=priority
                )
                messages.append(msg)
                self.message_queue.append(msg)
        
        logger.info(f"Agent {from_agent} broadcast to {len(messages)} agents: {message[:50]}...")
        return messages
    
    async def direct_message(self, from_agent: str, to_agent: str, message: str,
                           data: Optional[Dict[str, Any]] = None,
                           requires_response: bool = False,
                           priority: Literal["low", "medium", "high", "urgent"] = "medium") -> AgentMessage:
        """Send direct message between agents."""
        
        msg = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type="request" if requires_response else "info",
            content=message,
            data=data,
            requires_response=requires_response,
            priority=priority
        )
        
        self.message_queue.append(msg)
        
        # Track conversation
        conv_id = msg.conversation_id
        if conv_id not in self.conversation_history:
            self.conversation_history[conv_id] = []
        self.conversation_history[conv_id].append(msg)
        
        logger.info(f"Direct message: {from_agent} → {to_agent}: {message[:50]}...")
        return msg
    
    async def request_consultation(self, requesting_agent: str, consulting_agent: str,
                                 topic: str, context: Dict[str, Any]) -> AgentMessage:
        """Request consultation from specialist agent."""
        
        consultation_request = f"Consultation requested on: {topic}"
        
        return await self.direct_message(
            from_agent=requesting_agent,
            to_agent=consulting_agent,
            message=consultation_request,
            data={"topic": topic, "context": context},
            requires_response=True,
            priority="high"
        )
    
    async def escalate_alert(self, from_agent: str, alert_type: str, 
                           severity: Literal["low", "medium", "high", "critical"],
                           details: Dict[str, Any]) -> List[AgentMessage]:
        """Escalate alert to relevant agents based on severity."""
        
        alert_message = f"ALERT [{severity.upper()}]: {alert_type}"
        
        # Determine which agents should receive the alert
        alert_recipients = []
        
        if severity in ["high", "critical"]:
            # Critical alerts go to all agents
            alert_recipients = list(self.agents.keys())
        elif alert_type in ["risk", "compliance"]:
            # Risk/compliance alerts go to risk manager and compliance monitor
            alert_recipients = ["risk_manager", "compliance_monitor", "strategy_coordinator"]
        elif alert_type in ["market", "opportunity"]:
            # Market alerts go to intelligence and strategy agents
            alert_recipients = ["market_intelligence", "strategy_coordinator", "portfolio_manager"]
        else:
            # Default to strategy coordinator
            alert_recipients = ["strategy_coordinator"]
        
        # Remove the sender from recipients
        alert_recipients = [a for a in alert_recipients if a != from_agent]
        
        messages = []
        for recipient in alert_recipients:
            msg = AgentMessage(
                from_agent=from_agent,
                to_agent=recipient,
                message_type="alert",
                content=alert_message,
                data={"alert_type": alert_type, "severity": severity, "details": details},
                priority="urgent" if severity == "critical" else "high"
            )
            messages.append(msg)
            self.message_queue.append(msg)
        
        logger.warning(f"Alert escalated by {from_agent}: {alert_type} ({severity}) to {len(messages)} agents")
        return messages
    
    async def process_message_queue(self, state: EnhancedTradingState, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Process pending messages in the queue."""
        
        processed_messages = []
        messages_to_process = self.message_queue[:max_messages]
        
        for message in messages_to_process:
            try:
                # Add message to state for visibility
                state["messages"].append(
                    AIMessage(content=f"[{message.from_agent} → {message.to_agent}]: {message.content}")
                )
                
                # Update receiving agent's memory
                if message.to_agent in state["agent_memory"]:
                    state["agent_memory"][message.to_agent].remember_episode({
                        "type": "received_message",
                        "from": message.from_agent,
                        "content": message.content,
                        "data": message.data,
                        "priority": message.priority
                    })
                
                # Process message based on type
                response = await self._process_agent_message(message, state)
                
                result = {
                    "message_id": message.conversation_id,
                    "from": message.from_agent,
                    "to": message.to_agent,
                    "type": message.message_type,
                    "processed": True,
                    "response_generated": response is not None,
                    "timestamp": datetime.now()
                }
                
                if response:
                    # Add response to queue
                    self.message_queue.append(response)
                    result["response"] = response.content
                
                processed_messages.append(result)
                
            except Exception as e:
                logger.error(f"Error processing message {message.conversation_id}: {e}")
                processed_messages.append({
                    "message_id": message.conversation_id,
                    "error": str(e),
                    "processed": False,
                    "timestamp": datetime.now()
                })
        
        # Remove processed messages from queue
        self.message_queue = self.message_queue[len(messages_to_process):]
        
        return processed_messages
    
    async def _process_agent_message(self, message: AgentMessage, 
                                   state: EnhancedTradingState) -> Optional[AgentMessage]:
        """Process individual agent message and generate response if needed."""
        
        to_agent = message.to_agent
        
        # Simple response logic based on agent role and message type
        if message.message_type == "request" and message.requires_response:
            
            if to_agent == "risk_manager" and message.data:
                # Risk manager responds to risk-related requests
                topic = message.data.get("topic", "")
                if "risk" in topic.lower():
                    return AgentMessage(
                        from_agent=to_agent,
                        to_agent=message.from_agent,
                        message_type="response",
                        content=f"Risk assessment completed for {topic}. Current risk level: moderate.",
                        data={"risk_level": "moderate", "recommendations": ["Maintain current positions", "Monitor closely"]},
                        conversation_id=message.conversation_id
                    )
            
            elif to_agent == "market_intelligence" and message.data:
                # Market intelligence responds to market-related requests
                topic = message.data.get("topic", "")
                if "market" in topic.lower() or "analysis" in topic.lower():
                    return AgentMessage(
                        from_agent=to_agent,
                        to_agent=message.from_agent,
                        message_type="response",
                        content=f"Market analysis for {topic} indicates neutral conditions with moderate volatility.",
                        data={"market_sentiment": "neutral", "volatility": "moderate", "trend": "sideways"},
                        conversation_id=message.conversation_id
                    )
            
            elif to_agent == "portfolio_manager" and message.data:
                # Portfolio manager responds to portfolio-related requests
                topic = message.data.get("topic", "")
                if "portfolio" in topic.lower() or "diversification" in topic.lower():
                    return AgentMessage(
                        from_agent=to_agent,
                        to_agent=message.from_agent,
                        message_type="response",
                        content=f"Portfolio analysis for {topic}: Currently at target diversification levels.",
                        data={"portfolio_size": len(state["positions"]), "diversification_score": 0.75, "rebalancing_needed": False},
                        conversation_id=message.conversation_id
                    )
        
        return None
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all agents."""
        
        summary = {
            "total_agents": len(self.agents),
            "message_queue_size": len(self.message_queue),
            "active_conversations": len(self.conversation_history),
            "agent_details": {}
        }
        
        for agent_name, role in self.agents.items():
            summary["agent_details"][agent_name] = {
                "role": role.role,
                "capabilities": len(role.capabilities),
                "tools": len(role.tools),
                "performance": self.agent_performance.get(agent_name, {})
            }
        
        return summary
    
    async def coordinate_team_decision(self, decision_topic: str, 
                                     context: Dict[str, Any],
                                     required_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """Coordinate team decision-making process."""
        
        if required_agents is None:
            required_agents = ["market_intelligence", "risk_manager", "strategy_coordinator"]
        
        decision_id = str(uuid.uuid4())
        
        # Collect input from required agents
        agent_inputs = {}
        
        # Broadcast decision request
        await self.broadcast_message(
            from_agent="system",
            message=f"Team decision required: {decision_topic}",
            data={"decision_id": decision_id, "context": context, "required_agents": required_agents},
            priority="high"
        )
        
        # Simulate agent responses (in real implementation, would wait for actual responses)
        for agent_name in required_agents:
            if agent_name in self.agents:
                role = self.agents[agent_name]
                
                # Generate simulated response based on agent capabilities
                if "risk" in role.capabilities:
                    agent_inputs[agent_name] = {
                        "recommendation": "moderate_risk",
                        "confidence": 0.75,
                        "reasoning": f"Risk assessment from {role.role}"
                    }
                elif "market" in role.capabilities:
                    agent_inputs[agent_name] = {
                        "recommendation": "neutral_market",
                        "confidence": 0.65,
                        "reasoning": f"Market analysis from {role.role}"
                    }
                else:
                    agent_inputs[agent_name] = {
                        "recommendation": "proceed_cautiously",
                        "confidence": 0.70,
                        "reasoning": f"Strategic input from {role.role}"
                    }
        
        # Synthesize decision
        avg_confidence = sum(inp["confidence"] for inp in agent_inputs.values()) / len(agent_inputs)
        
        decision_result = {
            "decision_id": decision_id,
            "topic": decision_topic,
            "team_recommendation": "proceed" if avg_confidence > 0.6 else "hold",
            "confidence": avg_confidence,
            "agent_inputs": agent_inputs,
            "consensus_level": "high" if avg_confidence > 0.8 else "medium" if avg_confidence > 0.6 else "low",
            "timestamp": datetime.now()
        }
        
        logger.info(f"Team decision completed: {decision_topic} → {decision_result['team_recommendation']} (confidence: {avg_confidence:.2f})")
        
        return decision_result

# Global agent team instance
trading_agent_team = TradingAgentTeam()