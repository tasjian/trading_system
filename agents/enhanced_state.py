"""Enhanced state management for trading system using advanced LangGraph patterns."""

import uuid
from typing import TypedDict, Annotated, List, Dict, Optional, Any, Literal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
import operator

@dataclass
class TradingContext:
    """Rich context for trading decisions with memory and reasoning."""
    market_regime: Literal["bull", "bear", "sideways", "volatile"] = "sideways"
    sentiment_score: float = 0.0
    risk_level: Literal["low", "medium", "high", "critical"] = "medium"
    trading_session: str = ""
    reasoning_chain: List[str] = field(default_factory=list)
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    
    def add_reasoning(self, step: str, confidence: float = 0.5):
        """Add reasoning step with confidence."""
        self.reasoning_chain.append(f"{step} (confidence: {confidence:.2f})")
        
    def get_overall_confidence(self) -> float:
        """Calculate overall confidence from individual factors."""
        if not self.confidence_factors:
            return 0.5
        return sum(self.confidence_factors.values()) / len(self.confidence_factors)

@dataclass
class AgentMemory:
    """Memory system for agents with episodic and semantic components."""
    episodic: List[Dict[str, Any]] = field(default_factory=list)  # Recent events
    semantic: Dict[str, Any] = field(default_factory=dict)  # Learned patterns
    working: Dict[str, Any] = field(default_factory=dict)  # Current context
    
    def remember_episode(self, event: Dict[str, Any]):
        """Store episodic memory with timestamp."""
        event["timestamp"] = datetime.now()
        self.episodic.append(event)
        # Keep only last 100 episodes
        if len(self.episodic) > 100:
            self.episodic = self.episodic[-100:]
    
    def update_semantic(self, pattern: str, value: Any):
        """Update semantic knowledge."""
        self.semantic[pattern] = value
    
    def recall_similar(self, context: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Recall similar episodes based on context."""
        # Simple similarity matching - could be enhanced with embeddings
        similar = []
        for episode in self.episodic[-50:]:  # Search recent episodes
            if context.lower() in str(episode).lower():
                similar.append(episode)
        return similar[-limit:]

@dataclass 
class TradingSignalEnhanced:
    """Enhanced trading signal with rich metadata."""
    symbol: str
    action: Literal["buy", "sell", "hold", "reduce", "increase"]
    confidence: float
    quantity: Optional[float] = None
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: str = ""
    agent_source: str = ""
    context: Optional[TradingContext] = None
    supporting_evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    
    def add_evidence(self, evidence: str, confidence: float = 0.5):
        """Add supporting evidence with confidence."""
        self.supporting_evidence.append(f"{evidence} (conf: {confidence:.2f})")
        # Update overall confidence based on evidence
        if self.context:
            self.context.confidence_factors[evidence[:20]] = confidence

class EnhancedTradingState(TypedDict):
    """Enhanced trading state with advanced LangGraph features."""
    
    # Core message system with agent communication
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Agent coordination and control flow
    current_agent: str
    next_agent: Optional[str]
    agent_context: Dict[str, TradingContext]
    agent_memory: Dict[str, AgentMemory]
    
    # Enhanced portfolio management
    portfolio: Dict[str, Any]
    positions: Dict[str, Dict[str, Any]]
    orders: Annotated[List[Dict[str, Any]], operator.add]
    
    # Rich market data with analysis
    market_data: Dict[str, Any]
    market_analysis: Dict[str, Any]  # LLM-generated insights
    sentiment_data: Dict[str, Any]
    
    # Advanced signal management
    signals: List[TradingSignalEnhanced]
    signal_queue: List[TradingSignalEnhanced]  # Prioritized signals
    executed_signals: List[TradingSignalEnhanced]
    
    # Risk and compliance
    risk_metrics: Dict[str, float]
    risk_alerts: List[str]
    compliance_status: Dict[str, bool]
    circuit_breakers: Dict[str, bool]
    
    # Session and workflow management
    session_id: str
    workflow_stage: Literal["init", "analysis", "decision", "execution", "review", "halt"]
    last_update: datetime
    cycle_count: int
    
    # Performance tracking
    performance_metrics: Dict[str, float]
    trade_history: List[Dict[str, Any]]
    
    # Configuration and preferences
    trading_config: Dict[str, Any]
    user_preferences: Dict[str, Any]
    
    # Error handling and logging
    error_log: List[str]
    warnings: List[str]
    debug_info: Dict[str, Any]

def create_enhanced_initial_state(session_id: Optional[str] = None) -> EnhancedTradingState:
    """Create enhanced initial state with advanced features."""
    
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    # Initialize agent contexts - using enhanced agent names
    agent_names = [
        "market_intelligence", "sentiment_analysis", "risk_assessment", 
        "portfolio_manager", "signal_generator", "strategy_coordinator",
        "order_manager", "portfolio_tracker", "compliance_monitor",
        "market_monitor", "risk_manager"  # Include original names for compatibility
    ]
    
    agent_context = {}
    agent_memory = {}
    
    for agent in agent_names:
        agent_context[agent] = TradingContext(trading_session=session_id)
        agent_memory[agent] = AgentMemory()
    
    return EnhancedTradingState(
        messages=[],
        current_agent="market_monitor",
        next_agent=None,
        agent_context=agent_context,
        agent_memory=agent_memory,
        
        portfolio={
            "cash": 0.0,
            "equity": 0.0,
            "buying_power": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
        },
        positions={},
        orders=[],
        
        market_data={},
        market_analysis={},
        sentiment_data={},
        
        signals=[],
        signal_queue=[],
        executed_signals=[],
        
        risk_metrics={
            "portfolio_var": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "concentration_risk": 0.0,
        },
        risk_alerts=[],
        compliance_status={
            "position_limits": True,
            "risk_limits": True,
            "regulatory": True,
        },
        circuit_breakers={
            "daily_loss_limit": False,
            "volatility_spike": False,
            "correlation_breakdown": False,
            "market_disruption": False,
        },
        
        session_id=session_id,
        workflow_stage="init",
        last_update=datetime.now(),
        cycle_count=0,
        
        performance_metrics={
            "total_return": 0.0,
            "win_rate": 0.0,
            "avg_hold_time": 0.0,
            "profit_factor": 0.0,
        },
        trade_history=[],
        
        trading_config={
            "risk_tolerance": "moderate",
            "max_positions": 50,
            "target_positions": 35,
            "rebalance_threshold": 0.05,
        },
        user_preferences={
            "sectors_preferred": [],
            "sectors_avoided": [],
            "esg_filter": False,
        },
        
        error_log=[],
        warnings=[],
        debug_info={},
    )

def add_agent_communication(state: EnhancedTradingState, 
                          from_agent: str, 
                          to_agent: str, 
                          message: str,
                          data: Optional[Dict[str, Any]] = None) -> EnhancedTradingState:
    """Add inter-agent communication message."""
    
    comm_message = {
        "type": "agent_communication",
        "from": from_agent,
        "to": to_agent,
        "message": message,
        "data": data or {},
        "timestamp": datetime.now(),
    }
    
    # Add to receiving agent's memory
    if to_agent in state["agent_memory"]:
        state["agent_memory"][to_agent].remember_episode({
            "type": "received_message",
            "from": from_agent,
            "content": message,
            "data": data,
        })
    
    # Add to message stream for visibility
    state["messages"].append(
        AIMessage(content=f"[{from_agent} → {to_agent}]: {message}")
    )
    
    return state

def update_agent_context(state: EnhancedTradingState, 
                        agent: str, 
                        updates: Dict[str, Any]) -> EnhancedTradingState:
    """Update agent context with new information."""
    
    if agent in state["agent_context"]:
        context = state["agent_context"][agent]
        
        for key, value in updates.items():
            if hasattr(context, key):
                setattr(context, key, value)
            elif key == "reasoning":
                context.add_reasoning(str(value))
            elif key == "confidence":
                context.confidence_factors.update(value if isinstance(value, dict) else {"general": value})
    
    state["last_update"] = datetime.now()
    return state

def prioritize_signals(state: EnhancedTradingState) -> EnhancedTradingState:
    """Prioritize signals based on confidence, urgency, and context."""
    
    # Sort signals by priority, confidence, and timestamp
    def signal_score(signal: TradingSignalEnhanced) -> float:
        priority_weights = {"urgent": 4, "high": 3, "medium": 2, "low": 1}
        priority_score = priority_weights.get(signal.priority, 2)
        
        # Recent signals get bonus
        age_hours = (datetime.now() - signal.timestamp).total_seconds() / 3600
        recency_bonus = max(0, 1.0 - age_hours / 24)  # Decay over 24 hours
        
        return (priority_score * 10) + (signal.confidence * 5) + recency_bonus
    
    # Sort signals by score
    all_signals = state["signals"] + state["signal_queue"]
    sorted_signals = sorted(all_signals, key=signal_score, reverse=True)
    
    # Update signal queue with top signals
    state["signal_queue"] = sorted_signals[:20]  # Keep top 20
    state["signals"] = sorted_signals[20:]  # Rest in general signals
    
    return state

def get_agent_insights(state: EnhancedTradingState, agent: str) -> Dict[str, Any]:
    """Get insights and context from a specific agent."""
    
    if agent not in state["agent_context"] or agent not in state["agent_memory"]:
        return {}
    
    context = state["agent_context"][agent]
    memory = state["agent_memory"][agent]
    
    return {
        "context": {
            "market_regime": context.market_regime,
            "sentiment_score": context.sentiment_score,
            "risk_level": context.risk_level,
            "confidence": context.get_overall_confidence(),
            "reasoning": context.reasoning_chain[-5:],  # Last 5 reasoning steps
        },
        "memory": {
            "recent_episodes": len(memory.episodic),
            "semantic_knowledge": list(memory.semantic.keys()),
            "working_memory": memory.working,
        }
    }

def should_interrupt_for_human(state: EnhancedTradingState) -> bool:
    """Determine if human intervention is needed."""
    
    # Check for critical conditions
    critical_conditions = [
        any(state["circuit_breakers"].values()),  # Any circuit breaker active
        len(state["risk_alerts"]) > 5,  # Many risk alerts
        not all(state["compliance_status"].values()),  # Compliance issues
        state["workflow_stage"] == "halt",  # System halted
    ]
    
    return any(critical_conditions)

def create_human_interrupt(state: EnhancedTradingState, reason: str) -> Command:
    """Create human intervention interrupt."""
    
    interrupt_message = f"Human intervention needed: {reason}"
    
    # Add context for human decision
    context_info = {
        "current_stage": state["workflow_stage"],
        "active_breakers": [k for k, v in state["circuit_breakers"].items() if v],
        "risk_alerts": state["risk_alerts"][-5:],  # Last 5 alerts
        "portfolio_value": state["portfolio"].get("equity", 0),
        "active_positions": len(state["positions"]),
    }
    
    return interrupt(f"{interrupt_message}\n\nContext: {context_info}")