"""Enhanced Portfolio Construction Engine using LangGraph for agent orchestration."""

import asyncio
import logging
from typing import Dict, List, Any, TypedDict, Annotated
from datetime import datetime
import json

try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor
    from langchain_core.messages import HumanMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logging.warning("LangGraph not available. Install with: pip install langgraph")

from agents.portfolio_management import (
    PortfolioConstructionEngine as BaseEngine,
    PortfolioRecommendation, PortfolioAllocation, 
    RiskProfile, MarketRegime, AssetClass
)

logger = logging.getLogger(__name__)

class PortfolioState(TypedDict):
    """State for the portfolio construction graph."""
    candidate_symbols: List[str]
    portfolio_value: float
    risk_profile: RiskProfile
    max_positions: int
    
    # Market analysis results
    market_regime: MarketRegime
    sector_scores: Dict[AssetClass, float]
    risk_parity_weights: Dict[str, float]
    momentum_scores: Dict[str, float]
    value_scores: Dict[str, float]
    ml_scores: Dict[str, float]
    
    # Intermediate results
    combined_scores: Dict[str, float]
    selected_symbols: List[str]
    portfolio_allocations: List[PortfolioAllocation]
    
    # Final recommendation
    recommendation: PortfolioRecommendation
    
    # Chain of thought tracking
    chain_of_thought: List[Dict[str, Any]]
    agent_decisions: Dict[str, Any]

class LangGraphPortfolioEngine:
    """Enhanced portfolio engine using LangGraph for coordinated agent execution."""
    
    def __init__(self):
        self.base_engine = BaseEngine()
        self.chain_of_thought = []
        
        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_graph()
        else:
            self.graph = None
            logger.warning("LangGraph not available - using fallback to base engine")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for portfolio construction."""
        
        # Define the graph structure
        workflow = StateGraph(PortfolioState)
        
        # Add nodes (agents)
        workflow.add_node("market_regime_agent", self._market_regime_node)
        workflow.add_node("sector_rotation_agent", self._sector_rotation_node) 
        workflow.add_node("risk_parity_agent", self._risk_parity_node)
        workflow.add_node("momentum_agent", self._momentum_node)
        workflow.add_node("value_agent", self._value_node)
        workflow.add_node("ml_agent", self._ml_node)
        workflow.add_node("portfolio_optimizer", self._portfolio_optimizer_node)
        workflow.add_node("risk_validator", self._risk_validator_node)
        workflow.add_node("final_recommendation", self._final_recommendation_node)
        
        # Define the execution flow
        workflow.set_entry_point("market_regime_agent")
        
        # Parallel execution of analysis agents after market regime
        workflow.add_edge("market_regime_agent", "sector_rotation_agent")
        workflow.add_edge("market_regime_agent", "risk_parity_agent")
        workflow.add_edge("market_regime_agent", "momentum_agent") 
        workflow.add_edge("market_regime_agent", "value_agent")
        workflow.add_edge("market_regime_agent", "ml_agent")
        
        # All analysis agents flow to portfolio optimizer
        workflow.add_edge("sector_rotation_agent", "portfolio_optimizer")
        workflow.add_edge("risk_parity_agent", "portfolio_optimizer")
        workflow.add_edge("momentum_agent", "portfolio_optimizer")
        workflow.add_edge("value_agent", "portfolio_optimizer")
        workflow.add_edge("ml_agent", "portfolio_optimizer")
        
        # Portfolio construction flow
        workflow.add_edge("portfolio_optimizer", "risk_validator")
        workflow.add_edge("risk_validator", "final_recommendation")
        workflow.add_edge("final_recommendation", END)
        
        return workflow.compile()
    
    async def _market_regime_node(self, state: PortfolioState) -> PortfolioState:
        """Market regime analysis node."""
        self._add_to_chain_of_thought(state, "market_regime_agent", "Analyzing market regime...")
        
        try:
            regime = await self.base_engine.regime_analyst.analyze_market_regime()
            state["market_regime"] = regime
            
            decision = {
                "agent": "market_regime_agent",
                "regime_identified": regime.value,
                "confidence": 0.8,  # Would come from actual analysis
                "reasoning": f"Market regime classified as {regime.value} based on trend and volatility analysis"
            }
            state["agent_decisions"]["market_regime"] = decision
            
            self._add_to_chain_of_thought(
                state, "market_regime_agent", 
                f"✅ Market regime identified: {regime.value}", decision
            )
            
        except Exception as e:
            logger.error(f"Market regime analysis failed: {e}")
            state["market_regime"] = MarketRegime.SIDEWAYS_MARKET
            self._add_to_chain_of_thought(
                state, "market_regime_agent", 
                f"❌ Analysis failed, defaulting to sideways market: {e}"
            )
        
        return state
    
    async def _sector_rotation_node(self, state: PortfolioState) -> PortfolioState:
        """Sector rotation analysis node."""
        self._add_to_chain_of_thought(state, "sector_rotation_agent", "Analyzing sector opportunities...")
        
        try:
            sector_scores = await self.base_engine.sector_agent.analyze_sector_rotation(
                state["market_regime"]
            )
            state["sector_scores"] = sector_scores
            
            # Find top sectors
            top_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            decision = {
                "agent": "sector_rotation_agent",
                "top_sectors": [(sector.value, score) for sector, score in top_sectors],
                "market_regime_factor": state["market_regime"].value,
                "reasoning": f"Sector rotation analysis for {state['market_regime'].value} regime"
            }
            state["agent_decisions"]["sector_rotation"] = decision
            
            self._add_to_chain_of_thought(
                state, "sector_rotation_agent",
                f"✅ Top sectors: {[s[0] for s in top_sectors[:2]]}", decision
            )
            
        except Exception as e:
            logger.error(f"Sector rotation analysis failed: {e}")
            state["sector_scores"] = {}
            self._add_to_chain_of_thought(
                state, "sector_rotation_agent", 
                f"❌ Sector analysis failed: {e}"
            )
        
        return state
    
    async def _risk_parity_node(self, state: PortfolioState) -> PortfolioState:
        """Risk parity analysis node."""
        self._add_to_chain_of_thought(state, "risk_parity_agent", "Calculating risk parity weights...")
        
        try:
            weights = await self.base_engine.risk_parity_agent.calculate_risk_parity_weights(
                state["candidate_symbols"]
            )
            state["risk_parity_weights"] = weights
            
            # Calculate diversification
            diversification = await self.base_engine.risk_parity_agent.calculate_diversification_score(
                state["candidate_symbols"], weights
            )
            
            decision = {
                "agent": "risk_parity_agent", 
                "weights_calculated": len(weights),
                "diversification_score": diversification,
                "reasoning": "Risk parity weights calculated based on inverse volatility"
            }
            state["agent_decisions"]["risk_parity"] = decision
            
            self._add_to_chain_of_thought(
                state, "risk_parity_agent",
                f"✅ Risk parity weights calculated, diversification: {diversification:.3f}", decision
            )
            
        except Exception as e:
            logger.error(f"Risk parity analysis failed: {e}")
            state["risk_parity_weights"] = {}
            self._add_to_chain_of_thought(
                state, "risk_parity_agent",
                f"❌ Risk parity calculation failed: {e}"
            )
        
        return state
    
    async def _momentum_node(self, state: PortfolioState) -> PortfolioState:
        """Momentum analysis node."""
        self._add_to_chain_of_thought(state, "momentum_agent", "Analyzing momentum signals...")
        
        try:
            momentum_scores = await self.base_engine.momentum_agent.analyze_momentum_signals(
                state["candidate_symbols"]
            )
            state["momentum_scores"] = momentum_scores
            
            # Find top momentum stocks
            top_momentum = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            decision = {
                "agent": "momentum_agent",
                "top_momentum_stocks": top_momentum,
                "average_momentum": sum(momentum_scores.values()) / len(momentum_scores) if momentum_scores else 0,
                "reasoning": "Multi-timeframe momentum analysis completed"
            }
            state["agent_decisions"]["momentum"] = decision
            
            self._add_to_chain_of_thought(
                state, "momentum_agent",
                f"✅ Momentum analysis complete. Top: {[s[0] for s in top_momentum[:2]]}", decision
            )
            
        except Exception as e:
            logger.error(f"Momentum analysis failed: {e}")
            state["momentum_scores"] = {}
            self._add_to_chain_of_thought(
                state, "momentum_agent",
                f"❌ Momentum analysis failed: {e}"
            )
        
        return state
    
    async def _value_node(self, state: PortfolioState) -> PortfolioState:
        """Value analysis node."""
        self._add_to_chain_of_thought(state, "value_agent", "Analyzing value opportunities...")
        
        try:
            value_scores = await self.base_engine.value_agent.analyze_value_signals(
                state["candidate_symbols"]
            )
            state["value_scores"] = value_scores
            
            # Find top value stocks
            top_value = sorted(value_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            decision = {
                "agent": "value_agent",
                "top_value_stocks": top_value,
                "average_value": sum(value_scores.values()) / len(value_scores) if value_scores else 0,
                "reasoning": "Fundamental value analysis based on PE, metrics, and ratings"
            }
            state["agent_decisions"]["value"] = decision
            
            self._add_to_chain_of_thought(
                state, "value_agent",
                f"✅ Value analysis complete. Top: {[s[0] for s in top_value[:2]]}", decision
            )
            
        except Exception as e:
            logger.error(f"Value analysis failed: {e}")
            state["value_scores"] = {}
            self._add_to_chain_of_thought(
                state, "value_agent",
                f"❌ Value analysis failed: {e}"
            )
        
        return state
    
    async def _ml_node(self, state: PortfolioState) -> PortfolioState:
        """ML prediction analysis node."""
        self._add_to_chain_of_thought(state, "ml_agent", "Running ML predictions...")
        
        try:
            ml_scores = await self.base_engine.ml_agent.analyze_ml_predictions(
                state["candidate_symbols"]
            )
            state["ml_scores"] = ml_scores
            
            # Find top ML predictions
            top_ml = sorted(ml_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            decision = {
                "agent": "ml_agent",
                "top_ml_predictions": top_ml,
                "average_ml_score": sum(ml_scores.values()) / len(ml_scores) if ml_scores else 0,
                "reasoning": "H2O.ai machine learning predictions and analysis"
            }
            state["agent_decisions"]["ml_predictions"] = decision
            
            self._add_to_chain_of_thought(
                state, "ml_agent",
                f"✅ ML predictions complete. Top: {[s[0] for s in top_ml[:2]]}", decision
            )
            
        except Exception as e:
            logger.error(f"ML analysis failed: {e}")
            state["ml_scores"] = {}
            self._add_to_chain_of_thought(
                state, "ml_agent",
                f"❌ ML analysis failed: {e}"
            )
        
        return state
    
    async def _portfolio_optimizer_node(self, state: PortfolioState) -> PortfolioState:
        """Portfolio optimization node."""
        self._add_to_chain_of_thought(state, "portfolio_optimizer", "Optimizing portfolio allocation...")
        
        try:
            # Combine agent scores
            combined_scores = await self.base_engine._combine_agent_scores(
                state["candidate_symbols"], state["risk_profile"],
                state.get("sector_scores", {}), state.get("risk_parity_weights", {}),
                state.get("momentum_scores", {}), state.get("value_scores", {}),
                state.get("ml_scores", {}), state["market_regime"]
            )
            state["combined_scores"] = combined_scores
            
            # Select top positions
            selected_symbols = self.base_engine._select_top_positions(
                combined_scores, state["max_positions"]
            )
            state["selected_symbols"] = selected_symbols
            
            # Calculate allocations
            allocations = await self.base_engine._calculate_portfolio_allocations(
                selected_symbols, combined_scores, state["portfolio_value"], state["market_regime"]
            )
            state["portfolio_allocations"] = allocations
            
            decision = {
                "agent": "portfolio_optimizer",
                "symbols_selected": len(selected_symbols),
                "total_allocation": sum(alloc.target_weight for alloc in allocations),
                "top_selections": selected_symbols[:5],
                "reasoning": f"Selected {len(selected_symbols)} positions based on multi-agent consensus"
            }
            state["agent_decisions"]["portfolio_optimizer"] = decision
            
            self._add_to_chain_of_thought(
                state, "portfolio_optimizer", 
                f"✅ Portfolio optimized: {len(selected_symbols)} positions selected", decision
            )
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            state["combined_scores"] = {}
            state["selected_symbols"] = []
            state["portfolio_allocations"] = []
            self._add_to_chain_of_thought(
                state, "portfolio_optimizer",
                f"❌ Portfolio optimization failed: {e}"
            )
        
        return state
    
    async def _risk_validator_node(self, state: PortfolioState) -> PortfolioState:
        """Risk validation node."""
        self._add_to_chain_of_thought(state, "risk_validator", "Validating portfolio risk...")
        
        try:
            allocations = state.get("portfolio_allocations", [])
            
            # Calculate risk metrics
            total_allocation = sum(alloc.target_weight for alloc in allocations)
            max_single_position = max([alloc.target_weight for alloc in allocations]) if allocations else 0
            diversification_score = state["agent_decisions"].get("risk_parity", {}).get("diversification_score", 0)
            
            # Risk validation rules
            risk_issues = []
            if total_allocation > 0.95:
                risk_issues.append("Portfolio over-allocated (>95%)")
            if max_single_position > 0.2:
                risk_issues.append(f"Position too large: {max_single_position:.1%}")
            if diversification_score < 0.4:
                risk_issues.append(f"Poor diversification: {diversification_score:.3f}")
            
            decision = {
                "agent": "risk_validator",
                "total_allocation": total_allocation,
                "max_position": max_single_position,
                "diversification": diversification_score,
                "risk_issues": risk_issues,
                "risk_approved": len(risk_issues) == 0,
                "reasoning": "Portfolio risk validation against position size and diversification limits"
            }
            state["agent_decisions"]["risk_validator"] = decision
            
            if risk_issues:
                self._add_to_chain_of_thought(
                    state, "risk_validator",
                    f"⚠️ Risk issues found: {len(risk_issues)} issues", decision
                )
            else:
                self._add_to_chain_of_thought(
                    state, "risk_validator",
                    "✅ Portfolio risk validation passed", decision
                )
                
        except Exception as e:
            logger.error(f"Risk validation failed: {e}")
            self._add_to_chain_of_thought(
                state, "risk_validator",
                f"❌ Risk validation failed: {e}"
            )
        
        return state
    
    async def _final_recommendation_node(self, state: PortfolioState) -> PortfolioState:
        """Final recommendation node."""
        self._add_to_chain_of_thought(state, "final_recommendation", "Generating final recommendation...")
        
        try:
            # Calculate portfolio metrics
            portfolio_metrics = await self.base_engine._calculate_portfolio_metrics(
                state.get("portfolio_allocations", []), state["market_regime"]
            )
            
            # Calculate rebalancing urgency
            rebalance_urgency = self.base_engine._calculate_rebalance_urgency(
                state.get("portfolio_allocations", []), state["market_regime"]
            )
            
            # Generate agent consensus
            agents_consensus = self.base_engine._calculate_agent_consensus(
                state["risk_profile"], state.get("sector_scores", {}),
                state.get("momentum_scores", {}), state.get("value_scores", {}),
                state.get("ml_scores", {})
            )
            
            # Generate reasoning
            reasoning = self.base_engine._generate_portfolio_reasoning(
                state["market_regime"], state["risk_profile"], 
                state.get("selected_symbols", []), portfolio_metrics, agents_consensus
            )
            
            # Create final recommendation
            recommendation = PortfolioRecommendation(
                allocations=state.get("portfolio_allocations", []),
                target_risk_level=portfolio_metrics.get('risk_level', 0.5),
                expected_return=portfolio_metrics.get('expected_return', 0.08),
                expected_volatility=portfolio_metrics.get('expected_volatility', 0.15),
                diversification_score=portfolio_metrics.get('diversification_score', 0.5),
                market_regime=state["market_regime"],
                confidence=portfolio_metrics.get('confidence', 0.5),
                cash_allocation=portfolio_metrics.get('cash_allocation', 0.1),
                rebalance_urgency=rebalance_urgency,
                reasoning=reasoning,
                agents_consensus=agents_consensus,
                timestamp=datetime.now()
            )
            
            state["recommendation"] = recommendation
            
            decision = {
                "agent": "final_recommendation",
                "positions": len(recommendation.allocations),
                "expected_return": recommendation.expected_return,
                "volatility": recommendation.expected_volatility,
                "confidence": recommendation.confidence,
                "reasoning": "Final portfolio recommendation generated with multi-agent consensus"
            }
            state["agent_decisions"]["final_recommendation"] = decision
            
            self._add_to_chain_of_thought(
                state, "final_recommendation",
                f"✅ Final recommendation complete: {len(recommendation.allocations)} positions", decision
            )
            
        except Exception as e:
            logger.error(f"Final recommendation failed: {e}")
            # Create fallback recommendation
            from agents.portfolio_management import PortfolioRecommendation, MarketRegime
            fallback = PortfolioRecommendation(
                allocations=[], target_risk_level=0.5, expected_return=0.08, expected_volatility=0.15,
                diversification_score=0.0, market_regime=MarketRegime.SIDEWAYS_MARKET, confidence=0.0,
                cash_allocation=1.0, rebalance_urgency=0.0, reasoning="Portfolio construction failed",
                agents_consensus={}, timestamp=datetime.now()
            )
            state["recommendation"] = fallback
            
            self._add_to_chain_of_thought(
                state, "final_recommendation",
                f"❌ Recommendation failed, using fallback: {e}"
            )
        
        return state
    
    def _add_to_chain_of_thought(self, state: PortfolioState, agent: str, 
                                message: str, decision: Dict[str, Any] = None):
        """Add entry to chain of thought tracking."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "message": message,
            "decision": decision
        }
        
        if "chain_of_thought" not in state:
            state["chain_of_thought"] = []
        if "agent_decisions" not in state:
            state["agent_decisions"] = {}
            
        state["chain_of_thought"].append(entry)
    
    async def construct_portfolio_with_langgraph(self, candidate_symbols: List[str],
                                               portfolio_value: float,
                                               risk_profile: RiskProfile,
                                               max_positions: int) -> Dict[str, Any]:
        """Construct portfolio using LangGraph orchestration."""
        
        if not self.graph:
            logger.warning("LangGraph not available, using fallback")
            recommendation = await self.base_engine.construct_portfolio(
                candidate_symbols, portfolio_value, risk_profile, max_positions
            )
            return {
                "recommendation": recommendation,
                "chain_of_thought": [],
                "agent_decisions": {},
                "langgraph_used": False
            }
        
        # Initialize state
        initial_state = {
            "candidate_symbols": candidate_symbols,
            "portfolio_value": portfolio_value,
            "risk_profile": risk_profile,
            "max_positions": max_positions,
            "chain_of_thought": [],
            "agent_decisions": {}
        }
        
        try:
            # Execute the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                "recommendation": final_state.get("recommendation"),
                "chain_of_thought": final_state.get("chain_of_thought", []),
                "agent_decisions": final_state.get("agent_decisions", {}),
                "langgraph_used": True,
                "execution_flow": self._extract_execution_flow(final_state)
            }
            
        except Exception as e:
            logger.error(f"LangGraph execution failed: {e}")
            # Fallback to base engine
            recommendation = await self.base_engine.construct_portfolio(
                candidate_symbols, portfolio_value, risk_profile, max_positions
            )
            return {
                "recommendation": recommendation,
                "chain_of_thought": [{"error": str(e)}],
                "agent_decisions": {},
                "langgraph_used": False
            }
    
    def _extract_execution_flow(self, final_state: PortfolioState) -> List[str]:
        """Extract the execution flow from chain of thought."""
        flow = []
        for entry in final_state.get("chain_of_thought", []):
            flow.append(f"{entry['agent']}: {entry['message']}")
        return flow

# Global enhanced portfolio engine
if LANGGRAPH_AVAILABLE:
    langgraph_portfolio_engine = LangGraphPortfolioEngine()
else:
    langgraph_portfolio_engine = None

async def construct_portfolio_with_orchestration(candidate_symbols: List[str],
                                               portfolio_value: float,
                                               risk_profile: str = "moderate",
                                               max_positions: int = 10) -> Dict[str, Any]:
    """
    Construct portfolio with LangGraph orchestration and chain of thought tracking.
    
    Returns:
        Dict containing recommendation, chain_of_thought, and agent_decisions
    """
    if not langgraph_portfolio_engine:
        logger.warning("LangGraph engine not available")
        from agents.portfolio_management import construct_optimal_portfolio
        recommendation = await construct_optimal_portfolio(
            candidate_symbols, portfolio_value, risk_profile, max_positions
        )
        return {
            "recommendation": recommendation,
            "chain_of_thought": [],
            "agent_decisions": {},
            "langgraph_used": False
        }
    
    try:
        from agents.portfolio_management import RiskProfile
        risk_profile_enum = RiskProfile(risk_profile.lower())
        
        result = await langgraph_portfolio_engine.construct_portfolio_with_langgraph(
            candidate_symbols=candidate_symbols,
            portfolio_value=portfolio_value,
            risk_profile=risk_profile_enum,
            max_positions=max_positions
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Portfolio construction with orchestration failed: {e}")
        raise