"""LangGraph workflow for agentic trading system."""

import asyncio
import logging
from typing import Dict, Any, Literal
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from agents.state import TradingState, create_initial_state, update_state_timestamp
from agents.state import is_trading_halted, add_error_to_state
from config.settings import settings

logger = logging.getLogger(__name__)

class TradingWorkflow:
    """Main trading workflow orchestrator using LangGraph."""
    
    def __init__(self):
        """Initialize the trading workflow."""
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self) -> StateGraph:
        """Build the trading workflow graph."""
        
        # Create the workflow graph
        workflow = StateGraph(TradingState)
        
        # Add nodes (agent functions)
        workflow.add_node("market_monitor", self.market_monitor_agent)
        workflow.add_node("risk_assessor", self.risk_assessment_agent)
        workflow.add_node("signal_generator", self.signal_generation_agent)
        workflow.add_node("strategy_optimizer", self.strategy_optimization_agent)
        workflow.add_node("order_manager", self.order_management_agent)
        workflow.add_node("portfolio_tracker", self.portfolio_tracking_agent)
        workflow.add_node("emergency_handler", self.emergency_handler_agent)
        
        # Define the workflow routing
        workflow.add_edge(START, "market_monitor")
        workflow.add_edge("market_monitor", "risk_assessor")
        
        # Conditional routing based on risk assessment
        workflow.add_conditional_edges(
            "risk_assessor",
            self.risk_routing_logic,
            {
                "emergency": "emergency_handler",
                "proceed": "signal_generator",
                "halt": END
            }
        )
        
        workflow.add_edge("signal_generator", "strategy_optimizer")
        
        # Conditional routing based on signals
        workflow.add_conditional_edges(
            "strategy_optimizer",
            self.signal_routing_logic,
            {
                "execute": "order_manager",
                "hold": "portfolio_tracker",
                "reassess": "risk_assessor"
            }
        )
        
        workflow.add_edge("order_manager", "portfolio_tracker")
        workflow.add_edge("portfolio_tracker", END)
        workflow.add_edge("emergency_handler", END)
        
        return workflow
    
    async def market_monitor_agent(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor market conditions and update market data."""
        try:
            logger.info("Market Monitor Agent: Starting market analysis")
            
            # Import here to avoid circular imports
            from tools.alpaca_client import alpaca_client
            
            # Update account information
            account_info = alpaca_client.get_account_info()
            state["portfolio"].update({
                "cash": account_info["cash"],
                "equity": account_info["equity"],
                "buying_power": account_info["buying_power"]
            })
            
            # Check market status
            market_open = alpaca_client.is_market_open()
            state["market_data"]["market_open"] = market_open
            
            # Get current positions
            positions = alpaca_client.get_positions()
            state["portfolio"]["positions"] = {
                pos["symbol"]: pos for pos in positions
            }
            
            # Update watchlist data
            for symbol in state["watchlist"]:
                try:
                    market_data = alpaca_client.get_market_data(symbol, limit=50)
                    if not market_data.empty:
                        latest_data = market_data.iloc[-1]
                        state["market_data"]["symbols"] = state["market_data"].get("symbols", {})
                        state["market_data"]["symbols"][symbol] = {
                            "price": float(latest_data["close"]),
                            "volume": float(latest_data["volume"]),
                            "timestamp": datetime.now()
                        }
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
            
            # Add message to conversation
            state["messages"].append(
                AIMessage(content=f"Market monitor completed. Market open: {market_open}. "
                               f"Portfolio value: ${account_info['equity']:.2f}")
            )
            
            state["current_agent"] = "market_monitor"
            return update_state_timestamp(state)
            
        except Exception as e:
            error_msg = f"Market Monitor Agent error: {e}"
            logger.error(error_msg)
            return add_error_to_state(state, error_msg)
    
    async def risk_assessment_agent(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess portfolio risk and check limits."""
        try:
            logger.info("Risk Assessment Agent: Analyzing portfolio risk")
            
            # Check circuit breakers
            if is_trading_halted(state):
                logger.warning("Trading halted due to active circuit breakers")
                state["messages"].append(
                    AIMessage(content="Trading halted due to risk controls")
                )
                return update_state_timestamp(state)
            
            # Calculate portfolio risk metrics
            portfolio_value = state["portfolio"].get("equity", 0)
            cash = state["portfolio"].get("cash", 0)
            
            # Check daily loss limit
            if "initial_portfolio_value" not in state["trading_config"]:
                state["trading_config"]["initial_portfolio_value"] = portfolio_value
            
            initial_value = state["trading_config"]["initial_portfolio_value"]
            if initial_value > 0:
                daily_loss = (initial_value - portfolio_value) / initial_value
                max_daily_loss = state["risk_limits"]["max_daily_loss"]
                
                if daily_loss > max_daily_loss:
                    state["circuit_breakers"]["daily_loss_limit_hit"] = True
                    logger.warning(f"Daily loss limit hit: {daily_loss:.2%}")
            
            # Check position concentration
            positions = state["portfolio"].get("positions", {})
            total_exposure = sum(abs(pos.get("market_value", 0)) for pos in positions.values())
            
            for symbol, position in positions.items():
                market_value = abs(position.get("market_value", 0))
                if portfolio_value > 0:
                    exposure_pct = market_value / portfolio_value
                    max_position = state["risk_limits"]["max_position_size"]
                    
                    if exposure_pct > max_position:
                        state["circuit_breakers"]["position_size_exceeded"] = True
                        logger.warning(f"Position size exceeded for {symbol}: {exposure_pct:.2%}")
            
            # Calculate risk metrics
            from agents.state import RiskMetrics
            state["risk_metrics"] = RiskMetrics(
                portfolio_value=portfolio_value,
                cash_available=cash,
                total_exposure=total_exposure,
                concentration_risk={
                    symbol: abs(pos.get("market_value", 0)) / max(portfolio_value, 1)
                    for symbol, pos in positions.items()
                }
            )
            
            state["messages"].append(
                AIMessage(content=f"Risk assessment completed. Portfolio exposure: ${total_exposure:.2f}")
            )
            
            state["current_agent"] = "risk_assessor"
            return update_state_timestamp(state)
            
        except Exception as e:
            error_msg = f"Risk Assessment Agent error: {e}"
            logger.error(error_msg)
            return add_error_to_state(state, error_msg)
    
    async def signal_generation_agent(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals using advanced market analysis agents."""
        try:
            logger.info("Signal Generation Agent: Performing comprehensive market analysis")
            
            # Import market analysis agents
            from agents.market_analysis import market_analysis_factory
            
            # Get portfolio information for optimization
            portfolio_value = state["portfolio"].get("equity", 0)
            
            # Perform comprehensive analysis
            analysis_results = market_analysis_factory.get_portfolio_recommendations(
                symbols=state["watchlist"],
                portfolio_value=portfolio_value
            )
            
            # Extract individual symbol analysis
            symbol_analysis = analysis_results["symbol_analysis"]
            market_condition = analysis_results["market_condition"]
            optimal_weights = analysis_results["optimal_weights"]
            recommendations = analysis_results["recommendations"]
            
            # Update market conditions in state
            state["market_conditions"] = market_condition
            
            # Convert analysis results to trading signals
            signals = []
            for symbol, analysis in symbol_analysis.items():
                if analysis.action != "hold" and analysis.confidence > 0.3:
                    # Import here to avoid circular imports
                    from agents.state import TradingSignal
                    
                    # Calculate quantity based on optimal weights and risk management
                    quantity = self._calculate_position_size(
                        symbol, analysis, optimal_weights, portfolio_value
                    )
                    
                    signal = TradingSignal(
                        symbol=symbol,
                        action=analysis.action,
                        confidence=analysis.confidence,
                        price_target=analysis.price_target,
                        stop_loss=analysis.stop_loss,
                        quantity=quantity,
                        reasoning=f"{analysis.signal_type}: {analysis.reasoning}"
                    )
                    signals.append(signal)
            
            # Add signals to state
            state["signals"].extend(signals)
            
            # Create comprehensive summary
            signal_summary = f"Advanced analysis: {len(signals)} signals generated"
            if market_condition:
                signal_summary += f" | Market: {market_condition.market_trend}"
                signal_summary += f", Vol: {market_condition.volatility_regime}"
            
            if recommendations:
                signal_summary += f" | Recommendations: {len(recommendations)}"
            
            # Log detailed analysis results
            logger.info(f"Market Analysis Results:")
            logger.info(f"  - Market Trend: {market_condition.market_trend if market_condition else 'Unknown'}")
            logger.info(f"  - Volatility: {market_condition.volatility_regime if market_condition else 'Unknown'}")
            logger.info(f"  - Signals Generated: {len(signals)}")
            logger.info(f"  - Portfolio Recommendations: {len(recommendations)}")
            
            for signal in signals:
                logger.info(f"  - {signal.symbol}: {signal.action} (confidence: {signal.confidence:.2f})")
            
            state["messages"].append(AIMessage(content=signal_summary))
            state["current_agent"] = "signal_generator"
            return update_state_timestamp(state)
            
        except Exception as e:
            error_msg = f"Signal Generation Agent error: {e}"
            logger.error(error_msg)
            return add_error_to_state(state, error_msg)
    
    def _calculate_position_size(self, symbol: str, analysis, optimal_weights: Dict[str, float], 
                               portfolio_value: float) -> Optional[float]:
        """Calculate appropriate position size based on analysis and portfolio optimization."""
        try:
            if portfolio_value <= 0 or not analysis.indicators:
                return None
            
            # Get current price
            current_price = analysis.indicators.get('current_price', 0)
            if current_price <= 0:
                return None
            
            # Base position size from optimal weights
            optimal_weight = optimal_weights.get(symbol, settings.max_position_size / 2)
            base_position_value = portfolio_value * optimal_weight
            
            # Adjust based on signal confidence
            confidence_multiplier = min(1.0, analysis.confidence * 1.5)
            adjusted_position_value = base_position_value * confidence_multiplier
            
            # Calculate quantity
            quantity = adjusted_position_value / current_price
            
            # Apply minimum and maximum limits
            min_quantity = 1.0  # Minimum 1 share
            max_quantity = (portfolio_value * settings.max_position_size) / current_price
            
            return max(min_quantity, min(quantity, max_quantity))
            
        except Exception as e:
            logger.warning(f"Error calculating position size for {symbol}: {e}")
            return None
    
    async def strategy_optimization_agent(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize and prioritize trading signals."""
        try:
            logger.info("Strategy Optimization Agent: Optimizing trading strategy")
            
            # Get recent signals
            from agents.state import get_active_signals
            active_signals = get_active_signals(state, max_age_minutes=30)
            
            if not active_signals:
                state["messages"].append(AIMessage(content="No active signals to optimize"))
                state["current_agent"] = "strategy_optimizer"
                return update_state_timestamp(state)
            
            # Sort signals by confidence and prioritize
            active_signals.sort(key=lambda x: x.confidence, reverse=True)
            
            # Filter signals based on portfolio constraints
            portfolio_value = state["portfolio"].get("equity", 0)
            max_position_size = state["risk_limits"]["max_position_size"]
            
            optimized_signals = []
            for signal in active_signals[:3]:  # Top 3 signals
                # Calculate position size based on portfolio
                if portfolio_value > 0:
                    max_trade_value = portfolio_value * max_position_size
                    
                    # Estimate shares based on current price
                    symbol_data = state["market_data"]["symbols"].get(signal.symbol)
                    if symbol_data:
                        current_price = symbol_data["price"]
                        suggested_qty = max_trade_value / current_price
                        signal.quantity = min(suggested_qty, 100)  # Cap at 100 shares
                        optimized_signals.append(signal)
            
            # Update state with optimized signals
            state["signals"] = state["signals"][:-len(active_signals)] + optimized_signals
            
            state["messages"].append(
                AIMessage(content=f"Optimized {len(optimized_signals)} signals for execution")
            )
            
            state["current_agent"] = "strategy_optimizer"
            return update_state_timestamp(state)
            
        except Exception as e:
            error_msg = f"Strategy Optimization Agent error: {e}"
            logger.error(error_msg)
            return add_error_to_state(state, error_msg)
    
    async def order_management_agent(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading orders based on optimized signals."""
        try:
            logger.info("Order Management Agent: Executing trades")
            
            # Get signals ready for execution
            from agents.state import get_active_signals
            active_signals = get_active_signals(state, max_age_minutes=10)
            
            if not active_signals:
                state["messages"].append(AIMessage(content="No signals ready for execution"))
                state["current_agent"] = "order_manager"
                return update_state_timestamp(state)
            
            # Import here to avoid circular imports
            from tools.alpaca_client import alpaca_client
            
            executed_orders = []
            for signal in active_signals:
                if signal.action == "hold" or not signal.quantity:
                    continue
                
                try:
                    # Check if market is open
                    if not alpaca_client.is_market_open():
                        logger.info("Market closed, skipping order execution")
                        continue
                    
                    # Execute the order
                    order = alpaca_client.place_order(
                        symbol=signal.symbol,
                        qty=signal.quantity,
                        side=signal.action,
                        order_type="market"
                    )
                    
                    executed_orders.append({
                        "signal_id": signal.signal_id,
                        "order": order,
                        "timestamp": datetime.now()
                    })
                    
                    logger.info(f"Order executed: {signal.action} {signal.quantity} {signal.symbol}")
                    
                except Exception as e:
                    logger.error(f"Failed to execute order for {signal.symbol}: {e}")
                    add_error_to_state(state, f"Order execution failed for {signal.symbol}: {e}")
            
            # Update state with executed orders
            state["order_history"].extend(executed_orders)
            
            order_summary = f"Executed {len(executed_orders)} orders"
            state["messages"].append(AIMessage(content=order_summary))
            
            state["current_agent"] = "order_manager"
            return update_state_timestamp(state)
            
        except Exception as e:
            error_msg = f"Order Management Agent error: {e}"
            logger.error(error_msg)
            return add_error_to_state(state, error_msg)
    
    async def portfolio_tracking_agent(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Track portfolio performance and update statistics."""
        try:
            logger.info("Portfolio Tracking Agent: Updating portfolio metrics")
            
            # Import here to avoid circular imports
            from tools.alpaca_client import alpaca_client
            
            # Get updated account information
            account_info = alpaca_client.get_account_info()
            positions = alpaca_client.get_positions()
            
            # Update portfolio state
            state["portfolio"].update({
                "cash": account_info["cash"],
                "equity": account_info["equity"],
                "buying_power": account_info["buying_power"],
                "day_trades": account_info["day_trade_count"],
                "positions": {pos["symbol"]: pos for pos in positions}
            })
            
            # Calculate P&L
            current_value = account_info["equity"]
            initial_value = state["trading_config"].get("initial_portfolio_value", current_value)
            
            pnl = current_value - initial_value
            pnl_pct = (pnl / initial_value * 100) if initial_value > 0 else 0
            
            # Update trade statistics
            total_trades = len(state["order_history"])
            state["trade_statistics"]["total_trades"] = total_trades
            
            # Log portfolio update
            portfolio_summary = (f"Portfolio: ${current_value:.2f} "
                               f"(P&L: {pnl:+.2f}, {pnl_pct:+.2f}%)")
            
            state["messages"].append(AIMessage(content=portfolio_summary))
            
            state["current_agent"] = "portfolio_tracker"
            return update_state_timestamp(state)
            
        except Exception as e:
            error_msg = f"Portfolio Tracking Agent error: {e}"
            logger.error(error_msg)
            return add_error_to_state(state, error_msg)
    
    async def emergency_handler_agent(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency situations and risk violations."""
        try:
            logger.warning("Emergency Handler Agent: Responding to risk violation")
            
            # Import here to avoid circular imports
            from tools.alpaca_client import alpaca_client
            
            # Cancel all pending orders
            alpaca_client.cancel_all_orders()
            
            # Check if we need to close positions
            active_breakers = [name for name, active in state["circuit_breakers"].items() if active]
            
            if "daily_loss_limit_hit" in active_breakers:
                # Close all positions on daily loss limit
                positions = alpaca_client.get_positions()
                for position in positions:
                    try:
                        alpaca_client.close_position(position["symbol"])
                        logger.info(f"Emergency close position: {position['symbol']}")
                    except Exception as e:
                        logger.error(f"Failed to close position {position['symbol']}: {e}")
            
            emergency_msg = f"Emergency procedures activated: {', '.join(active_breakers)}"
            state["messages"].append(AIMessage(content=emergency_msg))
            
            state["current_agent"] = "emergency_handler"
            return update_state_timestamp(state)
            
        except Exception as e:
            error_msg = f"Emergency Handler Agent error: {e}"
            logger.error(error_msg)
            return add_error_to_state(state, error_msg)
    
    def risk_routing_logic(self, state: TradingState) -> Literal["emergency", "proceed", "halt"]:
        """Determine routing based on risk assessment."""
        
        # Check for active circuit breakers
        if any(state["circuit_breakers"].values()):
            return "emergency"
        
        # Check if market is closed
        if not state["market_data"].get("market_open", False):
            return "halt"
        
        # Check portfolio health
        portfolio_value = state["portfolio"].get("equity", 0)
        if portfolio_value <= 0:
            return "halt"
        
        return "proceed"
    
    def signal_routing_logic(self, state: TradingState) -> Literal["execute", "hold", "reassess"]:
        """Determine routing based on generated signals."""
        
        # Get recent signals
        from agents.state import get_active_signals
        active_signals = get_active_signals(state, max_age_minutes=30)
        
        if not active_signals:
            return "hold"
        
        # Check signal quality
        high_confidence_signals = [s for s in active_signals if s.confidence > 0.6]
        
        if high_confidence_signals:
            return "execute"
        
        return "reassess"
    
    async def run_cycle(self, session_id: str, input_message: str = "") -> Dict[str, Any]:
        """Run a complete trading cycle."""
        try:
            # Create or get existing state
            initial_state = create_initial_state(session_id)
            
            if input_message:
                initial_state["messages"].append(HumanMessage(content=input_message))
            
            # Run the workflow
            config = {"configurable": {"thread_id": session_id}}
            
            result = await self.app.ainvoke(initial_state, config)
            
            return {
                "status": "success",
                "session_id": session_id,
                "final_state": result,
                "portfolio_value": result["portfolio"].get("equity", 0),
                "active_positions": len(result["portfolio"].get("positions", {})),
                "signals_generated": len(result["signals"]),
                "orders_executed": len(result["order_history"])
            }
            
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id
            }

# Global workflow instance
trading_workflow = TradingWorkflow()