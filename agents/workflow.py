"""LangGraph workflow for agentic trading system."""

import asyncio
import logging
from typing import Dict, Any, Literal, Optional
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
        workflow.add_node("sentiment_analyzer", self.sentiment_analysis_agent)
        workflow.add_node("risk_assessor", self.risk_assessment_agent)
        workflow.add_node("signal_generator", self.signal_generation_agent)
        workflow.add_node("strategy_optimizer", self.strategy_optimization_agent)
        workflow.add_node("order_manager", self.order_management_agent)
        workflow.add_node("portfolio_tracker", self.portfolio_tracking_agent)
        workflow.add_node("emergency_handler", self.emergency_handler_agent)
        
        # Define the workflow routing
        workflow.add_edge(START, "market_monitor")
        workflow.add_edge("market_monitor", "sentiment_analyzer")
        workflow.add_edge("sentiment_analyzer", "risk_assessor")
        
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
    
    async def sentiment_analysis_agent(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze social media sentiment for active positions and candidate stocks."""
        try:
            from agents.sentiment_agent import sentiment_agent
            
            logger.info("Sentiment Analysis Agent: Analyzing social media sentiment")
            
            # Get current positions and candidate symbols
            current_positions = list(state.get("positions", {}).keys())
            candidate_symbols = state.get("candidate_symbols", [])
            all_symbols = list(set(current_positions + candidate_symbols))
            
            if not all_symbols:
                logger.warning("No symbols to analyze for sentiment")
                state["sentiment_data"] = {}
                state["current_agent"] = "sentiment_analyzer"
                return update_state_timestamp(state)
            
            # Run sentiment analysis cycle
            sentiment_outputs = await sentiment_agent.run_sentiment_cycle(
                tickers=all_symbols[:10],  # Limit to avoid rate limits
                platforms=['reddit']
            )
            
            # Process sentiment outputs
            sentiment_data = {}
            sentiment_signals = []
            
            for output in sentiment_outputs:
                ticker = output.ticker
                sentiment_data[ticker] = {
                    'sentiment': output.overall_sentiment.value,
                    'confidence': output.confidence,
                    'volume': output.volume,
                    'trend_signal': output.trend_signal.value,
                    'key_insights': output.key_insights,
                    'timestamp': output.timestamp.isoformat()
                }
                
                # Generate sentiment-based trading signals
                if output.confidence > 0.6:  # High confidence threshold
                    if output.overall_sentiment.value in ['positive', 'very_positive']:
                        if output.trend_signal.value in ['surge_positive', 'volume_spike']:
                            sentiment_signals.append({
                                'symbol': ticker,
                                'signal': 'BUY',
                                'strength': 0.7 if output.overall_sentiment.value == 'positive' else 0.9,
                                'reason': f'Positive sentiment surge: {", ".join(output.key_insights[:2])}',
                                'source': 'social_sentiment'
                            })
                    elif output.overall_sentiment.value in ['negative', 'very_negative']:
                        if output.trend_signal.value in ['surge_negative', 'coordination_detected']:
                            sentiment_signals.append({
                                'symbol': ticker,
                                'signal': 'SELL',
                                'strength': 0.6 if output.overall_sentiment.value == 'negative' else 0.8,
                                'reason': f'Negative sentiment surge: {", ".join(output.key_insights[:2])}',
                                'source': 'social_sentiment'
                            })
            
            # Update state with sentiment data
            state["sentiment_data"] = sentiment_data
            state["sentiment_signals"] = sentiment_signals
            
            # Add to existing signals
            existing_signals = state.get("trading_signals", [])
            existing_signals.extend(sentiment_signals)
            state["trading_signals"] = existing_signals
            
            logger.info(f"Sentiment analysis complete: {len(sentiment_outputs)} tickers analyzed, "
                       f"{len(sentiment_signals)} sentiment signals generated")
            
            state["current_agent"] = "sentiment_analyzer"
            return update_state_timestamp(state)
            
        except Exception as e:
            error_msg = f"Sentiment Analysis Agent error: {e}"
            logger.error(error_msg)
            # Don't fail the entire workflow for sentiment analysis errors
            state["sentiment_data"] = {}
            state["sentiment_signals"] = []
            state["current_agent"] = "sentiment_analyzer"
            return update_state_timestamp(state)
    
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
            
            # Check position concentration and diversification
            positions = state["portfolio"].get("positions", {})
            total_exposure = sum(abs(pos.get("market_value", 0)) for pos in positions.values())
            
            # Portfolio size checks
            num_positions = len(positions)
            if num_positions < settings.min_portfolio_size:
                logger.info(f"Portfolio under-diversified: {num_positions} positions (target: {settings.target_portfolio_size})")
            elif num_positions > settings.max_portfolio_size:
                logger.warning(f"Portfolio over-diversified: {num_positions} positions (max: {settings.max_portfolio_size})")
            
            # Check individual position sizes
            sector_allocation = {}
            for symbol, position in positions.items():
                market_value = abs(position.get("market_value", 0))
                if portfolio_value > 0:
                    exposure_pct = market_value / portfolio_value
                    max_position = state["risk_limits"]["max_position_size"]
                    
                    if exposure_pct > max_position:
                        state["circuit_breakers"]["position_size_exceeded"] = True
                        logger.warning(f"Position size exceeded for {symbol}: {exposure_pct:.2%}")
                    
                    # Track sector allocation for diversification
                    sector = position.get("sector", "Unknown")
                    sector_allocation[sector] = sector_allocation.get(sector, 0) + exposure_pct
            
            # Check sector concentration
            for sector, allocation in sector_allocation.items():
                if allocation > settings.max_sector_allocation:
                    logger.warning(f"Sector concentration risk - {sector}: {allocation:.2%} (max: {settings.max_sector_allocation:.2%})")
                    state["circuit_breakers"]["sector_concentration"] = True
            
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
        """Generate trading signals using LLM-enhanced multi-agent portfolio construction."""
        try:
            logger.info("Signal Generation Agent: Performing LLM-enhanced portfolio analysis")
            
            # Import LLM portfolio construction
            from agents.llm_portfolio_management import construct_llm_portfolio
            
            # Get portfolio information
            portfolio_value = state["portfolio"].get("equity", 100000.0)  # Default to 100k if not set
            risk_profile = state["trading_config"].get("risk_profile", "moderate")
            
            # Get expanded candidate symbols for diversified portfolio
            try:
                from simple_market_screener import SimpleMarketScreener
                screener = SimpleMarketScreener()
                all_candidate_symbols = screener.get_all_stocks()
                logger.info(f"Using expanded candidate universe of {len(all_candidate_symbols)} stocks")
            except Exception as e:
                logger.warning(f"Failed to get expanded stock universe, using watchlist: {e}")
                all_candidate_symbols = state["watchlist"]
            
            # Use LLM-enhanced portfolio construction with expanded universe
            recommendation = await construct_llm_portfolio(
                candidate_symbols=all_candidate_symbols[:100],  # Limit to top 100 for performance
                portfolio_value=portfolio_value,
                risk_profile=risk_profile,
                max_positions=settings.target_portfolio_size
            )
            
            # Update market regime in state
            state["market_conditions"] = recommendation.market_regime
            
            # Convert portfolio recommendation to trading signals
            signals = []
            for allocation in recommendation.allocations:
                if allocation.target_weight > 0.01:  # Only meaningful allocations
                    # Import here to avoid circular imports
                    from agents.state import TradingSignal
                    
                    # Calculate quantity based on target weight
                    target_value = portfolio_value * allocation.target_weight
                    
                    # Get current price from market data
                    symbol_data = state["market_data"]["symbols"].get(allocation.symbol)
                    if symbol_data:
                        current_price = symbol_data["price"]
                        quantity = target_value / current_price
                        
                        signal = TradingSignal(
                            symbol=allocation.symbol,
                            action=allocation.recommended_action,
                            confidence=allocation.confidence,
                            price_target=current_price * 1.1 if allocation.recommended_action == "buy" else current_price * 0.9,
                            stop_loss=current_price * 0.95 if allocation.recommended_action == "buy" else current_price * 1.05,
                            quantity=quantity,
                            reasoning=f"LLM Portfolio Construction: {allocation.reasoning}"
                        )
                        signals.append(signal)
            
            # Add signals to state
            state["signals"].extend(signals)
            
            # Create comprehensive summary
            signal_summary = f"LLM Portfolio Analysis: {len(signals)} signals generated"
            signal_summary += f" | Market Regime: {recommendation.market_regime.value}"
            signal_summary += f" | Expected Return: {recommendation.expected_return:.1%}"
            signal_summary += f" | Confidence: {recommendation.confidence:.1%}"
            
            # Log detailed analysis results
            logger.info(f"LLM Portfolio Construction Results:")
            logger.info(f"  - Market Regime: {recommendation.market_regime.value}")
            logger.info(f"  - Expected Return: {recommendation.expected_return:.1%}")
            logger.info(f"  - Portfolio Confidence: {recommendation.confidence:.1%}")
            logger.info(f"  - Cash Allocation: {recommendation.cash_allocation:.1%}")
            logger.info(f"  - Signals Generated: {len(signals)}")
            
            for signal in signals:
                logger.info(f"  - {signal.symbol}: {signal.action} ({signal.confidence:.2f} confidence, "
                          f"${signal.quantity * signal.price_target:.0f} value)")
            
            state["messages"].append(AIMessage(content=signal_summary))
            state["current_agent"] = "signal_generator"
            return update_state_timestamp(state)
            
        except Exception as e:
            error_msg = f"LLM Signal Generation Agent error: {e}"
            logger.error(error_msg)
            # Fallback to traditional analysis if LLM fails
            try:
                logger.info("Falling back to traditional market analysis")
                from agents.market_analysis import market_analysis_factory
                
                analysis_results = market_analysis_factory.get_portfolio_recommendations(
                    symbols=state["watchlist"],
                    portfolio_value=state["portfolio"].get("equity", 100000.0)
                )
                
                # Basic signal conversion
                signals = []
                for symbol, analysis in analysis_results["symbol_analysis"].items():
                    if analysis.action != "hold" and analysis.confidence > 0.3:
                        from agents.state import TradingSignal
                        
                        signal = TradingSignal(
                            symbol=symbol,
                            action=analysis.action,
                            confidence=analysis.confidence,
                            price_target=analysis.price_target,
                            stop_loss=analysis.stop_loss,
                            quantity=10,  # Default quantity
                            reasoning=f"Fallback analysis: {analysis.reasoning}"
                        )
                        signals.append(signal)
                
                state["signals"].extend(signals)
                state["messages"].append(AIMessage(content=f"Fallback analysis: {len(signals)} signals"))
                return update_state_timestamp(state)
                
            except Exception as fallback_error:
                logger.error(f"Fallback analysis also failed: {fallback_error}")
                return add_error_to_state(state, f"Both LLM and fallback analysis failed: {error_msg}")
    
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
            
            # Use diversified portfolio management for better optimization
            try:
                from agents.diversified_portfolio import diversified_portfolio_manager
                
                # Construct diversified portfolio if needed
                if len(state["portfolio"].get("positions", {})) < settings.min_portfolio_size:
                    logger.info("Constructing diversified portfolio with target size: {}".format(settings.target_portfolio_size))
                    
                    target_portfolio = await diversified_portfolio_manager.construct_diversified_portfolio(
                        portfolio_value=portfolio_value,
                        risk_tolerance="moderate"
                    )
                    
                    # Generate rebalancing orders
                    current_positions = state["portfolio"].get("positions", {})
                    rebalancing_orders = await diversified_portfolio_manager.generate_rebalancing_orders(
                        current_positions, portfolio_value
                    )
                    
                    # Convert rebalancing orders to signals
                    from agents.state import TradingSignal
                    optimized_signals = []
                    
                    for order in rebalancing_orders[:settings.max_portfolio_size]:  # Limit to max portfolio size
                        signal = TradingSignal(
                            symbol=order['symbol'],
                            action=order['side'],
                            confidence=0.8,  # High confidence for diversification
                            price_target=0.0,  # Market order
                            stop_loss=0.0,
                            quantity=order['quantity'],
                            reasoning=order['reason']
                        )
                        optimized_signals.append(signal)
                        
                else:
                    # Traditional optimization for existing portfolios
                    optimized_signals = []
                    max_new_positions = min(len(active_signals), 5)  # Limit new positions
                    
                    for signal in active_signals[:max_new_positions]:
                        # Calculate position size based on diversified approach
                        if portfolio_value > 0:
                            # Use smaller position sizes for better diversification
                            target_weight = min(settings.max_position_size, 1.0 / settings.target_portfolio_size * 2)
                            max_trade_value = portfolio_value * target_weight
                            
                            # Estimate shares based on current price
                            symbol_data = state["market_data"]["symbols"].get(signal.symbol)
                            if symbol_data:
                                current_price = symbol_data["price"]
                                suggested_qty = max_trade_value / current_price
                                signal.quantity = max(1, int(suggested_qty))  # At least 1 share
                                optimized_signals.append(signal)
                                
            except Exception as diversified_error:
                logger.warning(f"Diversified portfolio management failed, using fallback: {diversified_error}")
                
                # Fallback to original approach with updated limits
                optimized_signals = []
                max_signals = min(len(active_signals), 8)  # Increased from 3 to 8
                
                for signal in active_signals[:max_signals]:
                    if portfolio_value > 0:
                        # Use smaller position sizes for better diversification
                        diversified_max_position = settings.max_position_size
                        max_trade_value = portfolio_value * diversified_max_position
                        
                        symbol_data = state["market_data"]["symbols"].get(signal.symbol)
                        if symbol_data:
                            current_price = symbol_data["price"]
                            suggested_qty = max_trade_value / current_price
                            signal.quantity = max(1, int(suggested_qty))
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