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
        workflow.add_node("universe_filter", self.universe_filter_agent)
        workflow.add_node("sentiment_analyzer", self.sentiment_analysis_agent)
        workflow.add_node("risk_assessor", self.risk_assessment_agent)
        workflow.add_node("signal_generator", self.signal_generation_agent)
        workflow.add_node("strategy_optimizer", self.strategy_optimization_agent)
        workflow.add_node("order_manager", self.order_management_agent)
        workflow.add_node("portfolio_tracker", self.portfolio_tracking_agent)
        workflow.add_node("emergency_handler", self.emergency_handler_agent)
        
        # Define the workflow routing
        workflow.add_edge(START, "market_monitor")
        workflow.add_edge("market_monitor", "universe_filter")
        workflow.add_edge("universe_filter", "sentiment_analyzer")
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
            
            # Ensure market_data structure exists
            state["market_data"] = state.get("market_data", {})
            state["market_data"]["market_open"] = market_open
            state["market_data"]["symbols"] = state["market_data"].get("symbols", {})
            
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
    
    async def universe_filter_agent(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter stock universe from 11k+ stocks to 200-400 actionable candidates."""
        try:
            from core.universe_filter import filter_stock_universe
            
            logger.info("Universe Filter Agent: Filtering 11k+ stocks to actionable candidates")
            
            # Get base symbols from positions and watchlist
            current_positions = list(state.get("portfolio", {}).get("positions", {}).keys())
            watchlist_symbols = state.get("watchlist", [])
            
            # Apply universe filter to get actionable stocks
            logger.info("🔍 Running universe filter to identify actionable stocks...")
            filter_result = await filter_stock_universe(
                base_symbols=None,  # Use full universe
                max_symbols=400,    # Increased to 400 for better signal diversity
                include_watchlist=True
            )
            
            # Combine filtered universe with mandatory symbols (positions + watchlist)
            mandatory_symbols = set(current_positions + watchlist_symbols)
            filtered_symbols = list(set(filter_result.filtered_symbols) | mandatory_symbols)
            
            logger.info(f"Universe filter results:")
            logger.info(f"  📊 Original universe: {filter_result.total_symbols:,} stocks")
            logger.info(f"  ✨ Filtered to: {len(filter_result.filtered_symbols)} actionable candidates")
            logger.info(f"  📌 Added mandatory: {len(mandatory_symbols)} positions/watchlist")
            logger.info(f"  🎯 Final analysis set: {len(filtered_symbols)} symbols")
            logger.info(f"  📈 Signal breakdown: {filter_result.filter_summary}")
            logger.info(f"  ⚡ Processing efficiency: {((filter_result.total_symbols - len(filtered_symbols)) / filter_result.total_symbols * 100):.1f}% reduction")
            
            # Store filter results in state for sentiment analysis
            state["universe_filter_result"] = filter_result
            state["filtered_symbols"] = filtered_symbols[:200]  # Limit to top 200 for sentiment analysis
            state["current_agent"] = "universe_filter"
            
            return update_state_timestamp(state)
            
        except Exception as e:
            error_msg = f"Universe Filter Agent error: {e}"
            logger.error(error_msg)
            return add_error_to_state(state, error_msg)
    
    async def sentiment_analysis_agent(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze comprehensive sentiment for pre-filtered stocks only."""
        try:
            from agents.sentiment_agent import sentiment_agent
            
            logger.info("Sentiment Analysis Agent: Running comprehensive sentiment analysis on pre-filtered stocks")
            
            # Get pre-filtered symbols from universe filter step
            filtered_symbols = state.get("filtered_symbols", [])
            
            if not filtered_symbols:
                logger.warning("No filtered symbols provided by universe filter")
                state["sentiment_data"] = {}
                state["current_agent"] = "sentiment_analyzer" 
                return update_state_timestamp(state)
            
            logger.info(f"💭 Analyzing sentiment for {len(filtered_symbols)} pre-filtered symbols")
            
            # Run comprehensive sentiment analysis for each symbol
            sentiment_data = {}
            sentiment_signals = []
            
            # Limit to top 50 symbols for sentiment analysis to balance quality vs performance
            symbols_to_analyze = filtered_symbols[:50]
            logger.info(f"🎯 Processing top {len(symbols_to_analyze)} symbols: {', '.join(symbols_to_analyze[:5])}{'...' if len(symbols_to_analyze) > 5 else ''}")
            
            # Process symbols in parallel batches to speed up analysis
            batch_size = 8  # Increased batch size since we have fewer, higher-quality symbols
            
            async def analyze_symbol_sentiment(symbol: str):
                """Analyze sentiment for a single symbol."""
                try:
                    logger.info(f"Running comprehensive sentiment analysis for {symbol}")
                    
                    # Use the comprehensive sentiment analysis that includes earnings
                    comprehensive_sentiment = await sentiment_agent.analyze_comprehensive_sentiment(symbol)
                    return symbol, comprehensive_sentiment
                except Exception as e:
                    logger.error(f"Error analyzing sentiment for {symbol}: {e}")
                    return symbol, None
            
            # Process symbols in parallel batches
            for i in range(0, len(symbols_to_analyze), batch_size):
                batch = symbols_to_analyze[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {', '.join(batch)}")
                
                # Run batch in parallel
                batch_results = await asyncio.gather(
                    *[analyze_symbol_sentiment(symbol) for symbol in batch],
                    return_exceptions=True
                )
                
                # Process batch results
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch processing error: {result}")
                        continue
                    
                    symbol, comprehensive_sentiment = result
                    
                    if comprehensive_sentiment:
                        sentiment_data[symbol] = {
                            'overall_sentiment': comprehensive_sentiment.overall_sentiment,
                            'overall_score': comprehensive_sentiment.overall_score,
                            'confidence': comprehensive_sentiment.confidence,
                            'data_sources_count': comprehensive_sentiment.data_sources_count,
                            'news_articles_count': comprehensive_sentiment.news_articles_count,
                            'social_posts_count': comprehensive_sentiment.social_posts_count,
                            'has_recent_earnings': comprehensive_sentiment.has_recent_earnings,
                            'key_themes': comprehensive_sentiment.key_themes[:3],  # Top 3 themes
                            'risk_factors': comprehensive_sentiment.risk_factors[:2],  # Top 2 risks
                            'opportunities': comprehensive_sentiment.opportunities[:2],  # Top 2 opportunities
                            'timestamp': comprehensive_sentiment.timestamp.isoformat()
                        }
                        
                        # Log detailed sentiment breakdown
                        logger.info(f"{symbol} sentiment breakdown:")
                        logger.info(f"  Overall: {comprehensive_sentiment.overall_sentiment} (score: {comprehensive_sentiment.overall_score:.3f}, confidence: {comprehensive_sentiment.confidence:.3f})")
                        logger.info(f"  Sources: {comprehensive_sentiment.data_sources_count} (news: {comprehensive_sentiment.news_articles_count}, social: {comprehensive_sentiment.social_posts_count}, earnings: {'Yes' if comprehensive_sentiment.has_recent_earnings else 'No'})")
                        logger.info(f"  Key themes: {', '.join(comprehensive_sentiment.key_themes[:3])}")
                        
                        # Generate trading signals based on comprehensive sentiment
                        signal_strength = 0.0
                        signal_type = None
                        signal_reason = []
                        
                        # Earnings-based signals (highest priority)
                        if comprehensive_sentiment.has_recent_earnings:
                            if comprehensive_sentiment.overall_score > 0.4:
                                signal_strength += 0.5
                                signal_type = 'BUY'
                                signal_reason.append('Positive earnings sentiment')
                            elif comprehensive_sentiment.overall_score < -0.4:
                                signal_strength += 0.5
                                signal_type = 'SELL'
                                signal_reason.append('Negative earnings sentiment')
                        
                        # Overall sentiment signals
                        if comprehensive_sentiment.overall_sentiment in ['very_positive']:
                            signal_strength += 0.4
                            signal_type = 'BUY'
                            signal_reason.append('Very positive overall sentiment')
                        elif comprehensive_sentiment.overall_sentiment in ['positive']:
                            signal_strength += 0.2
                            signal_type = 'BUY'
                            signal_reason.append('Positive overall sentiment')
                        elif comprehensive_sentiment.overall_sentiment in ['very_negative']:
                            signal_strength += 0.4
                            signal_type = 'SELL'
                            signal_reason.append('Very negative overall sentiment')
                        elif comprehensive_sentiment.overall_sentiment in ['negative']:
                            signal_strength += 0.2
                            signal_type = 'SELL'
                            signal_reason.append('Negative overall sentiment')
                        
                        # Confidence and data quality boosts
                        if comprehensive_sentiment.confidence > 0.7:
                            signal_strength += 0.1
                            signal_reason.append('High confidence analysis')
                        
                        if comprehensive_sentiment.data_sources_count >= 3:
                            signal_strength += 0.1
                            signal_reason.append('Multiple data sources')
                        
                        # Generate signal if significant
                        if signal_type and signal_strength > 0.3:
                            sentiment_signals.append({
                                'symbol': symbol,
                                'signal': signal_type,
                                'strength': min(signal_strength, 1.0),
                                'reason': ', '.join(signal_reason),
                                'source': 'comprehensive_sentiment',
                                'has_earnings': comprehensive_sentiment.has_recent_earnings,
                                'sentiment_score': comprehensive_sentiment.overall_score
                            })
                            
                            logger.info(f"Generated {signal_type} signal for {symbol} (strength: {signal_strength:.2f}): {', '.join(signal_reason)}")
                    
                    else:
                        logger.warning(f"No comprehensive sentiment data for {symbol}")
                
                # Rate limiting between batches (reduced since we're processing in parallel)
                if i + batch_size < len(symbols_to_analyze):
                    await asyncio.sleep(2)  # 2 second delay between batches
            
            # Update state with comprehensive sentiment data
            state["sentiment_data"] = sentiment_data
            state["sentiment_signals"] = sentiment_signals
            
            # Add sentiment signals to existing trading signals
            existing_signals = state.get("trading_signals", [])
            existing_signals.extend(sentiment_signals)
            state["trading_signals"] = existing_signals
            
            logger.info(f"Comprehensive sentiment analysis complete: {len(sentiment_data)} symbols analyzed, "
                       f"{len(sentiment_signals)} sentiment signals generated")
            
            # Log summary of signals
            if sentiment_signals:
                buy_signals = [s for s in sentiment_signals if s['signal'] == 'BUY']
                sell_signals = [s for s in sentiment_signals if s['signal'] == 'SELL']
                earnings_signals = [s for s in sentiment_signals if s.get('has_earnings')]
                
                logger.info(f"Signal summary: {len(buy_signals)} BUY, {len(sell_signals)} SELL, {len(earnings_signals)} with earnings data")
                
                for signal in sentiment_signals:
                    logger.info(f"  {signal['signal']} {signal['symbol']} (strength: {signal['strength']:.2f}{'*' if signal.get('has_earnings') else ''}): {signal['reason']}")
            
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
            logger.info("Signal Generation Agent: Performing portfolio analysis")
            
            # Try LLM portfolio construction first
            try:
                from agents.llm_portfolio_management import construct_llm_portfolio
                return await self._llm_signal_generation(state, config)
            except ImportError:
                logger.info("LLM portfolio management not available, using fallback signal generation")
                return await self._fallback_signal_generation(state, config)
        
        except Exception as e:
            error_msg = f"Signal Generation Agent error: {e}"
            logger.error(error_msg)
            return add_error_to_state(state, error_msg)
    
    async def _llm_signal_generation(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """LLM-enhanced signal generation with sentiment integration."""
        try:
            from agents.llm_portfolio_management import construct_llm_portfolio
            
            logger.info("LLM Signal Generation: Integrating sentiment analysis with portfolio construction")
        
            # Get portfolio information
            portfolio_value = state["portfolio"].get("equity", 1000.0)  # Default to 1k if not set
            risk_profile = state["trading_config"].get("risk_profile", "moderate")
        
            # Get sentiment data from previous analysis
            sentiment_data = state.get("sentiment_data", {})
            sentiment_signals = state.get("sentiment_signals", [])
        
            logger.info(f"Integrating sentiment data for {len(sentiment_data)} symbols")
            if sentiment_signals:
                earnings_signals = [s for s in sentiment_signals if s.get('has_earnings')]
                logger.info(f"Found {len(sentiment_signals)} sentiment signals ({len(earnings_signals)} with earnings data)")
        
            # Get filtered candidate symbols for efficient processing
            universe_filter_result = state.get("universe_filter_result")
            if universe_filter_result:
                all_candidate_symbols = universe_filter_result.filtered_symbols
                logger.info(f"Using filtered candidate universe of {len(all_candidate_symbols)} actionable stocks")
                logger.info(f"Universe filter reduced processing from {universe_filter_result.total_symbols} to {len(all_candidate_symbols)} stocks")
            else:
                # Fallback to enhanced screener if filter not available
                logger.warning("Universe filter results not available, using fallback screener")
                try:
                    from enhanced_market_screener import EnhancedMarketScreener
                    screener = EnhancedMarketScreener()
                    all_candidate_symbols = screener.get_all_stocks()
                    logger.info(f"Using enhanced candidate universe of {len(all_candidate_symbols)} stocks")
                except Exception as e:
                    logger.warning(f"Failed to get enhanced stock universe: {e}")
                    try:
                        from simple_market_screener import SimpleMarketScreener
                        screener = SimpleMarketScreener()
                        all_candidate_symbols = screener.get_all_stocks()
                        logger.info(f"Using simple candidate universe of {len(all_candidate_symbols)} stocks")
                    except Exception as e2:
                        logger.warning(f"Failed to get simple stock universe, using watchlist: {e2}")
                        all_candidate_symbols = state["watchlist"]
        
            # Create comprehensive candidate universe including ALL symbols with sentiment data
            comprehensive_candidates = set(all_candidate_symbols)
            
            # Include ANY symbol we have sentiment data for (not just earnings symbols)
            if sentiment_data:
                # Get symbols from market data in state
                market_data_symbols = set(state.get("market_data", {}).get("symbols", {}).keys())
                
                # Include symbols from sentiment data, watchlist, or market data
                sentiment_symbols = [s for s in sentiment_data.keys() 
                                   if s in state["watchlist"] or s in market_data_symbols or s in all_candidate_symbols]
                comprehensive_candidates.update(sentiment_symbols)
                logger.info(f"Added {len(sentiment_symbols)} symbols with sentiment data to candidate universe")
            
            # Convert to list for LLM processing
            final_candidate_list = list(comprehensive_candidates)
            
            # Log candidate composition for transparency
            earnings_count = 0
            if sentiment_data:
                for s in final_candidate_list:
                    if s in sentiment_data:
                        sentiment = sentiment_data[s]
                        if isinstance(sentiment, dict):
                            has_earnings = sentiment.get('has_recent_earnings', False)
                        else:
                            has_earnings = getattr(sentiment, 'has_recent_earnings', False)
                        if has_earnings:
                            earnings_count += 1
            logger.info(f"Final candidate universe: {len(final_candidate_list)} symbols ({earnings_count} with earnings data)")
        
            # Use LLM-enhanced portfolio construction with comprehensive universe
            recommendation = await construct_llm_portfolio(
                candidate_symbols=final_candidate_list[:100],  # Limit to top 100 for performance
                portfolio_value=portfolio_value,
                risk_profile=risk_profile,
                max_positions=settings.target_portfolio_size,
                sentiment_data=sentiment_data  # Pass sentiment data to portfolio construction
            )
        
            # Update market regime in state
            state["market_conditions"] = recommendation.market_regime
        
            # Convert portfolio recommendation to trading signals
            signals = []
            logger.info(f"Converting {len(recommendation.allocations)} allocations to signals")
            
            for allocation in recommendation.allocations:
                logger.info(f"Allocation: {allocation.symbol} weight={allocation.target_weight:.3f} action={allocation.recommended_action}")
                if allocation.target_weight > 0.01:  # Only meaningful allocations
                    # Import here to avoid circular imports
                    from agents.state import TradingSignal
                
                    # Calculate quantity based on target weight
                    target_value = portfolio_value * allocation.target_weight
                
                    # Get current price from market data or fetch it
                    current_price = None
                    
                    # Safely check for existing market data
                    market_symbols = state.get("market_data", {}).get("symbols", {})
                    symbol_data = market_symbols.get(allocation.symbol)
                    
                    if symbol_data and "price" in symbol_data:
                        current_price = symbol_data["price"]
                    else:
                        # Fetch price from Alpaca for symbols not in market data
                        try:
                            from tools.alpaca_client import alpaca_client
                            current_price = alpaca_client.get_current_price(allocation.symbol)
                        except Exception as e:
                            logger.warning(f"Could not get price for {allocation.symbol}: {e}")
                            continue
                    
                    if current_price and current_price > 0:
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
            if "signals" not in state:
                state["signals"] = []
            
            # Debug: Log what we're adding
            logger.info(f"🔍 DEBUG: Adding {len(signals)} signals to state")
            for i, signal in enumerate(signals):
                logger.info(f"   Signal {i+1}: {signal.symbol} {signal.action} qty={signal.quantity:.2f} conf={signal.confidence:.2f}")
            
            state["signals"].extend(signals)
            
            # Debug: Log final state count
            total_signals_after = len(state.get("signals", []))
            logger.info(f"🔍 DEBUG: Total signals in state after adding: {total_signals_after}")
        
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
                    portfolio_value=state["portfolio"].get("equity", 1000.0)
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
    
    async def _fallback_signal_generation(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simple fallback signal generation using basic market data analysis."""
        try:
            logger.info("Using simple fallback signal generation")
            
            portfolio_value = state["portfolio"].get("equity", 26100.0)
            market_symbols = state["market_data"].get("symbols", {})
            
            if not market_symbols:
                logger.warning("No market data available for signal generation")
                state["signals"] = []
                state["messages"].append(AIMessage(content="No signals generated - no market data"))
                state["current_agent"] = "signal_generator"
                return update_state_timestamp(state)
            
            # Generate simple buy signals for available symbols
            signals = []
            target_position_value = portfolio_value * 0.20  # 20% positions
            
            for symbol, data in list(market_symbols.items())[:3]:  # Limit to 3 positions
                if data.get('price', 0) > 0:
                    price = data['price']
                    quantity = int(target_position_value / price)
                    
                    if quantity > 0:
                        # Create basic trading signal
                        from agents.state import TradingSignal
                        signal = TradingSignal(
                            symbol=symbol,
                            action="buy",
                            confidence=0.6,  # Moderate confidence
                            price_target=price * 1.05,  # 5% upside target
                            stop_loss=price * 0.95,     # 5% stop loss
                            quantity=quantity,
                            reasoning=f"Simple fallback signal: diversified position in {symbol}"
                        )
                        signals.append(signal)
            
            state["signals"] = signals
            
            signal_summary = f"Fallback analysis: {len(signals)} buy signals generated"
            state["messages"].append(AIMessage(content=signal_summary))
            
            logger.info(f"Fallback signal generation: {len(signals)} signals created")
            for signal in signals:
                logger.info(f"  - {signal.symbol}: {signal.action} {signal.quantity} shares at ${signal.price_target:.2f}")
            
            state["current_agent"] = "signal_generator" 
            return update_state_timestamp(state)
            
        except Exception as e:
            error_msg = f"Fallback signal generation error: {e}"
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
            
            # Simple strategy optimization: prioritize signals and validate quantities
            logger.info(f"🔍 DEBUG STRATEGY: Optimizing {len(active_signals)} active signals")
            
            # Sort signals by confidence (highest first)
            active_signals.sort(key=lambda x: x.confidence, reverse=True)
            
            # Take top signals based on available buying power and risk limits
            available_cash = state["portfolio"].get("buying_power", 0)
            max_positions = min(len(active_signals), 10)  # Limit to top 10 signals
            
            optimized_signals = []
            for signal in active_signals[:max_positions]:
                # Ensure signal has valid quantity (should already be set by LLM agent)
                if hasattr(signal, 'quantity') and signal.quantity and signal.quantity > 0:
                    optimized_signals.append(signal)
                    logger.info(f"✅ Keeping signal: {signal.symbol} {signal.action} {signal.quantity:.1f} shares (conf: {signal.confidence:.2f})")
                else:
                    logger.warning(f"❌ Skipping signal with invalid quantity: {signal.symbol} qty={getattr(signal, 'quantity', 'None')}")
            
            logger.info(f"🔍 DEBUG STRATEGY: Optimized to {len(optimized_signals)} valid signals")
            
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
            
            # Debug: Check all signals in state
            all_signals = state.get("signals", [])
            logger.info(f"🔍 DEBUG ORDER MGT: Total signals in state: {len(all_signals)}")
            logger.info(f"🔍 DEBUG ORDER MGT: Active signals (within 10 min): {len(active_signals)}")
            
            for i, signal in enumerate(all_signals):
                if hasattr(signal, 'timestamp'):
                    age_minutes = (datetime.now() - signal.timestamp).total_seconds() / 60
                    logger.info(f"Signal {i+1}: {signal.symbol} {signal.action} qty={getattr(signal, 'quantity', 'N/A')} age={age_minutes:.1f}min")
                else:
                    logger.warning(f"Signal {i+1}: {signal.symbol} missing timestamp attribute")
                    logger.warning(f"   Signal type: {type(signal)}, attributes: {dir(signal)}")
            
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
                    # Check if market is open (skip for paper trading as it's always available)
                    market_open = alpaca_client.is_market_open()
                    logger.info(f"Market status: {'Open' if market_open else 'Closed'}")
                    
                    # For paper trading, we can execute orders even when market is closed
                    # In live trading, you'd want to respect market hours
                    if not market_open:
                        logger.info("Market closed - executing paper trade anyway for testing")
                    
                    # Execute the order with advanced order type support
                    logger.info(f"Executing order: {signal.action} {signal.quantity:.2f} {signal.symbol}")
                    order = await self._execute_smart_order(signal, state)
                    
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
            state["executed_orders"] = executed_orders  # For immediate testing access
            
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
        
        # Allow trading even when market is closed - orders will be staged for market open
        # Paper trading accounts can execute at any time
        market_open = state["market_data"].get("market_open", False)
        if not market_open:
            logger.info("Market closed - will stage orders for market open")
        
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

    async def _execute_smart_order(self, signal, state: TradingState) -> Dict:
        """Execute order with intelligent order type selection."""
        from tools.alpaca_client import alpaca_client
        
        try:
            logger.info(f"Executing order for {signal.symbol}: {signal.action} {signal.quantity} shares")
            
            # For reliability, use market orders for all executions in paper trading
            # This ensures trades get filled immediately without price concerns
            order_result = alpaca_client.place_order(
                symbol=signal.symbol,
                qty=signal.quantity,
                side=signal.action,
                order_type="market",
                time_in_force="day"
            )
            
            logger.info(f"Order placed successfully: {order_result}")
            return order_result
                
        except Exception as e:
            logger.error(f"Smart order execution failed for {signal.symbol}: {e}")
            # Fallback to basic market order
            return alpaca_client.place_order(
                symbol=signal.symbol,
                qty=signal.quantity,
                side=signal.action,
                order_type="market"
            )

# Global workflow instance
trading_workflow = TradingWorkflow()