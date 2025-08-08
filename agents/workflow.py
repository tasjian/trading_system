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
            
            # Extract social media data from cached sentiment data if available
            cached_sentiment_data = state.get("cached_sentiment_data", {})
            cached_social_data = {}
            
            # Convert sentiment data to format expected by universe filter
            for symbol, sentiment_data in cached_sentiment_data.items():
                if isinstance(sentiment_data, dict):
                    # Extract social_sentiment data
                    social_sentiment = sentiment_data.get('social_sentiment', {})
                elif hasattr(sentiment_data, 'social_sentiment'):
                    social_sentiment = sentiment_data.social_sentiment
                else:
                    continue
                
                if social_sentiment:
                    # Create mock platform data structure for universe filter
                    symbol_social_data = {}
                    for platform, sentiment_analysis in social_sentiment.items():
                        # Create mock posts list based on platform having sentiment data
                        # Universe filter only needs the count, so we create a mock list
                        symbol_social_data[platform] = [{"mock": True}] * 10  # Assume 10 posts if sentiment exists
                    
                    if symbol_social_data:
                        cached_social_data[symbol] = symbol_social_data
            
            # Apply universe filter to get actionable stocks
            logger.info("🔍 Running universe filter to identify actionable stocks...")
            if cached_social_data:
                logger.info(f"Using cached social media data for {len(cached_social_data)} symbols")
            else:
                logger.info("No cached social media data available - universe filter will skip social signals")
            
            filter_result = await filter_stock_universe(
                base_symbols=None,  # Use full universe
                max_symbols=400,    # Increased to 400 for better signal diversity
                include_watchlist=True,
                cached_social_data=cached_social_data
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
        """Analyze comprehensive sentiment for pre-filtered stocks using enhanced sentiment engine."""
        try:
            from core.enhanced_sentiment_engine import get_enhanced_sentiment_engine
            from core.market_intelligence import UnifiedMarketIntelligence
            
            logger.info("Sentiment Analysis Agent: Running enhanced sentiment analysis with FinGPT on pre-filtered stocks")
            
            # Initialize enhanced sentiment engine
            sentiment_engine = get_enhanced_sentiment_engine()
            await sentiment_engine.initialize()
            
            # Get market intelligence for news data
            market_intel = UnifiedMarketIntelligence()
            
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
                """Analyze sentiment for a single symbol using enhanced engine."""
                try:
                    logger.info(f"Running enhanced FinGPT sentiment analysis for {symbol}")
                    
                    # Generate sample financial news text for sentiment analysis testing
                    # In a real implementation, this would fetch actual news data
                    sample_news_texts = {
                        'AAPL': 'Apple reports strong quarterly earnings beating analyst expectations with record iPhone sales and services revenue growth.',
                        'GOOGL': 'Google announces breakthrough in AI technology with new language model showing impressive performance improvements.',
                        'MSFT': 'Microsoft cloud revenue surges as enterprise customers accelerate digital transformation initiatives.',
                        'TSLA': 'Tesla delivery numbers disappoint investors as production challenges continue at Shanghai facility.',
                        'NVDA': 'NVIDIA stock jumps on strong AI chip demand as data center revenue exceeds expectations.',
                        'AMZN': 'Amazon Web Services growth slows while retail division shows improvement in profitability metrics.',
                        'META': 'Meta stock gains as user engagement metrics improve and metaverse investments show progress.',
                        'NFLX': 'Netflix subscriber growth beats estimates driven by popular original content and password sharing crackdown.'
                    }
                    
                    combined_text = sample_news_texts.get(symbol, f"Market analysis for {symbol} shows mixed sentiment with moderate trading volume and technical indicators suggesting neutral outlook.")
                    
                    # Use enhanced sentiment engine with FinGPT
                    comprehensive_sentiment = await asyncio.wait_for(
                        sentiment_engine.analyze_comprehensive_sentiment(combined_text, symbol),
                        timeout=45.0  # 45 second timeout per symbol for FinGPT processing
                    )
                    return symbol, comprehensive_sentiment
                except asyncio.TimeoutError:
                    logger.warning(f"Enhanced sentiment analysis timed out for {symbol}")
                    return symbol, None
                except asyncio.CancelledError:
                    logger.warning(f"Enhanced sentiment analysis cancelled for {symbol}")
                    return symbol, None
                except Exception as e:
                    logger.error(f"Error in enhanced sentiment analysis for {symbol}: {e}")
                    return symbol, None
            
            # Process symbols in parallel batches
            for i in range(0, len(symbols_to_analyze), batch_size):
                batch = symbols_to_analyze[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {', '.join(batch)}")
                
                # Run batch in parallel with timeout
                try:
                    batch_results = await asyncio.wait_for(
                        asyncio.gather(
                            *[analyze_symbol_sentiment(symbol) for symbol in batch],
                            return_exceptions=True
                        ),
                        timeout=60.0  # 60 second timeout for entire batch
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Batch {i//batch_size + 1} timed out after 60 seconds")
                    batch_results = [(symbol, None) for symbol in batch]
                
                # Process batch results
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch processing error: {result}")
                        continue
                    
                    symbol, comprehensive_sentiment = result
                    
                    if comprehensive_sentiment:
                        sentiment_data[symbol] = {
                            'overall_sentiment': comprehensive_sentiment.get('overall_sentiment', 'neutral'),
                            'overall_score': comprehensive_sentiment.get('overall_score', 0.0),
                            'confidence': comprehensive_sentiment.get('confidence', 0.5),
                            'ensemble_used': comprehensive_sentiment.get('ensemble_used', False),
                            'models_successful': comprehensive_sentiment.get('models_successful', 0),
                            'reasoning': comprehensive_sentiment.get('reasoning', 'Enhanced sentiment analysis'),
                            'model_results': comprehensive_sentiment.get('model_results', {}),
                            'analysis_timestamp': comprehensive_sentiment.get('analysis_timestamp')
                        }
                        
                        # Log detailed sentiment breakdown from enhanced engine
                        logger.info(f"{symbol} enhanced sentiment breakdown:")
                        logger.info(f"  Overall: {comprehensive_sentiment.get('overall_sentiment', 'neutral')} (score: {comprehensive_sentiment.get('overall_score', 0.0):.3f}, confidence: {comprehensive_sentiment.get('confidence', 0.5):.3f})")
                        logger.info(f"  Ensemble: {comprehensive_sentiment.get('ensemble_used', False)} with {comprehensive_sentiment.get('models_successful', 0)} successful models")
                        logger.info(f"  Models: {', '.join(comprehensive_sentiment.get('model_results', {}).keys())}")
                        
                        # Generate trading signals based on enhanced sentiment
                        signal_strength = 0.0
                        signal_type = None
                        signal_reason = []
                        
                        # Extract sentiment values
                        overall_sentiment = comprehensive_sentiment.get('overall_sentiment', 'neutral')
                        overall_score = comprehensive_sentiment.get('overall_score', 0.0)
                        confidence = comprehensive_sentiment.get('confidence', 0.5)
                        
                        # Strong signals based on score thresholds
                        if overall_score > 0.4 and confidence > 0.6:
                            signal_strength += 0.5
                            signal_type = 'BUY'
                            signal_reason.append('Strong positive sentiment with high confidence')
                        elif overall_score < -0.4 and confidence > 0.6:
                            signal_strength += 0.5
                            signal_type = 'SELL'
                            signal_reason.append('Strong negative sentiment with high confidence')
                        
                        # Overall sentiment signals
                        if overall_sentiment == 'positive':
                            signal_strength += 0.3
                            signal_type = 'BUY'
                            signal_reason.append('Positive overall sentiment')
                        elif overall_sentiment == 'negative':
                            signal_strength += 0.3
                            signal_type = 'SELL' 
                            signal_reason.append('Negative overall sentiment')
                        
                        # Enhanced extreme sentiment detection based on numerical scores
                        if overall_score <= -0.7:  # Extremely negative numerical score
                            signal_strength += 0.4
                            signal_type = 'SHORT'  # Override with SHORT if not already set
                            signal_reason.append('EXTREME negative sentiment score (≤-0.7)')
                        elif overall_score <= -0.5:  # Very negative numerical score
                            signal_strength += 0.3
                            signal_type = 'SHORT'
                            signal_reason.append('Strong negative sentiment score (≤-0.5)')
                        elif overall_score >= 0.5:  # Very positive numerical score
                            signal_strength += 0.3
                            signal_type = 'BUY'
                            signal_reason.append('Strong positive sentiment score (≥0.5)')
                        
                        # Crisis/panic sentiment detection through ensemble consensus
                        if (confidence > 0.8 and 
                            overall_score <= -0.6 and 
                            comprehensive_sentiment.get('models_successful', 0) >= 2):
                            signal_strength += 0.4  # Major boost for high-confidence extreme negativity
                            signal_type = 'SHORT'
                            signal_reason.append('CRISIS-LEVEL sentiment - Multi-model consensus')
                        
                        # Confidence and ensemble quality boosts
                        if confidence > 0.7:
                            signal_strength += 0.1
                            signal_reason.append('High confidence analysis')
                        
                        if comprehensive_sentiment.get('ensemble_used', False):
                            signal_strength += 0.1
                            signal_reason.append('Multi-model ensemble analysis')
                        
                        if comprehensive_sentiment.get('models_successful', 0) >= 2:
                            signal_strength += 0.1
                            signal_reason.append('Multiple models successful')
                        
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
                short_signals = [s for s in sentiment_signals if s['signal'] == 'SHORT']
                earnings_signals = [s for s in sentiment_signals if s.get('has_earnings')]
                extreme_negative_signals = [s for s in sentiment_signals if 'EXTREME' in s['reason']]
                crisis_signals = [s for s in sentiment_signals if 'CRISIS-LEVEL' in s['reason']]
                
                logger.info(f"Signal summary: {len(buy_signals)} BUY, {len(sell_signals)} SELL, {len(short_signals)} SHORT, {len(earnings_signals)} with earnings")
                if extreme_negative_signals:
                    logger.info(f"🔥 EXTREME SENTIMENT: {len(extreme_negative_signals)} extreme negative signals detected!")
                if crisis_signals:
                    logger.info(f"🚨 CRISIS SENTIMENT: {len(crisis_signals)} crisis-level panic signals detected!")
                
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
        """Generate trading signals using advanced online RL system with LLM fallbacks."""
        try:
            logger.info("Signal Generation Agent: Performing portfolio analysis")
            
            # PRIORITY 0: Evaluate existing positions for underperformance and generate sell/short signals
            underperformer_signals = await self._evaluate_existing_positions_for_selling(state)
            
            # PRIORITY 1: Try new online RL system (RE-ENABLED with enhanced error handling)
            try:
                logger.info("🤖 Attempting online RL signal generation...")
                rl_result = await self._online_rl_signal_generation(state, config)
                
                # Merge underperformer sell signals with RL buy signals
                if underperformer_signals:
                    current_signals = rl_result.get("signals", [])
                    combined_signals = list(underperformer_signals) + list(current_signals)
                    rl_result["signals"] = combined_signals
                    logger.info(f"🔄 Combined signals: {len(underperformer_signals)} sell/short + {len(current_signals)} RL = {len(combined_signals)} total")
                
                return rl_result
                
            except ImportError as e:
                logger.warning(f"⚠️ Online RL system not available (fallback to legacy): {e}")
            except Exception as e:
                logger.warning(f"⚠️ Online RL signal generation failed (fallback to legacy): {e}")
                # Log the full traceback for debugging but don't crash the system
                import traceback
                logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            # PRIORITY 2: Check legacy RL decisions  
            rl_decisions = state.get("rl_decisions")
            rl_enhanced = state.get("rl_enhanced", False)
            
            if rl_enhanced and rl_decisions and rl_decisions.get("allocations"):
                logger.info("🤖 Using legacy RL portfolio allocations for signal generation")
                
                # Add underperformer signals if not already added
                if not underperformer_signals:
                    underperformer_signals = await self._evaluate_existing_positions_for_selling(state)
                
                # Generate RL signals and merge with sell signals
                rl_result = await self._rl_signal_generation(state, config)
                
                # Merge underperformer sell signals with RL buy signals
                if underperformer_signals:
                    current_signals = rl_result.get("signals", [])
                    combined_signals = list(underperformer_signals) + list(current_signals)
                    rl_result["signals"] = combined_signals
                    logger.info(f"🔄 Combined legacy signals: {len(underperformer_signals)} sell/short + {len(current_signals)} RL = {len(combined_signals)} total")
                
                return rl_result
            
            # PRIORITY 3: Try LLM portfolio construction as fallback
            try:
                from agents.llm_portfolio_management import construct_llm_portfolio
                
                # Add underperformer signals if not already added
                if not underperformer_signals:
                    underperformer_signals = await self._evaluate_existing_positions_for_selling(state)
                
                # Generate LLM signals and merge with sell signals
                llm_result = await self._llm_signal_generation(state, config)
                
                # Merge underperformer sell signals with LLM buy signals
                if underperformer_signals:
                    current_signals = llm_result.get("signals", [])
                    combined_signals = list(underperformer_signals) + list(current_signals)
                    llm_result["signals"] = combined_signals
                    logger.info(f"🔄 Combined LLM signals: {len(underperformer_signals)} sell/short + {len(current_signals)} LLM = {len(combined_signals)} total")
                
                return llm_result
                
            except ImportError:
                logger.info("LLM portfolio management not available, using basic fallback signal generation")
                
                # Add underperformer signals if not already added
                if not underperformer_signals:
                    underperformer_signals = await self._evaluate_existing_positions_for_selling(state)
                
                fallback_result = await self._fallback_signal_generation(state, config)
                
                # Merge underperformer sell signals with fallback signals
                if underperformer_signals:
                    current_signals = fallback_result.get("signals", [])
                    combined_signals = list(underperformer_signals) + list(current_signals)
                    fallback_result["signals"] = combined_signals
                    logger.info(f"🔄 Combined fallback signals: {len(underperformer_signals)} sell/short + {len(current_signals)} fallback = {len(combined_signals)} total")
                
                return fallback_result
        
        except Exception as e:
            error_msg = f"Signal Generation Agent error: {e}"
            logger.error(error_msg)
            return add_error_to_state(state, error_msg)
        
        finally:
            # Ensure cleanup of resources
            try:
                await self._cleanup_signal_generation_resources()
            except Exception as cleanup_error:
                logger.warning(f"Cleanup warning: {cleanup_error}")
    
    async def _rl_signal_generation(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals directly from RL portfolio allocations."""
        try:
            logger.info("🚀 RL-Enhanced Signal Generation")
            
            portfolio_value = state["portfolio"].get("equity", 26100.0)
            available_cash = state["portfolio"].get("cash", 5715.18)
            
            rl_decisions = state.get("rl_decisions", {})
            rl_allocations = rl_decisions.get("allocations", [])
            
            signals = []
            
            for allocation in rl_allocations:
                symbol = allocation.get("symbol")
                target_weight = allocation.get("weight", 0)
                rl_confidence = allocation.get("confidence", 0.7)
                action = allocation.get("action", "buy")
                
                # Only process meaningful allocations
                if abs(target_weight) > 0.01:  # 1% minimum
                    # Determine action and position value
                    if target_weight > 0:
                        signal_action = "buy"
                        # Use available cash for buy positions (respect buying power limits)
                        # Handle margin accounts (negative cash) by using buying power instead
                        portfolio_info = state.get("portfolio", {})
                        buying_power = portfolio_info.get("buying_power", max(0, available_cash))
                        
                        if available_cash < 0:  # Margin account
                            # More aggressive with margin - use more of buying power but stay safe
                            max_position_value = min(buying_power * 0.9, portfolio_value * 0.12)  # Use 90% of buying power, max 12% per position
                            logger.info(f"Margin account detected: using buying power ${buying_power:.2f} instead of cash ${available_cash:.2f}")
                        else:  # Cash account
                            max_position_value = min(available_cash * 0.9, portfolio_value * 0.15)  # Max 15% per position or 90% of cash
                        
                        desired_position_value = portfolio_value * abs(target_weight)
                        position_value = min(max_position_value, desired_position_value)
                        
                        if position_value < desired_position_value:
                            logger.info(f"Position size limited for {symbol}: desired ${desired_position_value:.2f}, using ${position_value:.2f} (available cash: ${available_cash:.2f})")
                    else:
                        # Enhanced short selling strategy
                        signal_action = await self._determine_short_strategy(symbol, target_weight, state)
                        position_value = portfolio_value * abs(target_weight)
                    
                    # Get current price and create signal
                    try:
                        from tools.alpaca_client import alpaca_client
                        current_price = alpaca_client.get_current_price(symbol)
                        
                        if current_price and current_price > 0:
                            quantity = int(position_value / current_price)
                            
                            # Dynamic minimum position size based on available capital
                            min_position_value = min(25, buying_power * 0.1)  # Minimum $25 or 10% of buying power
                            if quantity > 0 and position_value >= min_position_value:
                                from agents.state import TradingSignal
                                
                                price_target = current_price * (1.05 if signal_action == "buy" else 0.95)
                                stop_loss = current_price * (0.95 if signal_action == "buy" else 1.05)
                                
                                signal = TradingSignal(
                                    symbol=symbol,
                                    action=signal_action,
                                    quantity=quantity,
                                    confidence=rl_confidence,
                                    reasoning=f"RL allocation: {target_weight:.1%} ({allocation.get('reasoning', 'RL-driven')}) - Strategy: {rl_decisions.get('strategy', 'rl_portfolio')}",
                                    price_target=price_target,
                                    stop_loss=stop_loss,
                                    timestamp=datetime.now()
                                )
                                
                                signals.append(signal)
                                logger.info(f"✅ RL Signal: {signal_action.upper()} {quantity} {symbol} @ ${current_price:.2f} (target: {target_weight:.1%})")
                    
                    except Exception as e:
                        logger.warning(f"Could not process RL allocation for {symbol}: {e}")
                        continue
            
            # Apply short selling risk management enhancements
            enhanced_signals = await self._enhance_short_signals_with_risk_management(signals, state)
            
            # Update state with enhanced signals
            state["signals"] = enhanced_signals
            state["current_agent"] = "signal_generator"
            
            # Count different signal types
            buy_signals = len([s for s in enhanced_signals if s.action.lower() in ['buy', 'long']])
            sell_signals = len([s for s in enhanced_signals if s.action.lower() in ['sell']])
            short_signals = len([s for s in enhanced_signals if s.action.lower() in ['sell_short', 'short']])
            
            logger.info(f"✅ Generated {len(enhanced_signals)} RL-based trading signals: {buy_signals} BUY, {sell_signals} SELL, {short_signals} SHORT")
            
            if signals:
                state["messages"].append(AIMessage(
                    content=f"Generated {len(signals)} RL-enhanced trading signals using {rl_decisions.get('strategy', 'RL portfolio')} strategy"
                ))
            else:
                state["messages"].append(AIMessage(
                    content="No viable RL trading signals generated - positions too small or prices unavailable"
                ))
            
            return update_state_timestamp(state)
            
        except Exception as e:
            logger.error(f"❌ RL signal generation failed: {e}")
            # Fall back to the existing fallback method
            return await self._fallback_signal_generation(state, config)
    
    async def _evaluate_existing_positions_for_selling(self, state: TradingState) -> List:
        """Evaluate existing portfolio positions and generate sell/short signals for underperformers."""
        try:
            logger.info("📊 Evaluating existing positions for underperformance...")
            
            # Import necessary modules
            from tools.alpaca_client import alpaca_client
            from agents.state import TradingSignal
            from datetime import datetime, timedelta
            
            sell_signals = []
            
            # Get current portfolio positions
            portfolio = state.get("portfolio", {})
            positions = portfolio.get("positions", {})
            
            if not positions:
                logger.info("No existing positions to evaluate")
                return []
            
            # Risk management thresholds
            STOP_LOSS_THRESHOLD = -0.05    # Sell if down 5% or more
            EXTREME_LOSS_THRESHOLD = -0.08  # Short if down 8% or more (extreme underperformance)
            CONCENTRATION_RISK_THRESHOLD = 0.15  # Sell if position > 15% of portfolio
            MOMENTUM_LOSS_THRESHOLD = -0.03  # Sell if down 3% with negative momentum
            
            portfolio_value = portfolio.get("equity", 100000)
            total_signals = 0
            
            for symbol, position_info in positions.items():
                try:
                    # Get position details
                    quantity = float(position_info.get("quantity", 0))
                    market_value = float(position_info.get("market_value", 0))
                    unrealized_pl = float(position_info.get("unrealized_pl", 0))
                    unrealized_plpc = float(position_info.get("unrealized_plpc", 0))
                    
                    if quantity == 0 or market_value == 0:
                        continue
                    
                    # Calculate position concentration
                    position_weight = abs(market_value) / max(portfolio_value, 1)
                    
                    # Get current price for momentum check
                    current_price = alpaca_client.get_current_price(symbol)
                    if not current_price or current_price <= 0:
                        continue
                    
                    # Determine signal based on risk factors
                    risk_factors = []
                    action = None
                    reasoning = []
                    confidence = 0.5
                    
                    # 1. Stop-loss check (most important)
                    if unrealized_plpc <= EXTREME_LOSS_THRESHOLD:
                        action = "sell_short"  # Extreme underperformance - short it
                        risk_factors.append(f"extreme_loss_{unrealized_plpc:.1%}")
                        reasoning.append(f"EXTREME LOSS: {unrealized_plpc:.1%} (threshold: {EXTREME_LOSS_THRESHOLD:.1%})")
                        confidence = 0.9
                        logger.warning(f"🚨 EXTREME UNDERPERFORMER: {symbol} down {unrealized_plpc:.1%} - recommending SHORT")
                        
                    elif unrealized_plpc <= STOP_LOSS_THRESHOLD:
                        action = "sell"
                        risk_factors.append(f"stop_loss_{unrealized_plpc:.1%}")
                        reasoning.append(f"STOP LOSS: {unrealized_plpc:.1%} (threshold: {STOP_LOSS_THRESHOLD:.1%})")
                        confidence = 0.8
                        logger.warning(f"⚠️ UNDERPERFORMER: {symbol} down {unrealized_plpc:.1%} - recommending SELL")
                    
                    # 2. Concentration risk check
                    elif position_weight > CONCENTRATION_RISK_THRESHOLD:
                        action = "sell"
                        risk_factors.append(f"concentration_{position_weight:.1%}")
                        reasoning.append(f"CONCENTRATION RISK: {position_weight:.1%} of portfolio (threshold: {CONCENTRATION_RISK_THRESHOLD:.1%})")
                        confidence = 0.7
                        logger.warning(f"📊 OVER-CONCENTRATED: {symbol} is {position_weight:.1%} of portfolio - recommending partial SELL")
                    
                    # 3. Momentum loss with existing loss
                    elif unrealized_plpc <= MOMENTUM_LOSS_THRESHOLD and unrealized_plpc < 0:
                        # Check if there's negative momentum (simple price trend check)
                        try:
                            # Get sentiment data for momentum proxy
                            sentiment_data = state.get("sentiment_data", {}).get(symbol, {})
                            sentiment_score = sentiment_data.get("overall_score", 0)
                            
                            if sentiment_score < -0.3:  # Negative sentiment + losses = sell
                                action = "sell"
                                risk_factors.append(f"momentum_loss_{unrealized_plpc:.1%}_sentiment_{sentiment_score:.2f}")
                                reasoning.append(f"MOMENTUM LOSS: {unrealized_plpc:.1%} with negative sentiment ({sentiment_score:.2f})")
                                confidence = 0.6
                                logger.info(f"📉 MOMENTUM LOSS: {symbol} down {unrealized_plpc:.1%} with negative sentiment - recommending SELL")
                        except Exception:
                            pass
                    
                    # Generate signal if action determined
                    if action:
                        # Calculate quantity to sell
                        if action == "sell_short":
                            # For shorting, we sell our position and then short more
                            sell_quantity = abs(quantity) * 1.5  # Sell position + short 50% more
                        elif position_weight > CONCENTRATION_RISK_THRESHOLD:
                            # For concentration risk, sell partial position
                            excess_weight = position_weight - (CONCENTRATION_RISK_THRESHOLD * 0.8)  # Target 80% of threshold
                            excess_value = excess_weight * portfolio_value
                            sell_quantity = int(min(abs(quantity), excess_value / current_price))
                        else:
                            # For stop-loss, sell entire position
                            sell_quantity = abs(quantity)
                        
                        # Create trading signal
                        signal = TradingSignal(
                            symbol=symbol,
                            action=action,
                            quantity=sell_quantity,
                            confidence=confidence,
                            reasoning=f"Position Risk Management: {'; '.join(reasoning)}",
                            price_target=current_price * (0.95 if action == "sell" else 0.90),  # Conservative target
                            stop_loss=current_price * (1.02 if action == "sell" else 1.05),   # Tight stop for sells
                            timestamp=datetime.now()
                        )
                        
                        sell_signals.append(signal)
                        total_signals += 1
                        
                        logger.info(f"🎯 SELL SIGNAL: {action.upper()} {sell_quantity:.2f} {symbol} @ ${current_price:.2f} "
                                   f"(P&L: {unrealized_plpc:.1%}, Value: ${market_value:.0f}, Weight: {position_weight:.1%})")
                
                except Exception as e:
                    logger.warning(f"Failed to evaluate position {symbol}: {e}")
                    continue
            
            # Summary
            if sell_signals:
                sell_count = len([s for s in sell_signals if s.action == "sell"])
                short_count = len([s for s in sell_signals if s.action == "sell_short"])
                logger.info(f"🔄 Generated {len(sell_signals)} position management signals: {sell_count} SELL, {short_count} SHORT")
            else:
                logger.info("✅ No underperforming positions requiring immediate action")
            
            return sell_signals
            
        except Exception as e:
            logger.error(f"❌ Position evaluation failed: {e}")
            return []
    
    async def _determine_short_strategy(self, symbol: str, target_weight: float, state: dict) -> str:
        """Determine the most appropriate short selling strategy based on market conditions and analysis."""
        try:
            from tools.alpaca_client import alpaca_client
            
            # Get market data and sentiment for informed short selling
            sentiment_data = state.get("sentiment_data", {}).get(symbol)
            market_data = state.get("market_data", {})
            
            # Default to regular sell
            short_strategy = "sell"
            
            # Enhanced short selling conditions with extreme sentiment detection
            extremely_negative_sentiment = False
            very_negative_sentiment = False
            strong_negative_sentiment = False
            crisis_level_sentiment = False
            technical_bearish_signal = False
            high_volatility = False
            
            # Check sentiment indicators with enhanced extreme detection
            if sentiment_data:
                if hasattr(sentiment_data, 'overall_score'):
                    sentiment_score = sentiment_data.overall_score
                    confidence = getattr(sentiment_data, 'confidence', 0.5)
                    data_sources = getattr(sentiment_data, 'data_sources_count', 1)
                elif isinstance(sentiment_data, dict):
                    sentiment_score = sentiment_data.get('overall_score', 0)
                    confidence = sentiment_data.get('confidence', 0.5)
                    data_sources = sentiment_data.get('data_sources_count', 1)
                else:
                    sentiment_score = 0
                    confidence = 0.5
                    data_sources = 1
                
                # Multi-tier sentiment analysis for short selling
                if sentiment_score <= -0.7:  # Extremely negative
                    extremely_negative_sentiment = True
                    logger.info(f"🔥 EXTREMELY negative sentiment detected for {symbol}: {sentiment_score:.2f}")
                elif sentiment_score <= -0.5:  # Very negative  
                    very_negative_sentiment = True
                    logger.info(f"📉 Very negative sentiment detected for {symbol}: {sentiment_score:.2f}")
                elif sentiment_score <= -0.3:  # Strong negative
                    strong_negative_sentiment = True
                    logger.info(f"📉 Strong negative sentiment detected for {symbol}: {sentiment_score:.2f}")
                
                # Crisis-level sentiment detection (high confidence + extreme negative + multiple sources)
                if (confidence > 0.8 and sentiment_score <= -0.6 and data_sources >= 4):
                    crisis_level_sentiment = True
                    logger.info(f"🚨 CRISIS-LEVEL sentiment detected for {symbol}: score={sentiment_score:.2f}, conf={confidence:.2f}, sources={data_sources}")
            
            # Check technical indicators for bearish signals
            try:
                current_price = alpaca_client.get_current_price(symbol)
                historical_data = alpaca_client.get_market_data(symbol, timeframe="1Day", limit=20)
                
                if current_price and historical_data is not None and len(historical_data) >= 10:
                    # Simple technical analysis for short selling signals
                    recent_high = historical_data['high'].tail(5).max()
                    recent_low = historical_data['low'].tail(5).min()
                    price_range = recent_high - recent_low
                    
                    # Check for breakdown below recent support
                    if current_price <= recent_low * 1.02:  # Within 2% of recent low
                        technical_bearish_signal = True
                        logger.info(f"📉 Technical breakdown detected for {symbol}: price ${current_price:.2f} near recent low ${recent_low:.2f}")
                    
                    # Check for high volatility (good for short selling)
                    if price_range / current_price > 0.15:  # 15% range indicates high volatility
                        high_volatility = True
                        logger.info(f"⚡ High volatility detected for {symbol}: {price_range/current_price:.1%} range")
                
            except Exception as e:
                logger.warning(f"Could not analyze technical indicators for {symbol}: {e}")
            
            # Enhanced confidence calculation with extreme sentiment weighting
            confidence_factors = 0
            strategy_reasoning = []
            
            # Sentiment-based confidence (tiered scoring)
            if crisis_level_sentiment:
                confidence_factors += 3.0  # Maximum confidence for crisis-level sentiment
                strategy_reasoning.append("CRISIS-LEVEL multi-source panic")
                logger.info(f"🚨 Short selling factor: CRISIS-LEVEL sentiment for {symbol}")
            elif extremely_negative_sentiment:
                confidence_factors += 2.5  # Very high confidence for extreme negativity
                strategy_reasoning.append("EXTREMELY negative sentiment")
                logger.info(f"🔥 Short selling factor: EXTREME negative sentiment for {symbol}")
            elif very_negative_sentiment:
                confidence_factors += 2.0  # High confidence for very negative
                strategy_reasoning.append("Very negative sentiment")
                logger.info(f"📉 Short selling factor: Very negative sentiment for {symbol}")
            elif strong_negative_sentiment:
                confidence_factors += 1.0  # Standard confidence for negative
                strategy_reasoning.append("Strong negative sentiment")
                logger.info(f"🎯 Short selling factor: Strong negative sentiment for {symbol}")
            
            # Technical analysis factors
            if technical_bearish_signal:
                confidence_factors += 1.0
                strategy_reasoning.append("Technical breakdown")
                logger.info(f"🎯 Short selling factor: Technical breakdown for {symbol}")
            
            if high_volatility:
                confidence_factors += 0.5
                strategy_reasoning.append("High volatility")
                logger.info(f"🎯 Short selling factor: High volatility for {symbol}")
            
            # Determine strategy based on enhanced confidence scoring
            if confidence_factors >= 3.0:
                short_strategy = "sell_short"  # Maximum aggression for crisis-level signals
                logger.info(f"🚨 MAXIMUM SHORT SELL strategy for {symbol} (confidence: {confidence_factors:.1f}) - {', '.join(strategy_reasoning)}")
            elif confidence_factors >= 2.0:
                short_strategy = "sell_short"  # Aggressive short selling for extreme sentiment
                logger.info(f"🔥 AGGRESSIVE SHORT SELL strategy for {symbol} (confidence: {confidence_factors:.1f}) - {', '.join(strategy_reasoning)}")
            elif confidence_factors >= 1.5:
                short_strategy = "sell_short"  # Moderate short selling
                logger.info(f"📉 MODERATE SHORT SELL strategy for {symbol} (confidence: {confidence_factors:.1f}) - {', '.join(strategy_reasoning)}")
            elif confidence_factors >= 1.0:
                short_strategy = "sell"  # Regular sell for basic negative sentiment
                logger.info(f"📉 Standard SELL strategy for {symbol} (confidence: {confidence_factors:.1f}) - {', '.join(strategy_reasoning)}")
            else:
                # Minimal conviction - just regular position exit
                short_strategy = "sell"
                logger.info(f"🤔 Low conviction SELL for {symbol} (confidence: {confidence_factors:.1f})")
            
            return short_strategy
            
        except Exception as e:
            logger.error(f"Short strategy analysis failed for {symbol}: {e}")
            return "sell"  # Fallback to regular sell
    
    async def _enhance_short_signals_with_risk_management(self, signals: list, state: dict) -> list:
        """Add advanced risk management to short selling signals."""
        try:
            enhanced_signals = []
            
            for signal in signals:
                if signal.action.lower() in ["sell_short", "short"]:
                    # Enhanced risk management for short positions with extreme sentiment handling
                    try:
                        from tools.alpaca_client import alpaca_client
                        current_price = alpaca_client.get_current_price(signal.symbol)
                        
                        if current_price:
                            # Check sentiment data for this symbol to adjust risk parameters
                            sentiment_data = state.get("sentiment_data", {}).get(signal.symbol)
                            sentiment_score = -0.3  # Default moderate negative
                            is_extreme_sentiment = False
                            is_crisis_sentiment = False
                            
                            if sentiment_data:
                                if hasattr(sentiment_data, 'overall_score'):
                                    sentiment_score = sentiment_data.overall_score
                                    confidence = getattr(sentiment_data, 'confidence', 0.5)
                                    data_sources = getattr(sentiment_data, 'data_sources_count', 1)
                                elif isinstance(sentiment_data, dict):
                                    sentiment_score = sentiment_data.get('overall_score', -0.3)
                                    confidence = sentiment_data.get('confidence', 0.5)
                                    data_sources = sentiment_data.get('data_sources_count', 1)
                                
                                # Identify extreme sentiment conditions
                                is_extreme_sentiment = sentiment_score <= -0.5
                                is_crisis_sentiment = (confidence > 0.8 and sentiment_score <= -0.6 and data_sources >= 4)
                            
                            # Adaptive risk management based on sentiment extremeness
                            if is_crisis_sentiment:
                                # Maximum aggression for crisis-level sentiment
                                stop_loss_pct = 0.06  # 6% stop loss (tighter)
                                profit_target_pct = 0.20  # 20% profit target (more aggressive)
                                position_size_multiplier = 1.0  # Full position size
                                risk_label = "CRISIS-LEVEL"
                            elif is_extreme_sentiment:
                                # High aggression for extreme negative sentiment
                                stop_loss_pct = 0.07  # 7% stop loss
                                profit_target_pct = 0.15  # 15% profit target
                                position_size_multiplier = 0.9  # 90% position size
                                risk_label = "EXTREME"
                            else:
                                # Standard short selling risk management
                                stop_loss_pct = 0.08  # 8% stop loss
                                profit_target_pct = 0.12  # 12% profit target
                                position_size_multiplier = 0.75  # 75% position size
                                risk_label = "STANDARD"
                            
                            # Apply risk management parameters
                            signal.stop_loss = current_price * (1 + stop_loss_pct)
                            signal.price_target = current_price * (1 - profit_target_pct)
                            signal.quantity = int(signal.quantity * position_size_multiplier)
                            
                            # Enhanced reasoning for short positions
                            original_reasoning = getattr(signal, 'reasoning', '')
                            signal.reasoning = (f"SHORT SELL ({risk_label}): {original_reasoning} | "
                                              f"SL: {stop_loss_pct:.0%}, PT: {profit_target_pct:.0%} | "
                                              f"Size: {position_size_multiplier:.0%} | Sentiment: {sentiment_score:.2f}")
                            
                            # Boost confidence for extreme sentiment shorts
                            if is_crisis_sentiment:
                                signal.confidence = min(0.98, signal.confidence * 1.3)
                            elif is_extreme_sentiment:
                                signal.confidence = min(0.95, signal.confidence * 1.2)
                            else:
                                signal.confidence = min(0.90, signal.confidence * 1.1)
                            
                            logger.info(f"🔥 Enhanced {risk_label} SHORT: {signal.symbol} qty={signal.quantity} "
                                      f"SL=${signal.stop_loss:.2f} PT=${signal.price_target:.2f} "
                                      f"sentiment={sentiment_score:.2f}")
                    
                    except Exception as e:
                        logger.warning(f"Could not enhance short signal for {signal.symbol}: {e}")
                
                enhanced_signals.append(signal)
            
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"Short signal enhancement failed: {e}")
            return signals  # Return original signals if enhancement fails
    
    async def _online_rl_signal_generation(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signals using the new online RL system with dual-agent architecture."""
        try:
            logger.info("🚀 Online RL Signal Generation - Advanced Learning System")
            
            # Import the integration layer
            from agents.online_rl_integration import generate_rl_enhanced_signals
            
            # Extract market data from state
            market_data = state.get("market_data", {})
            universe_results = state.get("universe_results", {})
            sentiment_data = state.get("sentiment_data", {})
            
            # Build comprehensive market data for RL
            rl_market_data = {}
            symbols = universe_results.get("actionable_symbols", [])
            
            for symbol in symbols:
                symbol_data = {
                    'price': market_data.get(symbol, {}).get('price', 0.0),
                    'price_change_pct': market_data.get(symbol, {}).get('price_change_pct', 0.0),
                    'volume': market_data.get(symbol, {}).get('volume', 0.0),
                    'avg_volume': market_data.get(symbol, {}).get('avg_volume', 0.0),
                    'rsi': market_data.get(symbol, {}).get('rsi', 50.0),
                    'macd': market_data.get(symbol, {}).get('macd', 0.0),
                    'bb_position': market_data.get(symbol, {}).get('bb_position', 0.5)
                }
                
                # Add sentiment data if available
                if symbol in sentiment_data:
                    sentiment_info = sentiment_data[symbol]
                    symbol_data['sentiment_score'] = sentiment_info.get('overall_score', 0.0)
                    symbol_data['news_count'] = len(sentiment_info.get('articles', []))
                else:
                    symbol_data['sentiment_score'] = 0.0
                    symbol_data['news_count'] = 0
                
                rl_market_data[symbol] = symbol_data
            
            # Extract portfolio data
            portfolio = state.get("portfolio", {})
            positions = state.get("positions", [])
            
            portfolio_data = {}
            for position in positions:
                symbol = position.get('symbol')
                portfolio_data[symbol] = {
                    'quantity': position.get('qty', 0.0),
                    'market_value': position.get('market_value', 0.0),
                    'cost_basis': position.get('cost_basis', 0.0)
                }
            
            portfolio_value = portfolio.get("equity", 100000.0)
            
            # Generate RL signals
            rl_signals = await generate_rl_enhanced_signals(
                rl_market_data,
                portfolio_data, 
                portfolio_value,
                symbols
            )
            
            # Convert to workflow format
            signals = []
            for rl_signal in rl_signals:
                # Get current price for quantity calculation
                current_price = rl_market_data.get(rl_signal['symbol'], {}).get('price', 0.0)
                
                if current_price > 0:
                    signal = {
                        'symbol': rl_signal['symbol'],
                        'action': rl_signal['action'],
                        'quantity': rl_signal['quantity'],
                        'price': current_price,
                        'confidence': rl_signal['confidence'],
                        'reasoning': rl_signal['reasoning'],
                        'strategy': 'online_rl',
                        'priority': 'high',  # RL signals get high priority
                        'rl_score': rl_signal.get('rl_score', 0.0),
                        'regime': rl_signal.get('regime', 'unknown'),
                        'uncertainty': rl_signal.get('uncertainty', 0.0),
                        'timestamp': datetime.now()
                    }
                    signals.append(signal)
            
            # Update state with RL signals
            state["trading_signals"] = signals
            state["signals_generated"] = len(signals)
            state["rl_enhanced"] = True
            state["signal_generation_method"] = "online_rl"
            
            logger.info(f"✨ Generated {len(signals)} online RL signals with dual-agent system")
            
            # Log signal summary
            if signals:
                buy_signals = len([s for s in signals if s['action'] == 'buy'])
                sell_signals = len([s for s in signals if s['action'] == 'sell'])
                avg_confidence = sum(s['confidence'] for s in signals) / len(signals)
                logger.info(f"Signal breakdown: {buy_signals} buy, {sell_signals} sell, avg confidence: {avg_confidence:.2f}")
            
            return update_state_timestamp(state)
            
        except Exception as e:
            logger.error(f"❌ Online RL signal generation failed: {e}")
            # Import error or system not available - fall back to legacy methods
            raise  # Re-raise to trigger fallback in main method
    
    async def _cleanup_signal_generation_resources(self):
        """Clean up resources used in signal generation."""
        try:
            from core.market_intelligence import cleanup_market_intelligence
            await cleanup_market_intelligence()
        except Exception as e:
            logger.debug(f"Market intelligence cleanup: {e}")
        
        try:
            from core.social_media_collector_optimized import SocialMediaCollector
            # Social media collector cleanup is handled in the collector itself
        except Exception as e:
            logger.debug(f"Social media cleanup: {e}")
        
        try:
            # Cleanup online RL agent if it exists
            from agents.online_rl_integration import cleanup_rl_agent
            await cleanup_rl_agent()
        except Exception as e:
            logger.debug(f"Online RL cleanup: {e}")
    
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
            
            # Get available buying power from state
            available_cash = state["portfolio"].get("buying_power", 0)
            logger.info(f"Available buying power: ${available_cash:,.2f}")
            
            if available_cash <= 0:
                logger.warning("No buying power available - skipping signal generation")
                return add_error_to_state(state, "No buying power available for new positions")
            
            for allocation in recommendation.allocations:
                logger.info(f"Allocation: {allocation.symbol} weight={allocation.target_weight:.3f} action={allocation.recommended_action}")
                if allocation.target_weight > 0.01:  # Only meaningful allocations
                    # Import here to avoid circular imports
                    from agents.state import TradingSignal
                
                    # Calculate quantity based on target weight using available cash instead of total portfolio
                    target_value = available_cash * allocation.target_weight
                
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
                        quantity = int(target_value / current_price)
                        
                        # Final safety check - ensure total cost doesn't exceed available cash
                        total_cost = quantity * current_price
                        if total_cost > available_cash:
                            quantity = int((available_cash * 0.95) / current_price)  # Use 95% for safety margin
                            logger.info(f"Adjusted {allocation.symbol} position to fit available cash: {quantity:.2f} shares (${total_cost:.2f})")
                        
                        # Minimum quantity check
                        if quantity < 1.0:
                            logger.warning(f"Skipping {allocation.symbol} - insufficient cash for minimum 1 share (need ${current_price:.2f}, have ${available_cash:.2f})")
                            continue
                    
                        signal = TradingSignal(
                            symbol=allocation.symbol,
                            action=allocation.recommended_action,
                            confidence=allocation.confidence,
                            price_target=current_price * 1.1 if allocation.recommended_action == "buy" else current_price * 0.9,
                            stop_loss=current_price * 0.95 if allocation.recommended_action == "buy" else current_price * 1.05,
                            quantity=quantity,
                            reasoning=f"LLM Portfolio Construction: {allocation.reasoning} (${total_cost:.2f} of ${available_cash:.2f} available)"
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
        """Simple fallback signal generation using sentiment signals."""
        try:
            logger.info("Using simple fallback signal generation")
            
            portfolio_value = state["portfolio"].get("equity", 26100.0)
            available_cash = state["portfolio"].get("cash", 5715.18)
            
            # Use sentiment signals as the primary source for trading signals
            sentiment_signals = state.get("sentiment_signals", [])
            trading_signals = state.get("trading_signals", [])
            
            # Check if RL decisions are available first (RL-Enhanced Path)
            rl_decisions = state.get("rl_decisions")
            rl_enhanced = state.get("rl_enhanced", False)
            
            # Only exit early if we have no sentiment signals AND no RL decisions
            if not sentiment_signals and not trading_signals and not (rl_enhanced and rl_decisions):
                logger.warning("No sentiment signals or RL decisions available for signal generation")
                state["signals"] = []
                state["messages"].append(AIMessage(content="No signals generated - no sentiment data or RL decisions"))
                state["current_agent"] = "signal_generator"
                return update_state_timestamp(state)
            
            logger.info(f"Signal generation sources: {len(sentiment_signals)} sentiment, {len(trading_signals)} trading, RL={'enabled' if rl_enhanced else 'disabled'}")
            
            signals = []
            
            if rl_enhanced and rl_decisions:
                # Enhanced RL Signal Generation Path (with short selling support)
                finrl_integrated = state.get("finrl_integrated", False)
                
                if finrl_integrated:
                    logger.info("🚀 Using FinRL enhanced portfolio allocations with advanced features")
                else:
                    logger.info("🤖 Using basic RL portfolio allocations for signal generation")
                
                rl_allocations = rl_decisions.get("allocations", [])
                advanced_features = rl_decisions.get("advanced_features", {})
                
                for allocation in rl_allocations:
                    symbol = allocation.get("symbol")
                    target_weight = allocation.get("weight", 0)
                    rl_confidence = allocation.get("confidence", 0.7)
                    action = allocation.get("action", "buy")
                    order_type = allocation.get("order_type", "market")
                    limit_price = allocation.get("limit_price")
                    
                    # Process both positive and negative allocations (for short selling)
                    if abs(target_weight) > 0.01:  # Only meaningful allocations
                        # Determine action based on weight and RL decision
                        if target_weight > 0:
                            signal_action = "buy"
                            position_value = available_cash * target_weight
                        else:
                            # Short selling support
                            if advanced_features.get("short_selling", False) and action in ["short", "sell"]:
                                signal_action = "short"
                                # For short positions, limit to available cash and position size limits
                                # Handle margin accounts (negative cash) by using buying power instead
                                portfolio_info = state.get("portfolio", {})
                                buying_power = portfolio_info.get("buying_power", max(0, available_cash))
                                
                                if available_cash < 0:  # Margin account
                                    max_position_value = min(buying_power * 0.8, portfolio_value * 0.10)  # Conservative for margin
                                    logger.info(f"Margin account detected for short: using buying power ${buying_power:.2f} instead of cash ${available_cash:.2f}")
                                else:  # Cash account
                                    max_position_value = min(available_cash * 0.9, portfolio_value * 0.15)  # Max 15% per position or 90% of cash
                                
                                desired_position_value = portfolio_value * abs(target_weight)
                                position_value = min(max_position_value, desired_position_value)
                                
                                if position_value < desired_position_value:
                                    logger.info(f"Short position size limited for {symbol}: desired ${desired_position_value:.2f}, using ${position_value:.2f} (available cash: ${available_cash:.2f})")
                            else:
                                signal_action = "sell"
                                position_value = portfolio_value * abs(target_weight)
                        
                        # Get current price
                        try:
                            from tools.alpaca_client import alpaca_client
                            current_price = alpaca_client.get_current_price(symbol)
                            
                            if current_price and current_price > 0:
                                quantity = int(position_value / current_price)
                                
                                # Validate position requirements
                                can_execute = False
                                if signal_action == "buy":
                                    can_execute = quantity > 0 and quantity * current_price <= available_cash
                                elif signal_action == "short":
                                    # Short selling validation (margin requirements)
                                    margin_requirement = quantity * current_price * 0.5  # 50% margin
                                    can_execute = quantity > 0 and margin_requirement <= available_cash
                                elif signal_action == "sell":
                                    can_execute = quantity > 0  # Assume we can sell existing positions
                                
                                if can_execute:
                                    from agents.state import TradingSignal
                                    
                                    # Set price targets based on position type
                                    if signal_action in ["buy"]:
                                        price_target = current_price * 1.05  # 5% upside for long
                                        stop_loss = current_price * 0.95     # 5% stop loss
                                    elif signal_action == "short":
                                        price_target = current_price * 0.95  # 5% downside target for short
                                        stop_loss = current_price * 1.05     # 5% stop loss (upward)
                                    else:  # sell
                                        price_target = current_price * 0.98  # Slightly below current
                                        stop_loss = current_price * 1.02     # Small upward stop
                                    
                                    # Override with limit price if provided
                                    if limit_price and order_type == "limit":
                                        price_target = limit_price
                                    
                                    signal = TradingSignal(
                                        symbol=symbol,
                                        action=signal_action,
                                        confidence=rl_confidence,
                                        price_target=price_target,
                                        stop_loss=stop_loss,
                                        quantity=quantity,
                                        reasoning=f"RL {action}: {target_weight:.1%} portfolio weight"
                                    )
                                    signals.append(signal)
                                    
                                    # Update available cash based on action
                                    if signal_action == "buy":
                                        available_cash -= quantity * current_price
                                    elif signal_action == "short":
                                        # Short selling reduces buying power by margin requirement
                                        available_cash -= quantity * current_price * 0.5
                                    
                                    # Enhanced logging
                                    order_info = f"({order_type}"
                                    if limit_price:
                                        order_info += f" @ ${limit_price:.2f}"
                                    order_info += ")"
                                    
                                    if finrl_integrated:
                                        logger.info(f"🚀 FinRL Signal: {symbol} {signal_action.upper()} {quantity} shares {order_info}")
                                    else:
                                        logger.info(f"🎯 RL Signal: {symbol} {signal_action.upper()} {quantity} shares ({target_weight:.1%})")
                                    
                                    # Log short selling specifically
                                    if signal_action == "short":
                                        logger.info(f"   ⚡ SHORT POSITION: {symbol} targeting ${price_target:.2f}")
                        
                        except Exception as e:
                            logger.warning(f"Could not get price for RL allocation {symbol}: {e}")
                            continue
                
                logger.info(f"✅ Generated {len(signals)} RL-enhanced trading signals (including shorts)")
                
            else:
                # Traditional Sentiment-Based Path (Fallback)
                logger.info("📊 Using sentiment-based signal generation (RL not available)")
                max_position_value = min(available_cash * 0.8, portfolio_value * 0.15)  # Max 15% per position
                
                # Process top sentiment signals (limit to 5 to avoid over-concentration)
                top_sentiment_signals = [s for s in sentiment_signals if s.get('signal') == 'BUY'][:5]
                
                logger.info(f"Converting {len(top_sentiment_signals)} sentiment signals to trading signals")
                
                for sentiment_signal in top_sentiment_signals:
                    symbol = sentiment_signal['symbol']
                    strength = sentiment_signal.get('strength', 0.3)
                    
                    # Get current price from Alpaca
                    try:
                        from tools.alpaca_client import alpaca_client
                        current_price = alpaca_client.get_current_price(symbol)
                        
                        if current_price and current_price > 0:
                            # Calculate position size based on sentiment strength
                            position_value = max_position_value * strength
                            quantity = int(position_value / current_price)
                            
                            if quantity > 0 and quantity * current_price <= available_cash:
                                # Create trading signal
                                from agents.state import TradingSignal
                                signal = TradingSignal(
                                    symbol=symbol,
                                    action="buy",
                                    confidence=strength,
                                    price_target=current_price * 1.05,  # 5% upside target
                                    stop_loss=current_price * 0.95,     # 5% stop loss
                                    quantity=quantity,
                                    reasoning=sentiment_signal.get('reason', f"Positive sentiment signal for {symbol}")
                                )
                                signals.append(signal)
                                available_cash -= quantity * current_price  # Reduce available cash
                                
                                logger.info(f"📊 Sentiment Signal: {symbol} - {quantity} shares at ${current_price:.2f}")
                        
                    except Exception as e:
                        logger.warning(f"Could not get price for {symbol}: {e}")
                        continue
            
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
                               portfolio_value: float, buying_power: float = None) -> Optional[float]:
        """Calculate appropriate position size based on analysis and portfolio optimization."""
        try:
            if portfolio_value <= 0 or not analysis.indicators:
                return None
            
            # Get current price
            current_price = analysis.indicators.get('current_price', 0)
            if current_price <= 0:
                return None
            
            # Use buying power if provided, otherwise fall back to portfolio value
            available_cash = buying_power if buying_power is not None else portfolio_value
            
            # Ensure we have sufficient cash
            if available_cash <= 0:
                logger.warning(f"No available cash for {symbol}: ${available_cash}")
                return None
            
            # Base position size from optimal weights - use available cash instead of total portfolio
            optimal_weight = optimal_weights.get(symbol, settings.max_position_size / 2)
            base_position_value = available_cash * optimal_weight
            
            # Adjust based on signal confidence
            confidence_multiplier = min(1.0, analysis.confidence * 1.5)
            adjusted_position_value = base_position_value * confidence_multiplier
            
            # Calculate quantity
            quantity = int(adjusted_position_value / current_price)
            
            # Apply minimum and maximum limits
            min_quantity = 1.0  # Minimum 1 share
            # Maximum based on available cash, not total portfolio
            max_quantity = int((available_cash * settings.max_position_size) / current_price)
            
            final_quantity = max(min_quantity, min(quantity, max_quantity))
            
            # Final safety check - ensure we don't exceed available cash
            total_cost = final_quantity * current_price
            if total_cost > available_cash:
                final_quantity = int(available_cash / current_price)
                logger.info(f"Adjusted {symbol} position size to fit available cash: {final_quantity:.2f} shares (${total_cost:.2f})")
            
            return final_quantity
            
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
        """Execute balanced trading orders with cash preservation."""
        try:
            logger.info("Order Management Agent: Executing balanced trades")
            
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
            
            if not active_signals:
                state["messages"].append(AIMessage(content="No signals ready for execution"))
                state["current_agent"] = "order_manager"
                return update_state_timestamp(state)
            
            # Import here to avoid circular imports
            from tools.alpaca_client import alpaca_client
            
            # Get portfolio information for balancing
            portfolio = state.get("portfolio", {})
            available_cash = max(0, portfolio.get("cash", 0))
            buying_power = portfolio.get("buying_power", 0)
            current_positions = portfolio.get("positions", {})
            
            logger.info(f"💰 Cash Management: Available=${available_cash:,.2f}, Buying Power=${buying_power:,.2f}")
            logger.info(f"📊 Current Positions: {len(current_positions)}")
            
            # Apply cash preservation and order balancing
            balanced_signals = await self._balance_orders_for_cash_preservation(
                active_signals, available_cash, buying_power, current_positions
            )
            
            logger.info(f"⚖️ Order Balancing: {len(active_signals)} → {len(balanced_signals)} signals")
            
            executed_orders = []
            for signal in balanced_signals:
                if signal.action == "hold" or not signal.quantity:
                    continue
                
                try:
                    # Check if market is open (skip for paper trading as it's always available)
                    market_open = alpaca_client.is_market_open()
                    
                    # For paper trading, we can execute orders even when market is closed
                    if not market_open:
                        logger.info("Market closed - executing paper trade anyway for testing")
                    
                    # Execute the order with advanced order type support
                    logger.info(f"Executing balanced order: {signal.action} {signal.quantity:.2f} {signal.symbol}")
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
            state["executed_orders"] = executed_orders
            
            # Calculate balance metrics
            buy_orders = len([o for o in executed_orders if 'buy' in str(o.get('order', {}).get('side', '')).lower()])
            sell_orders = len([o for o in executed_orders if 'sell' in str(o.get('order', {}).get('side', '')).lower()])
            
            order_summary = f"Executed {len(executed_orders)} balanced orders ({buy_orders} BUY, {sell_orders} SELL)"
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
            # Ensure quantity is always an integer (no fractional shares)
            original_quantity = signal.quantity
            integer_quantity = max(1, int(float(signal.quantity)))  # Convert to int, minimum 1 share
            
            logger.info(f"🔧 Quantity conversion: {signal.symbol} {original_quantity} → {integer_quantity}")
            
            # Force update the signal quantity
            signal.quantity = integer_quantity
            
            if original_quantity != integer_quantity:
                logger.info(f"✅ Rounded fractional quantity for {signal.symbol}: {original_quantity} → {integer_quantity} shares")
            
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
    
    async def llm_portfolio_agent(self, state: TradingState, config: Dict[str, Any]) -> TradingState:
        """LLM-powered portfolio management agent."""
        logger.info("🤖 Running LLM Portfolio Agent")
        
        try:
            # Import the LLM portfolio management module
            from agents.llm_portfolio_management import construct_llm_portfolio
            
            # Get current portfolio state
            current_positions = state.get("positions", {})
            portfolio_value = state.get("portfolio", {}).get("equity", 10000)
            
            # Get available candidate symbols
            candidate_symbols = state.get("filtered_symbols", state.get("watchlist", []))
            
            # Generate portfolio recommendations
            recommendations = await construct_llm_portfolio(
                candidate_symbols=candidate_symbols[:20],  # Limit for testing
                portfolio_value=portfolio_value,
                risk_profile="moderate",
                max_positions=5,
                sentiment_data=state.get("sentiment_data", {})
            )
            
            # Update state with LLM recommendations
            state["llm_recommendations"] = recommendations
            state["llm_analysis_complete"] = True
            
            logger.info(f"✅ LLM Portfolio Agent completed with {len(recommendations.allocations)} recommendations")
            
        except Exception as e:
            logger.error(f"❌ LLM Portfolio Agent error: {e}")
            state["llm_analysis_complete"] = False
            state["errors"].append(f"LLM Portfolio Agent: {str(e)}")
        
        return update_state_timestamp(state)
    
    async def _balance_orders_for_cash_preservation(self, signals, available_cash: float, 
                                                   buying_power: float, current_positions: dict) -> list:
        """Balance buy/sell orders to preserve cash and maintain adequate buying power."""
        try:
            from tools.alpaca_client import alpaca_client
            
            # Separate buy, sell, and short signals
            buy_signals = [s for s in signals if s.action.lower() in ['buy', 'long']]
            sell_signals = [s for s in signals if s.action.lower() in ['sell']]
            short_signals = [s for s in signals if s.action.lower() in ['sell_short', 'short']]
            all_sell_signals = sell_signals + short_signals
            
            logger.info(f"📊 Order Balancing Input: {len(buy_signals)} BUY, {len(sell_signals)} SELL, {len(short_signals)} SHORT signals")
            
            # Calculate total buy order value needed
            total_buy_value = 0
            valid_buy_signals = []
            
            for signal in buy_signals:
                try:
                    current_price = alpaca_client.get_current_price(signal.symbol)
                    if current_price and signal.quantity:
                        order_value = current_price * signal.quantity
                        total_buy_value += order_value
                        valid_buy_signals.append((signal, order_value))
                        logger.info(f"   BUY {signal.symbol}: {signal.quantity:.2f} @ ${current_price:.2f} = ${order_value:,.2f}")
                except Exception as e:
                    logger.warning(f"Could not price buy signal {signal.symbol}: {e}")
            
            # Calculate short selling margin requirements
            total_short_margin_needed = 0
            for signal in short_signals:
                try:
                    current_price = alpaca_client.get_current_price(signal.symbol)
                    if current_price and signal.quantity:
                        short_value = current_price * signal.quantity
                        margin_required = short_value * 0.5  # 50% margin requirement for shorts
                        total_short_margin_needed += margin_required
                        logger.info(f"   SHORT {signal.symbol}: {signal.quantity:.2f} @ ${current_price:.2f} = ${short_value:,.2f} (margin: ${margin_required:,.2f})")
                except Exception as e:
                    logger.warning(f"Could not calculate short margin for {signal.symbol}: {e}")
            
            # Calculate available cash for purchases (conservative approach)
            effective_cash = max(0, available_cash)
            if buying_power > effective_cash:
                # Use buying power but with margin safety buffer and short selling margin
                available_margin = buying_power - total_short_margin_needed
                effective_cash = min(available_margin * 0.7, effective_cash + (available_margin - effective_cash) * 0.5)
            
            logger.info(f"💰 Effective cash available: ${effective_cash:,.2f} (from cash=${available_cash:,.2f}, BP=${buying_power:,.2f})")
            logger.info(f"💸 Total buy orders needed: ${total_buy_value:,.2f}")
            logger.info(f"🔥 Short margin required: ${total_short_margin_needed:,.2f}")
            
            balanced_signals = []
            
            # Strategy 1: If cash is sufficient for buys and margin for shorts, execute all orders
            total_cash_needed = total_buy_value + total_short_margin_needed
            if total_buy_value <= effective_cash and total_short_margin_needed <= (buying_power * 0.8):
                logger.info("✅ Sufficient cash and margin available - executing all orders")
                balanced_signals.extend(buy_signals)
                balanced_signals.extend(sell_signals)
                balanced_signals.extend(short_signals)
                
            # Strategy 2: Cash/margin insufficient - need to balance orders strategically
            else:
                cash_deficit = total_buy_value - effective_cash
                margin_deficit = total_short_margin_needed - (buying_power * 0.8)
                logger.warning(f"⚠️ Resource deficit - Cash: ${cash_deficit:,.2f}, Margin: ${margin_deficit:,.2f} - implementing balancing strategy")
                
                # Add all sell signals first to free up cash (prioritize regular sells over shorts)
                balanced_signals.extend(sell_signals)
                
                # Add short signals only if margin is available
                if margin_deficit <= 0:
                    balanced_signals.extend(short_signals)
                    logger.info("✅ Adding all short positions - sufficient margin available")
                else:
                    # Prioritize short signals by confidence and fit within available margin
                    logger.warning(f"⚠️ Insufficient margin for all shorts - prioritizing by confidence")
                    short_signals_sorted = sorted(short_signals, key=lambda s: getattr(s, 'confidence', 0.5), reverse=True)
                    remaining_margin = buying_power * 0.8
                    
                    for signal in short_signals_sorted:
                        try:
                            current_price = alpaca_client.get_current_price(signal.symbol)
                            if current_price:
                                margin_needed = (current_price * signal.quantity) * 0.5
                                if margin_needed <= remaining_margin:
                                    balanced_signals.append(signal)
                                    remaining_margin -= margin_needed
                                    logger.info(f"✅ Including short: {signal.symbol} (margin: ${margin_needed:,.2f})")
                                else:
                                    logger.warning(f"❌ Skipping short: {signal.symbol} (needs ${margin_needed:,.2f}, have ${remaining_margin:,.2f})")
                        except Exception as e:
                            logger.warning(f"Could not process short signal {signal.symbol}: {e}")
                
                # Estimate cash from sell orders
                sell_proceeds = 0
                for signal in sell_signals:
                    try:
                        if signal.symbol in current_positions:
                            # Selling existing position
                            pos_qty = abs(float(current_positions[signal.symbol].get('qty', 0)))
                            current_price = alpaca_client.get_current_price(signal.symbol)
                            if current_price:
                                proceed_estimate = min(signal.quantity, pos_qty) * current_price
                                sell_proceeds += proceed_estimate
                                logger.info(f"   Sell proceeds from {signal.symbol}: ${proceed_estimate:,.2f}")
                    except Exception as e:
                        logger.warning(f"Could not estimate sell proceeds for {signal.symbol}: {e}")
                
                # Add sell proceeds to available cash
                total_available_cash = effective_cash + sell_proceeds * 0.9  # 90% of proceeds (conservative)
                logger.info(f"💰 Total cash after sells: ${total_available_cash:,.2f}")
                
                # Prioritize buy signals by confidence and fit within available cash
                valid_buy_signals.sort(key=lambda x: getattr(x[0], 'confidence', 0.5), reverse=True)
                
                remaining_cash = total_available_cash
                for signal, order_value in valid_buy_signals:
                    if order_value <= remaining_cash:
                        balanced_signals.append(signal)
                        remaining_cash -= order_value
                        logger.info(f"✅ Including buy order: {signal.symbol} ${order_value:,.2f} (remaining: ${remaining_cash:,.2f})")
                    else:
                        # Try to scale down the order to fit
                        if remaining_cash > 100:  # Minimum $100 order
                            try:
                                current_price = alpaca_client.get_current_price(signal.symbol)
                                if current_price:
                                    adjusted_qty = int(remaining_cash / current_price)
                                    if adjusted_qty > 0:
                                        signal.quantity = adjusted_qty
                                        signal.reasoning = f"{getattr(signal, 'reasoning', '')} [Cash-adjusted]"
                                        balanced_signals.append(signal)
                                        remaining_cash = 0
                                        logger.info(f"📉 Scaled buy order: {signal.symbol} to {adjusted_qty} shares")
                            except Exception as e:
                                logger.warning(f"Could not scale order for {signal.symbol}: {e}")
                        else:
                            logger.warning(f"❌ Skipping buy order: {signal.symbol} ${order_value:,.2f} (insufficient cash: ${remaining_cash:,.2f})")
                
                # Add forced sell signals if still not enough cash
                if remaining_cash < 0:
                    logger.warning("🚨 Still insufficient cash - adding forced position trims")
                    await self._add_forced_sell_signals(balanced_signals, current_positions, abs(remaining_cash))
            
            final_buys = len([s for s in balanced_signals if s.action.lower() in ['buy', 'long']])
            final_sells = len([s for s in balanced_signals if s.action.lower() in ['sell']])
            final_shorts = len([s for s in balanced_signals if s.action.lower() in ['sell_short', 'short']])
            
            logger.info(f"⚖️ Final order balance: {final_buys} BUY, {final_sells} SELL, {final_shorts} SHORT")
            
            return balanced_signals
            
        except Exception as e:
            logger.error(f"Order balancing failed: {e}")
            # Fallback: return original signals but prioritize sells, then shorts, then limited buys
            sell_signals = [s for s in signals if s.action.lower() in ['sell']]
            short_signals = [s for s in signals if s.action.lower() in ['sell_short', 'short']]
            buy_signals = [s for s in signals if s.action.lower() in ['buy', 'long']]
            return sell_signals + short_signals[:2] + buy_signals[:2]  # Conservative fallback
    
    async def _add_forced_sell_signals(self, balanced_signals: list, current_positions: dict, cash_needed: float):
        """Add forced sell signals to free up additional cash when needed."""
        try:
            from tools.alpaca_client import alpaca_client
            from agents.state import TradingSignal
            
            logger.info(f"🚨 Adding forced sells to free up ${cash_needed:,.2f}")
            
            # Sort positions by unrealized P&L (sell losers first to preserve winners)
            position_candidates = []
            
            for symbol, position in current_positions.items():
                try:
                    unrealized_pnl = float(position.get('unrealized_pl', 0))
                    qty = float(position.get('qty', 0))
                    market_value = abs(float(position.get('market_value', 0)))
                    
                    if qty != 0 and market_value > 200:  # Only positions worth > $200
                        position_candidates.append({
                            'symbol': symbol,
                            'qty': abs(qty),
                            'unrealized_pnl': unrealized_pnl,
                            'market_value': market_value,
                            'pnl_pct': (unrealized_pnl / market_value) if market_value > 0 else 0
                        })
                except Exception as e:
                    logger.warning(f"Could not analyze position {symbol}: {e}")
            
            # Sort by unrealized P&L (negative first = biggest losers first)
            position_candidates.sort(key=lambda x: x['pnl_pct'])
            
            cash_freed = 0
            forced_sells = 0
            
            for pos in position_candidates:
                if cash_freed >= cash_needed:
                    break
                    
                # Sell 50% of position to preserve some upside
                sell_qty = max(1, int(pos['qty'] * 0.5))
                sell_value = pos['market_value'] * 0.5
                
                # Create forced sell signal
                forced_signal = TradingSignal(
                    symbol=pos['symbol'],
                    action="sell",
                    confidence=0.9,  # High confidence for cash management
                    price_target=None,
                    quantity=sell_qty,
                    reasoning=f"Forced sell for cash management (P&L: {pos['pnl_pct']:+.1%})"
                )
                
                balanced_signals.insert(0, forced_signal)  # Insert at beginning for priority
                cash_freed += sell_value
                forced_sells += 1
                
                logger.info(f"🔥 Forced sell: {pos['symbol']} {sell_qty} shares ≈ ${sell_value:,.2f} "
                           f"(P&L: {pos['pnl_pct']:+.1%})")
            
            logger.info(f"✅ Added {forced_sells} forced sells to free up ≈${cash_freed:,.2f}")
            
        except Exception as e:
            logger.error(f"Failed to add forced sell signals: {e}")

# Global workflow instance
trading_workflow = TradingWorkflow()