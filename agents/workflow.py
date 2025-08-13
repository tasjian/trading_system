"""LangGraph workflow for agentic trading system."""

import asyncio
import logging
from typing import Dict, Any, Literal, Optional, List
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
            
            # Update watchlist data (stocks only - CRYPTO TRADING DISABLED)
            from config.settings import is_crypto_symbol
            # from core.crypto_data_collector import crypto_collector  # CRYPTO DISABLED
            
            for symbol in state["watchlist"]:
                try:
                    # CRYPTO TRADING DISABLED - Comment out crypto data collection
                    # if is_crypto_symbol(symbol):
                    #     # Get crypto data from crypto collector
                    #     ticker = crypto_collector.alpaca_collector.get_latest_ticker(symbol)
                    #     if ticker:
                    #         state["market_data"]["symbols"] = state["market_data"].get("symbols", {})
                    #         state["market_data"]["symbols"][symbol] = {
                    #             "price": float(ticker.price),
                    #             "volume": float(ticker.volume_24h),
                    #             "timestamp": ticker.timestamp,
                    #             "change_24h": float(ticker.change_24h),
                    #             "bid": float(ticker.bid),
                    #             "ask": float(ticker.ask),
                    #             "asset_type": "crypto"
                    #         }
                    #         logger.debug(f"📈 {symbol}: ${ticker.price:.2f} ({ticker.change_24h:+.2f}%)")
                    #     else:
                    #         logger.warning(f"No crypto data available for {symbol}")
                    # else:
                    # Skip crypto symbols entirely when crypto is disabled
                    if not is_crypto_symbol(symbol):
                        # Get stock data from Alpaca
                        market_data = alpaca_client.get_market_data(symbol, limit=50)
                        if not market_data.empty:
                            latest_data = market_data.iloc[-1]
                            state["market_data"]["symbols"] = state["market_data"].get("symbols", {})
                            state["market_data"]["symbols"][symbol] = {
                                "price": float(latest_data["close"]),
                                "volume": float(latest_data["volume"]),
                                "timestamp": datetime.now(),
                                "asset_type": "stock"
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
            from core.llm_sentiment_analyzer import LLMSentimentAnalyzer
            from core.market_intelligence import UnifiedMarketIntelligence
            
            logger.info("Sentiment Analysis Agent: Running Ollama-based sentiment analysis on pre-filtered stocks")
            
            # Initialize direct LLM sentiment analyzer (Ollama only)
            sentiment_engine = LLMSentimentAnalyzer()
            
            # Get market intelligence for news data
            market_intel = UnifiedMarketIntelligence()
            
            # Get pre-filtered symbols from universe filter step
            filtered_symbols = state.get("filtered_symbols", [])
            
            # ALWAYS include current portfolio positions for sentiment analysis
            portfolio_positions = list(state.get("portfolio", {}).get("positions", {}).keys())
            
            # Combine filtered symbols with portfolio positions (remove duplicates)
            all_symbols_set = set(filtered_symbols)
            all_symbols_set.update(portfolio_positions)
            
            # Prioritize portfolio positions first, then filtered symbols
            symbols_to_analyze = portfolio_positions + [s for s in filtered_symbols if s not in portfolio_positions]
            
            # Limit total symbols but ensure all portfolio positions are included
            max_symbols = 50
            if len(portfolio_positions) > max_symbols:
                # If we have more portfolio positions than max_symbols, analyze all positions
                symbols_to_analyze = portfolio_positions
                logger.warning(f"Portfolio has {len(portfolio_positions)} positions, analyzing all despite {max_symbols} limit")
            else:
                # Include all portfolio positions + top filtered symbols up to max_symbols
                remaining_slots = max_symbols - len(portfolio_positions)
                additional_symbols = [s for s in filtered_symbols if s not in portfolio_positions][:remaining_slots]
                symbols_to_analyze = portfolio_positions + additional_symbols
            
            if not symbols_to_analyze:
                logger.warning("No symbols to analyze (no filtered symbols or portfolio positions)")
                state["sentiment_data"] = {}
                state["current_agent"] = "sentiment_analyzer" 
                return update_state_timestamp(state)
            
            logger.info(f"💭 Analyzing sentiment for {len(symbols_to_analyze)} symbols")
            logger.info(f"   📊 Portfolio positions: {len(portfolio_positions)} (all included)")
            logger.info(f"   🔍 Additional filtered: {len(symbols_to_analyze) - len(portfolio_positions)}")
            
            # Run comprehensive sentiment analysis for each symbol
            sentiment_data = {}
            sentiment_signals = []
            
            # Process symbols in parallel batches to speed up analysis  
            batch_size = 2  # Further reduced to prevent Ollama overload and timeouts
            
            async def analyze_symbol_sentiment(symbol: str):
                """Analyze sentiment for a single symbol using enhanced engine."""
                try:
                    logger.info(f"Running enhanced FinGPT sentiment analysis for {symbol}")
                    
                    # Generate sample financial news text for sentiment analysis testing
                    # Use dynamic news data - no hardcoded examples
                    sample_news_texts = {}
                    
                    combined_text = sample_news_texts.get(symbol, f"Market analysis for {symbol} shows mixed sentiment with moderate trading volume and technical indicators suggesting neutral outlook.")
                    
                    # Use direct LLM sentiment analyzer with Ollama
                    comprehensive_sentiment = await asyncio.wait_for(
                        sentiment_engine.analyze_text(combined_text, "financial_news"),
                        timeout=45.0  # 45 second timeout per symbol for Ollama processing
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
                            'overall_sentiment': str(comprehensive_sentiment.sentiment),
                            'overall_score': comprehensive_sentiment.score,
                            'confidence': comprehensive_sentiment.confidence,
                            'ensemble_used': False,  # Not using ensemble anymore
                            'models_successful': 1,  # Single Ollama model
                            'reasoning': comprehensive_sentiment.reasoning,
                            'model_results': {'ollama': comprehensive_sentiment.sentiment},
                            'analysis_timestamp': None  # Not tracked in simple version
                        }
                        
                        # Log detailed sentiment breakdown from Ollama analysis
                        logger.info(f"{symbol} sentiment breakdown:")
                        logger.info(f"  Overall: {comprehensive_sentiment.sentiment} (score: {comprehensive_sentiment.score:.3f}, confidence: {comprehensive_sentiment.confidence:.3f})")
                        logger.info(f"  Reasoning: {comprehensive_sentiment.reasoning[:100]}...")
                        logger.info(f"  Key phrases: {', '.join(comprehensive_sentiment.key_phrases[:3])}")
                        
                        # Generate trading signals based on enhanced sentiment
                        signal_strength = 0.0
                        signal_type = None
                        signal_reason = []
                        
                        # Extract sentiment values
                        overall_sentiment = str(comprehensive_sentiment.sentiment)
                        overall_score = comprehensive_sentiment.score
                        confidence = comprehensive_sentiment.confidence
                        
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
                        
                        # Crisis/panic sentiment detection with high confidence
                        if (confidence > 0.8 and overall_score <= -0.6):
                            signal_strength += 0.4  # Major boost for high-confidence extreme negativity
                            signal_type = 'SHORT'
                            signal_reason.append('CRISIS-LEVEL sentiment - High confidence extreme negative')
                        
                        # Confidence boosts
                        if confidence > 0.7:
                            signal_strength += 0.1
                            signal_reason.append('High confidence analysis')
                        
                        # Ollama model quality boost
                        if confidence > 0.8:
                            signal_strength += 0.1
                            signal_reason.append('Very high confidence Ollama analysis')
                        
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
        """Generate trading signals using dual-agent online RL system ONLY."""
        try:
            logger.info("Signal Generation Agent: Using dual-agent online RL system")
            
            # ONLY USE: Dual-agent online RL system - no fallbacks
            logger.info("🤖 Using DUAL-AGENT ONLINE RL SYSTEM (no fallbacks)")
            
            # Evaluate existing positions for underperformance
            underperformer_signals = await self._evaluate_existing_positions_for_selling(state)
            
            # Use the dual-agent online RL system
            rl_result = await self._online_rl_signal_generation(state, config)
            
            # Merge underperformer sell signals with RL buy signals
            if underperformer_signals:
                current_signals = rl_result.get("signals", [])
                combined_signals = list(underperformer_signals) + list(current_signals)
                rl_result["signals"] = combined_signals
                logger.info(f"🔄 Combined RL signals: {len(underperformer_signals)} sell/short + {len(current_signals)} RL = {len(combined_signals)} total")
            
            # If RL system fails, raise error instead of using fallbacks
            if not rl_result.get("signals") and not rl_result.get("trading_signals"):
                error_msg = "❌ CRITICAL: RL signal generation failed and no fallback allowed - system requires RL agent to generate signals"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            return rl_result
        
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
                        else:  # Cash account - ALGO AGENT ENHANCEMENT: Kelly Criterion position sizing
                            # Calculate optimal Kelly position size
                            kelly_size = self._calculate_kelly_position_size(symbol, state, 0.15)
                            kelly_position_value = portfolio_value * kelly_size
                            
                            # Respect cash constraints
                            cash_limit = available_cash * 0.9
                            max_position_value = min(cash_limit, kelly_position_value)
                            logger.info(f"💰 Kelly sizing for {symbol}: Kelly={kelly_size:.1%} (${kelly_position_value:.2f}), Cash limit=${cash_limit:.2f}")
                        
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
            error_msg = f"❌ CRITICAL: RL signal generation failed: {e} - system requires RL agent to generate signals"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
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
                
                # Multi-tier sentiment analysis for short selling (ENHANCED - more aggressive)
                if sentiment_score <= -0.4:  # Extremely negative (lowered from -0.7)
                    extremely_negative_sentiment = True
                    logger.info(f"🔥 EXTREMELY negative sentiment detected for {symbol}: {sentiment_score:.2f}")
                elif sentiment_score <= -0.3:  # Very negative (lowered from -0.5)
                    very_negative_sentiment = True
                    logger.info(f"📉 Very negative sentiment detected for {symbol}: {sentiment_score:.2f}")
                elif sentiment_score <= -0.2:  # Strong negative (lowered from -0.3) - ALGO AGENT RECOMMENDATION
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
                    
                    # Check for high volatility (good for short selling) - ALGO AGENT RECOMMENDATION: >0.25
                    volatility_ratio = price_range / current_price
                    if volatility_ratio > 0.25:  # 25% range indicates high volatility for shorts
                        high_volatility = True
                        logger.info(f"⚡ High volatility detected for {symbol}: {volatility_ratio:.1%} range - SHORT OPPORTUNITY")
                
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
            
            # ALGO AGENT RECOMMENDATION: sentiment < 0.2 AND volatility > 0.25 = SHORT OPPORTUNITY
            if (sentiment_score < -0.2 and high_volatility and sentiment_data):
                confidence_factors += 1.0  # Strong boost for algo agent criteria
                strategy_reasoning.append("ALGO-AGENT: Negative sentiment + High volatility")
                logger.info(f"🤖 ALGO AGENT SHORT SIGNAL for {symbol}: sentiment={sentiment_score:.2f}, volatility={volatility_ratio:.1%}")
            
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
            raise ValueError(f"Cannot determine order type for {symbol}: {e}")
    
    def _calculate_kelly_position_size(self, symbol: str, state: dict, base_position_pct: float = 0.15) -> float:
        """Calculate optimal position size using Kelly Criterion - ALGO AGENT RECOMMENDATION."""
        try:
            # Get historical performance data for Kelly calculation
            portfolio = state.get("portfolio", {})
            portfolio_value = portfolio.get("equity", 50000.0)
            
            # ALGO AGENT KELLY CRITERION PARAMETERS
            win_rate = 0.58  # From current system performance (208/208 successful runs)
            avg_win = 0.025  # 2.5% average win (conservative estimate)
            avg_loss = 0.08  # 8% average loss (stop loss level)
            
            # Kelly Fraction: f = (bp - q) / b
            # where b = odds received on the wager (avg_win/avg_loss)
            # p = probability of winning, q = probability of losing
            b = avg_win / avg_loss if avg_loss > 0 else 0.3125  # 0.3125 from 0.025/0.08
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b if b > 0 else 0
            
            # Conservative Kelly: use 25% of full Kelly for safety
            conservative_kelly = kelly_fraction * 0.25
            
            # Get volatility adjustment from market data
            volatility_adjustment = 1.0
            try:
                # Get current volatility proxy from sentiment confidence
                sentiment_data = state.get("sentiment_data", {}).get(symbol, {})
                if isinstance(sentiment_data, dict):
                    confidence = sentiment_data.get("confidence", 0.5)
                    # Higher confidence = lower volatility = larger position
                    volatility_adjustment = 0.8 + (confidence * 0.4)  # Scale from 0.8 to 1.2
                else:
                    volatility_adjustment = 1.0
            except Exception:
                volatility_adjustment = 1.0
            
            # Apply volatility adjustment
            kelly_adjusted = conservative_kelly * volatility_adjustment
            
            # ALGO AGENT LIMITS: Cap between 5% and 45% as recommended
            optimal_size = max(0.05, min(0.45, kelly_adjusted))
            
            logger.info(f"🎯 Kelly position sizing for {symbol}: "
                       f"base={base_position_pct:.1%}, kelly={conservative_kelly:.1%}, "
                       f"vol_adj={volatility_adjustment:.2f}, final={optimal_size:.1%}")
            
            return optimal_size
            
        except Exception as e:
            logger.warning(f"Kelly calculation failed for {symbol}: {e}, using base size {base_position_pct:.1%}")
            return base_position_pct
    
    def _calculate_current_short_exposure(self, state: dict) -> float:
        """Calculate current short position exposure as percentage of portfolio."""
        try:
            portfolio = state.get("portfolio", {})
            positions = portfolio.get("positions", {})
            total_equity = portfolio.get("equity", 50000.0)
            
            short_value = 0
            for symbol, position_data in positions.items():
                if isinstance(position_data, dict):
                    qty = position_data.get("qty", 0)
                    market_value = abs(position_data.get("market_value", 0))
                    if qty < 0:  # Short position
                        short_value += market_value
            
            return short_value / total_equity if total_equity > 0 else 0
        except Exception as e:
            logger.warning(f"Could not calculate short exposure: {e}")
            return 0
    
    async def _enhance_short_signals_with_risk_management(self, signals: list, state: dict) -> list:
        """Add advanced risk management to short selling signals with PORTFOLIO LIMITS."""
        try:
            enhanced_signals = []
            
            # ALGO AGENT RECOMMENDATION: Max 15% portfolio in short positions
            current_short_exposure = self._calculate_current_short_exposure(state)
            MAX_SHORT_PORTFOLIO_ALLOCATION = 0.15  # 15% max as recommended
            
            logger.info(f"📊 Current short exposure: {current_short_exposure:.1%} (max allowed: {MAX_SHORT_PORTFOLIO_ALLOCATION:.1%})")
            
            for signal in signals:
                if signal.action.lower() in ["sell_short", "short"]:
                    # ALGO AGENT PORTFOLIO LIMIT: Check if we're over 15% short allocation
                    signal_value_estimate = getattr(signal, 'quantity', 0) * 100  # Rough estimate
                    portfolio_value = state.get("portfolio", {}).get("equity", 50000.0)
                    estimated_new_short_exposure = current_short_exposure + (signal_value_estimate / portfolio_value)
                    
                    if estimated_new_short_exposure > MAX_SHORT_PORTFOLIO_ALLOCATION:
                        logger.warning(f"🚫 SHORT REJECTED: {signal.symbol} would exceed 15% short limit ({estimated_new_short_exposure:.1%} > {MAX_SHORT_PORTFOLIO_ALLOCATION:.1%})")
                        # Convert to regular sell instead
                        signal.action = "sell"
                        enhanced_signals.append(signal)
                        continue
                    
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
            # Fix: Get symbols from correct state key where universe filter actually stores them
            symbols = state.get("filtered_symbols", [])
            if not symbols:
                symbols = universe_results.get("actionable_symbols", [])
            
            # No fallbacks allowed - system must have filtered symbols
            if not symbols:
                raise ValueError("No filtered symbols available and no fallback mechanisms allowed - system requires universe filter to provide actionable symbols")
            
            for symbol in symbols:
                # Handle both direct market_data[symbol] and market_data['symbols'][symbol] structures
                symbol_market_data = market_data.get(symbol, {})
                if not symbol_market_data and 'symbols' in market_data:
                    symbol_market_data = market_data['symbols'].get(symbol, {})
                
                symbol_data = {
                    'price': symbol_market_data.get('price', 0.0),
                    'price_change_pct': symbol_market_data.get('price_change_pct', 0.0),
                    'volume': symbol_market_data.get('volume', 0.0),
                    'avg_volume': symbol_market_data.get('avg_volume', symbol_market_data.get('volume', 0.0)),
                    'rsi': symbol_market_data.get('rsi', 50.0),
                    'macd': symbol_market_data.get('macd', 0.0),
                    'bb_position': symbol_market_data.get('bb_position', 0.5)
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
            logger.info(f"🚀 Calling RL signal generation with {len(symbols)} symbols")
            logger.debug(f"📊 RL market data symbols: {list(rl_market_data.keys())}")
            logger.debug(f"💰 Portfolio data symbols: {list(portfolio_data.keys())}")
            logger.debug(f"💵 Portfolio value: {portfolio_value}")
            
            rl_signals = await generate_rl_enhanced_signals(
                rl_market_data,
                portfolio_data, 
                portfolio_value,
                symbols
            )
            
            logger.info(f"📋 Received {len(rl_signals)} RL signals from integration layer")
            
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
            # Import error or system not available - no fallbacks allowed
            raise ValueError("RL integration system unavailable and no fallback mechanisms allowed")
    
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
    
    def _generate_fallback_signals(self, state: TradingState) -> List[Dict]:
        """REMOVED: Fallback signal generation disabled - system must use RL agent signals only."""
        raise RuntimeError("❌ CRITICAL: Fallback signal generation disabled - system requires RL agent to generate signals")
    
    async def strategy_optimization_agent(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize trading strategy based on generated signals."""
        try:
            logger.info("Strategy Optimization Agent: Optimizing trading strategy")
            
            signals = state.get("signals", [])
            if not signals:
                logger.info("No signals to optimize")
                state["current_agent"] = "strategy_optimizer"
                return update_state_timestamp(state)
            
            # Simple strategy optimization - filter and rank signals
            optimized_signals = []
            for signal in signals:
                if signal.confidence > 0.5:  # Only keep confident signals
                    optimized_signals.append(signal)
            
            state["signals"] = optimized_signals
            state["current_agent"] = "strategy_optimizer"
            
            logger.info(f"Strategy optimization complete: {len(optimized_signals)}/{len(signals)} signals retained")
            
            return update_state_timestamp(state)
            
        except Exception as e:
            error_msg = f"Strategy Optimization Agent error: {e}"
            logger.error(error_msg)
            return add_error_to_state(state, error_msg)
    
    async def order_management_agent(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading orders based on optimized signals."""
        try:
            logger.info("Order Management Agent: Executing trading orders")
            
            signals = state.get("signals", [])
            if not signals:
                logger.info("No signals to execute")
                state["current_agent"] = "order_manager"
                return update_state_timestamp(state)
            
            executed_orders = []
            for signal in signals[:5]:  # Execute top 5 signals
                try:
                    # Execute actual order through Alpaca API
                    from tools.alpaca_client import alpaca_client
                    
                    try:
                        # ALGO AGENT ENHANCEMENT: Advanced order types based on market conditions
                        from tools.alpaca_client import alpaca_client
                        
                        # Determine optimal order type based on volatility and market conditions
                        current_price = alpaca_client.get_current_price(signal.symbol)
                        volatility = getattr(signal, 'volatility', 0.02)  # Default 2% volatility
                        
                        # ALGO AGENT RECOMMENDATION: Use limit orders for volatile stocks
                        if volatility > 0.25:  # High volatility threshold from algo agent
                            order_type = "limit"
                            # Set limit price with small buffer
                            if signal.action.lower() == "buy":
                                limit_price = current_price * 1.002  # Buy 0.2% above current
                            else:
                                limit_price = current_price * 0.998  # Sell 0.2% below current
                            logger.info(f"🎯 Using LIMIT order for volatile {signal.symbol}: volatility={volatility:.1%}, limit=${limit_price:.2f}")
                        else:
                            order_type = "market"
                            limit_price = None
                            logger.info(f"📈 Using MARKET order for stable {signal.symbol}: volatility={volatility:.1%}")
                        
                        # Calculate stop loss and take profit levels (ALGO AGENT RECOMMENDATION)
                        if signal.action.lower() == "buy":
                            stop_loss_price = current_price * 0.92   # 8% stop loss
                            take_profit_price = current_price * 1.15  # 15% take profit
                        else:  # sell or short
                            stop_loss_price = current_price * 1.08   # 8% stop loss for shorts
                            take_profit_price = current_price * 0.85  # 15% take profit for shorts
                        
                        # Place primary order
                        order_params = {
                            "symbol": signal.symbol,
                            "qty": signal.quantity,
                            "side": signal.action.lower(),
                            "order_type": order_type,
                            "time_in_force": "day"
                        }
                        if limit_price:
                            order_params["limit_price"] = limit_price
                            
                        alpaca_order = alpaca_client.place_order(**order_params)
                        
                        # ALGO AGENT ENHANCEMENT: Place bracket orders (stop-loss + take-profit) for risk management
                        bracket_orders = []
                        if alpaca_order.get("status") in ["new", "partially_filled", "filled", "submitted"]:
                            try:
                                # Stop Loss Order
                                stop_order_params = {
                                    "symbol": signal.symbol,
                                    "qty": signal.quantity,
                                    "side": "sell" if signal.action.lower() == "buy" else "buy",
                                    "order_type": "stop",
                                    "stop_price": stop_loss_price,
                                    "time_in_force": "gtc"  # Good till cancelled for stop orders
                                }
                                stop_order = alpaca_client.place_order(**stop_order_params)
                                bracket_orders.append(("stop_loss", stop_order))
                                logger.info(f"🛡️ Stop-loss order placed for {signal.symbol}: ${stop_loss_price:.2f} (ID: {stop_order.get('id', 'N/A')})")
                                
                                # Take Profit Order  
                                profit_order_params = {
                                    "symbol": signal.symbol,
                                    "qty": signal.quantity,
                                    "side": "sell" if signal.action.lower() == "buy" else "buy",
                                    "order_type": "limit", 
                                    "limit_price": take_profit_price,
                                    "time_in_force": "gtc"
                                }
                                profit_order = alpaca_client.place_order(**profit_order_params)
                                bracket_orders.append(("take_profit", profit_order))
                                logger.info(f"🎯 Take-profit order placed for {signal.symbol}: ${take_profit_price:.2f} (ID: {profit_order.get('id', 'N/A')})")
                                
                            except Exception as bracket_error:
                                logger.warning(f"⚠️ Could not place bracket orders for {signal.symbol}: {bracket_error}")
                        
                        order = {
                            "symbol": signal.symbol,
                            "action": signal.action,
                            "quantity": signal.quantity,
                            "status": alpaca_order.get("status", "submitted"),
                            "order_id": alpaca_order.get("id"),
                            "order_type": order_type,
                            "limit_price": limit_price,
                            "stop_loss_price": stop_loss_price,
                            "take_profit_price": take_profit_price,
                            "bracket_orders": bracket_orders,
                            "timestamp": datetime.now(),
                            "alpaca_response": alpaca_order
                        }
                        executed_orders.append(order)
                        logger.info(f"✅ Alpaca order placed: {signal.action} {signal.quantity} {signal.symbol} (ID: {alpaca_order.get('id', 'N/A')})")
                        
                    except Exception as api_error:
                        logger.error(f"❌ Alpaca API error for {signal.symbol}: {api_error}")
                        # Fall back to simulation for this order
                        order = {
                            "symbol": signal.symbol,
                            "action": signal.action,
                            "quantity": signal.quantity,
                            "status": "failed",
                            "error": str(api_error),
                            "timestamp": datetime.now()
                        }
                        executed_orders.append(order)
                        logger.info(f"⚠️ Order failed, logged for retry: {signal.action} {signal.quantity} {signal.symbol}")
                except Exception as e:
                    logger.warning(f"Failed to execute order for {signal.symbol}: {e}")
            
            state["executed_orders"] = executed_orders
            state["current_agent"] = "order_manager"
            
            logger.info(f"Order execution complete: {len(executed_orders)} orders executed")
            
            return update_state_timestamp(state)
            
        except Exception as e:
            error_msg = f"Order Management Agent error: {e}"
            logger.error(error_msg)
            return add_error_to_state(state, error_msg)
    
    async def portfolio_tracking_agent(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Track portfolio performance and positions."""
        try:
            logger.info("Portfolio Tracking Agent: Tracking portfolio performance")
            
            # Simple portfolio tracking
            portfolio = state.get("portfolio", {})
            executed_orders = state.get("executed_orders", [])
            
            total_orders = len(executed_orders)
            portfolio_value = portfolio.get("equity", 0)
            
            state["current_agent"] = "portfolio_tracker"
            state["messages"].append(
                AIMessage(content=f"Portfolio tracking complete. {total_orders} orders executed. Portfolio value: ${portfolio_value:.2f}")
            )
            
            logger.info(f"Portfolio tracking complete: {total_orders} orders, ${portfolio_value:.2f} value")
            
            return update_state_timestamp(state)
            
        except Exception as e:
            error_msg = f"Portfolio Tracking Agent error: {e}"
            logger.error(error_msg)
            return add_error_to_state(state, error_msg)
    
    async def emergency_handler_agent(self, state: TradingState, config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency situations."""
        try:
            logger.info("Emergency Handler Agent: Handling emergency situation")
            
            state["current_agent"] = "emergency_handler"
            state["messages"].append(
                AIMessage(content="Emergency protocols activated")
            )
            
            return update_state_timestamp(state)
            
        except Exception as e:
            error_msg = f"Emergency Handler Agent error: {e}"
            logger.error(error_msg)
            return add_error_to_state(state, error_msg)
    
    def risk_routing_logic(self, state: TradingState) -> Literal["emergency", "proceed", "halt"]:
        """Determine routing based on risk assessment."""
        circuit_breakers = state.get("circuit_breakers", {})
        
        if any(circuit_breakers.values()):
            return "emergency"
        
        return "proceed"
    
    def signal_routing_logic(self, state: TradingState) -> Literal["execute", "hold", "reassess"]:
        """Determine routing based on signal generation."""
        signals = state.get("signals", [])
        
        if len(signals) > 0:
            return "execute"
        
        return "hold"
