"""
Pairs Trading Agent for LangGraph Workflow

This agent specializes in identifying, monitoring, and executing pairs trading strategies
within the enhanced trading system's multi-agent framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime, timedelta

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..strategies.pairs_trading import PairsStrategy, PairSignal, PairCandidate
from .enhanced_state import EnhancedTradingState
from ..tools.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


class PairsTradingAgent:
    """
    Specialized agent for pairs trading strategy execution
    
    Responsibilities:
    1. Identify cointegrated pairs from universe of stocks
    2. Monitor spread relationships and generate signals
    3. Execute pairs trades with proper risk management
    4. Provide strategy performance analysis
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        alpaca_client: AlpacaClient,
        strategy_config: Optional[Dict] = None
    ):
        self.llm = llm
        self.alpaca_client = alpaca_client
        
        # Initialize pairs strategy
        default_config = {
            'lookback_period': 252,
            'signal_window': 20,
            'entry_zscore': 2.0,
            'exit_zscore': 0.0,
            'stop_loss_zscore': 3.0,
            'max_positions': 3,  # Conservative for live trading
            'position_size': 5000.0,  # $5k per leg
            'correlation_threshold': 0.75,
            'cointegration_pvalue': 0.05
        }
        
        config = {**default_config, **(strategy_config or {})}
        self.strategy = PairsStrategy(**config)
        
        # Agent state
        self.last_pair_scan: Optional[datetime] = None
        self.scan_frequency = timedelta(hours=24)  # Daily pair scanning
        self.monitoring_symbols: List[str] = []
        
        # Performance tracking
        self.strategy_stats = {
            'total_pairs_traded': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0,
            'sharpe_ratio': 0.0
        }
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a Pairs Trading Specialist Agent in an automated trading system.

Your responsibilities:
1. Analyze market data to identify cointegrated pairs
2. Monitor existing pairs for trading signals
3. Execute pairs trades with proper risk management
4. Provide statistical analysis and performance metrics

Key principles:
- Focus on mean-reverting relationships between correlated securities
- Maintain dollar neutrality in all positions
- Use statistical significance testing for pair validation
- Implement strict risk controls and stop-losses

Current portfolio status:
Active pairs: {active_pairs}
Available capacity: {available_capacity}
Strategy performance: {strategy_performance}

Market context: {market_context}
"""),
            ("human", "{input}")
        ])
    
    async def analyze_pairs_opportunities(self, state: EnhancedTradingState) -> Dict[str, Any]:
        """
        Analyze current market for pairs trading opportunities
        """
        try:
            # Get universe of stocks for pair scanning
            universe = await self._get_trading_universe(state)
            
            if not universe:
                return {
                    'analysis': 'No suitable universe available for pairs analysis',
                    'action': 'wait',
                    'pairs_found': 0
                }
            
            # Get historical price data
            price_data = await self._fetch_price_data(universe, lookback_days=300)
            
            # Find cointegrated pairs
            pairs_found = self.strategy.find_pairs(universe, price_data)
            
            # Analyze top pairs
            analysis = await self._analyze_pairs_with_llm(pairs_found, state)
            
            # Update state
            self.last_pair_scan = datetime.now()
            self.monitoring_symbols = universe
            
            return {
                'analysis': analysis,
                'pairs_found': len(pairs_found),
                'top_pairs': [
                    {
                        'symbols': f"{p.symbol_a}-{p.symbol_b}",
                        'correlation': p.correlation,
                        'p_value': p.p_value,
                        'half_life': p.half_life,
                        'score': p.score
                    }
                    for p in pairs_found[:5]
                ],
                'action': 'monitor' if pairs_found else 'wait'
            }
            
        except Exception as e:
            logger.error(f"Error in pairs opportunity analysis: {e}")
            return {
                'analysis': f'Error in pairs analysis: {str(e)}',
                'action': 'error',
                'pairs_found': 0
            }
    
    async def monitor_pairs_signals(self, state: EnhancedTradingState) -> Dict[str, Any]:
        """
        Monitor existing pairs and active positions for trading signals
        """
        try:
            if not self.monitoring_symbols:
                return {
                    'signals': [],
                    'analysis': 'No pairs being monitored',
                    'action': 'scan_for_pairs'
                }
            
            # Get current market data
            current_data = await self._fetch_current_prices(self.monitoring_symbols)
            
            # Generate signals
            signals = self.strategy.generate_signals(current_data)
            
            if not signals:
                return {
                    'signals': [],
                    'analysis': 'No trading signals generated',
                    'action': 'continue_monitoring',
                    'active_positions': len(self.strategy.active_pairs)
                }
            
            # Analyze signals with LLM
            signal_analysis = await self._analyze_signals_with_llm(signals, state)
            
            return {
                'signals': [self._serialize_signal(s) for s in signals],
                'analysis': signal_analysis,
                'action': 'execute_signals' if signals else 'continue_monitoring',
                'active_positions': len(self.strategy.active_pairs)
            }
            
        except Exception as e:
            logger.error(f"Error monitoring pairs signals: {e}")
            return {
                'signals': [],
                'analysis': f'Error monitoring signals: {str(e)}',
                'action': 'error'
            }
    
    async def execute_pairs_trades(
        self, 
        signals: List[PairSignal], 
        state: EnhancedTradingState
    ) -> Dict[str, Any]:
        """
        Execute pairs trading signals with proper risk management
        """
        execution_results = []
        
        for signal in signals:
            try:
                # Pre-trade risk validation
                risk_check = await self._validate_pairs_risk(signal, state)
                
                if not risk_check['approved']:
                    execution_results.append({
                        'signal': self._serialize_signal(signal),
                        'status': 'rejected',
                        'reason': risk_check['reason']
                    })
                    continue
                
                # Execute the pairs trade
                execution_result = await self._execute_pairs_signal(signal, state)
                execution_results.append(execution_result)
                
                # Update strategy state
                if execution_result['status'] == 'executed':
                    self.strategy.execute_signal(signal)
                    self._update_performance_stats(signal, execution_result)
                
            except Exception as e:
                logger.error(f"Error executing pairs signal {signal.symbol_a}-{signal.symbol_b}: {e}")
                execution_results.append({
                    'signal': self._serialize_signal(signal),
                    'status': 'error',
                    'reason': str(e)
                })
        
        # Generate execution summary
        executed_count = sum(1 for r in execution_results if r['status'] == 'executed')
        rejected_count = sum(1 for r in execution_results if r['status'] == 'rejected')
        
        return {
            'execution_results': execution_results,
            'executed_count': executed_count,
            'rejected_count': rejected_count,
            'analysis': f"Executed {executed_count} pairs signals, rejected {rejected_count}",
            'action': 'continue_monitoring'
        }
    
    async def get_pairs_portfolio_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of pairs trading portfolio
        """
        portfolio_status = self.strategy.get_portfolio_status()
        
        return {
            **portfolio_status,
            'strategy_stats': self.strategy_stats,
            'last_scan': self.last_pair_scan.isoformat() if self.last_pair_scan else None,
            'monitoring_symbols_count': len(self.monitoring_symbols),
            'next_scan_due': (
                self.last_pair_scan + self.scan_frequency
            ).isoformat() if self.last_pair_scan else 'immediate'
        }
    
    async def _get_trading_universe(self, state: EnhancedTradingState) -> List[str]:
        """
        Get universe of stocks suitable for pairs trading
        """
        try:
            # Use liquid, large-cap stocks for pairs trading
            # You can expand this based on your preferences
            tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN', 'TSLA']
            energy_stocks = ['XOM', 'CVX', 'COP', 'EOG', 'SLB']
            finance_stocks = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C']
            retail_stocks = ['WMT', 'TGT', 'COST', 'HD', 'LOW']
            
            # Combine sectors for cross-sector and intra-sector pairs
            universe = tech_stocks + energy_stocks + finance_stocks + retail_stocks
            
            # Filter based on current positions to avoid conflicts
            # (This is a simplified approach - you might want more sophisticated filtering)
            return universe
            
        except Exception as e:
            logger.error(f"Error getting trading universe: {e}")
            return []
    
    async def _fetch_price_data(
        self, 
        symbols: List[str], 
        lookback_days: int = 300
    ) -> Dict[str, pd.Series]:
        """
        Fetch historical price data for symbols
        """
        try:
            price_data = {}
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            for symbol in symbols:
                try:
                    # Use Alpaca client to get historical data
                    bars = await self.alpaca_client.get_historical_bars(
                        symbol, 
                        start_date.isoformat(), 
                        end_date.isoformat(),
                        timeframe='1Day'
                    )
                    
                    if bars and len(bars) > 0:
                        # Convert to pandas Series
                        prices = pd.Series(
                            [bar.close for bar in bars],
                            index=[bar.timestamp for bar in bars]
                        )
                        price_data[symbol] = prices
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
                    continue
            
            return price_data
            
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            return {}
    
    async def _fetch_current_prices(self, symbols: List[str]) -> Dict[str, pd.Series]:
        """
        Fetch current/recent price data for signal generation
        """
        try:
            # For signal generation, we need recent price history (last 30 days)
            return await self._fetch_price_data(symbols, lookback_days=30)
            
        except Exception as e:
            logger.error(f"Error fetching current prices: {e}")
            return {}
    
    async def _analyze_pairs_with_llm(
        self, 
        pairs: List[PairCandidate], 
        state: EnhancedTradingState
    ) -> str:
        """
        Use LLM to analyze found pairs and provide insights
        """
        try:
            if not pairs:
                return "No statistically significant pairs found in current market conditions."
            
            pairs_summary = "\n".join([
                f"• {p.symbol_a}-{p.symbol_b}: corr={p.correlation:.3f}, "
                f"p-val={p.p_value:.4f}, half-life={p.half_life:.1f}d, score={p.score:.3f}"
                for p in pairs[:10]
            ])
            
            portfolio_status = self.strategy.get_portfolio_status()
            
            prompt = self.prompt_template.format_messages(
                active_pairs=portfolio_status['active_pairs'],
                available_capacity=portfolio_status['max_pairs'] - portfolio_status['active_pairs'],
                strategy_performance=str(self.strategy_stats),
                market_context=state.market_context.get('summary', 'Unknown'),
                input=f"""Analyze these pairs trading opportunities:

{pairs_summary}

Provide insights on:
1. Quality of statistical relationships
2. Sector diversification
3. Recommended prioritization
4. Risk considerations"""
            )
            
            response = await self.llm.ainvoke(prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in LLM pairs analysis: {e}")
            return f"Error analyzing pairs: {str(e)}"
    
    async def _analyze_signals_with_llm(
        self, 
        signals: List[PairSignal], 
        state: EnhancedTradingState
    ) -> str:
        """
        Use LLM to analyze trading signals
        """
        try:
            signals_summary = "\n".join([
                f"• {s.symbol_a}-{s.symbol_b}: {s.action}, z-score={s.zscore:.2f}, "
                f"confidence={s.confidence:.2f}"
                for s in signals
            ])
            
            prompt = self.prompt_template.format_messages(
                active_pairs=len(self.strategy.active_pairs),
                available_capacity=self.strategy.max_positions - len(self.strategy.active_pairs),
                strategy_performance=str(self.strategy_stats),
                market_context=state.market_context.get('summary', 'Unknown'),
                input=f"""Analyze these pairs trading signals:

{signals_summary}

Current market conditions: {state.market_context.get('sentiment', 'Unknown')}

Provide recommendation on signal execution considering:
1. Signal strength and confidence
2. Current market volatility
3. Portfolio risk management
4. Timing considerations"""
            )
            
            response = await self.llm.ainvoke(prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in LLM signal analysis: {e}")
            return f"Error analyzing signals: {str(e)}"
    
    async def _validate_pairs_risk(
        self, 
        signal: PairSignal, 
        state: EnhancedTradingState
    ) -> Dict[str, Any]:
        """
        Validate pairs trade against risk management rules
        """
        try:
            # Check position limits
            if len(self.strategy.active_pairs) >= self.strategy.max_positions:
                return {
                    'approved': False,
                    'reason': f'Maximum pairs positions reached ({self.strategy.max_positions})'
                }
            
            # Check portfolio exposure
            current_exposure = state.portfolio.get('total_equity', 0)
            max_pairs_exposure = current_exposure * 0.3  # Max 30% in pairs
            
            proposed_exposure = (
                abs(signal.suggested_quantity_a) * state.market_data.get(signal.symbol_a, {}).get('price', 0) +
                abs(signal.suggested_quantity_b) * state.market_data.get(signal.symbol_b, {}).get('price', 0)
            )
            
            if proposed_exposure > max_pairs_exposure:
                return {
                    'approved': False,
                    'reason': f'Trade would exceed pairs exposure limit'
                }
            
            # Check signal confidence
            if signal.confidence < 0.6:
                return {
                    'approved': False,
                    'reason': f'Signal confidence too low: {signal.confidence:.2f}'
                }
            
            return {'approved': True, 'reason': 'Risk checks passed'}
            
        except Exception as e:
            logger.error(f"Error in pairs risk validation: {e}")
            return {'approved': False, 'reason': f'Risk validation error: {str(e)}'}
    
    async def _execute_pairs_signal(
        self, 
        signal: PairSignal, 
        state: EnhancedTradingState
    ) -> Dict[str, Any]:
        """
        Execute a pairs trading signal
        """
        try:
            # This would integrate with the actual order management system
            # For now, return a simulated execution
            
            logger.info(f"Executing pairs signal: {signal.symbol_a}-{signal.symbol_b} {signal.action}")
            
            return {
                'signal': self._serialize_signal(signal),
                'status': 'executed',
                'timestamp': datetime.now().isoformat(),
                'order_ids': ['simulated_order_1', 'simulated_order_2'],
                'execution_price_a': state.market_data.get(signal.symbol_a, {}).get('price', 0),
                'execution_price_b': state.market_data.get(signal.symbol_b, {}).get('price', 0)
            }
            
        except Exception as e:
            logger.error(f"Error executing pairs signal: {e}")
            return {
                'signal': self._serialize_signal(signal),
                'status': 'error',
                'reason': str(e)
            }
    
    def _serialize_signal(self, signal: PairSignal) -> Dict[str, Any]:
        """
        Convert PairSignal to serializable dictionary
        """
        return {
            'symbol_a': signal.symbol_a,
            'symbol_b': signal.symbol_b,
            'action': signal.action,
            'zscore': signal.zscore,
            'spread': signal.spread,
            'confidence': signal.confidence,
            'quantity_a': signal.suggested_quantity_a,
            'quantity_b': signal.suggested_quantity_b
        }
    
    def _update_performance_stats(self, signal: PairSignal, execution_result: Dict[str, Any]):
        """
        Update strategy performance statistics
        """
        if execution_result['status'] == 'executed':
            self.strategy_stats['total_pairs_traded'] += 1
            
        # Additional performance tracking would be implemented here
        # based on the specific execution result and position tracking


# Integration functions for the LangGraph workflow

async def pairs_opportunity_analysis(state: EnhancedTradingState) -> EnhancedTradingState:
    """
    LangGraph node for pairs opportunity analysis
    """
    try:
        # Get pairs agent from state or create new one
        pairs_agent = state.agents.get('pairs_trading_agent')
        if not pairs_agent:
            logger.warning("Pairs trading agent not found in state")
            return state
        
        # Analyze opportunities
        analysis_result = await pairs_agent.analyze_pairs_opportunities(state)
        
        # Update state
        state.agent_outputs['pairs_trading_agent'] = analysis_result
        state.signals.append({
            'agent': 'pairs_trading_agent',
            'type': 'opportunity_analysis',
            'data': analysis_result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Add reasoning to chain
        state.reasoning_chain.append({
            'agent': 'pairs_trading_agent',
            'step': 'opportunity_analysis',
            'reasoning': analysis_result.get('analysis', ''),
            'confidence': 0.8,
            'timestamp': datetime.now().isoformat()
        })
        
        return state
        
    except Exception as e:
        logger.error(f"Error in pairs opportunity analysis: {e}")
        state.errors.append(f"Pairs opportunity analysis failed: {str(e)}")
        return state


async def pairs_signal_monitoring(state: EnhancedTradingState) -> EnhancedTradingState:
    """
    LangGraph node for pairs signal monitoring
    """
    try:
        pairs_agent = state.agents.get('pairs_trading_agent')
        if not pairs_agent:
            return state
        
        # Monitor for signals
        monitoring_result = await pairs_agent.monitor_pairs_signals(state)
        
        # Update state
        state.agent_outputs['pairs_trading_agent'] = monitoring_result
        
        # Add any generated signals to state
        if monitoring_result.get('signals'):
            for signal in monitoring_result['signals']:
                state.signals.append({
                    'agent': 'pairs_trading_agent',
                    'type': 'pairs_signal',
                    'data': signal,
                    'timestamp': datetime.now().isoformat()
                })
        
        return state
        
    except Exception as e:
        logger.error(f"Error in pairs signal monitoring: {e}")
        state.errors.append(f"Pairs signal monitoring failed: {str(e)}")
        return state