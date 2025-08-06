"""
FinRL Integration with ML4T Trading System
Integrates enhanced FinRL agents with the existing trading system architecture.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

from agents.finrl_enhanced_env import EnhancedFinRLTradingEnv, create_enhanced_finrl_env, OrderSide, OrderType
from agents.finrl_rl_agent import FinRLAgent, RLConfig, create_finrl_agent, train_finrl_agent
from agents.llm_rl_integration import LLMStateEnricher
from tools.alpaca_client import alpaca_client

logger = logging.getLogger(__name__)


@dataclass
class FinRLTradingConfig:
    """Configuration for FinRL trading integration."""
    
    # Environment settings
    symbols: List[str] = field(default_factory=list)
    initial_balance: float = 100000
    commission_rate: float = 0.001
    margin_requirement: float = 0.5
    short_borrow_rate: float = 0.03
    enable_short_selling: bool = True
    enable_limit_orders: bool = True
    max_position_size: float = 0.15
    
    # RL agent settings
    rl_config: RLConfig = field(default_factory=RLConfig)
    
    # Integration settings
    lookback_days: int = 252
    retraining_frequency: int = 1000  # Episodes
    paper_trading: bool = True
    risk_management_enabled: bool = True
    
    # Performance thresholds
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = -0.15
    min_win_rate: float = 0.4


class FinRLTradingOrchestrator:
    """
    Orchestrates FinRL-based trading with the ML4T system.
    
    Features:
    - Integration with existing ML4T workflow
    - LLM state enrichment for RL environment
    - Advanced order management with short selling
    - Risk management and performance monitoring
    - Continuous learning and model updates
    """
    
    def __init__(self, config: FinRLTradingConfig):
        self.config = config
        
        # Core components
        self.environment: Optional[EnhancedFinRLTradingEnv] = None
        self.agent: Optional[FinRLAgent] = None
        self.llm_enricher = LLMStateEnricher()
        
        # Trading state
        self.is_trained = False
        self.current_positions = {}
        self.pending_orders = []
        self.performance_history = []
        
        # Risk management
        self.circuit_breakers = {
            'max_drawdown': False,
            'low_sharpe': False,
            'excessive_losses': False
        }
        
        # Model management
        self.model_dir = Path("models/finrl_integration")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ FinRL Trading Orchestrator initialized")
        logger.info(f"   Symbols: {len(config.symbols)}")
        logger.info(f"   Short selling: {'enabled' if config.enable_short_selling else 'disabled'}")
        logger.info(f"   Paper trading: {'enabled' if config.paper_trading else 'LIVE TRADING'}")
    
    async def initialize_components(self, market_data: pd.DataFrame = None):
        """Initialize RL environment and agent."""
        try:
            # Create enhanced environment
            self.environment = create_enhanced_finrl_env(
                symbols=self.config.symbols,
                initial_balance=self.config.initial_balance,
                commission_rate=self.config.commission_rate,
                margin_requirement=self.config.margin_requirement,
                short_borrow_rate=self.config.short_borrow_rate,
                enable_short_selling=self.config.enable_short_selling,
                enable_limit_orders=self.config.enable_limit_orders,
                max_position_size=self.config.max_position_size
            )
            
            # Set market data if provided
            if market_data is not None:
                self.environment.set_data(market_data)
                logger.info(f"Market data loaded: {len(market_data)} rows")
            
            # Create RL agent
            if self.environment.observation_space is not None:
                self.agent = create_finrl_agent(self.environment, self.config.rl_config)
                logger.info("✅ FinRL components initialized")
            else:
                logger.warning("Environment observation space not defined - need market data")
                
        except Exception as e:
            logger.error(f"Failed to initialize FinRL components: {e}")
            raise
    
    async def train_agent(self, 
                         training_data: pd.DataFrame,
                         num_episodes: int = 1000,
                         eval_frequency: int = 100) -> Dict:
        """Train the RL agent on historical data."""
        if self.agent is None or self.environment is None:
            await self.initialize_components(training_data)
        
        # Set training data
        self.environment.set_data(training_data)
        
        logger.info(f"🚀 Starting FinRL agent training")
        logger.info(f"   Episodes: {num_episodes}")
        logger.info(f"   Training data: {len(training_data)} rows")
        
        # Train agent
        training_results = train_finrl_agent(
            self.agent, 
            self.environment, 
            num_episodes=num_episodes,
            eval_frequency=eval_frequency
        )
        
        # Mark as trained
        self.is_trained = True
        
        # Save training results
        results_path = self.model_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(training_results)
            json.dump(serializable_results, f, indent=2)
        
        logger.info("✅ FinRL agent training completed")
        return training_results
    
    async def make_trading_decisions(self, 
                                   current_state: Dict[str, Any],
                                   market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make trading decisions using the trained RL agent.
        
        This method integrates with the ML4T workflow to provide RL-based
        portfolio allocation decisions.
        """
        if not self.is_trained or self.agent is None:
            logger.warning("RL agent not trained - cannot make decisions")
            return self._get_fallback_decisions()
        
        try:
            # Check circuit breakers
            if any(self.circuit_breakers.values()):
                logger.warning("Circuit breakers active - using conservative strategy")
                return self._get_conservative_decisions()
            
            # Prepare environment state
            if self.environment is None:
                await self.initialize_components(market_data)
            
            # Set current market data
            self.environment.set_data(market_data)
            
            # Get current observation
            observation = self.environment._get_observation()
            
            # Get RL agent decision
            action, metadata = self.agent.get_action(observation, training=False)
            
            # Debug logging
            logger.info(f"🎯 RL Agent Actions: {action}")
            logger.info(f"   Action magnitudes: {[abs(a) for a in action]}")
            logger.info(f"   Max action: {max(abs(a) for a in action):.4f}")
            
            # Convert RL actions to trading decisions
            trading_decisions = await self._convert_actions_to_decisions(
                action, metadata, current_state
            )
            
            # Apply risk management
            if self.config.risk_management_enabled:
                trading_decisions = self._apply_risk_management(trading_decisions, current_state)
            
            # Log decision
            logger.info(f"🤖 RL Trading Decision: {len(trading_decisions.get('orders', []))} orders")
            for order in trading_decisions.get('orders', [])[:3]:  # Log first 3
                logger.info(f"   {order['symbol']}: {order['side']} {order['quantity']} @ {order.get('price', 'market')}")
            
            return trading_decisions
            
        except Exception as e:
            logger.error(f"Error making RL trading decisions: {e}")
            return self._get_fallback_decisions()
    
    async def _convert_actions_to_decisions(self, 
                                          actions: np.ndarray,
                                          metadata: Dict,
                                          current_state: Dict) -> Dict[str, Any]:
        """Convert RL actions to specific trading decisions."""
        
        orders = []
        portfolio_value = current_state.get('portfolio', {}).get('equity', self.config.initial_balance)
        current_prices = self._get_current_prices_from_state(current_state)
        
        logger.info(f"🔄 Converting actions to decisions:")
        logger.info(f"   Portfolio value: ${portfolio_value:,.2f}")
        logger.info(f"   Current prices: {current_prices}")
        logger.info(f"   Max position size: {self.config.max_position_size:.1%}")
        
        for i, action in enumerate(actions):
            if i >= len(self.config.symbols):
                break
            
            symbol = self.config.symbols[i]
            
            logger.info(f"   Processing {symbol}: action={action:.4f}")
            
            if abs(action) < 0.01:  # Ignore very small actions (lowered threshold)
                logger.info(f"     Skipping {symbol}: action too small ({abs(action):.4f} < 0.01)")
                continue
            
            if symbol not in current_prices:
                logger.warning(f"     No price data for {symbol}")
                continue
            
            current_price = current_prices[symbol]
            
            # Calculate position size based on action
            target_value = action * portfolio_value * self.config.max_position_size
            target_shares = target_value / current_price  # Don't convert to int yet
            
            logger.info(f"     Target value: ${target_value:.2f}, Target shares: {target_shares:.2f}")
            
            if abs(target_shares) < 0.1:  # Lower minimum threshold
                logger.info(f"     Skipping {symbol}: target shares too small ({abs(target_shares):.2f} < 0.1)")
                continue
            
            # Determine order type and side based on original action direction
            if action > 0:  # Positive action = buy/long
                side = OrderSide.BUY
                quantity = max(1, int(abs(target_shares)))
            else:  # Negative action = short
                if self.config.enable_short_selling:
                    side = OrderSide.SHORT
                else:
                    side = OrderSide.SELL
                quantity = max(1, int(abs(target_shares)))
            
            logger.info(f"     Order: {side.value} {quantity} {symbol} @ ${current_price:.2f}")
            
            # Create order
            order = {
                'symbol': symbol,
                'side': side.value,
                'quantity': quantity,
                'order_type': OrderType.MARKET.value,
                'price': current_price,
                'confidence': metadata.get('confidence', 0.7),
                'source': 'finrl_agent',
                'action_value': float(action)
            }
            
            # Add limit order if enabled and action is strong enough
            if (self.config.enable_limit_orders and 
                abs(action) > 0.3 and 
                metadata.get('confidence', 0) > 0.8):
                
                # Create limit order slightly better than current price
                if side == OrderSide.BUY:
                    limit_price = current_price * 0.995  # 0.5% below current
                else:
                    limit_price = current_price * 1.005  # 0.5% above current
                
                order['order_type'] = OrderType.LIMIT.value
                order['limit_price'] = limit_price
            
            orders.append(order)
        
        return {
            'orders': orders,
            'strategy': 'finrl_rl_optimized',
            'confidence': metadata.get('confidence', 0.7),
            'reasoning': f"RL agent decision with {len(orders)} positions",
            'risk_level': self._assess_risk_level(actions),
            'metadata': {
                'rl_actions': actions.tolist(),
                'agent_metadata': metadata,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _get_current_prices_from_state(self, state: Dict) -> Dict[str, float]:
        """Extract current prices from trading state."""
        prices = {}
        market_data = state.get('market_data', {})
        
        for symbol in self.config.symbols:
            if symbol in market_data:
                # Try to get current price from various sources
                symbol_data = market_data[symbol]
                if isinstance(symbol_data, dict):
                    price = (symbol_data.get('close') or 
                            symbol_data.get('price') or 
                            symbol_data.get('last'))
                else:
                    price = symbol_data  # Assume it's the price directly
                
                if price and float(price) > 0:
                    prices[symbol] = float(price)
                    
        # If no prices found, use default test prices
        if not prices:
            logger.warning("No market data prices found, using defaults for testing")
            default_prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GME': 80.0, 'NVDA': 120.0, 'TSLA': 200.0}
            for symbol in self.config.symbols:
                if symbol in default_prices:
                    prices[symbol] = default_prices[symbol]
        
        return prices
    
    def _assess_risk_level(self, actions: np.ndarray) -> str:
        """Assess risk level based on action magnitudes."""
        max_action = np.max(np.abs(actions))
        mean_action = np.mean(np.abs(actions))
        
        if max_action > 0.8 or mean_action > 0.5:
            return "high"
        elif max_action > 0.5 or mean_action > 0.3:
            return "medium"
        else:
            return "low"
    
    def _apply_risk_management(self, decisions: Dict, current_state: Dict) -> Dict:
        """Apply risk management rules to trading decisions."""
        orders = decisions.get('orders', [])
        filtered_orders = []
        
        portfolio = current_state.get('portfolio', {})
        cash = portfolio.get('cash', 0)
        equity = portfolio.get('equity', self.config.initial_balance)
        
        # Calculate total order value
        total_order_value = 0
        for order in orders:
            order_value = order['quantity'] * order['price']
            total_order_value += order_value
        
        # Risk management checks
        
        # 1. Cash availability check
        if total_order_value > cash * 1.5:  # Allow some margin
            logger.warning("Insufficient cash for all orders - scaling down")
            scale_factor = (cash * 1.5) / total_order_value
            for order in orders:
                order['quantity'] = max(1, int(order['quantity'] * scale_factor))
        
        # 2. Position concentration check
        position_values = {}
        for order in orders:
            symbol = order['symbol']
            order_value = order['quantity'] * order['price']
            position_values[symbol] = position_values.get(symbol, 0) + order_value
        
        for order in orders:
            symbol = order['symbol']
            position_value = position_values[symbol]
            
            # Check if position exceeds maximum size
            if position_value > equity * self.config.max_position_size:
                max_quantity = int((equity * self.config.max_position_size) / order['price'])
                if max_quantity >= 1:
                    order['quantity'] = min(order['quantity'], max_quantity)
                    filtered_orders.append(order)
                else:
                    logger.warning(f"Skipping {symbol} order - exceeds position size limit")
            else:
                filtered_orders.append(order)
        
        # 3. Short selling risk check
        if self.config.enable_short_selling:
            short_orders = [o for o in filtered_orders if o['side'] == 'short']
            total_short_value = sum(o['quantity'] * o['price'] for o in short_orders)
            
            if total_short_value > equity * 0.3:  # Max 30% of equity in shorts
                logger.warning("Excessive short exposure - reducing short positions")
                scale_factor = (equity * 0.3) / total_short_value
                for order in short_orders:
                    order['quantity'] = max(1, int(order['quantity'] * scale_factor))
        
        decisions['orders'] = filtered_orders
        decisions['risk_management_applied'] = True
        
        return decisions
    
    def _get_fallback_decisions(self) -> Dict[str, Any]:
        """Get fallback decisions when RL agent is not available."""
        return {
            'orders': [],
            'strategy': 'finrl_fallback',
            'confidence': 0.3,
            'reasoning': 'RL agent not available - no trades',
            'risk_level': 'low',
            'fallback': True
        }
    
    def _get_conservative_decisions(self) -> Dict[str, Any]:
        """Get conservative decisions when circuit breakers are active."""
        return {
            'orders': [],
            'strategy': 'finrl_conservative',
            'confidence': 0.5,
            'reasoning': 'Circuit breakers active - conservative strategy',
            'risk_level': 'low',
            'conservative_mode': True
        }
    
    async def update_performance_tracking(self, 
                                        execution_results: Dict,
                                        portfolio_state: Dict):
        """Update performance tracking and check circuit breakers."""
        
        # Calculate performance metrics
        current_equity = portfolio_state.get('equity', self.config.initial_balance)
        portfolio_return = (current_equity - self.config.initial_balance) / self.config.initial_balance
        
        # Add to performance history
        performance_entry = {
            'timestamp': datetime.now().isoformat(),
            'equity': current_equity,
            'portfolio_return': portfolio_return,
            'execution_results': execution_results
        }
        
        self.performance_history.append(performance_entry)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        # Check circuit breakers
        await self._check_circuit_breakers()
        
        # Retrain if needed
        if (self.agent and 
            len(self.performance_history) % self.config.retraining_frequency == 0):
            logger.info("Performance update - considering retraining")
    
    async def _check_circuit_breakers(self):
        """Check and update circuit breaker status."""
        if len(self.performance_history) < 10:
            return
        
        recent_performance = self.performance_history[-20:]  # Last 20 updates
        returns = [p['portfolio_return'] for p in recent_performance]
        
        # Max drawdown check
        peak = max(returns)
        current = returns[-1]
        drawdown = (current - peak) / peak if peak > 0 else 0
        
        if drawdown < self.config.max_drawdown_threshold:
            self.circuit_breakers['max_drawdown'] = True
            logger.warning(f"Max drawdown circuit breaker triggered: {drawdown:.2%}")
        else:
            self.circuit_breakers['max_drawdown'] = False
        
        # Sharpe ratio check (simplified)
        if len(returns) >= 10:
            return_changes = np.diff(returns)
            if len(return_changes) > 0:
                sharpe = np.mean(return_changes) / (np.std(return_changes) + 1e-8)
                
                if sharpe < self.config.min_sharpe_ratio:
                    self.circuit_breakers['low_sharpe'] = True
                    logger.warning(f"Low Sharpe ratio circuit breaker triggered: {sharpe:.3f}")
                else:
                    self.circuit_breakers['low_sharpe'] = False
        
        # Excessive losses check
        recent_losses = sum(1 for r in returns[-10:] if r < returns[0] * 0.95)
        if recent_losses >= 7:  # 7 out of 10 recent periods with losses
            self.circuit_breakers['excessive_losses'] = True
            logger.warning("Excessive losses circuit breaker triggered")
        else:
            self.circuit_breakers['excessive_losses'] = False
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    
    async def save_state(self, filepath: str = None):
        """Save orchestrator state."""
        if filepath is None:
            filepath = self.model_dir / f"orchestrator_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        state = {
            'config': {
                'symbols': self.config.symbols,
                'initial_balance': self.config.initial_balance,
                'enable_short_selling': self.config.enable_short_selling,
                'enable_limit_orders': self.config.enable_limit_orders,
                'max_position_size': self.config.max_position_size
            },
            'is_trained': self.is_trained,
            'circuit_breakers': self.circuit_breakers,
            'performance_history': self.performance_history[-100:],  # Last 100 entries
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Orchestrator state saved: {filepath}")
    
    async def load_state(self, filepath: str):
        """Load orchestrator state."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.is_trained = state.get('is_trained', False)
            self.circuit_breakers = state.get('circuit_breakers', {})
            self.performance_history = state.get('performance_history', [])
            
            logger.info(f"Orchestrator state loaded: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load orchestrator state: {e}")
            return False
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary."""
        summary = {
            'orchestrator_status': {
                'is_trained': self.is_trained,
                'symbols': self.config.symbols,
                'circuit_breakers': self.circuit_breakers,
                'paper_trading': self.config.paper_trading
            },
            'agent_status': {},
            'performance_summary': {}
        }
        
        # Agent status
        if self.agent:
            perf_stats = self.agent.get_performance_stats()
            summary['agent_status'] = {
                'total_episodes': perf_stats.get('total_episodes', 0),
                'training_steps': perf_stats.get('total_training_steps', 0),
                'average_reward': perf_stats.get('average_reward', 0),
                'buffer_utilization': perf_stats.get('buffer_utilization', 0)
            }
        
        # Performance summary
        if self.performance_history:
            recent_performance = self.performance_history[-10:]
            current_return = recent_performance[-1]['portfolio_return']
            avg_return = np.mean([p['portfolio_return'] for p in recent_performance])
            
            summary['performance_summary'] = {
                'current_portfolio_return': current_return,
                'recent_avg_return': avg_return,
                'total_updates': len(self.performance_history),
                'last_update': self.performance_history[-1]['timestamp']
            }
        
        return summary
    
    async def close(self):
        """Clean up resources."""
        if self.llm_enricher:
            await self.llm_enricher.close()
        
        if self.environment:
            self.environment.close()
        
        logger.info("✅ FinRL Trading Orchestrator closed")


# Integration functions for ML4T workflow
async def create_finrl_trading_system(symbols: List[str], **kwargs) -> FinRLTradingOrchestrator:
    """Create FinRL trading system for ML4T integration."""
    
    # Extract specific configuration parameters to avoid duplicates
    config_params = {
        'symbols': symbols,
        'initial_balance': kwargs.get('initial_balance', 100000),
        'commission_rate': kwargs.get('commission_rate', 0.001),
        'margin_requirement': kwargs.get('margin_requirement', 0.5),
        'short_borrow_rate': kwargs.get('short_borrow_rate', 0.03),
        'enable_short_selling': kwargs.get('enable_short_selling', True),
        'enable_limit_orders': kwargs.get('enable_limit_orders', True),
        'max_position_size': kwargs.get('max_position_size', 0.15),
        'paper_trading': kwargs.get('paper_trading', True),
        'risk_management_enabled': kwargs.get('risk_management_enabled', True),
    }
    
    # Add RL config if provided
    if 'rl_config' in kwargs:
        config_params['rl_config'] = kwargs['rl_config']
    
    config = FinRLTradingConfig(**config_params)
    
    orchestrator = FinRLTradingOrchestrator(config)
    return orchestrator


async def integrate_finrl_with_workflow(orchestrator: FinRLTradingOrchestrator,
                                      workflow_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integration point for ML4T workflow.
    
    This function is called from the continuous rebalancer to get
    RL-based trading decisions.
    """
    
    # Extract relevant data from workflow state
    sentiment_data = workflow_state.get('sentiment_data', {})
    market_data = workflow_state.get('market_data', {})
    portfolio = workflow_state.get('portfolio', {})
    
    # Convert market data to DataFrame format if needed
    if isinstance(market_data, dict) and market_data:
        # Convert to DataFrame format expected by FinRL environment
        market_df = pd.DataFrame(market_data)
    else:
        logger.warning("No market data available for FinRL integration")
        return orchestrator._get_fallback_decisions()
    
    # Make trading decisions
    trading_decisions = await orchestrator.make_trading_decisions(
        current_state=workflow_state,
        market_data=market_df
    )
    
    # Update workflow state with RL decisions
    workflow_state['finrl_decisions'] = trading_decisions
    workflow_state['finrl_enhanced'] = True
    
    return workflow_state


if __name__ == "__main__":
    # Test FinRL integration
    # Note: Using synthetic data generation instead of yfinance for testing
    
    async def test_finrl_integration():
        # Test configuration
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        # Create orchestrator
        orchestrator = await create_finrl_trading_system(
            symbols=symbols,
            initial_balance=100000,
            enable_short_selling=True,
            enable_limit_orders=True,
            paper_trading=True
        )
        
        # Generate synthetic test market data
        from datetime import datetime, timedelta
        import numpy as np
        
        days = 60
        dates = pd.bdate_range(start=datetime.now() - timedelta(days=days), periods=days)
        data_dict = {}
        
        for symbol in symbols:
            np.random.seed(hash(symbol) % 2**32)
            initial_price = np.random.uniform(50, 200)
            returns = np.random.normal(0.001, 0.02, days)
            prices = initial_price * np.exp(np.cumsum(returns))
            
            data_dict[f"{symbol}_open"] = prices * np.random.uniform(0.995, 1.005, days)
            data_dict[f"{symbol}_high"] = prices * np.random.uniform(1.0, 1.02, days)
            data_dict[f"{symbol}_low"] = prices * np.random.uniform(0.98, 1.0, days)
            data_dict[f"{symbol}_close"] = prices
            data_dict[f"{symbol}_volume"] = np.random.uniform(1000000, 10000000, days)
        
        market_data = pd.DataFrame(data_dict, index=dates)
        market_data = market_data.fillna(method='ffill').fillna(method='bfill')
        
        print("Testing FinRL Integration")
        print(f"Symbols: {symbols}")
        print(f"Market data shape: {market_data.shape}")
        
        # Initialize components
        await orchestrator.initialize_components(market_data)
        
        # Test training (short training for demo)
        print("\nTesting agent training...")
        training_results = await orchestrator.train_agent(
            training_data=market_data,
            num_episodes=50,  # Short training for test
            eval_frequency=25
        )
        
        print(f"Training completed - Episodes: {training_results['performance_stats']['total_episodes']}")
        
        # Test trading decisions
        print("\nTesting trading decisions...")
        test_state = {
            'portfolio': {'equity': 100000, 'cash': 50000},
            'market_data': {symbol: {'close': 150.0} for symbol in symbols},
            'sentiment_data': {}
        }
        
        decisions = await orchestrator.make_trading_decisions(test_state, market_data)
        
        print(f"Generated {len(decisions['orders'])} trading orders:")
        for order in decisions['orders'][:3]:
            print(f"  {order['symbol']}: {order['side']} {order['quantity']} @ {order.get('price', 'market')}")
        
        # Test status summary
        status = orchestrator.get_status_summary()
        print(f"\nOrchestrator Status:")
        print(f"  Trained: {status['orchestrator_status']['is_trained']}")
        print(f"  Circuit Breakers: {status['orchestrator_status']['circuit_breakers']}")
        
        # Test save/load
        print("\nTesting save/load...")
        await orchestrator.save_state()
        
        await orchestrator.close()
        print("✅ FinRL Integration test completed")
    
    asyncio.run(test_finrl_integration())