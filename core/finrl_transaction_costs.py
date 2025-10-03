#!/usr/bin/env python3
"""
FinRL Trading Environment with Realistic Transaction Costs

Extends FinRL's StockTradingEnv to include:
1. Commission costs proportional to trade value
2. Slippage as a function of trade size vs. average daily volume
3. Volume-aware trade size constraints (max % of ADV)
4. Adjusted execution prices
5. Cost tracking and metrics

Usage:
    from core.finrl_transaction_costs import TradingEnvWithCosts

    env = TradingEnvWithCosts(
        df=market_data,
        commission_rate=0.001,      # 10 bps (0.1%)
        slippage_coeff=0.0005,      # Slippage coefficient
        max_trade_pct_adv=0.10,     # Max 10% of ADV per trade
        track_costs=True            # Enable cost tracking
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionCostModel:
    """
    Realistic transaction cost model with commission and slippage.

    Commission Model:
        - Fixed percentage of trade value
        - Typical: 0.001 (0.1% or 10 bps)

    Slippage Model:
        - Square-root impact model (commonly used in academic literature)
        - Slippage = coeff * sqrt(trade_size / ADV)
        - Higher slippage for larger trades relative to ADV

    Volume Constraints:
        - Limit trades to max % of average daily volume
        - Prevents unrealistic large trades that would move market
    """

    def __init__(
        self,
        commission_rate: float = 0.001,      # 10 bps
        slippage_coeff: float = 0.0005,      # Calibrated for typical stocks
        max_trade_pct_adv: float = 0.10,     # Max 10% of ADV
        min_commission: float = 0.0,         # Minimum commission per trade
        use_bps_slippage: bool = True        # Express slippage in bps
    ):
        """
        Initialize transaction cost model.

        Args:
            commission_rate: Commission as % of trade value (e.g., 0.001 = 0.1%)
            slippage_coeff: Coefficient for square-root slippage model
            max_trade_pct_adv: Maximum trade size as % of ADV
            min_commission: Minimum commission per trade (e.g., $1.00)
            use_bps_slippage: If True, slippage_coeff is in bps (basis points)
        """
        self.commission_rate = commission_rate
        self.slippage_coeff = slippage_coeff
        self.max_trade_pct_adv = max_trade_pct_adv
        self.min_commission = min_commission
        self.use_bps_slippage = use_bps_slippage

        # Cost tracking
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_trades = 0
        self.constrained_trades = 0  # Trades limited by ADV constraint

        logger.info(f"Transaction Cost Model initialized:")
        logger.info(f"  Commission: {commission_rate*100:.3f}%")
        logger.info(f"  Slippage coefficient: {slippage_coeff}")
        logger.info(f"  Max trade size: {max_trade_pct_adv*100:.1f}% of ADV")

    def calculate_costs(
        self,
        trade_size: float,
        price: float,
        avg_daily_volume: float,
        bid_ask_spread: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate all transaction costs for a trade.

        Args:
            trade_size: Number of shares to trade (positive = buy, negative = sell)
            price: Current market price
            avg_daily_volume: Average daily volume for the stock
            bid_ask_spread: Optional bid-ask spread in dollars

        Returns:
            Dict with keys:
                - commission: Commission cost
                - slippage_bps: Slippage in basis points
                - slippage_dollars: Slippage cost in dollars
                - execution_price: Actual execution price
                - constrained_size: Final trade size after ADV constraint
                - was_constrained: Whether trade was limited by ADV
        """
        # Enforce volume constraint
        max_trade = self.max_trade_pct_adv * avg_daily_volume
        constrained_size = np.clip(trade_size, -max_trade, max_trade)
        was_constrained = abs(constrained_size) < abs(trade_size)

        if was_constrained:
            self.constrained_trades += 1

        # Calculate commission
        trade_value = abs(constrained_size) * price
        commission = max(trade_value * self.commission_rate, self.min_commission)

        # Calculate slippage (square-root model)
        if avg_daily_volume > 0:
            participation_rate = abs(constrained_size) / avg_daily_volume
            slippage_bps = self.slippage_coeff * np.sqrt(participation_rate * 10000)  # Scale to bps

            if not self.use_bps_slippage:
                # Convert to percentage
                slippage_bps = slippage_bps * 100
        else:
            slippage_bps = 0.0

        # Add bid-ask spread if provided
        if bid_ask_spread is not None:
            spread_cost_bps = (bid_ask_spread / (2 * price)) * 10000
            slippage_bps += spread_cost_bps

        # Calculate execution price
        slippage_pct = slippage_bps / 10000
        if constrained_size > 0:  # Buy
            execution_price = price * (1 + slippage_pct)
        elif constrained_size < 0:  # Sell
            execution_price = price * (1 - slippage_pct)
        else:
            execution_price = price

        # Calculate slippage cost in dollars
        slippage_dollars = abs(constrained_size) * (abs(execution_price - price))

        # Update tracking
        self.total_commission += commission
        self.total_slippage += slippage_dollars
        if abs(constrained_size) > 0:
            self.total_trades += 1

        return {
            'commission': commission,
            'slippage_bps': slippage_bps,
            'slippage_dollars': slippage_dollars,
            'execution_price': execution_price,
            'constrained_size': constrained_size,
            'was_constrained': was_constrained,
            'participation_rate': abs(constrained_size) / max(avg_daily_volume, 1)
        }

    def get_statistics(self) -> Dict[str, float]:
        """Get cumulative cost statistics."""
        avg_commission = self.total_commission / max(self.total_trades, 1)
        avg_slippage = self.total_slippage / max(self.total_trades, 1)
        constraint_rate = self.constrained_trades / max(self.total_trades, 1)

        return {
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_costs': self.total_commission + self.total_slippage,
            'total_trades': self.total_trades,
            'avg_commission_per_trade': avg_commission,
            'avg_slippage_per_trade': avg_slippage,
            'constrained_trades': self.constrained_trades,
            'constraint_rate': constraint_rate
        }

    def reset_statistics(self):
        """Reset cost tracking statistics."""
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_trades = 0
        self.constrained_trades = 0


class TradingEnvWithCosts:
    """
    Extended FinRL trading environment with realistic transaction costs.

    This wraps or extends the FinRL StockTradingEnv to add:
    - Commission costs
    - Market impact (slippage)
    - Volume constraints
    - Cost tracking and metrics

    Compatible with Stable-Baselines3 and FinRL agent wrapper.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int = 100,
        initial_amount: float = 1000000,
        transaction_cost_pct: float = 0.001,
        reward_scaling: float = 1e-4,
        state_space: int = None,
        action_space: int = None,
        tech_indicator_list: List[str] = None,
        turbulence_threshold: float = None,
        risk_indicator_col: str = 'turbulence',
        make_plots: bool = False,
        print_verbosity: int = 10,
        day: int = 0,
        initial: bool = True,
        previous_state: List = [],
        model_name: str = '',
        mode: str = '',
        iteration: str = '',
        # Transaction cost parameters
        commission_rate: float = 0.001,
        slippage_coeff: float = 0.0005,
        max_trade_pct_adv: float = 0.10,
        track_costs: bool = True,
        use_realistic_prices: bool = True
    ):
        """
        Initialize trading environment with transaction costs.

        Args:
            df: Market data DataFrame with OHLCV and indicators
            commission_rate: Commission as % of trade value
            slippage_coeff: Slippage coefficient for square-root model
            max_trade_pct_adv: Maximum trade size as % of ADV
            track_costs: Enable detailed cost tracking
            use_realistic_prices: Use mid-price instead of close for execution
            ... (other FinRL parameters)
        """
        # Store parameters
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list or []
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.day = day
        self.track_costs = track_costs
        self.use_realistic_prices = use_realistic_prices

        # Initialize transaction cost model
        self.cost_model = TransactionCostModel(
            commission_rate=commission_rate,
            slippage_coeff=slippage_coeff,
            max_trade_pct_adv=max_trade_pct_adv
        )

        # State variables
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_boolean = 0
        self.initial = initial
        self.previous_state = previous_state

        # Initialize portfolio
        self.state = self._initiate_state()

        # Action and observation spaces (Gym compatibility)
        self.action_space_dim = self.stock_dim
        self.observation_space_dim = len(self.state)

        # Cost tracking
        self.episode_costs = []
        self.episode_returns = []

        logger.info(f"Trading environment initialized with realistic costs")
        logger.info(f"  Stocks: {stock_dim}")
        logger.info(f"  Initial capital: ${initial_amount:,.2f}")
        logger.info(f"  Commission: {commission_rate*100:.3f}%")
        logger.info(f"  Max trade: {max_trade_pct_adv*100:.0f}% of ADV")

    def _initiate_state(self) -> np.ndarray:
        """Initialize state with cash, holdings, and indicators."""
        # Cash balance
        state = [self.initial_amount]

        # Holdings (0 for all stocks initially)
        state.extend([0] * self.stock_dim)

        # Stock prices
        state.extend(self.data['close'].values.tolist())

        # Technical indicators
        for indicator in self.tech_indicator_list:
            if indicator in self.data.columns:
                state.extend(self.data[indicator].values.tolist())

        return np.array(state)

    def _get_observation(self) -> np.ndarray:
        """Get current observation (state)."""
        # Update data
        self.data = self.df.loc[self.day, :]

        # Cash balance
        state = [self.state[0]]

        # Holdings
        state.extend(self.state[1:self.stock_dim + 1].tolist())

        # Current prices
        state.extend(self.data['close'].values.tolist())

        # Technical indicators
        for indicator in self.tech_indicator_list:
            if indicator in self.data.columns:
                state.extend(self.data[indicator].values.tolist())

        return np.array(state)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one trading step with realistic transaction costs.

        Args:
            actions: Array of target positions for each stock

        Returns:
            observation: New state
            reward: Reward (change in portfolio value minus costs)
            done: Whether episode is complete
            info: Additional information including costs
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            # End of episode
            obs = self.state
            reward = 0
            info = self._get_episode_summary()
            return obs, reward, True, info

        # Get current state
        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1:self.stock_dim + 1]) *
            np.array(self.state[self.stock_dim + 1:2 * self.stock_dim + 1])
        )

        # Get current prices and volumes
        prices = self.data['close'].values
        volumes = self.data['volume'].values if 'volume' in self.data.columns else np.ones(self.stock_dim) * 1e6

        # Calculate average daily volume (use 20-day average if available)
        avg_daily_volumes = volumes  # Simplified - in production, use rolling average

        # Current holdings
        current_holdings = np.array(self.state[1:self.stock_dim + 1])

        # Process each stock's action
        total_commission = 0.0
        total_slippage = 0.0
        execution_info = []

        for i in range(self.stock_dim):
            # Action is target position
            target_position = actions[i]
            current_position = current_holdings[i]
            trade_size = target_position - current_position

            if abs(trade_size) < 1e-6:  # No trade
                continue

            # Calculate transaction costs
            costs = self.cost_model.calculate_costs(
                trade_size=trade_size,
                price=prices[i],
                avg_daily_volume=avg_daily_volumes[i]
            )

            # Update position and cash
            final_trade_size = costs['constrained_size']
            execution_price = costs['execution_price']
            commission = costs['commission']

            # Update holdings
            self.state[i + 1] += final_trade_size

            # Update cash (deduct trade cost + commission)
            trade_cost = final_trade_size * execution_price
            self.state[0] -= (trade_cost + commission)

            # Track costs
            total_commission += commission
            total_slippage += costs['slippage_dollars']

            execution_info.append({
                'stock_idx': i,
                'trade_size': trade_size,
                'final_size': final_trade_size,
                'price': prices[i],
                'execution_price': execution_price,
                'commission': commission,
                'slippage_bps': costs['slippage_bps'],
                'was_constrained': costs['was_constrained']
            })

        # Calculate new portfolio value
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1:self.stock_dim + 1]) *
            np.array(prices)
        )

        # Reward = change in portfolio value (costs already deducted from cash)
        reward = (end_total_asset - begin_total_asset) * self.reward_scaling

        # Move to next day
        self.day += 1
        self.data = self.df.loc[self.day, :]

        # Get new observation
        obs = self._get_observation()

        # Track costs for episode
        if self.track_costs:
            self.episode_costs.append({
                'day': self.day,
                'commission': total_commission,
                'slippage': total_slippage,
                'total_cost': total_commission + total_slippage,
                'portfolio_value': end_total_asset,
                'num_trades': len(execution_info)
            })

        # Info dictionary
        info = {
            'commission': total_commission,
            'slippage': total_slippage,
            'total_cost': total_commission + total_slippage,
            'portfolio_value': end_total_asset,
            'num_trades': len(execution_info),
            'executions': execution_info
        }

        done = self.terminal

        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()
        self.terminal = False

        # Reset cost tracking
        self.episode_costs = []
        self.episode_returns = []

        return self.state

    def _get_episode_summary(self) -> Dict[str, Any]:
        """Get summary statistics for completed episode."""
        cost_stats = self.cost_model.get_statistics()

        total_costs = sum(c['total_cost'] for c in self.episode_costs)
        total_commission = sum(c['commission'] for c in self.episode_costs)
        total_slippage = sum(c['slippage'] for c in self.episode_costs)

        return {
            'episode_total_commission': total_commission,
            'episode_total_slippage': total_slippage,
            'episode_total_costs': total_costs,
            'episode_num_trades': cost_stats['total_trades'],
            'avg_cost_per_trade': total_costs / max(cost_stats['total_trades'], 1),
            'cost_statistics': cost_stats,
            'costs_as_pct_of_value': total_costs / max(self.initial_amount, 1) * 100
        }

    def render(self, mode='human'):
        """Render environment (for debugging)."""
        return self.state

    def save_asset_memory(self):
        """Save portfolio value history."""
        return self.episode_costs

    def save_action_memory(self):
        """Save action history."""
        return []  # Implement if needed


def create_env_with_costs(
    df: pd.DataFrame,
    stock_dim: int,
    commission_rate: float = 0.001,
    slippage_coeff: float = 0.0005,
    max_trade_pct_adv: float = 0.10,
    **kwargs
) -> TradingEnvWithCosts:
    """
    Factory function to create trading environment with costs.

    Args:
        df: Market data DataFrame
        stock_dim: Number of stocks
        commission_rate: Commission percentage
        slippage_coeff: Slippage coefficient
        max_trade_pct_adv: Max trade size as % of ADV
        **kwargs: Additional environment parameters

    Returns:
        Initialized TradingEnvWithCosts
    """
    return TradingEnvWithCosts(
        df=df,
        stock_dim=stock_dim,
        commission_rate=commission_rate,
        slippage_coeff=slippage_coeff,
        max_trade_pct_adv=max_trade_pct_adv,
        track_costs=True,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL']

    data = []
    for date in dates:
        for symbol in symbols:
            data.append({
                'date': date,
                'tic': symbol,
                'close': np.random.uniform(100, 200),
                'volume': np.random.uniform(1e6, 1e7),
                'macd': np.random.randn(),
                'rsi_30': np.random.uniform(30, 70)
            })

    df = pd.DataFrame(data)
    df = df.set_index('date')

    # Create environment
    env = create_env_with_costs(
        df=df,
        stock_dim=len(symbols),
        commission_rate=0.001,
        slippage_coeff=0.0005,
        max_trade_pct_adv=0.10
    )

    # Test environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    # Random actions
    for _ in range(10):
        action = np.random.randint(-10, 10, size=len(symbols))
        obs, reward, done, info = env.step(action)

        print(f"Step {env.day}:")
        print(f"  Commission: ${info['commission']:.2f}")
        print(f"  Slippage: ${info['slippage']:.2f}")
        print(f"  Total cost: ${info['total_cost']:.2f}")
        print(f"  Portfolio value: ${info['portfolio_value']:,.2f}")
        print(f"  Reward: {reward:.6f}")

        if done:
            break

    # Get episode summary
    summary = env._get_episode_summary()
    print("\nEpisode Summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
