"""
Pairs Trading Strategy Implementation

This module implements a statistical arbitrage strategy based on pairs trading,
which profits from mean reversion in the spread between two cointegrated securities.

Key Components:
1. Pair Selection - Statistical testing for cointegration
2. Spread Calculation - Beta estimation and spread monitoring
3. Signal Generation - Z-score based entry/exit signals
4. Risk Management - Stop-loss and position sizing controls
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.api import OLS, add_constant
from sklearn.linear_model import LinearRegression
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class PairCandidate:
    """Data structure for a potential trading pair"""
    symbol_a: str
    symbol_b: str
    correlation: float
    p_value: float  # Cointegration p-value
    beta: float  # Hedge ratio
    adf_stat: float  # Augmented Dickey-Fuller statistic
    half_life: float  # Mean reversion half-life in days
    score: float  # Overall pair quality score


@dataclass
class PairPosition:
    """Active pairs trading position"""
    symbol_a: str
    symbol_b: str
    beta: float
    entry_spread: float
    entry_zscore: float
    current_spread: float
    current_zscore: float
    position_side: str  # 'long_spread' or 'short_spread'
    quantity_a: int
    quantity_b: int
    entry_time: pd.Timestamp
    unrealized_pnl: float
    stop_loss_zscore: float


class PairSignal(NamedTuple):
    """Trading signal for pairs strategy"""
    symbol_a: str
    symbol_b: str
    action: str  # 'enter_long_spread', 'enter_short_spread', 'exit', 'stop_loss'
    zscore: float
    spread: float
    confidence: float
    suggested_quantity_a: int
    suggested_quantity_b: int


class PairsStrategy:
    """
    Statistical arbitrage pairs trading strategy
    
    This strategy identifies cointegrated pairs of securities and trades
    the mean-reverting spread between them.
    """
    
    def __init__(
        self,
        lookback_period: int = 252,  # 1 year for pair selection
        signal_window: int = 20,     # Rolling window for z-score calculation
        entry_zscore: float = 2.0,   # Entry threshold
        exit_zscore: float = 0.0,    # Exit threshold
        stop_loss_zscore: float = 3.0,  # Stop-loss threshold
        min_half_life: float = 1.0,     # Minimum mean reversion half-life (days)
        max_half_life: float = 30.0,    # Maximum mean reversion half-life (days)
        max_positions: int = 5,         # Maximum number of active pairs
        position_size: float = 10000.0, # Dollar amount per leg
        correlation_threshold: float = 0.7,  # Minimum correlation for pair selection
        cointegration_pvalue: float = 0.05,  # Maximum p-value for cointegration
    ):
        self.lookback_period = lookback_period
        self.signal_window = signal_window
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.max_positions = max_positions
        self.position_size = position_size
        self.correlation_threshold = correlation_threshold
        self.cointegration_pvalue = cointegration_pvalue
        
        # State tracking
        self.active_pairs: Dict[str, PairPosition] = {}
        self.pair_candidates: List[PairCandidate] = []
        self.price_history: Dict[str, pd.Series] = {}
        
    def find_pairs(self, symbols: List[str], price_data: Dict[str, pd.Series]) -> List[PairCandidate]:
        """
        Find cointegrated pairs from a universe of symbols
        
        Args:
            symbols: List of symbols to test for pairs
            price_data: Dictionary mapping symbols to price series
            
        Returns:
            List of ranked pair candidates
        """
        pairs = []
        
        # Test all symbol combinations
        for i, symbol_a in enumerate(symbols):
            for symbol_b in symbols[i+1:]:
                if symbol_a not in price_data or symbol_b not in price_data:
                    continue
                    
                pair_candidate = self._test_pair_cointegration(
                    symbol_a, symbol_b, price_data[symbol_a], price_data[symbol_b]
                )
                
                if pair_candidate and self._is_valid_pair(pair_candidate):
                    pairs.append(pair_candidate)
        
        # Rank pairs by quality score
        pairs.sort(key=lambda x: x.score, reverse=True)
        self.pair_candidates = pairs[:20]  # Keep top 20 pairs
        
        logger.info(f"Found {len(self.pair_candidates)} valid pairs from {len(symbols)} symbols")
        return self.pair_candidates
    
    def _test_pair_cointegration(
        self, 
        symbol_a: str, 
        symbol_b: str, 
        prices_a: pd.Series, 
        prices_b: pd.Series
    ) -> Optional[PairCandidate]:
        """
        Test a pair for cointegration using Engle-Granger method
        """
        try:
            # Align price series
            aligned_data = pd.DataFrame({
                'A': prices_a,
                'B': prices_b
            }).dropna()
            
            if len(aligned_data) < self.lookback_period:
                return None
            
            # Take most recent data for testing
            test_data = aligned_data.tail(self.lookback_period)
            prices_a_test = test_data['A']
            prices_b_test = test_data['B']
            
            # Calculate correlation
            correlation = prices_a_test.corr(prices_b_test)
            
            if abs(correlation) < self.correlation_threshold:
                return None
            
            # Perform cointegration test (Engle-Granger)
            # Step 1: Estimate the long-run relationship
            X = add_constant(prices_b_test)
            model = OLS(prices_a_test, X).fit()
            beta = model.params.iloc[1]  # Hedge ratio
            
            # Step 2: Test residuals for stationarity
            spread = prices_a_test - beta * prices_b_test
            adf_stat, p_value, _, _, _, _ = adfuller(spread, maxlag=1)
            
            # Calculate mean reversion half-life
            half_life = self._calculate_half_life(spread)
            
            # Calculate quality score
            score = self._calculate_pair_score(correlation, p_value, half_life, adf_stat)
            
            return PairCandidate(
                symbol_a=symbol_a,
                symbol_b=symbol_b,
                correlation=correlation,
                p_value=p_value,
                beta=beta,
                adf_stat=adf_stat,
                half_life=half_life,
                score=score
            )
            
        except Exception as e:
            logger.warning(f"Error testing pair {symbol_a}-{symbol_b}: {e}")
            return None
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate mean reversion half-life using Ornstein-Uhlenbeck process
        """
        try:
            # Lag the spread
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            
            # Align series
            min_len = min(len(spread_lag), len(spread_diff))
            spread_lag = spread_lag.iloc[-min_len:]
            spread_diff = spread_diff.iloc[-min_len:]
            
            # Regression: Δspread = α + β*spread_lag + ε
            X = add_constant(spread_lag)
            model = OLS(spread_diff, X).fit()
            beta = model.params.iloc[1]
            
            # Half-life = -ln(2) / β
            if beta < 0:
                half_life = -np.log(2) / beta
            else:
                half_life = np.inf
                
            return half_life
            
        except Exception:
            return np.inf
    
    def _calculate_pair_score(
        self, 
        correlation: float, 
        p_value: float, 
        half_life: float, 
        adf_stat: float
    ) -> float:
        """
        Calculate a quality score for the pair (higher = better)
        """
        # Correlation component (higher is better)
        corr_score = abs(correlation)
        
        # Cointegration strength (lower p-value is better)
        coint_score = max(0, 1 - p_value / self.cointegration_pvalue)
        
        # Mean reversion speed (optimal half-life range)
        if self.min_half_life <= half_life <= self.max_half_life:
            speed_score = 1.0 - (half_life - self.min_half_life) / (self.max_half_life - self.min_half_life)
        else:
            speed_score = 0.0
        
        # Stationarity strength (more negative ADF stat is better)
        stat_score = max(0, min(1, (-adf_stat + 1) / 5))
        
        # Weighted combination
        score = (
            0.3 * corr_score +
            0.3 * coint_score +
            0.2 * speed_score +
            0.2 * stat_score
        )
        
        return score
    
    def _is_valid_pair(self, pair: PairCandidate) -> bool:
        """
        Check if a pair meets our trading criteria
        """
        return (
            pair.p_value <= self.cointegration_pvalue and
            abs(pair.correlation) >= self.correlation_threshold and
            self.min_half_life <= pair.half_life <= self.max_half_life and
            not np.isnan(pair.beta) and
            pair.beta > 0  # Positive relationship
        )
    
    def calculate_spread_zscore(
        self, 
        symbol_a: str, 
        symbol_b: str, 
        beta: float,
        price_data: Dict[str, pd.Series]
    ) -> Tuple[float, float]:
        """
        Calculate current spread and z-score for a pair
        
        Returns:
            Tuple of (current_spread, current_zscore)
        """
        if symbol_a not in price_data or symbol_b not in price_data:
            return np.nan, np.nan
        
        # Get aligned price data
        prices_a = price_data[symbol_a]
        prices_b = price_data[symbol_b]
        
        # Calculate spread over signal window
        spread = prices_a - beta * prices_b
        recent_spread = spread.tail(self.signal_window)
        
        if len(recent_spread) < self.signal_window:
            return np.nan, np.nan
        
        # Calculate z-score
        current_spread = spread.iloc[-1]
        mean_spread = recent_spread.mean()
        std_spread = recent_spread.std()
        
        if std_spread == 0:
            return current_spread, 0.0
        
        zscore = (current_spread - mean_spread) / std_spread
        
        return current_spread, zscore
    
    def generate_signals(self, price_data: Dict[str, pd.Series]) -> List[PairSignal]:
        """
        Generate trading signals for all monitored pairs
        """
        signals = []
        
        # Check existing positions for exit signals
        for pair_key, position in list(self.active_pairs.items()):
            current_spread, current_zscore = self.calculate_spread_zscore(
                position.symbol_a, position.symbol_b, position.beta, price_data
            )
            
            if np.isnan(current_zscore):
                continue
            
            # Update position
            position.current_spread = current_spread
            position.current_zscore = current_zscore
            
            # Check for exit conditions
            signal = self._check_exit_conditions(position)
            if signal:
                signals.append(signal)
        
        # Check for new entry signals if we have capacity
        if len(self.active_pairs) < self.max_positions:
            entry_signals = self._check_entry_conditions(price_data)
            signals.extend(entry_signals)
        
        return signals
    
    def _check_exit_conditions(self, position: PairPosition) -> Optional[PairSignal]:
        """
        Check if an existing position should be closed
        """
        # Stop-loss condition
        if abs(position.current_zscore) >= self.stop_loss_zscore:
            return PairSignal(
                symbol_a=position.symbol_a,
                symbol_b=position.symbol_b,
                action='stop_loss',
                zscore=position.current_zscore,
                spread=position.current_spread,
                confidence=1.0,
                suggested_quantity_a=-position.quantity_a,
                suggested_quantity_b=-position.quantity_b
            )
        
        # Mean reversion exit condition
        if (
            (position.position_side == 'long_spread' and position.current_zscore <= self.exit_zscore) or
            (position.position_side == 'short_spread' and position.current_zscore >= -self.exit_zscore)
        ):
            return PairSignal(
                symbol_a=position.symbol_a,
                symbol_b=position.symbol_b,
                action='exit',
                zscore=position.current_zscore,
                spread=position.current_spread,
                confidence=0.8,
                suggested_quantity_a=-position.quantity_a,
                suggested_quantity_b=-position.quantity_b
            )
        
        return None
    
    def _check_entry_conditions(self, price_data: Dict[str, pd.Series]) -> List[PairSignal]:
        """
        Check for new entry opportunities
        """
        signals = []
        
        for pair in self.pair_candidates:
            # Skip if already have position in this pair
            pair_key = f"{pair.symbol_a}-{pair.symbol_b}"
            if pair_key in self.active_pairs:
                continue
            
            current_spread, current_zscore = self.calculate_spread_zscore(
                pair.symbol_a, pair.symbol_b, pair.beta, price_data
            )
            
            if np.isnan(current_zscore):
                continue
            
            # Check for entry conditions
            if current_zscore >= self.entry_zscore:
                # Short the spread (short A, long B)
                signal = self._create_entry_signal(
                    pair, 'enter_short_spread', current_spread, current_zscore, price_data
                )
                if signal:
                    signals.append(signal)
                    
            elif current_zscore <= -self.entry_zscore:
                # Long the spread (long A, short B)
                signal = self._create_entry_signal(
                    pair, 'enter_long_spread', current_spread, current_zscore, price_data
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _create_entry_signal(
        self, 
        pair: PairCandidate, 
        action: str, 
        spread: float, 
        zscore: float,
        price_data: Dict[str, pd.Series]
    ) -> Optional[PairSignal]:
        """
        Create an entry signal with position sizing
        """
        try:
            # Get current prices
            price_a = price_data[pair.symbol_a].iloc[-1]
            price_b = price_data[pair.symbol_b].iloc[-1]
            
            # Calculate position sizes for dollar neutrality
            if action == 'enter_long_spread':
                # Long A, Short B
                quantity_a = int(self.position_size / price_a)
                quantity_b = -int((quantity_a * price_a * pair.beta) / price_b)
            else:
                # Short A, Long B  
                quantity_a = -int(self.position_size / price_a)
                quantity_b = int((-quantity_a * price_a * pair.beta) / price_b)
            
            # Calculate confidence based on z-score magnitude
            confidence = min(1.0, abs(zscore) / (2 * self.entry_zscore))
            
            return PairSignal(
                symbol_a=pair.symbol_a,
                symbol_b=pair.symbol_b,
                action=action,
                zscore=zscore,
                spread=spread,
                confidence=confidence,
                suggested_quantity_a=quantity_a,
                suggested_quantity_b=quantity_b
            )
            
        except Exception as e:
            logger.warning(f"Error creating entry signal for {pair.symbol_a}-{pair.symbol_b}: {e}")
            return None
    
    def execute_signal(self, signal: PairSignal) -> bool:
        """
        Execute a pairs trading signal
        
        This is a placeholder - actual execution would integrate with
        the trading system's order management
        """
        pair_key = f"{signal.symbol_a}-{signal.symbol_b}"
        
        try:
            if signal.action in ['enter_long_spread', 'enter_short_spread']:
                # Create new position
                position = PairPosition(
                    symbol_a=signal.symbol_a,
                    symbol_b=signal.symbol_b,
                    beta=next(p.beta for p in self.pair_candidates 
                             if p.symbol_a == signal.symbol_a and p.symbol_b == signal.symbol_b),
                    entry_spread=signal.spread,
                    entry_zscore=signal.zscore,
                    current_spread=signal.spread,
                    current_zscore=signal.zscore,
                    position_side=signal.action.replace('enter_', ''),
                    quantity_a=signal.suggested_quantity_a,
                    quantity_b=signal.suggested_quantity_b,
                    entry_time=pd.Timestamp.now(),
                    unrealized_pnl=0.0,
                    stop_loss_zscore=self.stop_loss_zscore
                )
                
                self.active_pairs[pair_key] = position
                logger.info(f"Opened pairs position: {pair_key} ({signal.action})")
                
            elif signal.action in ['exit', 'stop_loss']:
                # Close existing position
                if pair_key in self.active_pairs:
                    del self.active_pairs[pair_key]
                    logger.info(f"Closed pairs position: {pair_key} ({signal.action})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing signal for {pair_key}: {e}")
            return False
    
    def get_portfolio_status(self) -> Dict:
        """
        Get current status of pairs trading portfolio
        """
        total_positions = len(self.active_pairs)
        total_exposure = sum(
            abs(pos.quantity_a * pos.current_spread) + abs(pos.quantity_b * pos.current_spread)
            for pos in self.active_pairs.values()
        )
        
        return {
            'active_pairs': total_positions,
            'max_pairs': self.max_positions,
            'total_exposure': total_exposure,
            'pairs_detail': [
                {
                    'pair': f"{pos.symbol_a}-{pos.symbol_b}",
                    'side': pos.position_side,
                    'entry_zscore': pos.entry_zscore,
                    'current_zscore': pos.current_zscore,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'entry_time': pos.entry_time.isoformat()
                }
                for pos in self.active_pairs.values()
            ]
        }


def create_pairs_strategy(config: Dict) -> PairsStrategy:
    """
    Factory function to create a pairs trading strategy with configuration
    """
    return PairsStrategy(
        lookback_period=config.get('lookback_period', 252),
        signal_window=config.get('signal_window', 20),
        entry_zscore=config.get('entry_zscore', 2.0),
        exit_zscore=config.get('exit_zscore', 0.0),
        stop_loss_zscore=config.get('stop_loss_zscore', 3.0),
        min_half_life=config.get('min_half_life', 1.0),
        max_half_life=config.get('max_half_life', 30.0),
        max_positions=config.get('max_positions', 5),
        position_size=config.get('position_size', 10000.0),
        correlation_threshold=config.get('correlation_threshold', 0.7),
        cointegration_pvalue=config.get('cointegration_pvalue', 0.05)
    )