#!/usr/bin/env python3
"""
Market Regime Detector

Detects market volatility and regime changes to inform retraining strategies.
Used to determine optimal lookback periods for incremental learning.

Key Features:
- Volatility detection (VIX, realized volatility)
- Regime classification (low/normal/high volatility)
- Adaptive lookback recommendations
- Historical volatility tracking
"""

import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Add trading system to path
sys.path.insert(0, '/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system')
from tools.alpaca_client import alpaca_client

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Market regime classification."""
    regime: str  # 'low', 'normal', 'high'
    volatility: float  # Current volatility level
    recommended_lookback: int  # Days for retraining
    vix_level: Optional[float] = None  # VIX if available
    confidence: float = 1.0  # Confidence in regime detection


class MarketRegimeDetector:
    """
    Detect market volatility regimes and recommend retraining parameters.

    Usage:
        detector = MarketRegimeDetector(symbols=['SPY', 'AAPL', 'MSFT'])
        regime = detector.detect_current_regime()

        if regime.regime == 'high':
            lookback_days = regime.recommended_lookback  # 30 days
        else:
            lookback_days = 60  # Normal conditions
    """

    # Volatility thresholds (annualized)
    LOW_VOL_THRESHOLD = 0.10  # 10% annualized volatility
    HIGH_VOL_THRESHOLD = 0.25  # 25% annualized volatility

    # VIX thresholds (if available)
    VIX_LOW = 15
    VIX_HIGH = 30

    # Lookback recommendations by regime
    LOOKBACK_MAP = {
        'low': 90,      # Stable market - longer lookback
        'normal': 60,   # Normal conditions - standard lookback
        'high': 30      # High volatility - shorter lookback for quick adaptation
    }

    def __init__(self, symbols: Optional[list] = None, lookback_days: int = 30):
        """
        Initialize market regime detector.

        Args:
            symbols: List of symbols to analyze (default: SPY as market proxy)
            lookback_days: Days of historical data for volatility calculation
        """
        self.symbols = symbols or ['SPY']  # Use SPY as market proxy
        self.lookback_days = lookback_days

        logger.info(f"📊 Initialized MarketRegimeDetector")
        logger.info(f"   Symbols: {self.symbols}")
        logger.info(f"   Lookback: {lookback_days} days")

    def detect_current_regime(self) -> MarketRegime:
        """
        Detect current market regime based on recent volatility.

        Returns:
            MarketRegime object with classification and recommendations
        """
        logger.info("🔍 Detecting current market regime...")

        try:
            # Calculate realized volatility
            volatility = self._calculate_realized_volatility()

            # Try to get VIX if available
            vix_level = self._get_vix_level()

            # Classify regime
            regime = self._classify_regime(volatility, vix_level)

            logger.info(f"✅ Market Regime: {regime.regime.upper()}")
            logger.info(f"   Volatility: {volatility:.2%}")
            if vix_level:
                logger.info(f"   VIX: {vix_level:.1f}")
            logger.info(f"   Recommended Lookback: {regime.recommended_lookback} days")

            return regime

        except Exception as e:
            logger.error(f"❌ Regime detection failed: {e}")
            # Return default normal regime
            return MarketRegime(
                regime='normal',
                volatility=0.20,
                recommended_lookback=60,
                confidence=0.5
            )

    def _calculate_realized_volatility(self) -> float:
        """
        Calculate realized volatility from recent market data.

        Returns:
            Annualized volatility as a float
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days + 10)  # Extra days for weekends

        volatilities = []

        for symbol in self.symbols:
            try:
                # Fetch historical data
                bars = alpaca_client.api.get_bars(
                    symbol,
                    '1Day',
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    limit=self.lookback_days
                ).df

                if bars.empty or len(bars) < 10:
                    logger.warning(f"⚠️ Insufficient data for {symbol}")
                    continue

                # Calculate daily returns
                bars['returns'] = bars['close'].pct_change()

                # Calculate realized volatility (annualized)
                daily_vol = bars['returns'].std()
                annualized_vol = daily_vol * np.sqrt(252)

                volatilities.append(annualized_vol)

            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch data for {symbol}: {e}")
                continue

        if not volatilities:
            logger.warning("⚠️ No volatility data available, using default")
            return 0.20  # Default 20% volatility

        # Use mean volatility across symbols
        avg_volatility = np.mean(volatilities)

        logger.debug(f"Calculated volatility: {avg_volatility:.2%}")

        return avg_volatility

    def _get_vix_level(self) -> Optional[float]:
        """
        Get current VIX level if available.

        Returns:
            VIX level as float, or None if unavailable
        """
        try:
            # Alpaca doesn't provide VIX directly
            # Could integrate with another data source or use SPY options implied volatility
            # For now, return None and rely on realized volatility
            return None

        except Exception as e:
            logger.debug(f"VIX not available: {e}")
            return None

    def _classify_regime(self, volatility: float, vix_level: Optional[float]) -> MarketRegime:
        """
        Classify market regime based on volatility metrics.

        Args:
            volatility: Realized volatility
            vix_level: VIX level (optional)

        Returns:
            MarketRegime object
        """
        # Primary classification based on realized volatility
        if volatility < self.LOW_VOL_THRESHOLD:
            regime_type = 'low'
        elif volatility > self.HIGH_VOL_THRESHOLD:
            regime_type = 'high'
        else:
            regime_type = 'normal'

        # Adjust based on VIX if available
        confidence = 1.0
        if vix_level:
            if vix_level < self.VIX_LOW and regime_type == 'low':
                confidence = 1.0  # Strong confirmation
            elif vix_level > self.VIX_HIGH and regime_type == 'high':
                confidence = 1.0  # Strong confirmation
            elif (vix_level < self.VIX_LOW and regime_type != 'low') or \
                 (vix_level > self.VIX_HIGH and regime_type != 'high'):
                confidence = 0.7  # Conflicting signals

        # Get recommended lookback
        recommended_lookback = self.LOOKBACK_MAP[regime_type]

        return MarketRegime(
            regime=regime_type,
            volatility=volatility,
            recommended_lookback=recommended_lookback,
            vix_level=vix_level,
            confidence=confidence
        )

    def get_adaptive_lookback(self) -> int:
        """
        Get adaptive lookback period based on current regime.

        Returns:
            Recommended lookback days
        """
        regime = self.detect_current_regime()
        return regime.recommended_lookback

    def should_retrain_urgently(self, threshold_volatility: float = 0.30) -> bool:
        """
        Determine if urgent retraining is needed due to extreme volatility.

        Args:
            threshold_volatility: Threshold for urgent retraining (default 30%)

        Returns:
            True if urgent retraining is recommended
        """
        volatility = self._calculate_realized_volatility()

        urgent = volatility > threshold_volatility

        if urgent:
            logger.warning(f"⚠️ URGENT RETRAINING RECOMMENDED!")
            logger.warning(f"   Current volatility: {volatility:.2%}")
            logger.warning(f"   Threshold: {threshold_volatility:.2%}")

        return urgent


def detect_regime() -> MarketRegime:
    """
    Convenience function to detect current market regime.

    Returns:
        MarketRegime object
    """
    detector = MarketRegimeDetector(symbols=['SPY', 'QQQ'])  # Use market proxies
    return detector.detect_current_regime()


if __name__ == "__main__":
    # Test the detector
    logging.basicConfig(level=logging.INFO)

    detector = MarketRegimeDetector(symbols=['SPY', 'QQQ', 'IWM'])
    regime = detector.detect_current_regime()

    print(f"\n{'='*60}")
    print(f"Current Market Regime: {regime.regime.upper()}")
    print(f"{'='*60}")
    print(f"Volatility: {regime.volatility:.2%}")
    print(f"Recommended Lookback: {regime.recommended_lookback} days")
    print(f"Confidence: {regime.confidence:.1%}")
    print(f"{'='*60}\n")

    # Check if urgent retraining needed
    if detector.should_retrain_urgently():
        print("🚨 URGENT RETRAINING RECOMMENDED 🚨")
    else:
        print("✅ Normal retraining schedule OK")
