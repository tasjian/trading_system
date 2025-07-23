"""
High Frequency Trading (HFT) Module

Specialized trading system for accounts with $25,000+ balance that can perform
unlimited day trading without Pattern Day Trader restrictions.
"""

from .hft_engine import hft_engine, HighFrequencyTradingEngine
from .hft_compliance import hft_compliance_manager, HFTComplianceManager

__all__ = [
    'hft_engine',
    'HighFrequencyTradingEngine', 
    'hft_compliance_manager',
    'HFTComplianceManager'
]

__version__ = "1.0.0"
__author__ = "Trading System Development Team"