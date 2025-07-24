#!/usr/bin/env python3
"""
Alpaca Fee Calculator

Calculates trading fees based on Alpaca's regulatory fee structure including
FINRA TAF, FINRA CAT, and SEC fees.
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FeeBreakdown:
    """Breakdown of trading fees."""
    commission_fees: float = 0.0
    finra_taf_fees: float = 0.0
    finra_cat_fees: float = 0.0
    sec_fees: float = 0.0
    total_fees: float = 0.0
    transactions_count: int = 0
    sell_transactions: int = 0
    sell_shares: int = 0

class AlpacaFeeCalculator:
    """Calculate Alpaca trading fees based on regulatory fee structure."""
    
    # Alpaca fee rates (2024)
    COMMISSION_RATE = 0.00  # Commission-free
    FINRA_TAF_RATE = 0.000166  # Per share on sells only
    FINRA_TAF_MAX = 8.30  # Maximum TAF fee per order
    FINRA_CAT_RATE = 0.000046  # Per transaction on buys and sells
    SEC_RATE_PER_MILLION = 0.00  # SEC fee per $1M of principal (sells only)
    
    def __init__(self):
        """Initialize fee calculator."""
        pass
    
    def calculate_fees_for_positions(self, positions: Dict[str, Dict]) -> FeeBreakdown:
        """
        Calculate estimated fees if all positions were traded.
        
        Args:
            positions: Dictionary of positions with symbol as key and position data as value
                      Expected format: {'symbol': {'shares': int, 'pnl': float, 'side': str}}
        
        Returns:
            FeeBreakdown: Complete breakdown of estimated fees
        """
        try:
            fee_breakdown = FeeBreakdown()
            
            # Count transactions and categorize
            for symbol, position_data in positions.items():
                shares = abs(position_data.get('shares', 0))
                side = position_data.get('side', 'unknown')
                
                if shares == 0:
                    continue
                
                fee_breakdown.transactions_count += 1
                
                # Calculate FINRA TAF (sells only)
                if side == 'short' or shares < 0:  # Sell transaction
                    fee_breakdown.sell_transactions += 1
                    fee_breakdown.sell_shares += shares
                    
                    taf_fee = min(shares * self.FINRA_TAF_RATE, self.FINRA_TAF_MAX)
                    fee_breakdown.finra_taf_fees += taf_fee
                
                # Calculate FINRA CAT (all transactions)
                fee_breakdown.finra_cat_fees += self.FINRA_CAT_RATE
            
            # Round up CAT fees to nearest penny (regulatory requirement)
            fee_breakdown.finra_cat_fees = max(0.01, fee_breakdown.finra_cat_fees)
            
            # Calculate SEC fees (sells only, minimal at these volumes)
            if fee_breakdown.sell_transactions > 0:
                # Estimate total sell value (simplified calculation)
                estimated_sell_value = fee_breakdown.sell_shares * 100  # Rough estimate
                sec_fee_raw = (estimated_sell_value / 1000000) * self.SEC_RATE_PER_MILLION
                fee_breakdown.sec_fees = max(0.01, sec_fee_raw)  # Minimum penny
            
            # Commission fees (always $0 for Alpaca)
            fee_breakdown.commission_fees = 0.0
            
            # Calculate total
            fee_breakdown.total_fees = (
                fee_breakdown.commission_fees +
                fee_breakdown.finra_taf_fees +
                fee_breakdown.finra_cat_fees +
                fee_breakdown.sec_fees
            )
            
            logger.debug(f"Calculated fees for {fee_breakdown.transactions_count} positions: ${fee_breakdown.total_fees:.4f}")
            
            return fee_breakdown
            
        except Exception as e:
            logger.error(f"Fee calculation error: {e}")
            return FeeBreakdown()
    
    def calculate_fees_for_trades(self, trades: List[Dict]) -> FeeBreakdown:
        """
        Calculate actual fees for executed trades.
        
        Args:
            trades: List of trade dictionaries with 'symbol', 'quantity', 'side', 'value'
        
        Returns:
            FeeBreakdown: Complete breakdown of actual fees
        """
        try:
            fee_breakdown = FeeBreakdown()
            
            for trade in trades:
                shares = abs(trade.get('quantity', 0))
                side = trade.get('side', '').lower()
                value = abs(trade.get('value', 0))
                
                if shares == 0:
                    continue
                
                fee_breakdown.transactions_count += 1
                
                # Calculate FINRA TAF (sells only)
                if side in ['sell', 'sell_short']:
                    fee_breakdown.sell_transactions += 1
                    fee_breakdown.sell_shares += shares
                    
                    taf_fee = min(shares * self.FINRA_TAF_RATE, self.FINRA_TAF_MAX)
                    fee_breakdown.finra_taf_fees += taf_fee
                
                # Calculate FINRA CAT (all transactions)
                fee_breakdown.finra_cat_fees += self.FINRA_CAT_RATE
            
            # Round up CAT fees to nearest penny
            fee_breakdown.finra_cat_fees = max(0.01, fee_breakdown.finra_cat_fees)
            
            # Calculate SEC fees for actual sell values
            total_sell_value = sum(abs(t.get('value', 0)) for t in trades if t.get('side', '').lower() in ['sell', 'sell_short'])
            if total_sell_value > 0:
                sec_fee_raw = (total_sell_value / 1000000) * self.SEC_RATE_PER_MILLION
                fee_breakdown.sec_fees = max(0.01, sec_fee_raw)
            
            # Commission fees (always $0 for Alpaca)
            fee_breakdown.commission_fees = 0.0
            
            # Calculate total
            fee_breakdown.total_fees = (
                fee_breakdown.commission_fees +
                fee_breakdown.finra_taf_fees +
                fee_breakdown.finra_cat_fees +
                fee_breakdown.sec_fees
            )
            
            return fee_breakdown
            
        except Exception as e:
            logger.error(f"Trade fee calculation error: {e}")
            return FeeBreakdown()
    
    def get_fee_summary_text(self, fee_breakdown: FeeBreakdown) -> str:
        """Generate human-readable fee summary."""
        if fee_breakdown.total_fees == 0:
            return "No trading fees (paper trading or no transactions)"
        
        summary = f"""
Trading Fee Summary:
• Commission: ${fee_breakdown.commission_fees:.2f} (Commission-free)
• FINRA TAF: ${fee_breakdown.finra_taf_fees:.4f} ({fee_breakdown.sell_shares} shares sold)
• FINRA CAT: ${fee_breakdown.finra_cat_fees:.2f} ({fee_breakdown.transactions_count} transactions)
• SEC Fees: ${fee_breakdown.sec_fees:.2f} (Regulatory minimum)
• Total Fees: ${fee_breakdown.total_fees:.4f}

Transactions: {fee_breakdown.transactions_count} total, {fee_breakdown.sell_transactions} sells
Fee Efficiency: Extremely low cost (regulatory fees only)
        """.strip()
        
        return summary
    
    def get_fee_efficiency_metrics(self, fee_breakdown: FeeBreakdown, total_pnl: float, portfolio_value: float) -> Dict[str, float]:
        """Calculate fee efficiency metrics."""
        try:
            metrics = {
                'total_fees': fee_breakdown.total_fees,
                'fee_to_pnl_ratio': (fee_breakdown.total_fees / abs(total_pnl)) * 100 if total_pnl != 0 else 0,
                'fee_to_portfolio_ratio': (fee_breakdown.total_fees / portfolio_value) * 100 if portfolio_value > 0 else 0,
                'cost_per_transaction': fee_breakdown.total_fees / fee_breakdown.transactions_count if fee_breakdown.transactions_count > 0 else 0,
                'cost_per_1000_traded': fee_breakdown.total_fees * (1000 / portfolio_value) if portfolio_value > 0 else 0
            }
            return metrics
        except Exception as e:
            logger.error(f"Fee efficiency calculation error: {e}")
            return {}

# Global fee calculator instance
fee_calculator = AlpacaFeeCalculator()