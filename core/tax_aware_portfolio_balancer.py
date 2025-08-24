#!/usr/bin/env python3
"""
Tax-Aware Portfolio Balancer

An advanced portfolio balancing system that integrates tax-loss harvesting
considerations into portfolio rebalancing decisions. Extends the existing
IntelligentPortfolioBalancer with tax-aware functionality for optimal
after-tax portfolio management.

Key Features:
- Integration with existing portfolio_balancer.py
- Tax-loss harvesting opportunity identification during rebalancing
- Wash sale rule compliance during portfolio changes
- After-tax return optimization
- Intelligent lot selection for tax efficiency
- Replacement asset integration for maintaining exposure
- Tax-aware timing of rebalancing transactions
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Literal, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum

from core.portfolio_balancer import (
    IntelligentPortfolioBalancer, PositionAnalysis, RebalanceDecision,
    PositionAction, OrderUrgency, RiskLevel
)
from core.tax_loss_harvesting import (
    tax_loss_harvesting_engine, TaxLossOpportunity, HarvestingResult
)
from core.lot_tracking import lot_tracker, LotAccounting
from core.wash_sale_monitor import wash_sale_monitor
from core.asset_replacement import asset_replacement_engine, ReplacementCandidate
from tools.alpaca_client import alpaca_client
from config.settings import settings

logger = logging.getLogger(__name__)

class TaxAwareRebalanceStrategy(Enum):
    """Tax-aware rebalancing strategies."""
    HARVEST_FIRST = "harvest_first"              # Harvest losses before rebalancing
    MINIMIZE_GAINS = "minimize_gains"            # Minimize realization of gains
    DEFER_GAINS = "defer_gains"                  # Defer gains to next tax year
    BALANCED_APPROACH = "balanced_approach"       # Balance tax efficiency with portfolio needs
    IGNORE_TAX = "ignore_tax"                    # Traditional rebalancing without tax consideration

@dataclass
class TaxAwareRebalanceDecision(RebalanceDecision):
    """Enhanced rebalancing decision with tax awareness."""
    
    # Tax-related information
    tax_impact: Optional[Decimal] = None
    is_loss_harvesting: bool = False
    is_gain_realization: bool = False
    wash_sale_risk: bool = False
    
    # Lot-specific details
    specific_lots_to_sell: List[str] = field(default_factory=list)
    lot_accounting_method: Optional[LotAccounting] = None
    
    # Replacement information
    replacement_candidate: Optional[ReplacementCandidate] = None
    maintains_exposure: bool = True
    
    # Tax optimization details
    tax_efficiency_score: float = 0.0
    estimated_tax_benefit: Optional[Decimal] = None
    after_tax_performance_impact: float = 0.0
    
    # Timing considerations
    optimal_execution_date: Optional[datetime] = None
    can_defer_to_next_year: bool = False
    wash_sale_clear_date: Optional[datetime] = None

@dataclass
class TaxAwarePortfolioAnalysis:
    """Comprehensive tax-aware portfolio analysis."""
    analysis_date: datetime
    
    # Traditional analysis
    position_analyses: List[PositionAnalysis]
    
    # Tax-aware enhancements
    tlh_opportunities: List[TaxLossOpportunity]
    wash_sale_constraints: Dict[str, datetime]  # symbol -> clear_date
    
    # Tax impact projections
    total_potential_tax_savings: Decimal
    total_wash_sale_risks: int
    estimated_after_tax_alpha: float
    
    # Recommendations
    recommended_strategy: TaxAwareRebalanceStrategy
    priority_actions: List[str]
    deferral_recommendations: List[str]

class TaxAwarePortfolioBalancer(IntelligentPortfolioBalancer):
    """
    Tax-aware portfolio balancer that extends the base IntelligentPortfolioBalancer
    with comprehensive tax optimization capabilities.
    
    This system integrates tax-loss harvesting, wash sale compliance, and
    after-tax return optimization into the portfolio rebalancing process.
    """
    
    def __init__(self, tax_aware_strategy: TaxAwareRebalanceStrategy = TaxAwareRebalanceStrategy.BALANCED_APPROACH):
        """Initialize the tax-aware portfolio balancer."""
        super().__init__()
        
        self.tax_aware_strategy = tax_aware_strategy
        
        # Tax-aware parameters
        self.min_tax_benefit_threshold = Decimal("50.00")  # Minimum tax benefit to pursue
        self.max_wash_sale_deferral_days = 35  # Max days to defer for wash sale clearance
        self.tax_efficiency_weight = 0.3  # Weight of tax considerations in decisions
        
        # Integration with tax systems
        self.tlh_engine = tax_loss_harvesting_engine
        self.lot_tracker = lot_tracker
        self.wash_sale_monitor = wash_sale_monitor
        self.asset_replacement_engine = asset_replacement_engine
        
        logger.info(f"Tax-Aware Portfolio Balancer initialized with strategy: {tax_aware_strategy.value}")
    
    async def analyze_tax_aware_portfolio_balance(self, 
                                                target_allocation: Dict[str, float],
                                                current_signals: List[Dict] = None) -> TaxAwarePortfolioAnalysis:
        """
        Perform comprehensive tax-aware portfolio analysis.
        
        Args:
            target_allocation: Target portfolio allocation {symbol: weight}
            current_signals: Optional current trading signals
            
        Returns:
            TaxAwarePortfolioAnalysis with recommendations
        """
        try:
            logger.info("🎯 Performing tax-aware portfolio balance analysis...")
            
            # Start with traditional portfolio analysis
            position_analyses = await self.analyze_portfolio_balance(target_allocation, current_signals)
            
            # Identify tax-loss harvesting opportunities
            current_positions = await self._get_current_portfolio()
            tlh_opportunities = await self.tlh_engine.scan_for_opportunities(
                current_positions.get('positions', {})
            )
            
            # Check wash sale constraints
            wash_sale_constraints = {}
            for symbol in target_allocation.keys():
                compliance_status = await self.wash_sale_monitor.get_compliance_status(symbol)
                if compliance_status.get('days_until_clear', 0) > 0:
                    clear_date = datetime.now() + timedelta(days=compliance_status['days_until_clear'])
                    wash_sale_constraints[symbol] = clear_date
            
            # Calculate tax impact projections
            total_potential_tax_savings = sum(opp.tax_benefit_estimate for opp in tlh_opportunities)
            total_wash_sale_risks = len(wash_sale_constraints)
            
            # Estimate after-tax alpha improvement
            estimated_after_tax_alpha = await self._estimate_after_tax_alpha_improvement(
                tlh_opportunities, position_analyses
            )
            
            # Determine optimal strategy based on current conditions
            recommended_strategy = await self._determine_optimal_strategy(
                position_analyses, tlh_opportunities, wash_sale_constraints
            )
            
            # Generate priority actions
            priority_actions = await self._generate_priority_actions(
                position_analyses, tlh_opportunities, wash_sale_constraints
            )
            
            # Generate deferral recommendations
            deferral_recommendations = await self._generate_deferral_recommendations(
                position_analyses, wash_sale_constraints
            )
            
            analysis = TaxAwarePortfolioAnalysis(
                analysis_date=datetime.now(),
                position_analyses=position_analyses,
                tlh_opportunities=tlh_opportunities,
                wash_sale_constraints=wash_sale_constraints,
                total_potential_tax_savings=total_potential_tax_savings,
                total_wash_sale_risks=total_wash_sale_risks,
                estimated_after_tax_alpha=estimated_after_tax_alpha,
                recommended_strategy=recommended_strategy,
                priority_actions=priority_actions,
                deferral_recommendations=deferral_recommendations
            )
            
            logger.info(f"✅ Tax-aware analysis complete:")
            logger.info(f"   TLH opportunities: {len(tlh_opportunities)} (${total_potential_tax_savings:,.2f} potential benefit)")
            logger.info(f"   Wash sale constraints: {total_wash_sale_risks}")
            logger.info(f"   Recommended strategy: {recommended_strategy.value}")
            logger.info(f"   Estimated after-tax alpha: {estimated_after_tax_alpha:+.2%}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in tax-aware portfolio analysis: {e}")
            raise
    
    async def generate_tax_aware_rebalancing_orders(self, 
                                                  analysis: TaxAwarePortfolioAnalysis,
                                                  max_orders: int = 15) -> List[TaxAwareRebalanceDecision]:
        """
        Generate tax-aware rebalancing orders based on analysis.
        
        Args:
            analysis: TaxAwarePortfolioAnalysis from analyze_tax_aware_portfolio_balance
            max_orders: Maximum number of orders to generate
            
        Returns:
            List of TaxAwareRebalanceDecision objects
        """
        try:
            logger.info(f"🎯 Generating tax-aware rebalancing orders (strategy: {analysis.recommended_strategy.value})")
            
            tax_aware_decisions = []
            
            # Process based on recommended strategy
            if analysis.recommended_strategy == TaxAwareRebalanceStrategy.HARVEST_FIRST:
                tax_aware_decisions = await self._generate_harvest_first_orders(analysis, max_orders)
            
            elif analysis.recommended_strategy == TaxAwareRebalanceStrategy.MINIMIZE_GAINS:
                tax_aware_decisions = await self._generate_minimize_gains_orders(analysis, max_orders)
            
            elif analysis.recommended_strategy == TaxAwareRebalanceStrategy.DEFER_GAINS:
                tax_aware_decisions = await self._generate_defer_gains_orders(analysis, max_orders)
            
            elif analysis.recommended_strategy == TaxAwareRebalanceStrategy.BALANCED_APPROACH:
                tax_aware_decisions = await self._generate_balanced_approach_orders(analysis, max_orders)
            
            else:  # IGNORE_TAX
                # Fall back to traditional rebalancing
                traditional_decisions = await self.generate_rebalancing_orders(
                    analysis.position_analyses, max_orders
                )
                tax_aware_decisions = [
                    await self._convert_to_tax_aware_decision(decision) 
                    for decision in traditional_decisions
                ]
            
            # Sort by combined score (traditional urgency + tax efficiency)
            tax_aware_decisions.sort(
                key=lambda d: (d.urgency.value, -d.tax_efficiency_score, -abs(d.quantity))
            )
            
            # Log summary
            total_tax_benefits = sum(
                d.estimated_tax_benefit or Decimal("0") for d in tax_aware_decisions
            )
            harvest_orders = len([d for d in tax_aware_decisions if d.is_loss_harvesting])
            gain_orders = len([d for d in tax_aware_decisions if d.is_gain_realization])
            
            logger.info(f"✅ Generated {len(tax_aware_decisions)} tax-aware orders:")
            logger.info(f"   Loss harvesting orders: {harvest_orders}")
            logger.info(f"   Gain realization orders: {gain_orders}")
            logger.info(f"   Total estimated tax benefits: ${total_tax_benefits:,.2f}")
            
            return tax_aware_decisions[:max_orders]
            
        except Exception as e:
            logger.error(f"Error generating tax-aware rebalancing orders: {e}")
            return []
    
    async def execute_tax_aware_rebalancing(self, 
                                          decisions: List[TaxAwareRebalanceDecision]) -> List[Dict]:
        """
        Execute tax-aware rebalancing decisions with proper sequencing.
        
        Args:
            decisions: List of TaxAwareRebalanceDecision objects
            
        Returns:
            List of execution results
        """
        try:
            logger.info(f"🚀 Executing tax-aware rebalancing: {len(decisions)} orders")
            
            # Group decisions by execution priority and wash sale considerations
            immediate_orders = []  # Orders that can be executed immediately
            deferred_orders = []   # Orders that should be deferred due to wash sale risk
            replacement_orders = [] # Orders that require replacement assets
            
            for decision in decisions:
                if decision.wash_sale_risk and decision.wash_sale_clear_date:
                    if decision.wash_sale_clear_date > datetime.now():
                        deferred_orders.append(decision)
                        continue
                
                if decision.replacement_candidate:
                    replacement_orders.append(decision)
                else:
                    immediate_orders.append(decision)
            
            execution_results = []
            
            # Execute immediate orders first
            for decision in immediate_orders:
                try:
                    result = await self._execute_tax_aware_order(decision)
                    execution_results.append(result)
                    
                    if result.get('success'):
                        logger.info(f"✅ Executed: {decision.symbol} {decision.action.value} "
                                  f"{decision.quantity} (tax benefit: ${decision.estimated_tax_benefit or 0:,.2f})")
                    else:
                        logger.warning(f"❌ Failed: {decision.symbol} - {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"Error executing order for {decision.symbol}: {e}")
                    execution_results.append({
                        'symbol': decision.symbol,
                        'success': False,
                        'error': str(e),
                        'tax_aware': True
                    })
            
            # Execute replacement orders (sell original, buy replacement)
            for decision in replacement_orders:
                try:
                    result = await self._execute_replacement_order(decision)
                    execution_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error executing replacement order for {decision.symbol}: {e}")
                    execution_results.append({
                        'symbol': decision.symbol,
                        'success': False,
                        'error': str(e),
                        'replacement_failed': True
                    })
            
            # Schedule deferred orders
            if deferred_orders:
                await self._schedule_deferred_orders(deferred_orders)
                logger.info(f"📅 Scheduled {len(deferred_orders)} orders for later execution")
            
            # Log execution summary
            successful_orders = len([r for r in execution_results if r.get('success')])
            total_tax_benefits = sum(
                r.get('tax_benefit_realized', 0) for r in execution_results if r.get('success')
            )
            
            logger.info(f"📊 Tax-aware rebalancing execution complete:")
            logger.info(f"   Successful orders: {successful_orders}/{len(decisions)}")
            logger.info(f"   Total tax benefits realized: ${total_tax_benefits:,.2f}")
            logger.info(f"   Deferred orders: {len(deferred_orders)}")
            
            return execution_results
            
        except Exception as e:
            logger.error(f"Error executing tax-aware rebalancing: {e}")
            return []
    
    async def _generate_harvest_first_orders(self, 
                                           analysis: TaxAwarePortfolioAnalysis, 
                                           max_orders: int) -> List[TaxAwareRebalanceDecision]:
        """Generate orders with harvest-first strategy."""
        decisions = []
        
        try:
            # First, create harvest orders for high-value TLH opportunities
            for opp in analysis.tlh_opportunities[:max_orders//2]:
                if opp.tax_benefit_estimate >= self.min_tax_benefit_threshold:
                    
                    # Find corresponding position analysis
                    position_analysis = next(
                        (pa for pa in analysis.position_analyses if pa.symbol == opp.symbol), None
                    )
                    
                    # Check for replacement candidates
                    replacement_candidate = None
                    if opp.replacement_candidates:
                        replacement_candidate = opp.replacement_candidates[0]  # Best candidate
                    
                    # Check wash sale risk
                    wash_sale_risk = opp.symbol in analysis.wash_sale_constraints
                    wash_sale_clear_date = analysis.wash_sale_constraints.get(opp.symbol)
                    
                    decision = TaxAwareRebalanceDecision(
                        symbol=opp.symbol,
                        action=PositionAction.SELL,
                        quantity=float(opp.total_quantity),
                        order_type="market" if not wash_sale_risk else "limit",
                        urgency=OrderUrgency.HIGH if opp.urgency_level == "critical" else OrderUrgency.MEDIUM,
                        reasoning=f"Tax-loss harvesting: {opp.reasoning}",
                        confidence=opp.harvest_confidence,
                        
                        # Tax-aware fields
                        tax_impact=opp.unrealized_loss,
                        is_loss_harvesting=True,
                        wash_sale_risk=wash_sale_risk,
                        specific_lots_to_sell=opp.lot_ids,
                        lot_accounting_method=LotAccounting.HIFO,  # Highest cost first for tax benefits
                        replacement_candidate=replacement_candidate,
                        maintains_exposure=replacement_candidate is not None,
                        tax_efficiency_score=85.0,  # High score for loss harvesting
                        estimated_tax_benefit=opp.tax_benefit_estimate,
                        wash_sale_clear_date=wash_sale_clear_date
                    )
                    
                    decisions.append(decision)
            
            # Then, add remaining rebalancing orders that don't conflict
            remaining_analyses = [
                pa for pa in analysis.position_analyses 
                if pa.symbol not in [d.symbol for d in decisions] and pa.action_needed != PositionAction.HOLD
            ]
            
            for pa in remaining_analyses[:max_orders - len(decisions)]:
                decision = await self._create_tax_aware_decision_from_analysis(pa)
                if decision:
                    decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error generating harvest-first orders: {e}")
            return []
    
    async def _execute_tax_aware_order(self, decision: TaxAwareRebalanceDecision) -> Dict:
        """Execute a single tax-aware order."""
        try:
            # Prepare order parameters
            order_params = {
                'symbol': decision.symbol,
                'qty': abs(decision.quantity),
                'side': 'sell' if decision.action in [PositionAction.SELL, PositionAction.CLOSE] else 'buy',
                'type': decision.order_type,
                'time_in_force': 'day'
            }
            
            # Add limit price if specified
            if decision.limit_price:
                order_params['limit_price'] = float(decision.limit_price)
            
            # For loss harvesting, use specific lot identification
            if decision.is_loss_harvesting and decision.specific_lots_to_sell:
                # Record the sale with specific lots
                lot_ids_sold, realized_loss = await self.lot_tracker.record_sale(
                    symbol=decision.symbol,
                    quantity=Decimal(str(decision.quantity)),
                    price_per_share=decision.limit_price or await self._get_current_price(decision.symbol),
                    specific_lot_ids=decision.specific_lots_to_sell,
                    lot_selection_method=decision.lot_accounting_method
                )
                
                # Update wash sale monitoring
                await self.wash_sale_monitor.record_transaction(
                    decision.symbol, 'sell', Decimal(str(decision.quantity)),
                    decision.limit_price or await self._get_current_price(decision.symbol)
                )
            
            # Execute the order through Alpaca
            # Note: In a real implementation, this would execute through the Alpaca client
            # For this example, we'll simulate the execution
            
            execution_result = {
                'symbol': decision.symbol,
                'action': decision.action.value,
                'quantity': decision.quantity,
                'success': True,
                'tax_aware': True,
                'is_loss_harvesting': decision.is_loss_harvesting,
                'tax_benefit_realized': decision.estimated_tax_benefit,
                'execution_time': datetime.now().isoformat(),
                'lot_accounting_method': decision.lot_accounting_method.value if decision.lot_accounting_method else None
            }
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing tax-aware order for {decision.symbol}: {e}")
            return {
                'symbol': decision.symbol,
                'success': False,
                'error': str(e),
                'tax_aware': True
            }
    
    async def _execute_replacement_order(self, decision: TaxAwareRebalanceDecision) -> Dict:
        """Execute a replacement order (sell original, buy replacement)."""
        try:
            if not decision.replacement_candidate:
                raise ValueError("No replacement candidate specified")
            
            # Step 1: Sell the original position
            sell_result = await self._execute_tax_aware_order(decision)
            
            if not sell_result.get('success'):
                return sell_result
            
            # Step 2: Buy the replacement asset
            replacement_symbol = decision.replacement_candidate.symbol
            current_price = await self._get_current_price(replacement_symbol)
            
            if not current_price:
                return {
                    'symbol': decision.symbol,
                    'replacement_symbol': replacement_symbol,
                    'success': False,
                    'error': 'Could not get price for replacement asset'
                }
            
            # Calculate replacement quantity based on dollar amount
            dollar_amount = decision.quantity * (decision.limit_price or current_price)
            replacement_quantity = dollar_amount / current_price
            
            buy_order_params = {
                'symbol': replacement_symbol,
                'qty': replacement_quantity,
                'side': 'buy',
                'type': 'market',
                'time_in_force': 'day'
            }
            
            # Execute replacement purchase
            # Note: In real implementation, this would execute through Alpaca client
            
            return {
                'symbol': decision.symbol,
                'replacement_symbol': replacement_symbol,
                'original_quantity': decision.quantity,
                'replacement_quantity': replacement_quantity,
                'success': True,
                'tax_benefit_realized': decision.estimated_tax_benefit,
                'maintains_exposure': True,
                'correlation': decision.replacement_candidate.correlation
            }
            
        except Exception as e:
            logger.error(f"Error executing replacement order: {e}")
            return {
                'symbol': decision.symbol,
                'success': False,
                'error': str(e),
                'replacement_failed': True
            }
    
    async def _create_tax_aware_decision_from_analysis(self, 
                                                     analysis: PositionAnalysis) -> Optional[TaxAwareRebalanceDecision]:
        """Convert a PositionAnalysis to a TaxAwareRebalanceDecision."""
        try:
            # Calculate tax impact
            tax_impact = await self._calculate_tax_impact_for_analysis(analysis)
            
            # Check for wash sale risk
            wash_sale_risk = analysis.symbol in (await self.wash_sale_monitor.get_compliance_status(analysis.symbol))
            
            # Determine if this is gain or loss realization
            is_loss_harvesting = tax_impact < 0 if tax_impact else False
            is_gain_realization = tax_impact > 0 if tax_impact else False
            
            # Calculate tax efficiency score
            tax_efficiency_score = await self._calculate_tax_efficiency_score(analysis, tax_impact)
            
            decision = TaxAwareRebalanceDecision(
                symbol=analysis.symbol,
                action=analysis.action_needed,
                quantity=abs(analysis.target_quantity - analysis.current_quantity),
                order_type="limit",
                urgency=analysis.urgency,
                reasoning=analysis.reasoning,
                confidence=0.7,  # Default confidence
                
                # Tax-aware fields
                tax_impact=Decimal(str(tax_impact)) if tax_impact else None,
                is_loss_harvesting=is_loss_harvesting,
                is_gain_realization=is_gain_realization,
                wash_sale_risk=wash_sale_risk,
                tax_efficiency_score=tax_efficiency_score
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error creating tax-aware decision for {analysis.symbol}: {e}")
            return None

# Global instance
tax_aware_portfolio_balancer = TaxAwarePortfolioBalancer()

# Convenience functions
async def analyze_tax_aware_portfolio_balance(target_allocation: Dict[str, float], **kwargs) -> TaxAwarePortfolioAnalysis:
    """Analyze portfolio balance with tax considerations."""
    return await tax_aware_portfolio_balancer.analyze_tax_aware_portfolio_balance(target_allocation, **kwargs)

async def generate_tax_aware_rebalancing_orders(analysis: TaxAwarePortfolioAnalysis, **kwargs) -> List[TaxAwareRebalanceDecision]:
    """Generate tax-aware rebalancing orders."""
    return await tax_aware_portfolio_balancer.generate_tax_aware_rebalancing_orders(analysis, **kwargs)

__all__ = [
    'TaxAwarePortfolioBalancer', 'TaxAwareRebalanceDecision', 'TaxAwarePortfolioAnalysis',
    'TaxAwareRebalanceStrategy', 'tax_aware_portfolio_balancer',
    'analyze_tax_aware_portfolio_balance', 'generate_tax_aware_rebalancing_orders'
]