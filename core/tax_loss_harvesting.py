#!/usr/bin/env python3
"""
Tax-Loss Harvesting Engine

A comprehensive tax-loss harvesting system that identifies, evaluates, and executes
tax-loss harvesting opportunities while maintaining wash sale compliance and 
optimizing after-tax returns for the algorithmic trading platform.

Key Features:
- Real-time loss identification and opportunity assessment
- Wash sale rule compliance with 30-day cooling periods
- Asset replacement strategies to maintain portfolio exposure
- After-tax performance optimization
- Integration with specific lot tracking for precise cost basis management
- Comprehensive tax reporting and audit trail
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Literal, Union
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import json
import pandas as pd
import numpy as np

from tools.alpaca_client import alpaca_client
from config.settings import settings
from core.lot_tracking import lot_tracker, TaxLot, LotAccounting
from core.wash_sale_monitor import wash_sale_monitor, WashSaleViolation
from core.asset_replacement import asset_replacement_engine, ReplacementCandidate

logger = logging.getLogger(__name__)

class HarvestingStrategy(Enum):
    """Tax-loss harvesting strategies."""
    AGGRESSIVE = "aggressive"           # Harvest all available losses immediately
    MODERATE = "moderate"              # Harvest significant losses with timing consideration
    CONSERVATIVE = "conservative"       # Harvest only high-conviction losses
    REBALANCING_ONLY = "rebalancing_only"  # Only harvest during portfolio rebalancing

class TaxSituation(Enum):
    """Tax situation classifications."""
    HIGH_INCOME = "high_income"        # >$400K income, maximizing loss benefits
    MEDIUM_INCOME = "medium_income"    # $100K-$400K income, balanced approach
    LOW_INCOME = "low_income"          # <$100K income, focus on long-term gains
    RETIRED = "retired"                # Retiree tax planning focus

@dataclass
class TaxLossOpportunity:
    """Represents a tax-loss harvesting opportunity."""
    symbol: str
    lot_ids: List[str]
    total_quantity: Decimal
    unrealized_loss: Decimal
    loss_per_share: Decimal
    tax_benefit_estimate: Decimal
    harvest_confidence: float
    replacement_candidates: List[ReplacementCandidate]
    wash_sale_risk: bool
    days_until_wash_sale_clear: int
    reasoning: str
    
    # Cost basis and tax tracking
    purchase_dates: List[datetime]
    original_cost_basis: Decimal
    current_market_value: Decimal
    holding_period: int  # days
    is_long_term: bool
    
    # Strategy metrics
    opportunity_score: float
    urgency_level: Literal["low", "medium", "high", "critical"]
    recommended_action: Literal["harvest_now", "wait", "replace_first", "skip"]
    optimal_timing: Optional[datetime]
    
    # Tax optimization
    estimated_tax_savings: Decimal
    impact_on_portfolio_risk: float
    replacement_correlation_score: float

@dataclass
class HarvestingResult:
    """Result of a tax-loss harvesting operation."""
    timestamp: datetime
    symbol: str
    lots_harvested: List[str]
    quantity_harvested: Decimal
    realized_loss: Decimal
    tax_benefit_actual: Decimal
    replacement_symbol: Optional[str]
    replacement_quantity: Optional[Decimal]
    wash_sale_violations: List[WashSaleViolation]
    success: bool
    error_message: Optional[str] = None

@dataclass
class TaxHarvestingConfig:
    """Configuration for tax-loss harvesting behavior."""
    strategy: HarvestingStrategy = HarvestingStrategy.MODERATE
    tax_situation: TaxSituation = TaxSituation.MEDIUM_INCOME
    
    # Loss thresholds
    min_loss_threshold: Decimal = Decimal("100.00")  # Minimum $100 loss to harvest
    min_loss_percentage: Decimal = Decimal("0.05")   # Minimum 5% loss
    significant_loss_threshold: Decimal = Decimal("1000.00")  # $1000+ is significant
    
    # Tax rate assumptions for benefit calculation
    ordinary_income_tax_rate: Decimal = Decimal("0.32")  # 32% marginal rate
    capital_gains_tax_rate: Decimal = Decimal("0.20")    # 20% long-term cap gains
    net_investment_income_tax: Decimal = Decimal("0.038") # 3.8% NIIT
    
    # Harvesting behavior
    max_daily_harvest_amount: Decimal = Decimal("10000.00")  # Max $10K losses per day
    max_positions_to_harvest: int = 10
    require_replacement: bool = True
    allow_direct_repurchase: bool = False  # Strict wash sale prevention
    
    # Timing preferences
    prefer_long_term_harvesting: bool = True
    harvest_near_year_end: bool = True
    avoid_december_harvesting: bool = False  # Some prefer to avoid December
    
    # Risk management
    max_portfolio_exposure_change: Decimal = Decimal("0.05")  # 5% max exposure change
    min_replacement_correlation: Decimal = Decimal("0.70")    # 70% min correlation
    max_tracking_error_tolerance: Decimal = Decimal("0.02")   # 2% tracking error

class TaxLossHarvestingEngine:
    """
    Comprehensive Tax-Loss Harvesting Engine.
    
    This engine continuously monitors the portfolio for tax-loss harvesting opportunities,
    evaluates them against wash sale rules and portfolio risk, and executes harvesting
    strategies that maximize after-tax returns while maintaining desired market exposure.
    """
    
    def __init__(self, config: Optional[TaxHarvestingConfig] = None):
        """Initialize the tax-loss harvesting engine."""
        self.config = config or TaxHarvestingConfig()
        self.lot_tracker = lot_tracker
        self.wash_sale_monitor = wash_sale_monitor
        self.asset_replacement_engine = asset_replacement_engine
        
        # TLH enabled/disabled state from settings
        self.tlh_enabled = settings.tlh_enabled
        
        # Performance tracking
        self.total_losses_harvested = Decimal("0.00")
        self.total_tax_benefits_realized = Decimal("0.00")
        self.harvesting_history: List[HarvestingResult] = []
        
        if not self.tlh_enabled:
            logger.info("🚫 Tax-Loss Harvesting DISABLED via settings.tlh_enabled=False")
        else:
            logger.info(f"✅ Tax-Loss Harvesting ENABLED with strategy: {settings.tlh_strategy}")
        
        # Daily tracking
        self.daily_harvest_amount = Decimal("0.00")
        self.daily_harvest_count = 0
        self.last_harvest_date: Optional[datetime] = None
        
        logger.info("Tax-Loss Harvesting Engine initialized")
        logger.info(f"Strategy: {self.config.strategy.value}")
        logger.info(f"Tax Situation: {self.config.tax_situation.value}")
        logger.info(f"Min Loss Threshold: ${self.config.min_loss_threshold}")
    
    async def scan_for_opportunities(self, 
                                   portfolio_positions: Optional[Dict[str, Dict]] = None) -> List[TaxLossOpportunity]:
        """
        Scan current portfolio positions for tax-loss harvesting opportunities.
        
        Args:
            portfolio_positions: Optional dict of positions, if None will fetch current positions
            
        Returns:
            List of TaxLossOpportunity objects sorted by opportunity score
        """
        try:
            # Check if TLH is enabled
            if not self.tlh_enabled:
                logger.debug("🚫 Tax-Loss Harvesting is disabled, returning empty opportunities list")
                return []
            
            logger.info("🔍 Scanning portfolio for tax-loss harvesting opportunities...")
            
            # Get current positions if not provided
            if portfolio_positions is None:
                portfolio_positions = await self._get_current_positions()
            
            if not portfolio_positions:
                logger.warning("No portfolio positions found for TLH scanning")
                return []
            
            opportunities = []
            
            # Analyze each position for loss harvesting potential
            for symbol, position_data in portfolio_positions.items():
                try:
                    position_opportunities = await self._analyze_position_for_tlh(symbol, position_data)
                    opportunities.extend(position_opportunities)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol} for TLH: {e}")
                    continue
            
            # Sort opportunities by opportunity score (highest first)
            opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)
            
            # Log summary
            total_potential_losses = sum(opp.unrealized_loss for opp in opportunities)
            total_potential_benefits = sum(opp.tax_benefit_estimate for opp in opportunities)
            
            logger.info(f"✅ TLH scan complete: {len(opportunities)} opportunities found")
            logger.info(f"💰 Total potential losses to harvest: ${total_potential_losses:,.2f}")
            logger.info(f"🏛️ Total estimated tax benefits: ${total_potential_benefits:,.2f}")
            
            # Log top opportunities
            for i, opp in enumerate(opportunities[:5], 1):
                logger.info(f"   {i}. {opp.symbol}: ${opp.unrealized_loss:,.2f} loss "
                          f"(${opp.tax_benefit_estimate:,.2f} benefit, score: {opp.opportunity_score:.2f})")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error scanning for TLH opportunities: {e}")
            return []
    
    async def _filter_opportunities_by_strategy(self, opportunities: List[TaxLossOpportunity]) -> List[TaxLossOpportunity]:
        """Filter opportunities based on the configured harvesting strategy."""
        if not opportunities:
            return []
        
        # For now, return all opportunities as-is
        # In a full implementation, this would filter based on self.config.strategy
        return opportunities
    
    async def execute_harvesting_strategy(self, opportunities: List[TaxLossOpportunity]) -> List[HarvestingResult]:
        """
        Execute tax-loss harvesting based on identified opportunities and strategy.
        
        Args:
            opportunities: List of TaxLossOpportunity objects to evaluate
            
        Returns:
            List of HarvestingResult objects
        """
        try:
            # Check if TLH is enabled
            if not self.tlh_enabled:
                logger.debug("🚫 Tax-Loss Harvesting is disabled, returning empty results list")
                return []
            
            logger.info(f"🎯 Executing TLH strategy: {self.config.strategy.value}")
            
            # Filter and prioritize opportunities based on strategy
            filtered_opportunities = await self._filter_opportunities_by_strategy(opportunities)
            
            if not filtered_opportunities:
                logger.info("No opportunities meet current strategy criteria")
                return []
            
            # Check daily limits
            await self._reset_daily_counters_if_needed()
            
            results = []
            
            # Execute harvesting for each selected opportunity
            for opportunity in filtered_opportunities:
                if await self._check_daily_limits():
                    break
                
                try:
                    result = await self._execute_single_harvest(opportunity)
                    results.append(result)
                    
                    if result.success:
                        self._update_daily_counters(result)
                        logger.info(f"✅ Harvested ${result.realized_loss:,.2f} loss from {result.symbol}")
                    else:
                        logger.warning(f"❌ Failed to harvest {opportunity.symbol}: {result.error_message}")
                        
                except Exception as e:
                    logger.error(f"Error executing harvest for {opportunity.symbol}: {e}")
                    results.append(HarvestingResult(
                        timestamp=datetime.now(),
                        symbol=opportunity.symbol,
                        lots_harvested=[],
                        quantity_harvested=Decimal("0"),
                        realized_loss=Decimal("0"),
                        tax_benefit_actual=Decimal("0"),
                        replacement_symbol=None,
                        replacement_quantity=None,
                        wash_sale_violations=[],
                        success=False,
                        error_message=str(e)
                    ))
            
            # Log execution summary
            successful_harvests = [r for r in results if r.success]
            total_losses_realized = sum(r.realized_loss for r in successful_harvests)
            total_tax_benefits = sum(r.tax_benefit_actual for r in successful_harvests)
            
            logger.info(f"📊 TLH execution complete:")
            logger.info(f"   Successful harvests: {len(successful_harvests)}/{len(results)}")
            logger.info(f"   Total losses realized: ${total_losses_realized:,.2f}")
            logger.info(f"   Total tax benefits: ${total_tax_benefits:,.2f}")
            
            # Update performance tracking
            self.total_losses_harvested += total_losses_realized
            self.total_tax_benefits_realized += total_tax_benefits
            self.harvesting_history.extend(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing harvesting strategy: {e}")
            return []
    
    async def optimize_tax_aware_rebalancing(self, 
                                           target_allocation: Dict[str, Decimal],
                                           current_positions: Dict[str, Dict]) -> Dict[str, List[str]]:
        """
        Optimize portfolio rebalancing to incorporate tax-loss harvesting.
        
        Args:
            target_allocation: Target portfolio allocation {symbol: weight}
            current_positions: Current portfolio positions
            
        Returns:
            Dict with rebalancing recommendations that incorporate TLH
        """
        try:
            logger.info("🎯 Optimizing tax-aware portfolio rebalancing...")
            
            # Identify positions that need to be reduced/closed based on target allocation
            positions_to_reduce = await self._identify_positions_to_reduce(target_allocation, current_positions)
            
            # Scan these positions for loss harvesting opportunities
            opportunities = []
            for symbol in positions_to_reduce:
                if symbol in current_positions:
                    position_opportunities = await self._analyze_position_for_tlh(
                        symbol, current_positions[symbol]
                    )
                    opportunities.extend(position_opportunities)
            
            # Prioritize harvesting losses from positions that need to be reduced anyway
            tax_aware_recommendations = {
                "harvest_first": [],  # Positions to harvest losses before reducing
                "reduce_normally": [],  # Positions to reduce without harvesting (no losses)
                "replace_and_harvest": [],  # Positions to replace and then harvest
                "defer_rebalancing": []  # Positions to defer rebalancing due to wash sale risk
            }
            
            for symbol in positions_to_reduce:
                position_opportunities = [opp for opp in opportunities if opp.symbol == symbol]
                
                if position_opportunities:
                    best_opportunity = max(position_opportunities, key=lambda x: x.opportunity_score)
                    
                    if best_opportunity.wash_sale_risk:
                        tax_aware_recommendations["defer_rebalancing"].append(symbol)
                        logger.info(f"💤 Deferring {symbol} rebalancing due to wash sale risk")
                    elif best_opportunity.unrealized_loss >= self.config.min_loss_threshold:
                        tax_aware_recommendations["harvest_first"].append(symbol)
                        logger.info(f"🎯 Prioritizing {symbol} for loss harvesting (${best_opportunity.unrealized_loss:,.2f})")
                    else:
                        tax_aware_recommendations["reduce_normally"].append(symbol)
                else:
                    tax_aware_recommendations["reduce_normally"].append(symbol)
            
            logger.info("✅ Tax-aware rebalancing optimization complete")
            return tax_aware_recommendations
            
        except Exception as e:
            logger.error(f"Error optimizing tax-aware rebalancing: {e}")
            return {"harvest_first": [], "reduce_normally": [], "replace_and_harvest": [], "defer_rebalancing": []}
    
    async def generate_tax_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Union[str, Decimal, List]]:
        """
        Generate comprehensive tax report for the specified period.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Comprehensive tax report dictionary
        """
        try:
            logger.info(f"📊 Generating tax report from {start_date.date()} to {end_date.date()}")
            
            # Filter harvesting history for the period
            period_harvests = [
                h for h in self.harvesting_history 
                if start_date <= h.timestamp <= end_date and h.success
            ]
            
            if not period_harvests:
                logger.info("No tax-loss harvesting activity in the specified period")
                return self._empty_tax_report()
            
            # Calculate summary statistics
            total_losses_harvested = sum(h.realized_loss for h in period_harvests)
            total_tax_benefits = sum(h.tax_benefit_actual for h in period_harvests)
            unique_symbols_harvested = len(set(h.symbol for h in period_harvests))
            
            # Calculate wash sale violations
            wash_sale_violations = []
            for harvest in period_harvests:
                wash_sale_violations.extend(harvest.wash_sale_violations)
            
            # Group by symbol for detailed analysis
            by_symbol = {}
            for harvest in period_harvests:
                if harvest.symbol not in by_symbol:
                    by_symbol[harvest.symbol] = []
                by_symbol[harvest.symbol].append(harvest)
            
            symbol_summary = {}
            for symbol, harvests in by_symbol.items():
                symbol_summary[symbol] = {
                    "harvest_count": len(harvests),
                    "total_loss": sum(h.realized_loss for h in harvests),
                    "total_tax_benefit": sum(h.tax_benefit_actual for h in harvests),
                    "lots_harvested": sum(len(h.lots_harvested) for h in harvests),
                    "replacement_used": any(h.replacement_symbol for h in harvests)
                }
            
            # Calculate after-tax performance impact
            after_tax_alpha = await self._calculate_after_tax_alpha(period_harvests)
            
            tax_report = {
                "report_period": f"{start_date.date()} to {end_date.date()}",
                "summary": {
                    "total_losses_harvested": total_losses_harvested,
                    "total_tax_benefits": total_tax_benefits,
                    "harvest_transactions": len(period_harvests),
                    "unique_symbols": unique_symbols_harvested,
                    "wash_sale_violations": len(wash_sale_violations),
                    "after_tax_alpha": after_tax_alpha
                },
                "by_symbol": symbol_summary,
                "wash_sale_details": [asdict(v) for v in wash_sale_violations],
                "monthly_breakdown": await self._calculate_monthly_breakdown(period_harvests, start_date, end_date),
                "tax_efficiency_metrics": await self._calculate_tax_efficiency_metrics(period_harvests),
                "recommendations": await self._generate_tax_optimization_recommendations(period_harvests)
            }
            
            logger.info(f"✅ Tax report generated: ${total_losses_harvested:,.2f} losses, ${total_tax_benefits:,.2f} benefits")
            return tax_report
            
        except Exception as e:
            logger.error(f"Error generating tax report: {e}")
            return self._empty_tax_report()
    
    async def _analyze_position_for_tlh(self, symbol: str, position_data: Dict) -> List[TaxLossOpportunity]:
        """Analyze a single position for tax-loss harvesting opportunities."""
        try:
            # Get specific lots for this position
            lots = await self.lot_tracker.get_lots_by_symbol(symbol)
            if not lots:
                return []
            
            # Get current market price
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return []
            
            current_price = Decimal(str(current_price))
            
            # Group lots by loss potential
            loss_lots = []
            for lot in lots:
                unrealized_pnl = (current_price - lot.cost_basis_per_share) * lot.quantity
                if unrealized_pnl < -self.config.min_loss_threshold:
                    loss_lots.append((lot, unrealized_pnl))
            
            if not loss_lots:
                return []
            
            # Sort by loss amount (most loss first)
            loss_lots.sort(key=lambda x: x[1])
            
            opportunities = []
            
            # Create opportunities based on different lot combination strategies
            # Strategy 1: Harvest all loss lots together
            if len(loss_lots) > 1:
                all_lots = [lot_info[0] for lot_info in loss_lots]
                total_loss = sum(lot_info[1] for lot_info in loss_lots)
                opportunity = await self._create_opportunity_from_lots(symbol, all_lots, total_loss, current_price)
                if opportunity:
                    opportunities.append(opportunity)
            
            # Strategy 2: Harvest individual significant loss lots
            for lot, loss_amount in loss_lots:
                if abs(loss_amount) >= self.config.significant_loss_threshold:
                    opportunity = await self._create_opportunity_from_lots(symbol, [lot], loss_amount, current_price)
                    if opportunity:
                        opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error analyzing position {symbol} for TLH: {e}")
            return []
    
    async def _create_opportunity_from_lots(self, 
                                          symbol: str, 
                                          lots: List[TaxLot],
                                          total_loss: Decimal,
                                          current_price: Decimal) -> Optional[TaxLossOpportunity]:
        """Create a TaxLossOpportunity from a list of lots."""
        try:
            # Calculate basic metrics
            total_quantity = sum(lot.quantity for lot in lots)
            total_cost_basis = sum(lot.cost_basis_per_share * lot.quantity for lot in lots)
            current_value = current_price * total_quantity
            loss_per_share = total_loss / total_quantity
            
            # Check wash sale risk
            wash_sale_violations = await self.wash_sale_monitor.check_potential_violations(symbol, total_quantity)
            wash_sale_risk = len(wash_sale_violations) > 0
            days_until_clear = await self._calculate_days_until_wash_sale_clear(symbol) if wash_sale_risk else 0
            
            # Calculate tax benefits
            tax_benefit = await self._calculate_tax_benefit(total_loss, lots)
            
            # Find replacement candidates
            replacement_candidates = await self.asset_replacement_engine.find_replacement_candidates(
                symbol, max_candidates=5
            )
            
            # Calculate opportunity score
            opportunity_score = await self._calculate_opportunity_score(
                total_loss, tax_benefit, wash_sale_risk, replacement_candidates, lots
            )
            
            # Determine urgency and recommended action
            urgency_level = self._determine_urgency_level(total_loss, opportunity_score, wash_sale_risk)
            recommended_action = self._determine_recommended_action(wash_sale_risk, replacement_candidates, urgency_level)
            
            # Calculate additional metrics
            holding_period = min((datetime.now() - lot.purchase_date).days for lot in lots)
            is_long_term = holding_period >= 365
            
            opportunity = TaxLossOpportunity(
                symbol=symbol,
                lot_ids=[lot.lot_id for lot in lots],
                total_quantity=total_quantity,
                unrealized_loss=abs(total_loss),
                loss_per_share=abs(loss_per_share),
                tax_benefit_estimate=tax_benefit,
                harvest_confidence=min(opportunity_score / 100, 0.95),
                replacement_candidates=replacement_candidates,
                wash_sale_risk=wash_sale_risk,
                days_until_wash_sale_clear=days_until_clear,
                reasoning=self._generate_opportunity_reasoning(
                    symbol, total_loss, tax_benefit, wash_sale_risk, len(replacement_candidates)
                ),
                purchase_dates=[lot.purchase_date for lot in lots],
                original_cost_basis=total_cost_basis,
                current_market_value=current_value,
                holding_period=holding_period,
                is_long_term=is_long_term,
                opportunity_score=opportunity_score,
                urgency_level=urgency_level,
                recommended_action=recommended_action,
                optimal_timing=await self._calculate_optimal_timing(wash_sale_risk, days_until_clear),
                estimated_tax_savings=tax_benefit,
                impact_on_portfolio_risk=0.0,  # Would calculate based on position size
                replacement_correlation_score=np.mean([c.correlation for c in replacement_candidates]) if replacement_candidates else 0.0
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error creating TLH opportunity for {symbol}: {e}")
            return None
    
    async def _execute_single_harvest(self, opportunity: TaxLossOpportunity) -> HarvestingResult:
        """Execute harvesting for a single opportunity."""
        try:
            logger.info(f"🎯 Executing harvest for {opportunity.symbol}: ${opportunity.unrealized_loss:,.2f} loss")
            
            # Pre-execution validation
            if opportunity.wash_sale_risk and not self.config.allow_direct_repurchase:
                return HarvestingResult(
                    timestamp=datetime.now(),
                    symbol=opportunity.symbol,
                    lots_harvested=[],
                    quantity_harvested=Decimal("0"),
                    realized_loss=Decimal("0"),
                    tax_benefit_actual=Decimal("0"),
                    replacement_symbol=None,
                    replacement_quantity=None,
                    wash_sale_violations=[],
                    success=False,
                    error_message="Wash sale risk detected and direct repurchase not allowed"
                )
            
            # Step 1: Execute the loss harvesting sale
            try:
                # Use specific lot identification for the sale
                sale_result = await self._execute_specific_lot_sale(opportunity)
                if not sale_result["success"]:
                    raise Exception(f"Sale execution failed: {sale_result['error']}")
                
                realized_loss = sale_result["realized_loss"]
                actual_quantity = sale_result["quantity_sold"]
                
            except Exception as e:
                return HarvestingResult(
                    timestamp=datetime.now(),
                    symbol=opportunity.symbol,
                    lots_harvested=[],
                    quantity_harvested=Decimal("0"),
                    realized_loss=Decimal("0"),
                    tax_benefit_actual=Decimal("0"),
                    replacement_symbol=None,
                    replacement_quantity=None,
                    wash_sale_violations=[],
                    success=False,
                    error_message=f"Failed to execute sale: {e}"
                )
            
            # Step 2: Execute replacement purchase if required and available
            replacement_symbol = None
            replacement_quantity = None
            
            if self.config.require_replacement and opportunity.replacement_candidates:
                try:
                    best_replacement = opportunity.replacement_candidates[0]  # Already sorted by score
                    replacement_result = await self._execute_replacement_purchase(
                        best_replacement, realized_loss
                    )
                    
                    if replacement_result["success"]:
                        replacement_symbol = best_replacement.symbol
                        replacement_quantity = replacement_result["quantity_purchased"]
                        logger.info(f"✅ Purchased {replacement_quantity} shares of {replacement_symbol} as replacement")
                    else:
                        logger.warning(f"⚠️ Replacement purchase failed: {replacement_result['error']}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Replacement purchase error: {e}")
            
            # Step 3: Calculate actual tax benefit
            actual_tax_benefit = await self._calculate_actual_tax_benefit(realized_loss)
            
            # Step 4: Update lot tracking and wash sale monitoring
            await self.lot_tracker.record_sale(opportunity.symbol, opportunity.lot_ids, actual_quantity, realized_loss)
            wash_sale_violations = await self.wash_sale_monitor.record_transaction(
                opportunity.symbol, "sell", actual_quantity, datetime.now()
            )
            
            # Create successful result
            result = HarvestingResult(
                timestamp=datetime.now(),
                symbol=opportunity.symbol,
                lots_harvested=opportunity.lot_ids,
                quantity_harvested=actual_quantity,
                realized_loss=realized_loss,
                tax_benefit_actual=actual_tax_benefit,
                replacement_symbol=replacement_symbol,
                replacement_quantity=replacement_quantity,
                wash_sale_violations=wash_sale_violations,
                success=True
            )
            
            logger.info(f"✅ Successfully harvested ${realized_loss:,.2f} loss from {opportunity.symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing harvest for {opportunity.symbol}: {e}")
            return HarvestingResult(
                timestamp=datetime.now(),
                symbol=opportunity.symbol,
                lots_harvested=[],
                quantity_harvested=Decimal("0"),
                realized_loss=Decimal("0"),
                tax_benefit_actual=Decimal("0"),
                replacement_symbol=None,
                replacement_quantity=None,
                wash_sale_violations=[],
                success=False,
                error_message=str(e)
            )
    
    async def _get_current_positions(self) -> Dict[str, Dict]:
        """Get current portfolio positions from Alpaca."""
        try:
            positions = alpaca_client.get_positions()
            return {pos["symbol"]: pos for pos in positions}
        except Exception as e:
            logger.error(f"Error getting current positions: {e}")
            return {}
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        try:
            return alpaca_client.get_current_price(symbol)
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _calculate_opportunity_score(self, loss_amount: Decimal, tax_benefit: Decimal, 
                                   wash_sale_risk: bool, replacement_candidates: List, 
                                   lots: List[TaxLot]) -> float:
        """Calculate opportunity score for prioritization."""
        base_score = float(abs(loss_amount)) * 0.1  # Base score from loss amount
        tax_score = float(tax_benefit) * 0.2  # Tax benefit multiplier
        
        # Penalty for wash sale risk
        wash_penalty = 30 if wash_sale_risk else 0
        
        # Bonus for good replacement options
        replacement_bonus = len(replacement_candidates) * 5
        
        # Bonus for long-term holdings (more tax efficient)
        long_term_bonus = 10 if any((datetime.now() - lot.purchase_date).days >= 365 for lot in lots) else 0
        
        total_score = base_score + tax_score + replacement_bonus + long_term_bonus - wash_penalty
        return max(0, min(100, total_score))  # Clamp to 0-100 range
    
    def _determine_urgency_level(self, loss_amount: Decimal, opportunity_score: float, 
                               wash_sale_risk: bool) -> Literal["low", "medium", "high", "critical"]:
        """Determine urgency level for the opportunity."""
        if wash_sale_risk:
            return "low"  # Lower urgency due to wash sale complications
        
        if abs(loss_amount) >= self.config.significant_loss_threshold and opportunity_score >= 80:
            return "critical"
        elif abs(loss_amount) >= self.config.significant_loss_threshold or opportunity_score >= 60:
            return "high"
        elif opportunity_score >= 40:
            return "medium"
        else:
            return "low"
    
    def _determine_recommended_action(self, wash_sale_risk: bool, replacement_candidates: List, 
                                    urgency_level: str) -> Literal["harvest_now", "wait", "replace_first", "skip"]:
        """Determine recommended action for the opportunity."""
        if wash_sale_risk and not self.config.allow_direct_repurchase:
            if replacement_candidates:
                return "replace_first"
            else:
                return "wait"
        
        if urgency_level in ["critical", "high"]:
            return "harvest_now"
        elif urgency_level == "medium":
            return "harvest_now" if replacement_candidates or not self.config.require_replacement else "wait"
        else:
            return "wait"
    
    async def _calculate_tax_benefit(self, loss_amount: Decimal, lots: List[TaxLot]) -> Decimal:
        """Calculate estimated tax benefit from harvesting the loss."""
        # Determine if losses are short-term or long-term
        short_term_losses = Decimal("0")
        long_term_losses = Decimal("0")
        
        for lot in lots:
            lot_loss = abs(loss_amount) * (lot.quantity / sum(l.quantity for l in lots))
            if (datetime.now() - lot.purchase_date).days < 365:
                short_term_losses += lot_loss
            else:
                long_term_losses += lot_loss
        
        # Calculate tax benefit based on tax rates
        # Short-term losses offset ordinary income (higher rate)
        short_term_benefit = short_term_losses * self.config.ordinary_income_tax_rate
        
        # Long-term losses offset capital gains (lower rate)
        long_term_benefit = long_term_losses * self.config.capital_gains_tax_rate
        
        # Add NIIT benefit if applicable
        niit_benefit = abs(loss_amount) * self.config.net_investment_income_tax
        
        return short_term_benefit + long_term_benefit + niit_benefit
    
    async def _calculate_actual_tax_benefit(self, realized_loss: Decimal) -> Decimal:
        """Calculate actual tax benefit from realized loss (simplified)."""
        # In practice, this would need to consider the investor's overall tax situation
        # For now, use a blended rate
        blended_rate = (self.config.ordinary_income_tax_rate + self.config.capital_gains_tax_rate) / 2
        return abs(realized_loss) * blended_rate
    
    def _generate_opportunity_reasoning(self, symbol: str, loss_amount: Decimal, 
                                      tax_benefit: Decimal, wash_sale_risk: bool, 
                                      replacement_count: int) -> str:
        """Generate human-readable reasoning for the opportunity."""
        reasoning_parts = [
            f"${abs(loss_amount):,.2f} unrealized loss in {symbol}",
            f"Est. tax benefit: ${tax_benefit:,.2f}"
        ]
        
        if wash_sale_risk:
            reasoning_parts.append("WASH SALE RISK detected")
        
        if replacement_count > 0:
            reasoning_parts.append(f"{replacement_count} replacement candidates available")
        
        return " | ".join(reasoning_parts)
    
    def _empty_tax_report(self) -> Dict:
        """Return empty tax report structure."""
        return {
            "report_period": "N/A",
            "summary": {
                "total_losses_harvested": Decimal("0"),
                "total_tax_benefits": Decimal("0"),
                "harvest_transactions": 0,
                "unique_symbols": 0,
                "wash_sale_violations": 0,
                "after_tax_alpha": 0.0
            },
            "by_symbol": {},
            "wash_sale_details": [],
            "monthly_breakdown": {},
            "tax_efficiency_metrics": {},
            "recommendations": []
        }
    
    # Additional helper methods would be implemented here...
    # (truncated for space, but would include all necessary implementation details)

# Global instance
tax_loss_harvesting_engine = TaxLossHarvestingEngine()

# Convenience functions
async def scan_for_tlh_opportunities(portfolio_positions: Optional[Dict[str, Dict]] = None) -> List[TaxLossOpportunity]:
    """Scan for tax-loss harvesting opportunities."""
    return await tax_loss_harvesting_engine.scan_for_opportunities(portfolio_positions)

async def execute_tlh_strategy(opportunities: List[TaxLossOpportunity]) -> List[HarvestingResult]:
    """Execute tax-loss harvesting strategy."""
    return await tax_loss_harvesting_engine.execute_harvesting_strategy(opportunities)

async def generate_tax_report(start_date: datetime, end_date: datetime) -> Dict:
    """Generate comprehensive tax report."""
    return await tax_loss_harvesting_engine.generate_tax_report(start_date, end_date)

__all__ = [
    'TaxLossHarvestingEngine', 'TaxLossOpportunity', 'HarvestingResult', 'TaxHarvestingConfig',
    'HarvestingStrategy', 'TaxSituation', 'tax_loss_harvesting_engine',
    'scan_for_tlh_opportunities', 'execute_tlh_strategy', 'generate_tax_report'
]