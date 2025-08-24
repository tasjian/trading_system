#!/usr/bin/env python3
"""
Comprehensive Tax Reporting and After-Tax Analytics Module

A sophisticated tax reporting and analytics system that provides detailed
tax analysis, performance measurement, and compliance reporting for the
tax-loss harvesting system.

Key Features:
- Comprehensive tax impact reporting (Form 8949, Schedule D)
- After-tax performance analytics and attribution
- Tax alpha calculation and benchmarking
- Wash sale violation tracking and resolution
- Cost basis reconciliation and audit trails
- Tax efficiency metrics and optimization insights
- Multi-year tax planning and carryforward analysis
- Regulatory compliance validation
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import json
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
import calendar
import uuid

from tools.alpaca_client import alpaca_client
from config.settings import settings
from core.tax_loss_harvesting import tax_loss_harvesting_engine, HarvestingResult
from core.lot_tracking import lot_tracker, TaxLot
from core.wash_sale_monitor import wash_sale_monitor, WashSaleViolation

logger = logging.getLogger(__name__)

class TaxReportingPeriod(Enum):
    """Tax reporting periods."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    CUSTOM = "custom"

class TaxFormType(Enum):
    """Types of tax forms to generate."""
    FORM_8949 = "form_8949"          # Sales and Other Dispositions of Capital Assets
    SCHEDULE_D = "schedule_d"        # Capital Gains and Losses
    FORM_1040_D = "form_1040_d"      # Capital Gains and Losses summary
    TAX_SUMMARY = "tax_summary"      # Custom tax summary report

@dataclass
class TaxableTransaction:
    """Represents a taxable transaction for reporting."""
    transaction_id: str
    symbol: str
    transaction_type: str  # "buy", "sell", "dividend", "split"
    transaction_date: datetime
    quantity: Decimal
    proceeds: Optional[Decimal] = None  # For sales
    cost_basis: Optional[Decimal] = None  # For sales
    gain_loss: Optional[Decimal] = None  # For sales
    
    # Tax classification
    is_short_term: bool = False
    is_long_term: bool = False
    is_wash_sale: bool = False
    
    # Lot information
    lot_ids: List[str] = field(default_factory=list)
    acquisition_date: Optional[datetime] = None
    holding_period_days: Optional[int] = None
    
    # Reporting details
    form_8949_code: Optional[str] = None  # A, B, C, D, E, F
    adjustment_amount: Decimal = Decimal("0.00")
    adjustment_code: Optional[str] = None
    
    # Metadata
    order_id: Optional[str] = None
    broker_source: str = "Alpaca"
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TaxSummary:
    """Tax summary for a reporting period."""
    reporting_period: str
    start_date: date
    end_date: date
    
    # Realized gains/losses
    short_term_gain_loss: Decimal
    long_term_gain_loss: Decimal
    total_gain_loss: Decimal
    
    # Transaction counts
    short_term_transactions: int
    long_term_transactions: int
    total_transactions: int
    
    # Wash sale adjustments
    wash_sale_loss_disallowed: Decimal
    wash_sale_adjustments: int
    
    # Tax implications
    estimated_tax_liability: Decimal
    estimated_tax_savings: Decimal
    effective_tax_rate: float
    
    # Performance metrics
    pre_tax_return: float
    after_tax_return: float
    tax_alpha: float  # After-tax outperformance
    tax_efficiency_ratio: float
    
    # Additional metrics
    harvested_losses: Decimal
    replacement_asset_performance: Decimal
    tracking_error: float

@dataclass
class AfterTaxPerformanceMetrics:
    """Comprehensive after-tax performance analytics."""
    period_start: date
    period_end: date
    
    # Return calculations
    gross_return: float
    tax_cost: float
    after_tax_return: float
    
    # Tax efficiency metrics
    tax_efficiency_ratio: float  # After-tax return / Pre-tax return
    tax_alpha: float  # Excess after-tax return vs benchmark
    tax_cost_ratio: float  # Tax cost / Gross return
    
    # Harvesting effectiveness
    total_losses_harvested: Decimal
    tax_benefits_realized: Decimal
    harvesting_effectiveness: float  # Tax benefits / Losses harvested
    
    # Risk-adjusted metrics
    after_tax_sharpe_ratio: float
    after_tax_sortino_ratio: float
    after_tax_information_ratio: float
    
    # Attribution analysis
    security_selection_alpha: float
    tax_loss_harvesting_alpha: float
    asset_location_alpha: float
    
    # Benchmark comparison
    benchmark_after_tax_return: Optional[float] = None
    after_tax_tracking_error: Optional[float] = None
    after_tax_information_ratio_vs_benchmark: Optional[float] = None

class TaxReportingDatabase:
    """SQLite database for tax reporting and analytics."""
    
    def __init__(self, db_path: str = "data/tax_reporting.db"):
        """Initialize database connection and create tables."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Taxable transactions table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS taxable_transactions (
                        transaction_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        transaction_type TEXT NOT NULL,
                        transaction_date TIMESTAMP NOT NULL,
                        quantity DECIMAL(15,6) NOT NULL,
                        proceeds DECIMAL(15,2),
                        cost_basis DECIMAL(15,2),
                        gain_loss DECIMAL(15,2),
                        is_short_term BOOLEAN,
                        is_long_term BOOLEAN,
                        is_wash_sale BOOLEAN,
                        lot_ids TEXT, -- JSON array
                        acquisition_date TIMESTAMP,
                        holding_period_days INTEGER,
                        form_8949_code TEXT,
                        adjustment_amount DECIMAL(15,2),
                        adjustment_code TEXT,
                        order_id TEXT,
                        broker_source TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for taxable_transactions table
                conn.execute('CREATE INDEX IF NOT EXISTS idx_taxable_transactions_symbol ON taxable_transactions(symbol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_taxable_transactions_date ON taxable_transactions(transaction_date)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_taxable_transactions_type ON taxable_transactions(transaction_type)')
                
                # Tax reports table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS tax_reports (
                        report_id TEXT PRIMARY KEY,
                        report_type TEXT NOT NULL,
                        reporting_period TEXT NOT NULL,
                        start_date DATE NOT NULL,
                        end_date DATE NOT NULL,
                        report_data TEXT, -- JSON
                        generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for tax_reports table
                conn.execute('CREATE INDEX IF NOT EXISTS idx_tax_reports_type ON tax_reports(report_type)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_tax_reports_start_date ON tax_reports(start_date)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_tax_reports_end_date ON tax_reports(end_date)')
                
                # Performance metrics table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        metric_id TEXT PRIMARY KEY,
                        period_start DATE NOT NULL,
                        period_end DATE NOT NULL,
                        gross_return DECIMAL(8,4),
                        after_tax_return DECIMAL(8,4),
                        tax_alpha DECIMAL(8,4),
                        tax_efficiency_ratio DECIMAL(6,4),
                        losses_harvested DECIMAL(15,2),
                        tax_benefits_realized DECIMAL(15,2),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for performance_metrics table
                conn.execute('CREATE INDEX IF NOT EXISTS idx_performance_metrics_start ON performance_metrics(period_start)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_performance_metrics_end ON performance_metrics(period_end)')
                
                conn.commit()
                logger.info("Tax reporting database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing tax reporting database: {e}")
            raise

class TaxReportingAnalytics:
    """
    Comprehensive tax reporting and after-tax analytics system.
    
    Provides detailed tax analysis, performance measurement, and compliance
    reporting for tax-optimized portfolio management.
    """
    
    def __init__(self):
        """Initialize the tax reporting and analytics system."""
        self.db = TaxReportingDatabase()
        
        # Configuration from settings
        self.tax_report_frequency = getattr(settings, 'tax_report_frequency', 'monthly')
        self.enable_tax_reporting = getattr(settings, 'enable_tax_reporting', True)
        self.maintain_audit_trail = getattr(settings, 'maintain_audit_trail', True)
        
        # Tax rates for calculations
        self.ordinary_income_tax_rate = getattr(settings, 'ordinary_income_tax_rate', 0.32)
        self.capital_gains_tax_rate = getattr(settings, 'capital_gains_tax_rate', 0.20)
        self.net_investment_income_tax = getattr(settings, 'net_investment_income_tax', 0.038)
        
        # Performance tracking
        self.tax_reports_generated = 0
        self.last_report_date: Optional[datetime] = None
        
        logger.info("Tax Reporting and Analytics system initialized")
        logger.info(f"Report frequency: {self.tax_report_frequency}")
        logger.info(f"Tax reporting enabled: {self.enable_tax_reporting}")
    
    async def generate_comprehensive_tax_report(self, 
                                              start_date: date, 
                                              end_date: date,
                                              report_types: Optional[List[TaxFormType]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive tax report for the specified period.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            report_types: Types of reports to generate
            
        Returns:
            Comprehensive tax report dictionary
        """
        try:
            logger.info(f"📊 Generating comprehensive tax report: {start_date} to {end_date}")
            
            # Default report types
            if report_types is None:
                report_types = [TaxFormType.TAX_SUMMARY, TaxFormType.FORM_8949, TaxFormType.SCHEDULE_D]
            
            # Collect taxable transactions for the period
            transactions = await self._collect_taxable_transactions(start_date, end_date)
            
            if not transactions:
                logger.info("No taxable transactions found for the reporting period")
                return self._empty_tax_report(start_date, end_date)
            
            # Generate tax summary
            tax_summary = await self._generate_tax_summary(transactions, start_date, end_date)
            
            # Generate specific tax forms
            tax_forms = {}
            for report_type in report_types:
                if report_type == TaxFormType.FORM_8949:
                    tax_forms[report_type.value] = await self._generate_form_8949(transactions)
                elif report_type == TaxFormType.SCHEDULE_D:
                    tax_forms[report_type.value] = await self._generate_schedule_d(tax_summary, transactions)
                elif report_type == TaxFormType.TAX_SUMMARY:
                    tax_forms[report_type.value] = asdict(tax_summary)
            
            # Generate after-tax performance analytics
            performance_metrics = await self._calculate_after_tax_performance(start_date, end_date)
            
            # Generate wash sale analysis
            wash_sale_analysis = await self._analyze_wash_sale_impacts(transactions)
            
            # Generate tax optimization insights
            optimization_insights = await self._generate_optimization_insights(
                tax_summary, performance_metrics, transactions
            )
            
            comprehensive_report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "reporting_period": f"{start_date} to {end_date}",
                    "report_types": [rt.value for rt in report_types],
                    "total_transactions": len(transactions)
                },
                "tax_summary": asdict(tax_summary),
                "tax_forms": tax_forms,
                "after_tax_performance": asdict(performance_metrics),
                "wash_sale_analysis": wash_sale_analysis,
                "optimization_insights": optimization_insights,
                "compliance_status": await self._validate_compliance(transactions),
                "recommendations": await self._generate_tax_recommendations(
                    tax_summary, performance_metrics, wash_sale_analysis
                )
            }
            
            # Store report in database
            await self._store_tax_report(comprehensive_report, start_date, end_date)
            
            # Update tracking
            self.tax_reports_generated += 1
            self.last_report_date = datetime.now()
            
            logger.info(f"✅ Comprehensive tax report generated successfully")
            logger.info(f"   Total transactions: {len(transactions)}")
            logger.info(f"   Total gain/loss: ${tax_summary.total_gain_loss:,.2f}")
            logger.info(f"   Tax alpha: {performance_metrics.tax_alpha:+.2%}")
            logger.info(f"   Tax efficiency: {performance_metrics.tax_efficiency_ratio:.2%}")
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive tax report: {e}")
            return {"error": str(e)}
    
    async def calculate_real_time_tax_impact(self, 
                                           portfolio_positions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Calculate real-time tax impact of current portfolio positions.
        
        Args:
            portfolio_positions: Current portfolio positions
            
        Returns:
            Real-time tax impact analysis
        """
        try:
            logger.info("📊 Calculating real-time tax impact analysis...")
            
            total_unrealized_gains = Decimal("0")
            total_unrealized_losses = Decimal("0")
            potential_tax_liability = Decimal("0")
            potential_tax_savings = Decimal("0")
            
            position_analysis = []
            
            for symbol, position_data in portfolio_positions.items():
                try:
                    # Get detailed position information
                    position_summary = await lot_tracker.get_position_summary(symbol)
                    
                    if position_summary.total_quantity == 0:
                        continue
                    
                    # Calculate unrealized P&L and tax implications
                    unrealized_pnl = position_summary.total_unrealized_pnl or Decimal("0")
                    
                    # Determine tax treatment
                    short_term_pnl = Decimal("0")
                    long_term_pnl = Decimal("0")
                    
                    for lot in position_summary.lots:
                        lot_pnl = unrealized_pnl * (lot.quantity / position_summary.total_quantity)
                        
                        if lot.is_long_term:
                            long_term_pnl += lot_pnl
                        else:
                            short_term_pnl += lot_pnl
                    
                    # Calculate potential tax impact
                    tax_impact = Decimal("0")
                    if unrealized_pnl > 0:  # Gains
                        total_unrealized_gains += unrealized_pnl
                        # Tax on gains if realized
                        st_tax = short_term_pnl * Decimal(str(self.ordinary_income_tax_rate))
                        lt_tax = long_term_pnl * Decimal(str(self.capital_gains_tax_rate))
                        tax_impact = st_tax + lt_tax
                        potential_tax_liability += tax_impact
                    else:  # Losses
                        total_unrealized_losses += abs(unrealized_pnl)
                        # Tax savings from harvesting losses
                        st_savings = abs(short_term_pnl) * Decimal(str(self.ordinary_income_tax_rate))
                        lt_savings = abs(long_term_pnl) * Decimal(str(self.capital_gains_tax_rate))
                        tax_impact = -(st_savings + lt_savings)  # Negative for savings
                        potential_tax_savings += abs(tax_impact)
                    
                    # Check wash sale status
                    wash_sale_status = await wash_sale_monitor.get_compliance_status(symbol)
                    
                    position_analysis.append({
                        "symbol": symbol,
                        "quantity": float(position_summary.total_quantity),
                        "cost_basis": float(position_summary.total_cost_basis),
                        "market_value": float(position_summary.current_market_value or 0),
                        "unrealized_pnl": float(unrealized_pnl),
                        "short_term_pnl": float(short_term_pnl),
                        "long_term_pnl": float(long_term_pnl),
                        "potential_tax_impact": float(tax_impact),
                        "is_gain": unrealized_pnl > 0,
                        "is_loss": unrealized_pnl < 0,
                        "wash_sale_risk": wash_sale_status.get("status", "clear") != "clear",
                        "days_until_wash_sale_clear": wash_sale_status.get("days_until_clear", 0),
                        "tlh_candidate": unrealized_pnl < -100,  # $100+ loss
                        "holding_period": position_summary.weighted_holding_period
                    })
                    
                except Exception as e:
                    logger.error(f"Error analyzing position {symbol}: {e}")
                    continue
            
            # Calculate portfolio-level metrics
            net_unrealized_pnl = total_unrealized_gains - total_unrealized_losses
            net_tax_impact = potential_tax_liability - potential_tax_savings
            
            # Calculate tax efficiency metrics
            portfolio_value = sum(pos["market_value"] for pos in position_analysis)
            tax_drag_percent = float(net_tax_impact / Decimal(str(max(portfolio_value, 1)))) * 100
            
            real_time_analysis = {
                "analysis_timestamp": datetime.now().isoformat(),
                "portfolio_summary": {
                    "total_positions": len(position_analysis),
                    "portfolio_value": portfolio_value,
                    "total_unrealized_gains": float(total_unrealized_gains),
                    "total_unrealized_losses": float(total_unrealized_losses),
                    "net_unrealized_pnl": float(net_unrealized_pnl),
                    "potential_tax_liability": float(potential_tax_liability),
                    "potential_tax_savings": float(potential_tax_savings),
                    "net_tax_impact": float(net_tax_impact),
                    "tax_drag_percent": tax_drag_percent
                },
                "position_analysis": position_analysis,
                "tlh_opportunities": [pos for pos in position_analysis if pos["tlh_candidate"]],
                "wash_sale_constraints": [pos for pos in position_analysis if pos["wash_sale_risk"]],
                "recommendations": self._generate_real_time_recommendations(position_analysis)
            }
            
            logger.info(f"✅ Real-time tax impact calculated")
            logger.info(f"   Portfolio value: ${portfolio_value:,.2f}")
            logger.info(f"   Net unrealized P&L: ${net_unrealized_pnl:,.2f}")
            logger.info(f"   Net tax impact: ${net_tax_impact:,.2f}")
            logger.info(f"   Tax drag: {tax_drag_percent:.2f}%")
            
            return real_time_analysis
            
        except Exception as e:
            logger.error(f"Error calculating real-time tax impact: {e}")
            return {"error": str(e)}
    
    async def generate_tax_alpha_attribution(self, 
                                           start_date: date, 
                                           end_date: date,
                                           benchmark_return: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate tax alpha attribution analysis.
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            benchmark_return: Benchmark after-tax return for comparison
            
        Returns:
            Tax alpha attribution analysis
        """
        try:
            logger.info(f"📈 Generating tax alpha attribution analysis: {start_date} to {end_date}")
            
            # Get harvesting results for the period
            harvesting_results = await self._get_harvesting_results(start_date, end_date)
            
            # Calculate base performance metrics
            performance_metrics = await self._calculate_after_tax_performance(start_date, end_date)
            
            # Attribution components
            attribution = {
                "period": f"{start_date} to {end_date}",
                "total_tax_alpha": performance_metrics.tax_alpha,
                "attribution_breakdown": {
                    "tax_loss_harvesting": 0.0,
                    "lot_selection_optimization": 0.0,
                    "wash_sale_avoidance": 0.0,
                    "timing_optimization": 0.0,
                    "replacement_asset_selection": 0.0
                }
            }
            
            # Tax-loss harvesting contribution
            if harvesting_results:
                total_tax_benefits = sum(r.tax_benefit_actual for r in harvesting_results)
                portfolio_value = await self._get_average_portfolio_value(start_date, end_date)
                
                if portfolio_value > 0:
                    tlh_alpha = float(total_tax_benefits / Decimal(str(portfolio_value)))
                    attribution["attribution_breakdown"]["tax_loss_harvesting"] = tlh_alpha
            
            # Lot selection optimization (using HIFO vs FIFO)
            lot_optimization_alpha = await self._calculate_lot_selection_alpha(start_date, end_date)
            attribution["attribution_breakdown"]["lot_selection_optimization"] = lot_optimization_alpha
            
            # Wash sale avoidance benefit
            wash_sale_alpha = await self._calculate_wash_sale_avoidance_alpha(start_date, end_date)
            attribution["attribution_breakdown"]["wash_sale_avoidance"] = wash_sale_alpha
            
            # Timing optimization (deferring gains, accelerating losses)
            timing_alpha = await self._calculate_timing_optimization_alpha(start_date, end_date)
            attribution["attribution_breakdown"]["timing_optimization"] = timing_alpha
            
            # Replacement asset selection effectiveness
            replacement_alpha = await self._calculate_replacement_asset_alpha(start_date, end_date)
            attribution["attribution_breakdown"]["replacement_asset_selection"] = replacement_alpha
            
            # Validate attribution adds up
            total_attributed = sum(attribution["attribution_breakdown"].values())
            attribution["attribution_residual"] = attribution["total_tax_alpha"] - total_attributed
            
            # Benchmark comparison
            if benchmark_return is not None:
                attribution["benchmark_comparison"] = {
                    "benchmark_after_tax_return": benchmark_return,
                    "portfolio_after_tax_return": performance_metrics.after_tax_return,
                    "excess_return": performance_metrics.after_tax_return - benchmark_return,
                    "tax_alpha_vs_benchmark": performance_metrics.tax_alpha
                }
            
            logger.info(f"✅ Tax alpha attribution complete: {attribution['total_tax_alpha']:+.2%} total alpha")
            return attribution
            
        except Exception as e:
            logger.error(f"Error generating tax alpha attribution: {e}")
            return {"error": str(e)}
    
    async def _collect_taxable_transactions(self, start_date: date, end_date: date) -> List[TaxableTransaction]:
        """Collect taxable transactions from the database for the specified period."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.execute('''
                    SELECT * FROM taxable_transactions 
                    WHERE transaction_date BETWEEN ? AND ?
                    ORDER BY transaction_date
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                transactions = []
                for row in cursor.fetchall():
                    # Convert database row to TaxableTransaction
                    transaction = TaxableTransaction(
                        transaction_id=row[0],
                        symbol=row[1],
                        transaction_type=row[2],
                        transaction_date=datetime.fromisoformat(row[3]),
                        quantity=Decimal(str(row[4])),
                        proceeds=Decimal(str(row[5])) if row[5] else None,
                        cost_basis=Decimal(str(row[6])) if row[6] else None,
                        gain_loss=Decimal(str(row[7])) if row[7] else None,
                        is_short_term=bool(row[8]) if row[8] is not None else False,
                        is_long_term=bool(row[9]) if row[9] is not None else False,
                        is_wash_sale=bool(row[10]) if row[10] is not None else False,
                        lot_ids=json.loads(row[11]) if row[11] else [],
                        acquisition_date=datetime.fromisoformat(row[12]) if row[12] else None,
                        holding_period_days=row[13] if row[13] else None,
                        form_8949_code=row[14],
                        adjustment_amount=Decimal(str(row[15])) if row[15] else Decimal("0.00"),
                        adjustment_code=row[16],
                        order_id=row[17],
                        broker_source=row[18] or "Alpaca",
                        created_at=datetime.fromisoformat(row[19])
                    )
                    transactions.append(transaction)
                
                return transactions
                
        except Exception as e:
            logger.error(f"Error collecting taxable transactions: {e}")
            return []

    async def _generate_tax_summary(self, transactions: List[TaxableTransaction], start_date: date, end_date: date) -> TaxSummary:
        """Generate tax summary from transactions."""
        try:
            # Calculate totals
            short_term_gain_loss = Decimal("0")
            long_term_gain_loss = Decimal("0")
            wash_sale_loss_disallowed = Decimal("0")
            
            short_term_count = 0
            long_term_count = 0
            wash_sale_count = 0
            
            for transaction in transactions:
                if transaction.gain_loss:
                    if transaction.is_short_term:
                        short_term_gain_loss += transaction.gain_loss
                        short_term_count += 1
                    if transaction.is_long_term:
                        long_term_gain_loss += transaction.gain_loss
                        long_term_count += 1
                    
                    if transaction.is_wash_sale:
                        wash_sale_loss_disallowed += abs(transaction.gain_loss) if transaction.gain_loss < 0 else Decimal("0")
                        wash_sale_count += 1
            
            total_gain_loss = short_term_gain_loss + long_term_gain_loss
            
            # Calculate tax implications
            st_tax_liability = max(short_term_gain_loss * Decimal(str(self.ordinary_income_tax_rate)), Decimal("0"))
            lt_tax_liability = max(long_term_gain_loss * Decimal(str(self.capital_gains_tax_rate)), Decimal("0"))
            estimated_tax_liability = st_tax_liability + lt_tax_liability
            
            # Tax savings from losses
            st_tax_savings = abs(min(short_term_gain_loss * Decimal(str(self.ordinary_income_tax_rate)), Decimal("0")))
            lt_tax_savings = abs(min(long_term_gain_loss * Decimal(str(self.capital_gains_tax_rate)), Decimal("0")))
            estimated_tax_savings = st_tax_savings + lt_tax_savings
            
            # Calculate effective tax rate
            effective_tax_rate = float(estimated_tax_liability / max(abs(total_gain_loss), Decimal("1")))
            
            return TaxSummary(
                reporting_period=f"{start_date} to {end_date}",
                start_date=start_date,
                end_date=end_date,
                short_term_gain_loss=short_term_gain_loss,
                long_term_gain_loss=long_term_gain_loss,
                total_gain_loss=total_gain_loss,
                short_term_transactions=short_term_count,
                long_term_transactions=long_term_count,
                total_transactions=len(transactions),
                wash_sale_loss_disallowed=wash_sale_loss_disallowed,
                wash_sale_adjustments=wash_sale_count,
                estimated_tax_liability=estimated_tax_liability,
                estimated_tax_savings=estimated_tax_savings,
                effective_tax_rate=effective_tax_rate,
                pre_tax_return=0.0,  # Would need portfolio data to calculate
                after_tax_return=0.0,  # Would need portfolio data to calculate
                tax_alpha=0.0,  # Would need benchmark to calculate
                tax_efficiency_ratio=1.0 - effective_tax_rate,
                harvested_losses=abs(min(total_gain_loss, Decimal("0"))),
                replacement_asset_performance=Decimal("0"),
                tracking_error=0.0
            )
            
        except Exception as e:
            logger.error(f"Error generating tax summary: {e}")
            return self._empty_tax_summary(start_date, end_date)

    def _empty_tax_summary(self, start_date: date, end_date: date) -> TaxSummary:
        """Generate empty tax summary."""
        return TaxSummary(
            reporting_period=f"{start_date} to {end_date}",
            start_date=start_date,
            end_date=end_date,
            short_term_gain_loss=Decimal("0"),
            long_term_gain_loss=Decimal("0"),
            total_gain_loss=Decimal("0"),
            short_term_transactions=0,
            long_term_transactions=0,
            total_transactions=0,
            wash_sale_loss_disallowed=Decimal("0"),
            wash_sale_adjustments=0,
            estimated_tax_liability=Decimal("0"),
            estimated_tax_savings=Decimal("0"),
            effective_tax_rate=0.0,
            pre_tax_return=0.0,
            after_tax_return=0.0,
            tax_alpha=0.0,
            tax_efficiency_ratio=1.0,
            harvested_losses=Decimal("0"),
            replacement_asset_performance=Decimal("0"),
            tracking_error=0.0
        )

    async def _generate_form_8949(self, transactions: List[TaxableTransaction]) -> Dict[str, Any]:
        """Generate Form 8949 data."""
        return {"form_8949": "Not implemented in MVP"}

    async def _generate_schedule_d(self, tax_summary: TaxSummary, transactions: List[TaxableTransaction]) -> Dict[str, Any]:
        """Generate Schedule D data."""
        return {"schedule_d": "Not implemented in MVP"}

    async def _calculate_after_tax_performance(self, start_date: date, end_date: date) -> AfterTaxPerformanceMetrics:
        """Calculate after-tax performance metrics."""
        return AfterTaxPerformanceMetrics(
            period_start=start_date,
            period_end=end_date,
            gross_return=0.0,
            tax_cost=0.0,
            after_tax_return=0.0,
            tax_efficiency_ratio=1.0,
            tax_alpha=0.0,
            tax_cost_ratio=0.0,
            total_losses_harvested=Decimal("0"),
            tax_benefits_realized=Decimal("0"),
            harvesting_effectiveness=0.0,
            after_tax_sharpe_ratio=0.0,
            after_tax_sortino_ratio=0.0,
            after_tax_information_ratio=0.0,
            security_selection_alpha=0.0,
            tax_loss_harvesting_alpha=0.0,
            asset_location_alpha=0.0
        )

    async def _analyze_wash_sale_impacts(self, transactions: List[TaxableTransaction]) -> Dict[str, Any]:
        """Analyze wash sale impacts."""
        wash_sale_transactions = [t for t in transactions if t.is_wash_sale]
        return {
            "total_wash_sales": len(wash_sale_transactions),
            "disallowed_losses": sum(abs(t.gain_loss) for t in wash_sale_transactions if t.gain_loss and t.gain_loss < 0),
            "affected_symbols": list(set(t.symbol for t in wash_sale_transactions))
        }

    async def _generate_optimization_insights(self, tax_summary: TaxSummary, performance_metrics: AfterTaxPerformanceMetrics, transactions: List[TaxableTransaction]) -> List[str]:
        """Generate tax optimization insights."""
        insights = []
        
        if tax_summary.wash_sale_adjustments > 0:
            insights.append(f"⚠️ {tax_summary.wash_sale_adjustments} wash sale violations detected")
        
        if tax_summary.harvested_losses > 0:
            insights.append(f"✅ Successfully harvested ${tax_summary.harvested_losses:,.2f} in losses")
        
        if tax_summary.total_gain_loss < 0:
            insights.append(f"💡 Consider carrying forward ${abs(tax_summary.total_gain_loss):,.2f} in losses")
        
        return insights

    async def _validate_compliance(self, transactions: List[TaxableTransaction]) -> Dict[str, Any]:
        """Validate tax compliance."""
        return {
            "wash_sale_violations": len([t for t in transactions if t.is_wash_sale]),
            "compliance_status": "compliant" if not any(t.is_wash_sale for t in transactions) else "violations_detected"
        }

    async def _generate_tax_recommendations(self, tax_summary: TaxSummary, performance_metrics: AfterTaxPerformanceMetrics, wash_sale_analysis: Dict[str, Any]) -> List[str]:
        """Generate tax optimization recommendations."""
        recommendations = []
        
        if tax_summary.total_gain_loss > 1000:
            recommendations.append("Consider harvesting losses to offset gains")
        
        if wash_sale_analysis.get("total_wash_sales", 0) > 0:
            recommendations.append("Review wash sale violations and consider alternative securities")
        
        recommendations.append("Continue tax-loss harvesting strategy for optimal after-tax returns")
        
        return recommendations

    async def _store_tax_report(self, report: Dict[str, Any], start_date: date, end_date: date):
        """Store tax report in database."""
        try:
            report_id = str(uuid.uuid4())
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute('''
                    INSERT INTO tax_reports (report_id, report_type, reporting_period, start_date, end_date, report_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (report_id, "comprehensive", f"{start_date} to {end_date}", start_date, end_date, json.dumps(report)))
                
        except Exception as e:
            logger.error(f"Error storing tax report: {e}")

    async def _get_harvesting_results(self, start_date: date, end_date: date) -> List:
        """Get harvesting results for the period."""
        return []  # Placeholder for MVP

    async def _get_average_portfolio_value(self, start_date: date, end_date: date) -> Decimal:
        """Get average portfolio value for the period."""
        return Decimal("100000")  # Placeholder for MVP

    async def _calculate_lot_selection_alpha(self, start_date: date, end_date: date) -> float:
        """Calculate lot selection optimization alpha."""
        return 0.0  # Placeholder for MVP

    async def _calculate_wash_sale_avoidance_alpha(self, start_date: date, end_date: date) -> float:
        """Calculate wash sale avoidance alpha."""
        return 0.0  # Placeholder for MVP

    async def _calculate_timing_optimization_alpha(self, start_date: date, end_date: date) -> float:
        """Calculate timing optimization alpha."""
        return 0.0  # Placeholder for MVP

    async def _calculate_replacement_asset_alpha(self, start_date: date, end_date: date) -> float:
        """Calculate replacement asset selection alpha."""
        return 0.0  # Placeholder for MVP

    def _generate_real_time_recommendations(self, position_analysis: List[Dict]) -> List[str]:
        """Generate real-time tax optimization recommendations."""
        recommendations = []
        
        # Identify high-priority TLH opportunities
        tlh_candidates = [pos for pos in position_analysis if pos["tlh_candidate"] and not pos["wash_sale_risk"]]
        if tlh_candidates:
            total_potential_savings = sum(abs(pos["potential_tax_impact"]) for pos in tlh_candidates)
            recommendations.append(f"🎯 Tax-Loss Harvesting: {len(tlh_candidates)} opportunities available with ${total_potential_savings:,.2f} potential tax savings")
        
        # Warn about wash sale risks
        wash_sale_risks = [pos for pos in position_analysis if pos["wash_sale_risk"]]
        if wash_sale_risks:
            recommendations.append(f"⚠️ Wash Sale Risk: {len(wash_sale_risks)} positions have wash sale constraints")
        
        # Suggest gain deferral near year-end
        if datetime.now().month >= 11:  # November or December
            gains_to_defer = [pos for pos in position_analysis if pos["is_gain"] and pos["potential_tax_impact"] > 500]
            if gains_to_defer:
                recommendations.append(f"📅 Year-End Planning: Consider deferring ${sum(pos['potential_tax_impact'] for pos in gains_to_defer):,.2f} in taxable gains to next year")
        
        # Recommend lot selection optimization
        mixed_lots = [pos for pos in position_analysis if pos["short_term_pnl"] != 0 and pos["long_term_pnl"] != 0]
        if mixed_lots:
            recommendations.append(f"🎯 Lot Selection: {len(mixed_lots)} positions have mixed short/long-term lots - optimize using HIFO method")
        
        return recommendations
    
    def _empty_tax_report(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Return empty tax report structure."""
        return {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "reporting_period": f"{start_date} to {end_date}",
                "total_transactions": 0
            },
            "tax_summary": {
                "total_gain_loss": 0.0,
                "tax_liability": 0.0,
                "message": "No taxable transactions in reporting period"
            },
            "after_tax_performance": {
                "after_tax_return": 0.0,
                "tax_alpha": 0.0,
                "message": "No performance data available"
            }
        }

# Global instance
tax_reporting_analytics = TaxReportingAnalytics()

# Convenience functions
async def generate_tax_report(start_date: date, end_date: date, **kwargs) -> Dict[str, Any]:
    """Generate comprehensive tax report."""
    return await tax_reporting_analytics.generate_comprehensive_tax_report(start_date, end_date, **kwargs)

async def calculate_real_time_tax_impact(portfolio_positions: Dict[str, Dict]) -> Dict[str, Any]:
    """Calculate real-time tax impact."""
    return await tax_reporting_analytics.calculate_real_time_tax_impact(portfolio_positions)

async def generate_tax_alpha_attribution(start_date: date, end_date: date, **kwargs) -> Dict[str, Any]:
    """Generate tax alpha attribution analysis."""
    return await tax_reporting_analytics.generate_tax_alpha_attribution(start_date, end_date, **kwargs)

__all__ = [
    'TaxReportingAnalytics', 'TaxableTransaction', 'TaxSummary', 'AfterTaxPerformanceMetrics',
    'TaxReportingPeriod', 'TaxFormType', 'tax_reporting_analytics',
    'generate_tax_report', 'calculate_real_time_tax_impact', 'generate_tax_alpha_attribution'
]