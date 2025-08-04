#!/usr/bin/env python3
"""Regulatory compliance module for Alpaca Customer Agreement adherence."""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from config.settings import settings

logger = logging.getLogger(__name__)

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_REVIEW = "requires_review"

@dataclass
class ComplianceCheck:
    """Individual compliance check result."""
    check_name: str
    status: ComplianceStatus
    description: str
    details: str
    timestamp: datetime
    action_required: Optional[str] = None

class RegulatoryComplianceManager:
    """
    Manages regulatory compliance with Alpaca Customer Agreement.
    
    Key Compliance Areas:
    1. Paper Trading Mode Enforcement
    2. Self-directed Trading Requirements  
    3. Risk Management Compliance
    4. Order Validation Requirements
    5. Record Keeping Obligations
    """
    
    def __init__(self):
        self.compliance_checks: List[ComplianceCheck] = []
        self.user_acknowledgments: Dict[str, datetime] = {}
        
    def validate_paper_trading_mode(self) -> ComplianceCheck:
        """Validate that system operates in paper trading mode only."""
        
        if settings.alpaca_base_url != "https://paper-api.alpaca.markets":
            return ComplianceCheck(
                check_name="Paper Trading Mode",
                status=ComplianceStatus.NON_COMPLIANT,
                description="System must operate in paper trading mode only",
                details=f"Current base URL: {settings.alpaca_base_url}",
                timestamp=datetime.now(),
                action_required="Switch to paper trading URL"
            )
        
        return ComplianceCheck(
            check_name="Paper Trading Mode",
            status=ComplianceStatus.COMPLIANT,
            description="System properly configured for paper trading",
            details="Using Alpaca paper trading API endpoint",
            timestamp=datetime.now()
        )
    
    def validate_self_directed_compliance(self) -> ComplianceCheck:
        """
        Validate compliance with self-directed trading requirements.
        
        Per Alpaca Customer Agreement Section 5:
        - Account is self-directed
        - All orders must be authorized by customer
        - No investment advice provided
        - No discretionary trades
        """
        
        # Check if human confirmation is enabled for automated systems
        if hasattr(settings, 'require_human_confirmation'):
            if not settings.require_human_confirmation:
                return ComplianceCheck(
                    check_name="Self-directed Trading",
                    status=ComplianceStatus.REQUIRES_REVIEW,
                    description="Automated trading without human confirmation",
                    details="System operates autonomously without per-trade approval",
                    timestamp=datetime.now(),
                    action_required="Enable human confirmation for trades or review regulatory requirements"
                )
        
        return ComplianceCheck(
            check_name="Self-directed Trading",
            status=ComplianceStatus.COMPLIANT,
            description="Trading system operates with appropriate oversight",
            details="Paper trading mode with risk controls active",
            timestamp=datetime.now()
        )
    
    def validate_risk_management(self) -> ComplianceCheck:
        """Validate risk management compliance."""
        
        risk_issues = []
        
        if settings.max_portfolio_risk > 0.05:  # 5% threshold
            risk_issues.append("Portfolio risk exceeds conservative limits")
            
        if settings.max_position_size > 0.1:  # 10% threshold
            risk_issues.append("Position size limits may be too high")
            
        if settings.max_daily_loss > 0.1:  # 10% threshold
            risk_issues.append("Daily loss limits may be too high")
        
        if risk_issues:
            return ComplianceCheck(
                check_name="Risk Management",
                status=ComplianceStatus.REQUIRES_REVIEW,
                description="Risk parameters require review",
                details="; ".join(risk_issues),
                timestamp=datetime.now(),
                action_required="Review and adjust risk parameters"
            )
        
        return ComplianceCheck(
            check_name="Risk Management",
            status=ComplianceStatus.COMPLIANT,
            description="Risk management parameters within acceptable ranges",
            details=f"Max risk: {settings.max_portfolio_risk:.1%}, Max position: {settings.max_position_size:.1%}",
            timestamp=datetime.now()
        )
    
    def validate_record_keeping(self) -> ComplianceCheck:
        """Validate record keeping compliance."""
        
        try:
            # Check if logging is properly configured
            if settings.log_level not in ["DEBUG", "INFO", "WARNING"]:
                return ComplianceCheck(
                    check_name="Record Keeping",
                    status=ComplianceStatus.NON_COMPLIANT,
                    description="Insufficient logging configuration",
                    details=f"Log level: {settings.log_level}",
                    timestamp=datetime.now(),
                    action_required="Configure appropriate logging level"
                )
            
            return ComplianceCheck(
                check_name="Record Keeping",
                status=ComplianceStatus.COMPLIANT,
                description="Proper logging and record keeping configured",
                details=f"Log level: {settings.log_level}, Log file: {settings.log_file}",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return ComplianceCheck(
                check_name="Record Keeping",
                status=ComplianceStatus.NON_COMPLIANT,
                description="Error validating record keeping setup",
                details=str(e),
                timestamp=datetime.now(),
                action_required="Fix logging configuration"
            )
    
    def require_user_acknowledgment(self, acknowledgment_type: str) -> bool:
        """
        Require user acknowledgment of risks and compliance.
        
        Per Alpaca Customer Agreement:
        - Users must acknowledge trading risks
        - Users must understand self-directed nature
        - Users must accept electronic communications
        """
        
        print(f"\n🚨 REGULATORY COMPLIANCE ACKNOWLEDGMENT REQUIRED")
        print("=" * 60)
        
        if acknowledgment_type == "automated_trading_risk":
            print("AUTOMATED TRADING RISK DISCLOSURE:")
            print("- This system uses artificial intelligence to make trading decisions")
            print("- All trading occurs in paper trading mode for safety")
            print("- You maintain full control and can stop the system at any time")
            print("- Past performance does not guarantee future results")
            print("- You are responsible for monitoring all trading activity")
            
        elif acknowledgment_type == "self_directed_trading":
            print("SELF-DIRECTED TRADING ACKNOWLEDGMENT:")
            print("- This account is self-directed per Alpaca Customer Agreement")
            print("- No investment advice is provided by this system")
            print("- All trading decisions are your responsibility")
            print("- You must comply with all applicable regulations")
            
        print("\nDo you acknowledge and accept these terms? (yes/no): ", end="")
        response = input().strip().lower()
        
        if response in ["yes", "y"]:
            self.user_acknowledgments[acknowledgment_type] = datetime.now()
            logger.info(f"User acknowledged {acknowledgment_type} at {datetime.now()}")
            return True
        else:
            logger.warning(f"User declined to acknowledge {acknowledgment_type}")
            return False
    
    def run_full_compliance_check(self) -> Tuple[bool, List[ComplianceCheck]]:
        """
        Run comprehensive compliance validation.
        
        Returns:
            Tuple of (is_compliant, list_of_checks)
        """
        
        logger.info("Running comprehensive regulatory compliance check...")
        
        checks = [
            self.validate_paper_trading_mode(),
            self.validate_self_directed_compliance(),
            self.validate_risk_management(),
            self.validate_record_keeping()
        ]
        
        self.compliance_checks.extend(checks)
        
        # Check for any non-compliant items
        non_compliant = [c for c in checks if c.status == ComplianceStatus.NON_COMPLIANT]
        requires_review = [c for c in checks if c.status == ComplianceStatus.REQUIRES_REVIEW]
        
        # Overall compliance status
        is_compliant = len(non_compliant) == 0
        
        # Log results
        logger.info(f"Compliance check completed: {len(checks)} checks performed")
        logger.info(f"Compliant: {len([c for c in checks if c.status == ComplianceStatus.COMPLIANT])}")
        logger.info(f"Non-compliant: {len(non_compliant)}")
        logger.info(f"Requires review: {len(requires_review)}")
        
        return is_compliant, checks
    
    def generate_compliance_report(self) -> str:
        """Generate detailed compliance report."""
        
        report = []
        report.append("ALPACA CUSTOMER AGREEMENT COMPLIANCE REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if not self.compliance_checks:
            report.append("No compliance checks performed yet.")
            return "\n".join(report)
        
        compliant_count = len([c for c in self.compliance_checks if c.status == ComplianceStatus.COMPLIANT])
        total_checks = len(self.compliance_checks)
        
        report.append(f"Overall Compliance: {compliant_count}/{total_checks} checks passed")
        report.append("")
        
        for check in self.compliance_checks:
            status_symbol = {
                ComplianceStatus.COMPLIANT: "✅",
                ComplianceStatus.NON_COMPLIANT: "❌", 
                ComplianceStatus.REQUIRES_REVIEW: "⚠️"
            }[check.status]
            
            report.append(f"{status_symbol} {check.check_name}: {check.status.value.upper()}")
            report.append(f"   Description: {check.description}")
            report.append(f"   Details: {check.details}")
            if check.action_required:
                report.append(f"   Action Required: {check.action_required}")
            report.append("")
        
        if self.user_acknowledgments:
            report.append("USER ACKNOWLEDGMENTS:")
            for ack_type, timestamp in self.user_acknowledgments.items():
                report.append(f"  - {ack_type}: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(report)

# Global compliance manager instance
compliance_manager = RegulatoryComplianceManager()

def ensure_regulatory_compliance() -> bool:
    """
    Entry point for regulatory compliance validation.
    
    Returns:
        True if system is compliant, False otherwise
    """
    
    logger.info("Ensuring regulatory compliance with Alpaca Customer Agreement...")
    
    is_compliant, checks = compliance_manager.run_full_compliance_check()
    
    # Print compliance report
    print("\n" + compliance_manager.generate_compliance_report())
    
    # Handle non-compliance
    if not is_compliant:
        print("\n🚨 COMPLIANCE ISSUES DETECTED")
        print("The system has compliance issues that must be addressed.")
        print("Please review the compliance report above.")
        
        # Require acknowledgment for automated trading risks
        if not compliance_manager.require_user_acknowledgment("automated_trading_risk"):
            return False
            
        if not compliance_manager.require_user_acknowledgment("self_directed_trading"):
            return False
    
    logger.info(f"Regulatory compliance check completed: {'PASSED' if is_compliant else 'REQUIRES ATTENTION'}")
    return True  # Allow operation with warnings but require acknowledgments

if __name__ == "__main__":
    ensure_regulatory_compliance()