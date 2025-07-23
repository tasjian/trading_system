#!/usr/bin/env python3
"""
High Frequency Trading Compliance Module

Specialized compliance checks for HFT operations that maintain $25,000 minimum
balance and operate under different regulatory requirements than standard accounts.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from config.hft_settings import hft_settings
from tools.alpaca_client import alpaca_client

logger = logging.getLogger(__name__)

@dataclass
class HFTComplianceResult:
    """Result of HFT compliance check."""
    compliant: bool
    violations: List[str]
    warnings: List[str]
    checks_performed: int
    compliance_score: float
    recommendations: List[str]

class HFTComplianceManager:
    """Manages compliance for high-frequency trading operations."""
    
    def __init__(self):
        """Initialize HFT compliance manager."""
        self.compliance_history: List[HFTComplianceResult] = []
        self.violation_count: Dict[str, int] = {}
        self.last_check_time: Optional[datetime] = None
    
    async def perform_hft_compliance_check(self, account_balance: float, 
                                         positions: Dict, 
                                         recent_trades: List[Dict]) -> HFTComplianceResult:
        """Perform comprehensive HFT compliance check."""
        
        logger.info("🔍 Performing HFT compliance check...")
        
        violations = []
        warnings = []
        recommendations = []
        checks_performed = 0
        
        try:
            # 1. Pattern Day Trader (PDT) Compliance
            pdt_result = await self._check_pdt_compliance(account_balance)
            checks_performed += 1
            if not pdt_result["compliant"]:
                violations.extend(pdt_result["violations"])
            if pdt_result["warnings"]:
                warnings.extend(pdt_result["warnings"])
            if pdt_result["recommendations"]:
                recommendations.extend(pdt_result["recommendations"])
            
            # 2. Minimum Balance Maintenance
            balance_result = self._check_minimum_balance(account_balance)
            checks_performed += 1
            if not balance_result["compliant"]:
                violations.extend(balance_result["violations"])
            if balance_result["warnings"]:
                warnings.extend(balance_result["warnings"])
            
            # 3. Position Size Limits
            position_result = self._check_position_limits(account_balance, positions)
            checks_performed += 1
            if not position_result["compliant"]:
                violations.extend(position_result["violations"])
            if position_result["warnings"]:
                warnings.extend(position_result["warnings"])
            
            # 4. Trading Frequency Limits
            frequency_result = self._check_trading_frequency(recent_trades)
            checks_performed += 1
            if not frequency_result["compliant"]:
                violations.extend(frequency_result["violations"])
            if frequency_result["warnings"]:
                warnings.extend(frequency_result["warnings"])
            
            # 5. Risk Management Compliance
            risk_result = self._check_risk_management(account_balance, positions)
            checks_performed += 1
            if not risk_result["compliant"]:
                violations.extend(risk_result["violations"])
            if risk_result["warnings"]:
                warnings.extend(risk_result["warnings"])
            
            # 6. Wash Sale Prevention
            wash_sale_result = self._check_wash_sale_prevention(recent_trades)
            checks_performed += 1
            if not wash_sale_result["compliant"]:
                violations.extend(wash_sale_result["violations"])
            if wash_sale_result["warnings"]:
                warnings.extend(wash_sale_result["warnings"])
            
            # Calculate compliance score
            compliance_score = 1.0 - (len(violations) / max(checks_performed, 1))
            compliant = len(violations) == 0
            
            # Generate recommendations
            if compliance_score < 1.0:
                recommendations.extend(self._generate_compliance_recommendations(violations, warnings))
            
            result = HFTComplianceResult(
                compliant=compliant,
                violations=violations,
                warnings=warnings,
                checks_performed=checks_performed,
                compliance_score=compliance_score,
                recommendations=recommendations
            )
            
            # Update compliance history
            self.compliance_history.append(result)
            self.last_check_time = datetime.now()
            
            # Update violation counts
            for violation in violations:
                self.violation_count[violation] = self.violation_count.get(violation, 0) + 1
            
            # Log results
            self._log_compliance_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"HFT compliance check error: {e}")
            return HFTComplianceResult(
                compliant=False,
                violations=[f"Compliance check failed: {e}"],
                warnings=[],
                checks_performed=checks_performed,
                compliance_score=0.0,
                recommendations=["Contact compliance team immediately"]
            )
    
    async def _check_pdt_compliance(self, account_balance: float) -> Dict:
        """Check Pattern Day Trader compliance requirements."""
        violations = []
        warnings = []
        recommendations = []
        
        try:
            # Get account information
            account_info = alpaca_client.get_account_info()
            
            # Check if account has PDT status
            is_pdt = account_info.get("pattern_day_trader", False)
            day_trade_count = account_info.get("day_trade_count", 0)
            
            # For HFT, we need PDT status OR account balance >= $25,000
            if not is_pdt and account_balance < 25000:
                violations.append("Account lacks PDT status and balance below $25,000 for unlimited day trading")
                recommendations.append("Maintain account balance above $25,000 or obtain PDT designation")
            
            # Check day trade buying power
            buying_power = account_info.get("buying_power", 0)
            day_trade_buying_power = account_info.get("daytrading_buying_power", buying_power)
            
            if day_trade_buying_power < account_balance * 4:  # Should have 4:1 leverage for PDT
                warnings.append(f"Day trading buying power may be limited: ${day_trade_buying_power:,.2f}")
                recommendations.append("Verify day trading buying power is adequate for HFT operations")
            
            # Check recent day trade count
            if day_trade_count >= 3 and not is_pdt:
                warnings.append(f"Day trade count: {day_trade_count} - approaching PDT designation")
            
            return {
                "compliant": len(violations) == 0,
                "violations": violations,
                "warnings": warnings,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"PDT compliance check error: {e}")
            return {
                "compliant": False,
                "violations": [f"PDT check failed: {e}"],
                "warnings": [],
                "recommendations": ["Verify account status with broker"]
            }
    
    def _check_minimum_balance(self, account_balance: float) -> Dict:
        """Check minimum balance requirements for HFT."""
        violations = []
        warnings = []
        
        # Critical minimum balance check
        if account_balance < hft_settings.minimum_account_balance:
            violations.append(f"Account balance ${account_balance:,.2f} below required minimum ${hft_settings.minimum_account_balance:,.2f}")
        
        # Warning for approaching minimum
        warning_threshold = hft_settings.minimum_account_balance * 1.1  # 10% buffer
        if account_balance < warning_threshold:
            warnings.append(f"Account balance approaching minimum: ${account_balance:,.2f} (minimum: ${hft_settings.minimum_account_balance:,.2f})")
        
        # Check cash reserve
        # Note: This is approximate since we don't have cash breakdown here
        estimated_cash_reserve = account_balance * 0.1  # Assume 10% cash
        if estimated_cash_reserve < hft_settings.minimum_cash_reserve:
            warnings.append(f"Cash reserve may be insufficient for HFT operations")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "recommendations": []
        }
    
    def _check_position_limits(self, account_balance: float, positions: Dict) -> Dict:
        """Check position size and concentration limits."""
        violations = []
        warnings = []
        
        try:
            total_positions = len(positions)
            max_position_value = 0
            total_exposure = 0
            
            for symbol, position in positions.items():
                market_value = abs(position.get("market_value", 0))
                total_exposure += market_value
                max_position_value = max(max_position_value, market_value)
            
            # Check maximum positions
            if total_positions > hft_settings.max_positions:
                violations.append(f"Too many positions: {total_positions} (max: {hft_settings.max_positions})")
            
            # Check individual position size
            if account_balance > 0:
                max_position_pct = max_position_value / account_balance
                if max_position_pct > hft_settings.max_position_size:
                    violations.append(f"Position size exceeded: {max_position_pct:.2%} (max: {hft_settings.max_position_size:.2%})")
            
            # Check total exposure
            if account_balance > 0:
                exposure_pct = total_exposure / account_balance
                if exposure_pct > hft_settings.max_account_utilization:
                    violations.append(f"Account utilization too high: {exposure_pct:.2%} (max: {hft_settings.max_account_utilization:.2%})")
            
            # Warning for high concentration
            if total_positions > 0:
                avg_position_size = total_exposure / total_positions
                if avg_position_size < hft_settings.min_position_size:
                    warnings.append(f"Average position size very small: ${avg_position_size:.2f}")
            
            return {
                "compliant": len(violations) == 0,
                "violations": violations,
                "warnings": warnings,
                "recommendations": []
            }
            
        except Exception as e:
            logger.error(f"Position limits check error: {e}")
            return {
                "compliant": False,
                "violations": [f"Position limits check failed: {e}"],
                "warnings": [],
                "recommendations": []
            }
    
    def _check_trading_frequency(self, recent_trades: List[Dict]) -> Dict:
        """Check trading frequency limits."""
        violations = []
        warnings = []
        
        try:
            now = datetime.now()
            
            # Count trades in different time windows
            trades_last_minute = 0
            trades_last_hour = 0
            trades_today = 0
            
            for trade in recent_trades:
                trade_time = trade.get("timestamp", now)
                if isinstance(trade_time, str):
                    trade_time = datetime.fromisoformat(trade_time.replace('Z', '+00:00'))
                
                time_diff = now - trade_time
                
                if time_diff <= timedelta(minutes=1):
                    trades_last_minute += 1
                if time_diff <= timedelta(hours=1):
                    trades_last_hour += 1
                if time_diff <= timedelta(days=1):
                    trades_today += 1
            
            # Check frequency limits
            if trades_last_minute > hft_settings.max_trades_per_minute:
                violations.append(f"Too many trades per minute: {trades_last_minute} (max: {hft_settings.max_trades_per_minute})")
            
            if trades_last_hour > hft_settings.max_trades_per_hour:
                violations.append(f"Too many trades per hour: {trades_last_hour} (max: {hft_settings.max_trades_per_hour})")
            
            if trades_today > hft_settings.max_trades_per_day:
                violations.append(f"Too many trades today: {trades_today} (max: {hft_settings.max_trades_per_day})")
            
            # Warnings for approaching limits
            if trades_last_hour > hft_settings.max_trades_per_hour * 0.8:
                warnings.append(f"Approaching hourly trade limit: {trades_last_hour}/{hft_settings.max_trades_per_hour}")
            
            if trades_today > hft_settings.max_trades_per_day * 0.8:
                warnings.append(f"Approaching daily trade limit: {trades_today}/{hft_settings.max_trades_per_day}")
            
            return {
                "compliant": len(violations) == 0,
                "violations": violations,
                "warnings": warnings,
                "recommendations": []
            }
            
        except Exception as e:
            logger.error(f"Trading frequency check error: {e}")
            return {
                "compliant": False,
                "violations": [f"Trading frequency check failed: {e}"],
                "warnings": [],
                "recommendations": []
            }
    
    def _check_risk_management(self, account_balance: float, positions: Dict) -> Dict:
        """Check risk management compliance."""
        violations = []
        warnings = []
        
        try:
            # Check if risk monitoring is enabled
            if not hft_settings.risk_limit_monitoring:
                violations.append("Risk limit monitoring is disabled")
            
            # Check position limit monitoring
            if not hft_settings.position_limit_monitoring:
                violations.append("Position limit monitoring is disabled")
            
            # Estimate current drawdown (simplified)
            total_unrealized_pnl = sum(pos.get("unrealized_pl", 0) for pos in positions.values())
            if total_unrealized_pnl < 0:
                drawdown_pct = abs(total_unrealized_pnl) / account_balance
                if drawdown_pct > hft_settings.max_drawdown:
                    violations.append(f"Maximum drawdown exceeded: {drawdown_pct:.2%} (max: {hft_settings.max_drawdown:.2%})")
                elif drawdown_pct > hft_settings.max_drawdown * 0.8:
                    warnings.append(f"Approaching maximum drawdown: {drawdown_pct:.2%}")
            
            return {
                "compliant": len(violations) == 0,
                "violations": violations,
                "warnings": warnings,
                "recommendations": []
            }
            
        except Exception as e:
            logger.error(f"Risk management check error: {e}")
            return {
                "compliant": False,
                "violations": [f"Risk management check failed: {e}"],
                "warnings": [],
                "recommendations": []
            }
    
    def _check_wash_sale_prevention(self, recent_trades: List[Dict]) -> Dict:
        """Check wash sale prevention compliance."""
        violations = []
        warnings = []
        
        try:
            if not hft_settings.wash_sale_prevention:
                violations.append("Wash sale prevention is disabled")
                return {
                    "compliant": False,
                    "violations": violations,
                    "warnings": warnings,
                    "recommendations": ["Enable wash sale prevention"]
                }
            
            # Simple wash sale detection (buy and sell same symbol within 30 days)
            symbol_trades = {}
            now = datetime.now()
            
            for trade in recent_trades:
                symbol = trade.get("symbol", "")
                side = trade.get("side", "")
                trade_time = trade.get("timestamp", now)
                
                if isinstance(trade_time, str):
                    trade_time = datetime.fromisoformat(trade_time.replace('Z', '+00:00'))
                
                # Only check trades within last 30 days
                if (now - trade_time).days <= 30:
                    if symbol not in symbol_trades:
                        symbol_trades[symbol] = {"buy": [], "sell": []}
                    
                    symbol_trades[symbol][side].append(trade_time)
            
            # Check for potential wash sales
            for symbol, trades in symbol_trades.items():
                buy_times = trades.get("buy", [])
                sell_times = trades.get("sell", [])
                
                for buy_time in buy_times:
                    for sell_time in sell_times:
                        time_diff = abs((buy_time - sell_time).days)
                        if time_diff <= 30:
                            warnings.append(f"Potential wash sale pattern detected for {symbol}")
                            break
            
            return {
                "compliant": len(violations) == 0,
                "violations": violations,
                "warnings": warnings,
                "recommendations": []
            }
            
        except Exception as e:
            logger.error(f"Wash sale check error: {e}")
            return {
                "compliant": False,
                "violations": [f"Wash sale check failed: {e}"],
                "warnings": [],
                "recommendations": []
            }
    
    def _generate_compliance_recommendations(self, violations: List[str], warnings: List[str]) -> List[str]:
        """Generate compliance recommendations based on violations and warnings."""
        recommendations = []
        
        # Balance-related recommendations
        if any("balance" in v.lower() for v in violations):
            recommendations.append("Consider depositing additional funds to maintain minimum balance")
            recommendations.append("Reduce position sizes to conserve capital")
        
        # Position-related recommendations
        if any("position" in v.lower() for v in violations):
            recommendations.append("Reduce individual position sizes")
            recommendations.append("Implement better position sizing algorithms")
            recommendations.append("Consider closing some positions to reduce concentration")
        
        # Trading frequency recommendations
        if any("trades" in v.lower() for v in violations):
            recommendations.append("Implement trade throttling mechanisms")
            recommendations.append("Improve signal quality to reduce trade frequency")
            recommendations.append("Consider increasing minimum time between trades")
        
        # Risk management recommendations
        if any("risk" in v.lower() or "drawdown" in v.lower() for v in violations):
            recommendations.append("Tighten stop-loss parameters")
            recommendations.append("Reduce overall portfolio risk exposure")
            recommendations.append("Implement additional risk monitoring")
        
        # General recommendations for warnings
        if warnings:
            recommendations.append("Monitor compliance metrics more frequently")
            recommendations.append("Consider implementing preventive measures before limits are reached")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _log_compliance_results(self, result: HFTComplianceResult) -> None:
        """Log compliance check results."""
        
        if result.compliant:
            logger.info("✅ HFT Compliance Check: PASSED")
        else:
            logger.warning("❌ HFT Compliance Check: FAILED")
        
        logger.info(f"   Checks Performed: {result.checks_performed}")
        logger.info(f"   Compliance Score: {result.compliance_score:.2%}")
        
        if result.violations:
            logger.warning(f"   Violations ({len(result.violations)}):")
            for violation in result.violations:
                logger.warning(f"     • {violation}")
        
        if result.warnings:
            logger.info(f"   Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                logger.info(f"     • {warning}")
        
        if result.recommendations:
            logger.info(f"   Recommendations ({len(result.recommendations)}):")
            for rec in result.recommendations:
                logger.info(f"     • {rec}")
    
    def get_compliance_summary(self) -> Dict:
        """Get summary of compliance history."""
        if not self.compliance_history:
            return {"status": "No compliance checks performed"}
        
        recent_results = self.compliance_history[-10:]  # Last 10 checks
        avg_score = sum(r.compliance_score for r in recent_results) / len(recent_results)
        
        return {
            "total_checks": len(self.compliance_history),
            "recent_average_score": avg_score,
            "last_check": self.last_check_time,
            "frequent_violations": dict(sorted(self.violation_count.items(), key=lambda x: x[1], reverse=True)[:5]),
            "compliance_trend": "improving" if len(recent_results) > 1 and recent_results[-1].compliance_score > recent_results[-2].compliance_score else "declining"
        }

# Global HFT compliance manager
hft_compliance_manager = HFTComplianceManager()