"""Enhanced risk control and circuit breaker system."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from tools.alpaca_client import alpaca_client
from config.settings import settings

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskAlert:
    level: RiskLevel
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False

class RiskMonitor:
    """Advanced risk monitoring and circuit breaker system."""
    
    def __init__(self):
        self.alerts: List[RiskAlert] = []
        self.circuit_breakers = {
            "daily_loss_limit": False,
            "position_concentration": False,
            "volatility_spike": False,
            "correlation_risk": False,
            "liquidity_risk": False,
            "margin_call": False
        }
        
        self.risk_thresholds = {
            "max_daily_loss": settings.max_daily_loss,
            "max_position_size": settings.max_position_size,
            "max_portfolio_risk": settings.max_portfolio_risk,
            "max_correlation": 0.8,
            "min_liquidity_ratio": 0.1,
            "max_volatility": 0.5,
            "max_drawdown": 0.15
        }
        
        self.trading_halt_active = False
        self.last_assessment = None
    
    def assess_portfolio_risk(self) -> Dict[str, any]:
        """Comprehensive portfolio risk assessment."""
        try:
            assessment = {
                "timestamp": datetime.now(),
                "overall_risk": RiskLevel.LOW,
                "risk_score": 0.0,
                "circuit_breakers": self.circuit_breakers.copy(),
                "metrics": {},
                "alerts": [],
                "recommendations": []
            }
            
            # Get current portfolio data
            account_info = alpaca_client.get_account_info()
            positions = alpaca_client.get_positions()
            
            portfolio_value = account_info["equity"]
            cash = account_info["cash"]
            
            # Risk Metric 1: Daily Loss Assessment
            daily_loss_risk = self._assess_daily_loss(account_info, assessment)
            
            # Risk Metric 2: Position Concentration
            concentration_risk = self._assess_concentration(positions, portfolio_value, assessment)
            
            # Risk Metric 3: Liquidity Risk
            liquidity_risk = self._assess_liquidity(cash, portfolio_value, assessment)
            
            # Risk Metric 4: Correlation Risk
            correlation_risk = self._assess_correlation(positions, assessment)
            
            # Risk Metric 5: Volatility Risk
            volatility_risk = self._assess_volatility(positions, assessment)
            
            # Calculate overall risk score (0-100)
            risk_components = [daily_loss_risk, concentration_risk, liquidity_risk, 
                             correlation_risk, volatility_risk]
            assessment["risk_score"] = sum(risk_components) / len(risk_components)
            
            # Determine overall risk level
            if assessment["risk_score"] >= 80:
                assessment["overall_risk"] = RiskLevel.CRITICAL
            elif assessment["risk_score"] >= 60:
                assessment["overall_risk"] = RiskLevel.HIGH
            elif assessment["risk_score"] >= 40:
                assessment["overall_risk"] = RiskLevel.MEDIUM
            else:
                assessment["overall_risk"] = RiskLevel.LOW
            
            # Generate recommendations
            self._generate_recommendations(assessment)
            
            # Update circuit breakers
            self._update_circuit_breakers(assessment)
            
            self.last_assessment = assessment
            return assessment
            
        except Exception as e:
            logger.error(f"Error in portfolio risk assessment: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(),
                "overall_risk": RiskLevel.CRITICAL
            }
    
    def _assess_daily_loss(self, account_info: dict, assessment: dict) -> float:
        """Assess daily loss risk."""
        try:
            current_equity = account_info["equity"]
            
            # Use initial equity from session or estimate
            initial_equity = getattr(self, 'initial_equity', current_equity)
            if not hasattr(self, 'initial_equity'):
                self.initial_equity = current_equity
            
            if initial_equity <= 0:
                return 0.0
            
            daily_loss = (initial_equity - current_equity) / initial_equity
            max_daily_loss = self.risk_thresholds["max_daily_loss"]
            
            assessment["metrics"]["daily_loss"] = {
                "current": daily_loss,
                "threshold": max_daily_loss,
                "status": "CRITICAL" if daily_loss > max_daily_loss else "OK"
            }
            
            if daily_loss > max_daily_loss:
                alert = RiskAlert(
                    level=RiskLevel.CRITICAL,
                    message=f"Daily loss limit exceeded: {daily_loss:.2%}",
                    metric="daily_loss",
                    value=daily_loss,
                    threshold=max_daily_loss,
                    timestamp=datetime.now()
                )
                assessment["alerts"].append(alert)
                self.circuit_breakers["daily_loss_limit"] = True
                return 100.0
            
            # Scale risk score based on proximity to limit
            risk_score = min(100.0, (daily_loss / max_daily_loss) * 100)
            return max(0.0, risk_score)
            
        except Exception as e:
            logger.error(f"Error assessing daily loss: {e}")
            return 50.0  # Medium risk on error
    
    def _assess_concentration(self, positions: List[dict], portfolio_value: float, 
                           assessment: dict) -> float:
        """Assess position concentration risk."""
        try:
            if portfolio_value <= 0 or not positions:
                assessment["metrics"]["concentration"] = {"status": "OK", "max_exposure": 0.0}
                return 0.0
            
            # Calculate position exposures
            exposures = {}
            max_exposure = 0.0
            
            for pos in positions:
                symbol = pos["symbol"]
                market_value = abs(pos["market_value"])
                exposure = market_value / portfolio_value
                exposures[symbol] = exposure
                max_exposure = max(max_exposure, exposure)
            
            max_position_size = self.risk_thresholds["max_position_size"]
            
            assessment["metrics"]["concentration"] = {
                "exposures": exposures,
                "max_exposure": max_exposure,
                "threshold": max_position_size,
                "status": "CRITICAL" if max_exposure > max_position_size else "OK"
            }
            
            if max_exposure > max_position_size:
                largest_position = max(exposures, key=exposures.get)
                alert = RiskAlert(
                    level=RiskLevel.HIGH,
                    message=f"Position concentration risk: {largest_position} at {max_exposure:.2%}",
                    metric="concentration",
                    value=max_exposure,
                    threshold=max_position_size,
                    timestamp=datetime.now()
                )
                assessment["alerts"].append(alert)
                self.circuit_breakers["position_concentration"] = True
                return 80.0
            
            # Scale risk score
            risk_score = min(100.0, (max_exposure / max_position_size) * 80)
            return risk_score
            
        except Exception as e:
            logger.error(f"Error assessing concentration: {e}")
            return 40.0
    
    def _assess_liquidity(self, cash: float, portfolio_value: float, assessment: dict) -> float:
        """Assess liquidity risk."""
        try:
            if portfolio_value <= 0:
                assessment["metrics"]["liquidity"] = {"status": "OK", "ratio": 1.0}
                return 0.0
            
            liquidity_ratio = cash / portfolio_value
            min_liquidity = self.risk_thresholds["min_liquidity_ratio"]
            
            assessment["metrics"]["liquidity"] = {
                "cash": cash,
                "ratio": liquidity_ratio,
                "threshold": min_liquidity,
                "status": "CRITICAL" if liquidity_ratio < min_liquidity else "OK"
            }
            
            if liquidity_ratio < min_liquidity:
                alert = RiskAlert(
                    level=RiskLevel.MEDIUM,
                    message=f"Low liquidity: {liquidity_ratio:.2%} cash ratio",
                    metric="liquidity",
                    value=liquidity_ratio,
                    threshold=min_liquidity,
                    timestamp=datetime.now()
                )
                assessment["alerts"].append(alert)
                self.circuit_breakers["liquidity_risk"] = True
                return 60.0
            
            # Higher cash ratio = lower risk
            risk_score = max(0.0, 100 * (1 - liquidity_ratio))
            return min(60.0, risk_score)
            
        except Exception as e:
            logger.error(f"Error assessing liquidity: {e}")
            return 30.0
    
    def _assess_correlation(self, positions: List[dict], assessment: dict) -> float:
        """Assess correlation risk between positions."""
        try:
            if len(positions) < 2:
                assessment["metrics"]["correlation"] = {"status": "OK", "max_correlation": 0.0}
                return 0.0
            
            # Simplified correlation assessment based on sector concentration
            # In a real system, you'd calculate actual price correlations
            symbols = [pos["symbol"] for pos in positions]
            
            # Simple heuristic: use sector information if available, otherwise assume moderate correlation
            try:
                # Get sector information dynamically from market data
                tech_count = 0
                if hasattr(self, 'alpaca_client') and self.alpaca_client:
                    # Would need sector data from Alpaca or another source
                    # For now, use a conservative estimate
                    tech_count = len(symbols) // 3  # Assume 1/3 might be tech-related
                else:
                    tech_count = len(symbols) // 3
            except Exception:
                tech_count = len(symbols) // 3  # Conservative fallback
            
            correlation_risk = tech_count / len(symbols) if len(symbols) > 0 else 0
            max_correlation = self.risk_thresholds["max_correlation"]
            
            assessment["metrics"]["correlation"] = {
                "estimated_correlation": correlation_risk,
                "tech_concentration": tech_count / len(symbols),
                "threshold": max_correlation,
                "status": "HIGH" if correlation_risk > max_correlation else "OK"
            }
            
            if correlation_risk > max_correlation:
                alert = RiskAlert(
                    level=RiskLevel.MEDIUM,
                    message=f"High correlation risk: {correlation_risk:.2%} in similar assets",
                    metric="correlation",
                    value=correlation_risk,
                    threshold=max_correlation,
                    timestamp=datetime.now()
                )
                assessment["alerts"].append(alert)
                return 50.0
            
            return correlation_risk * 60.0
            
        except Exception as e:
            logger.error(f"Error assessing correlation: {e}")
            return 20.0
    
    def _assess_volatility(self, positions: List[dict], assessment: dict) -> float:
        """Assess portfolio volatility risk."""
        try:
            if not positions:
                assessment["metrics"]["volatility"] = {"status": "OK", "estimate": 0.0}
                return 0.0
            
            # Get recent price data for volatility estimation
            total_volatility = 0.0
            valid_positions = 0
            
            for pos in positions:
                try:
                    symbol = pos["symbol"]
                    market_data = alpaca_client.get_market_data(symbol, limit=20)
                    
                    if len(market_data) >= 10:
                        returns = market_data["close"].pct_change().dropna()
                        volatility = returns.std() * (252 ** 0.5)  # Annualized
                        total_volatility += volatility
                        valid_positions += 1
                        
                except Exception:
                    continue
            
            avg_volatility = total_volatility / valid_positions if valid_positions > 0 else 0.0
            max_volatility = self.risk_thresholds["max_volatility"]
            
            assessment["metrics"]["volatility"] = {
                "estimated_volatility": avg_volatility,
                "threshold": max_volatility,
                "status": "HIGH" if avg_volatility > max_volatility else "OK"
            }
            
            if avg_volatility > max_volatility:
                alert = RiskAlert(
                    level=RiskLevel.MEDIUM,
                    message=f"High portfolio volatility: {avg_volatility:.2%}",
                    metric="volatility",
                    value=avg_volatility,
                    threshold=max_volatility,
                    timestamp=datetime.now()
                )
                assessment["alerts"].append(alert)
                return 70.0
            
            return min(70.0, (avg_volatility / max_volatility) * 70)
            
        except Exception as e:
            logger.error(f"Error assessing volatility: {e}")
            return 25.0
    
    def _generate_recommendations(self, assessment: dict):
        """Generate risk management recommendations."""
        recommendations = []
        risk_score = assessment["risk_score"]
        
        if risk_score >= 80:
            recommendations.append("IMMEDIATE ACTION: Consider closing positions to reduce risk")
            recommendations.append("Halt new position entries until risk reduces")
        elif risk_score >= 60:
            recommendations.append("Reduce position sizes on new trades")
            recommendations.append("Consider taking profits on winning positions")
        elif risk_score >= 40:
            recommendations.append("Monitor positions closely")
            recommendations.append("Consider defensive strategies")
        else:
            recommendations.append("Risk levels are acceptable")
            recommendations.append("Continue with normal trading operations")
        
        # Specific recommendations based on alerts
        for alert in assessment["alerts"]:
            if alert.metric == "daily_loss":
                recommendations.append("Stop all trading until tomorrow")
            elif alert.metric == "concentration":
                recommendations.append("Diversify portfolio to reduce concentration")
            elif alert.metric == "liquidity":
                recommendations.append("Increase cash reserves")
        
        assessment["recommendations"] = recommendations
    
    def _update_circuit_breakers(self, assessment: dict):
        """Update circuit breaker states based on assessment."""
        # Clear previous breakers
        for breaker in self.circuit_breakers:
            self.circuit_breakers[breaker] = False
        
        # Set breakers based on current assessment
        for alert in assessment["alerts"]:
            if alert.level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                if alert.metric == "daily_loss":
                    self.circuit_breakers["daily_loss_limit"] = True
                elif alert.metric == "concentration":
                    self.circuit_breakers["position_concentration"] = True
                elif alert.metric == "liquidity":
                    self.circuit_breakers["liquidity_risk"] = True
                elif alert.metric == "correlation":
                    self.circuit_breakers["correlation_risk"] = True
                elif alert.metric == "volatility":
                    self.circuit_breakers["volatility_spike"] = True
    
    def is_trading_halted(self) -> bool:
        """Check if trading should be halted due to risk controls."""
        critical_breakers = ["daily_loss_limit", "margin_call"]
        return any(self.circuit_breakers.get(breaker, False) for breaker in critical_breakers)
    
    def can_place_trade(self, symbol: str, quantity: float, side: str) -> Tuple[bool, str]:
        """Check if a specific trade can be placed given current risk state."""
        try:
            if self.is_trading_halted():
                return False, "Trading halted due to risk controls"
            
            # Perform trade-specific risk checks
            account_info = alpaca_client.get_account_info()
            portfolio_value = account_info["equity"]
            
            # Get current price estimate
            market_data = alpaca_client.get_market_data(symbol, limit=1)
            if market_data.empty:
                return False, f"Cannot get price data for {symbol}"
            
            current_price = market_data.iloc[-1]["close"]
            trade_value = quantity * current_price
            
            # Check position size limit
            if side.lower() == "buy":
                position_size_pct = trade_value / portfolio_value if portfolio_value > 0 else 0
                if position_size_pct > self.risk_thresholds["max_position_size"]:
                    return False, f"Position size too large: {position_size_pct:.2%}"
            
            # Check available buying power
            if side.lower() == "buy" and trade_value > account_info["buying_power"]:
                return False, "Insufficient buying power"
            
            return True, "Trade approved"
            
        except Exception as e:
            logger.error(f"Error checking trade approval: {e}")
            return False, f"Error in risk check: {e}"
    
    def get_risk_summary(self) -> dict:
        """Get a summary of current risk status."""
        if not self.last_assessment:
            return {"status": "No assessment available"}
        
        assessment = self.last_assessment
        return {
            "overall_risk": assessment["overall_risk"].value,
            "risk_score": assessment["risk_score"],
            "trading_halted": self.is_trading_halted(),
            "active_alerts": len(assessment["alerts"]),
            "circuit_breakers": sum(self.circuit_breakers.values()),
            "last_update": assessment["timestamp"].isoformat()
        }

# Global risk monitor instance
risk_monitor = RiskMonitor()