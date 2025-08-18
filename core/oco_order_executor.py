#!/usr/bin/env python3
"""
OCO Order Execution System

Executes OCO orders as part of the portfolio rebalancing process,
integrating with the existing trading engine and risk management systems.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from tools.alpaca_client import alpaca_client
from core.oco_order_manager import oco_manager, OCOType
from core.portfolio_balancer import RebalanceDecision, PositionAction
from config.settings import settings

logger = logging.getLogger(__name__)

class OCOOrderExecutor:
    """Executes OCO orders with proper risk management and monitoring."""
    
    def __init__(self):
        """Initialize OCO order executor."""
        self.execution_history: List[Dict] = []
        self.failed_executions: List[Dict] = []
        
    async def execute_rebalance_decisions(self, decisions: List[RebalanceDecision]) -> List[Dict]:
        """
        Execute a list of rebalancing decisions, handling OCO orders appropriately.
        
        Args:
            decisions: List of RebalanceDecision objects
            
        Returns:
            List of execution results
        """
        execution_results = []
        
        logger.info(f"Executing {len(decisions)} rebalancing decisions")
        
        for decision in decisions:
            try:
                result = await self._execute_single_decision(decision)
                execution_results.append(result)
                
                # Add delay between orders to avoid rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to execute decision for {decision.symbol}: {e}")
                execution_results.append({
                    "symbol": decision.symbol,
                    "success": False,
                    "error": str(e),
                    "decision": decision
                })
        
        # Log execution summary
        successful_executions = [r for r in execution_results if r.get("success", False)]
        failed_executions = [r for r in execution_results if not r.get("success", False)]
        
        logger.info(f"Execution complete: {len(successful_executions)} successful, {len(failed_executions)} failed")
        
        return execution_results
    
    async def _execute_single_decision(self, decision: RebalanceDecision) -> Dict:
        """Execute a single rebalancing decision."""
        start_time = datetime.now()
        
        try:
            # Check if this is an OCO order
            if decision.use_oco and decision.oco_type:
                return await self._execute_oco_decision(decision, start_time)
            else:
                return await self._execute_standard_decision(decision, start_time)
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            error_result = {
                "symbol": decision.symbol,
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "decision": decision,
                "timestamp": datetime.now().isoformat()
            }
            
            self.failed_executions.append(error_result)
            return error_result
    
    async def _execute_oco_decision(self, decision: RebalanceDecision, start_time: datetime) -> Dict:
        """Execute an OCO-based rebalancing decision."""
        
        logger.info(f"Executing OCO {decision.oco_type} order for {decision.symbol}")
        
        try:
            if decision.oco_type == "bracket":
                # Execute bracket OCO order
                oco_order = await oco_manager.place_bracket_oco(
                    symbol=decision.symbol,
                    quantity=decision.quantity,
                    side=decision.action.value,
                    take_profit_percent=None,  # Use calculated prices
                    stop_loss_percent=None,
                    reasoning=decision.reasoning
                )
                
                if oco_order:
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    result = {
                        "symbol": decision.symbol,
                        "success": True,
                        "oco_order_id": oco_order.oco_id,
                        "parent_order_id": oco_order.parent_order_id,
                        "take_profit_price": oco_order.take_profit_price,
                        "stop_loss_price": oco_order.stop_loss_price,
                        "execution_time": execution_time,
                        "order_type": "oco_bracket",
                        "decision": decision,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    self.execution_history.append(result)
                    return result
                else:
                    raise Exception("OCO bracket order placement failed")
            
            elif decision.oco_type == "breakout":
                # Execute breakout OCO order
                upper_breakout_pct = ((decision.take_profit_price / decision.limit_price) - 1) if decision.limit_price else 0.02
                lower_breakout_pct = (1 - (decision.stop_loss_price / decision.limit_price)) if decision.limit_price else 0.02
                
                oco_order = await oco_manager.place_breakout_oco(
                    symbol=decision.symbol,
                    quantity=decision.quantity,
                    upper_breakout_percent=upper_breakout_pct,
                    lower_breakout_percent=lower_breakout_pct,
                    reasoning=decision.reasoning
                )
                
                if oco_order:
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    result = {
                        "symbol": decision.symbol,
                        "success": True,
                        "oco_order_id": oco_order.oco_id,
                        "buy_order_id": oco_order.buy_order_id,
                        "sell_order_id": oco_order.sell_order_id,
                        "upper_breakout_price": oco_order.upper_breakout_price,
                        "lower_breakout_price": oco_order.lower_breakout_price,
                        "execution_time": execution_time,
                        "order_type": "oco_breakout",
                        "decision": decision,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    self.execution_history.append(result)
                    return result
                else:
                    raise Exception("OCO breakout order placement failed")
            
            else:
                raise Exception(f"Unsupported OCO type: {decision.oco_type}")
        
        except Exception as e:
            logger.error(f"OCO execution failed for {decision.symbol}: {e}")
            raise
    
    async def _execute_standard_decision(self, decision: RebalanceDecision, start_time: datetime) -> Dict:
        """Execute a standard (non-OCO) rebalancing decision."""
        
        logger.info(f"Executing standard {decision.order_type} order for {decision.symbol}")
        
        try:
            # Map PositionAction to string for alpaca_client
            side_mapping = {
                PositionAction.BUY: "buy",
                PositionAction.SELL: "sell",
                PositionAction.SHORT: "sell_short",
                PositionAction.COVER: "buy"
            }
            
            order_side = side_mapping.get(decision.action, "buy")
            
            # Place standard order
            order_result = alpaca_client.place_order(
                symbol=decision.symbol,
                qty=decision.quantity,
                side=order_side,
                order_type=decision.order_type,
                limit_price=decision.limit_price,
                stop_price=decision.stop_price,
                time_in_force="gtc"
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "symbol": decision.symbol,
                "success": True,
                "order_id": order_result["id"],
                "quantity": order_result["qty"],
                "side": order_result["side"],
                "order_type": order_result["order_type"],
                "status": order_result["status"],
                "execution_time": execution_time,
                "decision": decision,
                "timestamp": datetime.now().isoformat()
            }
            
            self.execution_history.append(result)
            return result
        
        except Exception as e:
            logger.error(f"Standard order execution failed for {decision.symbol}: {e}")
            raise
    
    def get_execution_summary(self) -> Dict:
        """Get summary of order execution performance."""
        
        total_executions = len(self.execution_history)
        failed_executions = len(self.failed_executions)
        
        if total_executions == 0:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "oco_orders": 0,
                "standard_orders": 0
            }
        
        # Calculate success rate
        success_rate = total_executions / (total_executions + failed_executions) if (total_executions + failed_executions) > 0 else 0.0
        
        # Calculate average execution time
        execution_times = [r.get("execution_time", 0) for r in self.execution_history]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
        
        # Count order types
        oco_orders = len([r for r in self.execution_history if r.get("order_type", "").startswith("oco")])
        standard_orders = total_executions - oco_orders
        
        return {
            "total_executions": total_executions,
            "failed_executions": failed_executions,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "oco_orders": oco_orders,
            "standard_orders": standard_orders,
            "oco_percentage": (oco_orders / total_executions * 100) if total_executions > 0 else 0.0
        }
    
    async def monitor_oco_executions(self, check_interval: int = 60) -> None:
        """Monitor OCO order executions and update status."""
        
        logger.info("Starting OCO execution monitoring")
        
        while True:
            try:
                await asyncio.sleep(check_interval)
                
                # Get active OCO orders from execution history
                active_oco_results = [
                    r for r in self.execution_history 
                    if r.get("order_type", "").startswith("oco") and 
                       r.get("success", False)
                ]
                
                for result in active_oco_results:
                    oco_order_id = result.get("oco_order_id")
                    if oco_order_id:
                        # Check OCO status
                        oco_status = await oco_manager.get_oco_status(oco_order_id)
                        if oco_status:
                            # Update result with latest status
                            result["oco_status"] = oco_status.get("status")
                            result["last_checked"] = datetime.now().isoformat()
                
                logger.debug(f"Monitored {len(active_oco_results)} active OCO executions")
            
            except asyncio.CancelledError:
                logger.info("OCO execution monitoring stopped")
                break
            except Exception as e:
                logger.error(f"Error in OCO execution monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

# Global OCO executor instance
oco_executor = OCOOrderExecutor()

# Convenience functions
async def execute_rebalancing_with_oco(decisions: List[RebalanceDecision]) -> List[Dict]:
    """Execute rebalancing decisions with OCO support."""
    return await oco_executor.execute_rebalance_decisions(decisions)

async def get_oco_execution_summary() -> Dict:
    """Get OCO execution performance summary."""
    return oco_executor.get_execution_summary()

async def start_oco_execution_monitoring():
    """Start OCO execution monitoring."""
    asyncio.create_task(oco_executor.monitor_oco_executions())

__all__ = [
    'OCOOrderExecutor', 'oco_executor', 'execute_rebalancing_with_oco',
    'get_oco_execution_summary', 'start_oco_execution_monitoring'
]