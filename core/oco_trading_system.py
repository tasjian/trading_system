#!/usr/bin/env python3
"""
Comprehensive OCO Trading System Integration

This module provides the main interface for OCO trading functionality,
integrating all OCO components with the existing trading system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from tools.alpaca_client import alpaca_client
from core.oco_order_manager import oco_manager, OCOConfiguration, OCOType, OCOStatus
from core.oco_order_executor import oco_executor
from core.portfolio_balancer import portfolio_balancer, RebalanceDecision
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class OCOTradingMetrics:
    """Metrics for OCO trading performance."""
    total_oco_orders: int = 0
    active_oco_orders: int = 0
    completed_oco_orders: int = 0
    cancelled_oco_orders: int = 0
    success_rate: float = 0.0
    average_profit_loss_ratio: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    average_holding_time_hours: float = 0.0

class OCOTradingSystem:
    """
    Comprehensive OCO trading system that integrates all OCO functionality
    with the existing algorithmic trading infrastructure.
    """
    
    def __init__(self):
        """Initialize the OCO trading system."""
        self.is_initialized = False
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.system_metrics = OCOTradingMetrics()
        
        # Risk management integration
        self.risk_limits = {
            "max_concurrent_oco_orders": 50,
            "max_oco_exposure_percent": 0.30,  # 30% of portfolio in OCO orders
            "min_time_between_oco_minutes": 5,  # Minimum time between OCO orders for same symbol
        }
        
        # Last OCO order timestamp per symbol
        self.last_oco_timestamp: Dict[str, datetime] = {}
        
    async def initialize(self) -> bool:
        """Initialize the OCO trading system and start monitoring."""
        try:
            if self.is_initialized:
                logger.info("OCO trading system already initialized")
                return True
            
            logger.info("Initializing OCO Trading System...")
            
            # Validate configuration
            if not self._validate_configuration():
                logger.error("OCO configuration validation failed")
                return False
            
            # Initialize OCO manager with configuration
            oco_config = OCOConfiguration()
            oco_manager.config = oco_config
            
            # Start monitoring services
            await self._start_monitoring_services()
            
            # Initialize metrics
            await self._update_system_metrics()
            
            self.is_initialized = True
            logger.info("✅ OCO Trading System initialized successfully")
            
            # Log system status
            await self._log_system_status()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OCO trading system: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the OCO trading system gracefully."""
        logger.info("Shutting down OCO Trading System...")
        
        # Cancel all monitoring tasks
        for task_name, task in self.monitoring_tasks.items():
            if not task.done():
                logger.info(f"Cancelling {task_name}")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop OCO manager monitoring
        await oco_manager.stop_monitoring()
        
        self.monitoring_tasks.clear()
        self.is_initialized = False
        
        logger.info("✅ OCO Trading System shutdown complete")
    
    async def place_protective_oco_for_portfolio(self) -> Dict[str, Any]:
        """
        Automatically place protective OCO orders for all significant positions
        that don't already have protection.
        
        Returns:
            Dictionary with placement results and statistics
        """
        try:
            logger.info("🛡️ Placing protective OCO orders for portfolio...")
            
            if not self.is_initialized:
                await self.initialize()
            
            # Get current positions
            positions = alpaca_client.get_positions()
            account_info = alpaca_client.get_account_info()
            portfolio_value = account_info.get("portfolio_value", 1)
            
            placement_results = {
                "total_positions": len(positions),
                "oco_placed": 0,
                "oco_skipped": 0,
                "errors": 0,
                "details": []
            }
            
            for position in positions:
                try:
                    symbol = position['symbol']
                    position_value = abs(float(position.get('market_value', 0)))
                    position_weight = position_value / portfolio_value if portfolio_value > 0 else 0
                    
                    # Check if position is significant enough for OCO
                    if position_weight < settings.oco_max_position_size * 0.3:  # 30% of max OCO size
                        placement_results["oco_skipped"] += 1
                        placement_results["details"].append({
                            "symbol": symbol,
                            "action": "skipped",
                            "reason": f"Position too small: {position_weight:.2%}"
                        })
                        continue
                    
                    # Check if position already has OCO
                    if symbol in oco_manager.position_oco_mapping:
                        placement_results["oco_skipped"] += 1
                        placement_results["details"].append({
                            "symbol": symbol,
                            "action": "skipped",
                            "reason": "Already has OCO protection"
                        })
                        continue
                    
                    # Check rate limiting
                    if not self._check_oco_rate_limit(symbol):
                        placement_results["oco_skipped"] += 1
                        placement_results["details"].append({
                            "symbol": symbol,
                            "action": "skipped",
                            "reason": "Rate limited"
                        })
                        continue
                    
                    # Place protective OCO
                    oco_order = await oco_manager.auto_place_position_oco(symbol, position)
                    
                    if oco_order:
                        placement_results["oco_placed"] += 1
                        placement_results["details"].append({
                            "symbol": symbol,
                            "action": "oco_placed",
                            "oco_id": oco_order.oco_id,
                            "take_profit": oco_order.take_profit_price,
                            "stop_loss": oco_order.stop_loss_price
                        })
                        
                        # Update rate limiting
                        self.last_oco_timestamp[symbol] = datetime.now()
                        
                        logger.info(f"✅ Protective OCO placed for {symbol}: {oco_order.oco_id}")
                    else:
                        placement_results["oco_skipped"] += 1
                        placement_results["details"].append({
                            "symbol": symbol,
                            "action": "skipped",
                            "reason": "OCO placement failed"
                        })
                
                except Exception as e:
                    logger.error(f"Error placing OCO for {symbol}: {e}")
                    placement_results["errors"] += 1
                    placement_results["details"].append({
                        "symbol": symbol,
                        "action": "error",
                        "error": str(e)
                    })
            
            logger.info(f"🎯 Protective OCO placement complete: "
                       f"{placement_results['oco_placed']} placed, "
                       f"{placement_results['oco_skipped']} skipped, "
                       f"{placement_results['errors']} errors")
            
            return placement_results
            
        except Exception as e:
            logger.error(f"Failed to place protective OCO orders: {e}")
            return {"error": str(e)}
    
    async def execute_breakout_strategy(self, symbols: List[str], 
                                      volatility_threshold: float = 0.02) -> Dict[str, Any]:
        """
        Execute breakout OCO strategies for specified symbols.
        
        Args:
            symbols: List of symbols to create breakout strategies for
            volatility_threshold: Minimum volatility threshold for breakout setup
            
        Returns:
            Dictionary with breakout strategy results
        """
        try:
            logger.info(f"🚀 Executing breakout OCO strategy for {len(symbols)} symbols...")
            
            if not self.is_initialized:
                await self.initialize()
            
            breakout_results = {
                "total_symbols": len(symbols),
                "breakouts_placed": 0,
                "breakouts_skipped": 0,
                "errors": 0,
                "details": []
            }
            
            # Get account info for position sizing
            account_info = alpaca_client.get_account_info()
            portfolio_value = account_info.get("portfolio_value", 1)
            
            for symbol in symbols:
                try:
                    # Check if already have position or OCO for this symbol
                    positions = alpaca_client.get_positions()
                    has_position = any(pos['symbol'] == symbol for pos in positions)
                    
                    if has_position or symbol in oco_manager.position_oco_mapping:
                        breakout_results["breakouts_skipped"] += 1
                        breakout_results["details"].append({
                            "symbol": symbol,
                            "action": "skipped",
                            "reason": "Already has position or OCO"
                        })
                        continue
                    
                    # Check volatility requirement
                    current_price = alpaca_client.get_current_price(symbol)
                    if not current_price:
                        breakout_results["breakouts_skipped"] += 1
                        breakout_results["details"].append({
                            "symbol": symbol,
                            "action": "skipped",
                            "reason": "Could not get current price"
                        })
                        continue
                    
                    # Calculate position size (smaller for breakout strategies)
                    position_value = portfolio_value * (settings.oco_max_position_size * 0.5)  # 50% of max OCO size
                    quantity = position_value / current_price
                    
                    # Check rate limiting
                    if not self._check_oco_rate_limit(symbol):
                        breakout_results["breakouts_skipped"] += 1
                        breakout_results["details"].append({
                            "symbol": symbol,
                            "action": "skipped",
                            "reason": "Rate limited"
                        })
                        continue
                    
                    # Place breakout OCO
                    oco_order = await oco_manager.place_breakout_oco(
                        symbol=symbol,
                        quantity=quantity,
                        upper_breakout_percent=volatility_threshold * 1.5,
                        lower_breakout_percent=volatility_threshold * 1.5,
                        reasoning=f"Breakout OCO strategy: {symbol} volatility-based"
                    )
                    
                    if oco_order:
                        breakout_results["breakouts_placed"] += 1
                        breakout_results["details"].append({
                            "symbol": symbol,
                            "action": "breakout_placed",
                            "oco_id": oco_order.oco_id,
                            "upper_breakout": oco_order.upper_breakout_price,
                            "lower_breakout": oco_order.lower_breakout_price,
                            "quantity": quantity
                        })
                        
                        # Update rate limiting
                        self.last_oco_timestamp[symbol] = datetime.now()
                        
                        logger.info(f"✅ Breakout OCO placed for {symbol}: {oco_order.oco_id}")
                    else:
                        breakout_results["breakouts_skipped"] += 1
                        breakout_results["details"].append({
                            "symbol": symbol,
                            "action": "skipped",
                            "reason": "Breakout OCO placement failed"
                        })
                
                except Exception as e:
                    logger.error(f"Error placing breakout OCO for {symbol}: {e}")
                    breakout_results["errors"] += 1
                    breakout_results["details"].append({
                        "symbol": symbol,
                        "action": "error",
                        "error": str(e)
                    })
            
            logger.info(f"🎯 Breakout OCO strategy complete: "
                       f"{breakout_results['breakouts_placed']} placed, "
                       f"{breakout_results['breakouts_skipped']} skipped, "
                       f"{breakout_results['errors']} errors")
            
            return breakout_results
            
        except Exception as e:
            logger.error(f"Failed to execute breakout strategy: {e}")
            return {"error": str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive OCO trading system status."""
        try:
            await self._update_system_metrics()
            
            # Get OCO manager status
            oco_summary = oco_manager.get_active_oco_summary()
            
            # Get executor status
            executor_summary = oco_executor.get_execution_summary()
            
            # Get risk status
            risk_status = await self._get_risk_status()
            
            return {
                "system_initialized": self.is_initialized,
                "timestamp": datetime.now().isoformat(),
                "configuration": {
                    "oco_enabled": settings.oco_enabled,
                    "auto_placement_enabled": settings.oco_auto_placement_enabled,
                    "breakout_enabled": settings.oco_breakout_enabled,
                    "max_daily_orders": settings.oco_max_daily_orders,
                    "default_take_profit_percent": settings.oco_default_take_profit_percent,
                    "default_stop_loss_percent": settings.oco_default_stop_loss_percent
                },
                "oco_orders": oco_summary,
                "execution_metrics": executor_summary,
                "risk_status": risk_status,
                "system_metrics": {
                    "total_oco_orders": self.system_metrics.total_oco_orders,
                    "active_oco_orders": self.system_metrics.active_oco_orders,
                    "success_rate": self.system_metrics.success_rate,
                    "average_profit_loss_ratio": self.system_metrics.average_profit_loss_ratio
                },
                "monitoring_tasks": {
                    task_name: not task.done() 
                    for task_name, task in self.monitoring_tasks.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    # Private methods
    
    def _validate_configuration(self) -> bool:
        """Validate OCO system configuration."""
        try:
            # Check required settings
            if not settings.oco_enabled:
                logger.warning("OCO trading is disabled in configuration")
                return True  # Still valid, just disabled
            
            # Validate percentage values
            if not (0.001 <= settings.oco_default_take_profit_percent <= 0.10):
                logger.error("Invalid take profit percentage: must be between 0.1% and 10%")
                return False
            
            if not (0.001 <= settings.oco_default_stop_loss_percent <= 0.10):
                logger.error("Invalid stop loss percentage: must be between 0.1% and 10%")
                return False
            
            # Validate profit:loss ratio
            profit_loss_ratio = settings.oco_default_take_profit_percent / settings.oco_default_stop_loss_percent
            if profit_loss_ratio < settings.oco_min_profit_ratio:
                logger.warning(f"Profit:loss ratio ({profit_loss_ratio:.2f}) below minimum ({settings.oco_min_profit_ratio})")
            
            # Validate position size limits
            if not (0.01 <= settings.oco_max_position_size <= 0.50):
                logger.error("Invalid OCO max position size: must be between 1% and 50%")
                return False
            
            logger.info("✅ OCO configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    async def _start_monitoring_services(self) -> None:
        """Start all OCO monitoring services."""
        
        # Start OCO order monitoring
        await oco_manager.start_monitoring()
        
        # Start execution monitoring
        self.monitoring_tasks["oco_execution_monitor"] = asyncio.create_task(
            oco_executor.monitor_oco_executions()
        )
        
        # Start system metrics monitoring
        self.monitoring_tasks["system_metrics_monitor"] = asyncio.create_task(
            self._monitor_system_metrics()
        )
        
        # Start risk monitoring
        self.monitoring_tasks["risk_monitor"] = asyncio.create_task(
            self._monitor_risk_limits()
        )
        
        logger.info("✅ All OCO monitoring services started")
    
    async def _monitor_system_metrics(self) -> None:
        """Monitor and update system metrics periodically."""
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                await self._update_system_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system metrics monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_risk_limits(self) -> None:
        """Monitor OCO risk limits and take action if necessary."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                risk_status = await self._get_risk_status()
                
                # Check OCO exposure limit
                if risk_status.get("oco_exposure_percent", 0) > self.risk_limits["max_oco_exposure_percent"]:
                    logger.warning(f"OCO exposure ({risk_status['oco_exposure_percent']:.1%}) "
                                 f"exceeds limit ({self.risk_limits['max_oco_exposure_percent']:.1%})")
                    
                    # Consider cancelling some OCO orders or pausing new placements
                    # This would be implemented based on specific risk management policies
                
                # Check concurrent OCO orders limit
                active_count = len(oco_manager.active_oco_orders)
                if active_count > self.risk_limits["max_concurrent_oco_orders"]:
                    logger.warning(f"Too many concurrent OCO orders: {active_count} > "
                                 f"{self.risk_limits['max_concurrent_oco_orders']}")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _update_system_metrics(self) -> None:
        """Update system performance metrics."""
        try:
            # Get OCO order statistics
            oco_summary = oco_manager.get_active_oco_summary()
            
            self.system_metrics.active_oco_orders = oco_summary.get("total_active", 0)
            self.system_metrics.total_oco_orders = (
                len(oco_manager.oco_history) + self.system_metrics.active_oco_orders
            )
            
            # Calculate success rate
            completed_orders = len([o for o in oco_manager.oco_history if o.status == OCOStatus.COMPLETED])
            total_historical = len(oco_manager.oco_history)
            
            if total_historical > 0:
                self.system_metrics.success_rate = completed_orders / total_historical
                self.system_metrics.completed_oco_orders = completed_orders
                self.system_metrics.cancelled_oco_orders = total_historical - completed_orders
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    async def _get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status for OCO orders."""
        try:
            # Get portfolio information
            account_info = alpaca_client.get_account_info()
            portfolio_value = account_info.get("portfolio_value", 1)
            
            # Calculate OCO exposure
            total_oco_exposure = 0.0
            for oco_order in oco_manager.active_oco_orders.values():
                if oco_order.entry_price and oco_order.quantity:
                    position_value = oco_order.entry_price * oco_order.quantity
                    total_oco_exposure += position_value
            
            oco_exposure_percent = total_oco_exposure / portfolio_value if portfolio_value > 0 else 0
            
            return {
                "total_oco_exposure": total_oco_exposure,
                "oco_exposure_percent": oco_exposure_percent,
                "active_oco_count": len(oco_manager.active_oco_orders),
                "daily_oco_count": oco_manager.daily_oco_count,
                "within_limits": (
                    oco_exposure_percent <= self.risk_limits["max_oco_exposure_percent"] and
                    len(oco_manager.active_oco_orders) <= self.risk_limits["max_concurrent_oco_orders"] and
                    oco_manager.daily_oco_count <= settings.oco_max_daily_orders
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get risk status: {e}")
            return {"error": str(e)}
    
    def _check_oco_rate_limit(self, symbol: str) -> bool:
        """Check if OCO order can be placed for symbol based on rate limiting."""
        if symbol not in self.last_oco_timestamp:
            return True
        
        time_since_last = datetime.now() - self.last_oco_timestamp[symbol]
        min_interval = timedelta(minutes=self.risk_limits["min_time_between_oco_minutes"])
        
        return time_since_last >= min_interval
    
    async def _log_system_status(self) -> None:
        """Log current system status."""
        try:
            status = await self.get_system_status()
            
            logger.info("📊 OCO Trading System Status:")
            logger.info(f"  - OCO Enabled: {status['configuration']['oco_enabled']}")
            logger.info(f"  - Auto Placement: {status['configuration']['auto_placement_enabled']}")
            logger.info(f"  - Active OCO Orders: {status['oco_orders']['total_active']}")
            logger.info(f"  - Daily OCO Count: {status['oco_orders']['daily_count']}")
            logger.info(f"  - Success Rate: {status['system_metrics']['success_rate']:.1%}")
            
        except Exception as e:
            logger.error(f"Failed to log system status: {e}")

# Global OCO trading system instance
oco_trading_system = OCOTradingSystem()

# Convenience functions for easy integration
async def initialize_oco_system() -> bool:
    """Initialize the OCO trading system."""
    return await oco_trading_system.initialize()

async def shutdown_oco_system() -> None:
    """Shutdown the OCO trading system."""
    await oco_trading_system.shutdown()

async def place_portfolio_protection() -> Dict[str, Any]:
    """Place protective OCO orders for the entire portfolio."""
    return await oco_trading_system.place_protective_oco_for_portfolio()

async def execute_breakout_strategies(symbols: List[str]) -> Dict[str, Any]:
    """Execute breakout OCO strategies for specified symbols."""
    return await oco_trading_system.execute_breakout_strategy(symbols)

async def get_oco_system_status() -> Dict[str, Any]:
    """Get comprehensive OCO system status."""
    return await oco_trading_system.get_system_status()

__all__ = [
    'OCOTradingSystem', 'OCOTradingMetrics', 'oco_trading_system',
    'initialize_oco_system', 'shutdown_oco_system', 'place_portfolio_protection',
    'execute_breakout_strategies', 'get_oco_system_status'
]