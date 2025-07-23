"""Integration bridge between enhanced agents and existing trading system."""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from agents.enhanced_workflow import enhanced_trading_workflow
from agents.enhanced_state import create_enhanced_initial_state
from agents.workflow import trading_workflow  # Original workflow
from agents.state import create_initial_state  # Original state

logger = logging.getLogger(__name__)

class EnhancedTradingSystemBridge:
    """Bridge to integrate enhanced agents with existing trading system."""
    
    def __init__(self, use_enhanced: bool = True):
        """Initialize bridge with option to use enhanced or original system."""
        self.use_enhanced = use_enhanced
        self.enhanced_workflow = enhanced_trading_workflow
        self.original_workflow = trading_workflow
        
    async def run_trading_cycle(self, session_id: str, input_message: str = "") -> Dict[str, Any]:
        """Run trading cycle using appropriate system."""
        
        if self.use_enhanced:
            logger.info("Running enhanced trading cycle")
            return await self._run_enhanced_cycle(session_id, input_message)
        else:
            logger.info("Running original trading cycle")
            return await self._run_original_cycle(session_id, input_message)
    
    async def _run_enhanced_cycle(self, session_id: str, input_message: str = "") -> Dict[str, Any]:
        """Run enhanced trading cycle with fallback to original."""
        try:
            # Try enhanced system first
            result = await self.enhanced_workflow.run_enhanced_cycle(session_id, input_message)
            
            if result["status"] == "success":
                logger.info("Enhanced trading cycle completed successfully")
                return self._format_enhanced_result(result)
            else:
                logger.warning("Enhanced cycle failed, falling back to original system")
                return await self._run_original_cycle(session_id, input_message)
                
        except Exception as e:
            logger.error(f"Enhanced trading cycle error: {e}")
            logger.info("Falling back to original trading system")
            return await self._run_original_cycle(session_id, input_message)
    
    async def _run_original_cycle(self, session_id: str, input_message: str = "") -> Dict[str, Any]:
        """Run original trading cycle as fallback."""
        try:
            result = await self.original_workflow.run_cycle(session_id, input_message)
            return self._format_original_result(result)
            
        except Exception as e:
            logger.error(f"Original trading cycle error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id,
                "system_used": "original_fallback_failed"
            }
    
    def _format_enhanced_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format enhanced system result for compatibility."""
        
        return {
            "status": result["status"],
            "session_id": result["session_id"],
            "system_used": "enhanced",
            "portfolio_value": result.get("portfolio_value", 0),
            "active_positions": result.get("active_positions", 0),
            "signals_generated": result.get("signals_generated", 0),
            "workflow_stage": result.get("workflow_stage", "completed"),
            "cycle_count": result.get("cycle_count", 1),
            "enhanced_features": {
                "agent_communication": True,
                "tool_integration": True,
                "advanced_state_management": True,
                "team_coordination": result.get("team_performance", {})
            },
            "timestamp": datetime.now()
        }
    
    def _format_original_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format original system result for compatibility."""
        
        return {
            "status": result["status"],
            "session_id": result["session_id"],
            "system_used": "original",
            "portfolio_value": result.get("portfolio_value", 0),
            "active_positions": result.get("active_positions", 0),
            "signals_generated": result.get("signals_generated", 0),
            "orders_executed": result.get("orders_executed", 0),
            "enhanced_features": {
                "agent_communication": False,
                "tool_integration": False,
                "advanced_state_management": False,
                "team_coordination": {}
            },
            "timestamp": datetime.now()
        }
    
    def switch_system(self, use_enhanced: bool):
        """Switch between enhanced and original system."""
        self.use_enhanced = use_enhanced
        logger.info(f"Switched to {'enhanced' if use_enhanced else 'original'} trading system")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of both systems."""
        
        return {
            "current_system": "enhanced" if self.use_enhanced else "original",
            "enhanced_available": True,
            "original_available": True,
            "enhanced_features": {
                "multi_agent_communication": "Available",
                "structured_tool_integration": "Available", 
                "advanced_state_management": "Available",
                "human_intervention_support": "Available",
                "memory_and_reasoning": "Available"
            },
            "integration_status": "Active",
            "fallback_configured": True
        }

# Create compatibility wrapper for existing main.py
class BackwardCompatibleWorkflow:
    """Wrapper to make enhanced system compatible with existing main.py."""
    
    def __init__(self):
        self.bridge = EnhancedTradingSystemBridge(use_enhanced=True)
    
    async def run_cycle(self, session_id: str, input_message: str = "") -> Dict[str, Any]:
        """Run cycle with same interface as original workflow."""
        result = await self.bridge.run_trading_cycle(session_id, input_message)
        
        # Convert to format expected by main.py
        return {
            "status": result["status"],
            "session_id": result["session_id"],
            "final_state": {"portfolio": {"equity": result["portfolio_value"]}},
            "portfolio_value": result["portfolio_value"],
            "active_positions": result["active_positions"],
            "signals_generated": result["signals_generated"],
            "orders_executed": result.get("orders_executed", 0)
        }

# Global instances for integration
trading_system_bridge = EnhancedTradingSystemBridge(use_enhanced=True)
backward_compatible_workflow = BackwardCompatibleWorkflow()

# For drop-in replacement in main.py, you can use:
# from agents.enhanced_integration import backward_compatible_workflow as trading_workflow