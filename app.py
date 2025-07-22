#!/usr/bin/env python3
"""
Production Web Application for LLM-Enhanced Trading System
Optimized for AWS App Runner deployment
"""

import asyncio
import logging
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

# Import our trading system
from main import TradingSystemApp
from config.settings import settings, validate_settings
from tools.llm_client import llm_client

# Configure production logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/app.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

# Global trading app instance
trading_app: Optional[TradingSystemApp] = None
background_tasks_running = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management for production"""
    global trading_app
    
    logger.info("🚀 Starting LLM-Enhanced Trading System")
    
    try:
        # Validate configuration
        validate_settings()
        
        # Initialize trading system
        trading_app = TradingSystemApp()
        
        # Test system components
        await test_system_health()
        
        logger.info("✅ Trading system initialized successfully")
        
        yield  # Application runs here
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize trading system: {e}")
        raise
    finally:
        # Cleanup
        if trading_app:
            logger.info("🛑 Shutting down trading system")
            trading_app.shutdown()

# Create FastAPI app with lifespan
app = FastAPI(
    title="LLM-Enhanced Trading System",
    description="AI-powered algorithmic trading system with LLM decision making",
    version="1.0.0",
    lifespan=lifespan
)

# Add security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure for your domain in production
)

# Pydantic models for API
class TradingCycleRequest(BaseModel):
    message: Optional[str] = "Manual trading cycle"

class ManualTradeRequest(BaseModel):
    symbol: str
    action: str  # "buy", "sell", "close"
    quantity: Optional[float] = None

class SystemHealthResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, Any]

async def test_system_health():
    """Test critical system components"""
    try:
        # Test LLM connectivity
        connections = await llm_client.test_connection()
        if not any(connections.values()):
            raise ValueError("No LLM providers available")
        
        # Test trading system initialization
        if not trading_app:
            raise ValueError("Trading system not initialized")
        
        logger.info("✅ System health check passed")
        
    except Exception as e:
        logger.error(f"❌ System health check failed: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with system information"""
    return """
    <html>
        <head>
            <title>LLM-Enhanced Trading System</title>
        </head>
        <body>
            <h1>🤖 LLM-Enhanced Trading System</h1>
            <h2>🚀 Production Deployment - AWS App Runner</h2>
            <p><strong>Status:</strong> Online</p>
            <p><strong>Version:</strong> 1.0.0</p>
            <p><strong>Mode:</strong> Paper Trading (Safe)</p>
            
            <h3>API Endpoints:</h3>
            <ul>
                <li><a href="/health">/health</a> - System health check</li>
                <li><a href="/status">/status</a> - Trading system status</li>
                <li><a href="/docs">/docs</a> - Interactive API documentation</li>
            </ul>
            
            <h3>Features:</h3>
            <ul>
                <li>✅ LLM-Enhanced Decision Making</li>
                <li>✅ Multi-Agent Portfolio Construction</li>
                <li>✅ Advanced Risk Management</li>
                <li>✅ Real-time Market Analysis</li>
                <li>✅ Paper Trading Safety</li>
            </ul>
        </body>
    </html>
    """

@app.get("/health", response_model=SystemHealthResponse)
async def health_check():
    """Health check endpoint for AWS App Runner"""
    try:
        components = {}
        
        # Check trading system
        if trading_app:
            status = trading_app.get_status()
            components["trading_system"] = {
                "status": "healthy",
                "running": status.get("system", {}).get("running", False),
                "session_id": status.get("system", {}).get("session_id"),
                "cycles": status.get("system", {}).get("cycle_count", 0)
            }
        else:
            components["trading_system"] = {"status": "unhealthy", "error": "Not initialized"}
        
        # Check LLM client
        try:
            llm_stats = llm_client.get_usage_stats()
            components["llm_client"] = {
                "status": "healthy",
                "provider": llm_stats.get("preferred_provider"),
                "requests": llm_stats.get("total_requests", 0),
                "cost": llm_stats.get("total_cost_estimate", 0.0)
            }
        except Exception as e:
            components["llm_client"] = {"status": "unhealthy", "error": str(e)}
        
        # Overall health
        overall_status = "healthy" if all(
            comp.get("status") == "healthy" for comp in components.values()
        ) else "unhealthy"
        
        return SystemHealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            components=components
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/status")
async def get_system_status():
    """Get detailed trading system status"""
    if not trading_app:
        raise HTTPException(status_code=503, detail="Trading system not initialized")
    
    try:
        status = trading_app.get_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trading/cycle")
async def run_trading_cycle(request: TradingCycleRequest, background_tasks: BackgroundTasks):
    """Trigger a manual trading cycle"""
    if not trading_app:
        raise HTTPException(status_code=503, detail="Trading system not initialized")
    
    try:
        # Run cycle in background to avoid timeout
        background_tasks.add_task(
            execute_trading_cycle,
            request.message
        )
        
        return {
            "status": "accepted",
            "message": "Trading cycle initiated",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error initiating trading cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def execute_trading_cycle(message: str):
    """Execute trading cycle in background"""
    global trading_app
    
    try:
        if trading_app:
            result = await trading_app.run_single_cycle(message)
            logger.info(f"Trading cycle completed: {result.get('status')}")
        else:
            logger.error("Trading system not available for cycle execution")
    except Exception as e:
        logger.error(f"Background trading cycle failed: {e}")

@app.post("/trading/manual")
async def manual_trade(request: ManualTradeRequest):
    """Execute a manual trade"""
    if not trading_app:
        raise HTTPException(status_code=503, detail="Trading system not initialized")
    
    try:
        result = await trading_app.manual_trade(
            symbol=request.symbol.upper(),
            action=request.action.lower(),
            quantity=request.quantity
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Manual trade error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get system performance metrics"""
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "uptime_seconds": time.time() - start_time,
                "memory_usage": get_memory_usage(),
                "cpu_usage": get_cpu_usage()
            }
        }
        
        if trading_app:
            status = trading_app.get_status()
            metrics["trading"] = {
                "portfolio_value": status.get("account", {}).get("equity", 0),
                "positions": status.get("portfolio", {}).get("positions", 0),
                "cycles": status.get("system", {}).get("cycle_count", 0)
            }
        
        # LLM metrics
        if llm_client:
            llm_stats = llm_client.get_usage_stats()
            metrics["llm"] = llm_stats
        
        return JSONResponse(content=metrics)
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_memory_usage() -> float:
    """Get memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

def get_cpu_usage() -> float:
    """Get CPU usage percentage"""
    try:
        import psutil
        return psutil.cpu_percent(interval=1)
    except ImportError:
        return 0.0

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

# Track start time
start_time = time.time()

if __name__ == "__main__":
    # Production server configuration
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🌐 Starting production server on {host}:{port}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True,
        workers=1,  # Single worker for trading system
        loop="asyncio"
    )