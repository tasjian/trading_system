#!/usr/bin/env python3
"""
Trading System API Server
Provides REST API endpoints for the Android frontend application.
Integrates with the existing trading system to serve portfolio and order data.
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from tools.alpaca_client import AlpacaClient
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Trading System API", version="1.0.0")

# Enable CORS for Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Alpaca client
alpaca_client = AlpacaClient()

# Data Models
class PortfolioData(BaseModel):
    portfolio_value: float
    daily_change: float
    daily_change_percent: float
    cash: float
    equity: float
    buying_power: float
    last_updated: str

class OrderData(BaseModel):
    id: str
    symbol: str
    side: str
    qty: str
    order_type: str
    limit_price: Optional[str] = None
    stop_price: Optional[str] = None
    status: str
    filled_qty: Optional[str] = None
    filled_avg_price: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None
    submitted_at: Optional[str] = None
    filled_at: Optional[str] = None

class HealthStatus(BaseModel):
    status: str
    timestamp: str
    system_health: str

# API Endpoints

@app.get("/api/health", response_model=HealthStatus)
async def get_health():
    """Get system health status"""
    try:
        # Check if Alpaca client is working
        account = alpaca_client.get_account_info()
        system_health = "operational" if account else "degraded"
        
        return HealthStatus(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            system_health=system_health
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthStatus(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            system_health="error"
        )

@app.get("/api/portfolio", response_model=PortfolioData)
async def get_portfolio():
    """Get current portfolio data"""
    try:
        # Get account information from Alpaca
        account = alpaca_client.get_account_info()
        
        if not account:
            raise HTTPException(status_code=503, detail="Unable to fetch account data")
        
        # Calculate values
        portfolio_value = float(account.get('portfolio_value', 0))
        equity = float(account.get('equity', 0))
        cash = float(account.get('cash', 0))
        buying_power = float(account.get('buying_power', 0))
        
        # Get previous close portfolio value for daily change calculation
        # This is a simplified calculation - you might want to store this data
        try:
            prev_close = float(account.get('last_equity', equity))
            daily_change = equity - prev_close
            daily_change_percent = (daily_change / prev_close * 100) if prev_close > 0 else 0.0
        except:
            daily_change = 0.0
            daily_change_percent = 0.0
        
        return PortfolioData(
            portfolio_value=portfolio_value,
            daily_change=daily_change,
            daily_change_percent=daily_change_percent,
            cash=cash,
            equity=equity,
            buying_power=buying_power,
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching portfolio: {str(e)}")

@app.get("/api/orders/recent", response_model=List[OrderData])
async def get_recent_orders(limit: int = 10):
    """Get recent orders"""
    try:
        # Get orders from Alpaca
        orders = alpaca_client.get_orders(
            status='all',
            limit=limit
        )
        
        if not orders:
            return []
        
        # Convert to our format
        order_list = []
        for order in orders:
            order_data = OrderData(
                id=order.get('id', ''),
                symbol=order.get('symbol', ''),
                side=order.get('side', ''),
                qty=str(order.get('qty', 0)),
                order_type=order.get('order_type', ''),
                limit_price=str(order.get('limit_price')) if order.get('limit_price') else None,
                stop_price=str(order.get('stop_price')) if order.get('stop_price') else None,
                status=order.get('status', ''),
                filled_qty=str(order.get('filled_qty')) if order.get('filled_qty') else None,
                filled_avg_price=str(order.get('filled_avg_price')) if order.get('filled_avg_price') else None,
                created_at=str(order.get('created_at', datetime.now().isoformat())),
                updated_at=str(order.get('updated_at')) if order.get('updated_at') else None,
                submitted_at=str(order.get('submitted_at')) if order.get('submitted_at') else None,
                filled_at=str(order.get('filled_at')) if order.get('filled_at') else None
            )
            order_list.append(order_data)
        
        return order_list
        
    except Exception as e:
        logger.error(f"Error fetching orders: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching orders: {str(e)}")

@app.get("/api/positions")
async def get_positions():
    """Get current positions"""
    try:
        positions = alpaca_client.get_positions()
        
        if not positions:
            return []
        
        position_list = []
        for pos in positions:
            qty = float(pos.get('qty', 0))
            position_data = {
                "symbol": pos.get('symbol', ''),
                "qty": str(qty),
                "side": "long" if qty > 0 else "short",
                "market_value": str(pos.get('market_value', 0)),
                "cost_basis": str(pos.get('cost_basis', 0)),
                "unrealized_pl": str(pos.get('unrealized_pl', 0)),
                "unrealized_plpc": str(pos.get('unrealized_plpc', 0)),
                "current_price": str(pos.get('current_price', 0)),
                "avg_entry_price": str(pos.get('avg_entry_price', 0))
            }
            position_list.append(position_data)
        
        return position_list
        
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching positions: {str(e)}")

@app.get("/api/system/status")
async def get_system_status():
    """Get trading system status"""
    try:
        # Check if continuous rebalancer is running
        rebalancer_running = False
        try:
            with open("data/continuous_rebalancer_state.json", "r") as f:
                state = json.load(f)
                last_run = state.get("last_run_time")
                if last_run:
                    # Check if last run was within the last hour
                    from datetime import datetime, timedelta
                    last_run_dt = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
                    rebalancer_running = (datetime.now(last_run_dt.tzinfo) - last_run_dt).total_seconds() < 3600
        except:
            pass
        
        return {
            "rebalancer_running": rebalancer_running,
            "market_open": alpaca_client.get_clock().is_open,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "rebalancer_running": False,
            "market_open": False,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Trading System API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": [
            "/api/health",
            "/api/portfolio", 
            "/api/orders/recent",
            "/api/positions",
            "/api/system/status"
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting Trading System API Server...")
    print("📱 Android app should connect to: http://10.0.2.2:8000")
    print("🌐 Web browser access: http://localhost:8000")
    print("📋 API docs: http://localhost:8000/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Bind to all interfaces so emulator can access
        port=8000,
        log_level="info"
    )