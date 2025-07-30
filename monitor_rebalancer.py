#!/usr/bin/env python3
"""
Continuous Rebalancer Monitor
Real-time monitoring dashboard for the continuous rebalancing system.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

def load_rebalancer_state() -> Optional[Dict[str, Any]]:
    """Load the current rebalancer state."""
    state_file = Path("data/continuous_rebalancer_state.json")
    try:
        if state_file.exists():
            with open(state_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading state: {e}")
    return None

def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def format_timestamp(iso_string: Optional[str]) -> str:
    """Format ISO timestamp for display."""
    if not iso_string:
        return "Never"
    try:
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return iso_string

def display_status_dashboard(state: Dict[str, Any]):
    """Display the monitoring dashboard."""
    # Clear screen
    print("\033[2J\033[H")
    
    print("🔄 CONTINUOUS REBALANCER MONITOR")
    print("=" * 70)
    print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    health = state.get("health", {})
    
    # System Status
    print("📊 SYSTEM STATUS")
    print("-" * 30)
    print(f"Uptime: {format_duration(health.get('uptime_hours', 0) * 3600)}")
    print(f"Total Runs: {health.get('total_runs', 0)}")
    print(f"Successful: {health.get('successful_runs', 0)}")
    print(f"Failed: {health.get('failed_runs', 0)}")
    
    total_runs = health.get('total_runs', 0)
    if total_runs > 0:
        success_rate = (health.get('successful_runs', 0) / total_runs) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    
    print(f"Avg Runtime: {format_duration(health.get('avg_runtime_seconds', 0))}")
    print()
    
    # Current Status
    print("⚡ CURRENT STATUS")
    print("-" * 30)
    
    consecutive_failures = health.get('consecutive_failures', 0)
    if consecutive_failures > 0:
        print(f"⚠️  Consecutive Failures: {consecutive_failures}")
    else:
        print("✅ System Healthy")
    
    last_success = format_timestamp(health.get('last_success'))
    last_failure = format_timestamp(health.get('last_failure'))
    print(f"Last Success: {last_success}")
    print(f"Last Failure: {last_failure}")
    
    api_hits = health.get('api_rate_limit_hits', 0)
    if api_hits > 0:
        print(f"⚠️  API Rate Limit Hits: {api_hits}")
    
    last_run = format_timestamp(state.get('last_run_time'))
    print(f"Last Run: {last_run}")
    
    # Estimate next run
    next_run = state.get('next_run_estimate')
    if next_run:
        next_run_formatted = format_timestamp(next_run)
        print(f"Next Run (Est): {next_run_formatted}")
    
    print()
    
    # Recent Results
    recent_results = state.get("results_history", [])
    if recent_results:
        print("📈 RECENT RESULTS")
        print("-" * 30)
        print("Time                Dur    Sigs  Orders  Success")
        print("-" * 50)
        
        for result in recent_results[-10:]:  # Last 10 results
            timestamp = format_timestamp(result.get('timestamp'))
            duration = format_duration(result.get('duration_seconds', 0))
            signals = result.get('signals_generated', 0)
            orders = result.get('orders_executed', 0)
            success = "✅" if result.get('success', False) else "❌"
            
            # Format for display
            time_short = timestamp.split()[1] if ' ' in timestamp else timestamp
            print(f"{time_short:<12} {duration:>6} {signals:>5} {orders:>7}  {success}")
        
        print()
    
    # Performance Metrics
    if recent_results:
        print("📊 PERFORMANCE METRICS")
        print("-" * 30)
        
        # Calculate recent performance
        recent_success = sum(1 for r in recent_results[-10:] if r.get('success', False))
        recent_total = len(recent_results[-10:])
        recent_success_rate = (recent_success / recent_total) * 100 if recent_total > 0 else 0
        
        total_signals = sum(r.get('signals_generated', 0) for r in recent_results[-10:])
        total_orders = sum(r.get('orders_executed', 0) for r in recent_results[-10:])
        
        print(f"Recent Success Rate (10 runs): {recent_success_rate:.1f}%")
        print(f"Recent Signals Generated: {total_signals}")
        print(f"Recent Orders Executed: {total_orders}")
        
        # Portfolio value trend
        portfolio_values = [r.get('portfolio_value', 0) for r in recent_results[-5:] if r.get('portfolio_value')]
        if len(portfolio_values) >= 2:
            value_change = portfolio_values[-1] - portfolio_values[0]
            value_change_pct = (value_change / portfolio_values[0]) * 100 if portfolio_values[0] > 0 else 0
            print(f"Portfolio Value Trend: ${value_change:+.2f} ({value_change_pct:+.2f}%)")
        
        print()
    
    # Warnings and Alerts
    alerts = []
    
    if consecutive_failures >= 3:
        alerts.append(f"🚨 HIGH: {consecutive_failures} consecutive failures")
    
    if api_hits > 10:
        alerts.append(f"⚠️  MEDIUM: {api_hits} API rate limit hits")
    
    if health.get('uptime_hours', 0) > 0:
        failure_rate = health.get('failed_runs', 0) / health.get('total_runs', 1)
        if failure_rate > 0.3:
            alerts.append(f"⚠️  MEDIUM: High failure rate ({failure_rate:.1%})")
    
    if alerts:
        print("🚨 ALERTS")
        print("-" * 30)
        for alert in alerts:
            print(alert)
        print()
    
    print("Press Ctrl+C to stop monitoring")
    print("Refreshing every 30 seconds...")

async def monitor_continuous_rebalancer():
    """Monitor the continuous rebalancer in real-time."""
    print("🔍 Starting Continuous Rebalancer Monitor...")
    print("Monitoring system state and performance...")
    print()
    
    try:
        while True:
            state = load_rebalancer_state()
            
            if state:
                display_status_dashboard(state)
            else:
                print("❌ No rebalancer state found.")
                print("Make sure the continuous rebalancer is running.")
                print("State file: data/continuous_rebalancer_state.json")
            
            # Wait 30 seconds before refresh
            await asyncio.sleep(30)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")

if __name__ == "__main__":
    asyncio.run(monitor_continuous_rebalancer())