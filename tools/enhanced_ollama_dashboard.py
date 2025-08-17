#!/usr/bin/env python3
"""
Enhanced Ollama Performance Dashboard

Real-time monitoring dashboard for the enhanced Ollama implementation
specifically designed for trading system performance requirements.

Features:
- Real-time performance metrics
- Trading-specific KPIs
- Cache efficiency monitoring  
- Express lane performance tracking
- Circuit breaker status
- Performance recommendations
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import threading
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Single point-in-time performance snapshot."""
    timestamp: datetime
    express_requests_per_minute: float
    standard_requests_per_minute: float
    batch_requests_per_minute: float
    avg_express_latency_ms: float
    avg_standard_latency_ms: float
    l1_cache_hit_rate: float
    l2_cache_hit_rate: float
    queue_utilization: float
    circuit_breaker_state: str
    active_workers: int
    memory_usage_mb: float
    trading_cycle_impact_ms: float


class EnhancedOllamaPerformanceDashboard:
    """Real-time performance dashboard for enhanced Ollama implementation."""
    
    def __init__(self, history_minutes: int = 60):
        self.history_minutes = history_minutes
        self.snapshots: deque = deque(maxlen=history_minutes * 6)  # 10-second intervals
        self.last_metrics = {}
        self.trading_kpis = {}
        self.performance_trends = defaultdict(list)
        self.dashboard_active = False
        self._lock = threading.Lock()
        
        # Performance thresholds for trading
        self.thresholds = {
            "express_latency_warning_ms": 1000,      # 1 second
            "express_latency_critical_ms": 2000,     # 2 seconds  
            "standard_latency_warning_ms": 5000,     # 5 seconds
            "queue_utilization_warning": 0.7,        # 70%
            "queue_utilization_critical": 0.9,       # 90%
            "cache_hit_rate_warning": 0.3,           # 30%
            "memory_usage_warning_mb": 1500,         # 1.5GB
            "trading_cycle_impact_warning_ms": 10    # 10ms
        }
    
    async def start_monitoring(self, update_interval: float = 10.0):
        """Start real-time monitoring."""
        if self.dashboard_active:
            logger.warning("Dashboard monitoring already active")
            return
        
        self.dashboard_active = True
        logger.info("🔍 Starting enhanced Ollama performance dashboard...")
        
        while self.dashboard_active:
            try:
                await self.collect_performance_snapshot()
                await asyncio.sleep(update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dashboard monitoring error: {e}")
                await asyncio.sleep(5)
        
        logger.info("Enhanced Ollama dashboard monitoring stopped")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.dashboard_active = False
    
    async def collect_performance_snapshot(self):
        """Collect current performance snapshot."""
        try:
            # Get enhanced bridge metrics
            from tools.enhanced_ollama_bridge import get_enhanced_ollama_bridge
            
            bridge = await get_enhanced_ollama_bridge()
            metrics = await bridge.get_enhanced_metrics()
            
            # Calculate rates from deltas
            current_time = datetime.now()
            trading_enhancements = metrics.get("trading_enhancements", {})
            
            # Request rates (calculate from previous snapshot)
            express_rate = self._calculate_rate("express_requests_total", trading_enhancements.get("express_requests_total", 0))
            batch_rate = self._calculate_rate("batch_requests_total", trading_enhancements.get("batch_requests_total", 0))
            standard_rate = self._calculate_rate("total_processed", metrics.get("queue_metrics", {}).get("total_processed", 0)) - express_rate - batch_rate
            
            # Cache statistics
            cache_tiers = trading_enhancements.get("cache_tiers", {})
            l1_stats = cache_tiers.get("l1_stats", {})
            l2_stats = cache_tiers.get("l2_stats", {})
            
            l1_hit_rate = l1_stats.get("hit_rate", 0)
            l2_hit_rate = metrics.get("cache_hit_rate", 0)  # From base bridge
            
            # System metrics
            system_metrics = metrics.get("system", {})
            queue_utilization = system_metrics.get("queue_size", 0) / max(system_metrics.get("queue_capacity", 1), 1)
            
            # Circuit breaker status
            circuit_breaker = metrics.get("circuit_breaker", {})
            cb_state = circuit_breaker.get("state", "UNKNOWN")
            
            # Performance metrics
            queue_metrics = metrics.get("queue_metrics", {})
            avg_processing_time = queue_metrics.get("avg_processing_time", 0) * 1000  # Convert to ms
            
            # Estimate express latency (would need actual measurement in production)
            express_latency = avg_processing_time * 0.5  # Express lane is typically faster
            
            # Trading cycle impact (estimated)
            trading_cycle_impact = queue_utilization * 10  # Rough estimate in ms
            
            # Memory usage (estimated)
            memory_usage = system_metrics.get("cache_size", 0) * 0.001  # Rough estimate
            
            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=current_time,
                express_requests_per_minute=express_rate * 60,
                standard_requests_per_minute=standard_rate * 60,
                batch_requests_per_minute=batch_rate * 60,
                avg_express_latency_ms=express_latency,
                avg_standard_latency_ms=avg_processing_time,
                l1_cache_hit_rate=l1_hit_rate,
                l2_cache_hit_rate=l2_hit_rate,
                queue_utilization=queue_utilization,
                circuit_breaker_state=cb_state,
                active_workers=system_metrics.get("active_workers", 0),
                memory_usage_mb=memory_usage,
                trading_cycle_impact_ms=trading_cycle_impact
            )
            
            # Store snapshot
            with self._lock:
                self.snapshots.append(snapshot)
                self.last_metrics = metrics
            
            # Update performance trends
            self._update_performance_trends(snapshot)
            
            # Calculate trading KPIs
            await self._calculate_trading_kpis()
            
        except Exception as e:
            logger.error(f"Failed to collect performance snapshot: {e}")
    
    def _calculate_rate(self, metric_name: str, current_value: int) -> float:
        """Calculate rate per second for a cumulative metric."""
        now = time.time()
        
        if metric_name not in self.last_metrics:
            self.last_metrics[metric_name] = {"value": current_value, "timestamp": now}
            return 0.0
        
        last_data = self.last_metrics[metric_name]
        time_delta = now - last_data["timestamp"]
        value_delta = current_value - last_data["value"]
        
        if time_delta > 0:
            rate = value_delta / time_delta
        else:
            rate = 0.0
        
        # Update last data
        self.last_metrics[metric_name] = {"value": current_value, "timestamp": now}
        
        return max(0.0, rate)  # Ensure non-negative
    
    def _update_performance_trends(self, snapshot: PerformanceSnapshot):
        """Update performance trend calculations."""
        # Keep last 30 minutes of data for trends
        max_trend_points = 180  # 30 minutes at 10-second intervals
        
        trends = {
            "express_latency": snapshot.avg_express_latency_ms,
            "standard_latency": snapshot.avg_standard_latency_ms,
            "l1_cache_hit_rate": snapshot.l1_cache_hit_rate,
            "queue_utilization": snapshot.queue_utilization,
            "trading_cycle_impact": snapshot.trading_cycle_impact_ms
        }
        
        for metric, value in trends.items():
            trend_data = self.performance_trends[metric]
            trend_data.append({"timestamp": snapshot.timestamp, "value": value})
            
            # Keep only recent data
            while len(trend_data) > max_trend_points:
                trend_data.pop(0)
    
    async def _calculate_trading_kpis(self):
        """Calculate trading-specific KPIs."""
        if not self.snapshots:
            return
        
        # Get recent snapshots (last 5 minutes)
        recent_cutoff = datetime.now() - timedelta(minutes=5)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= recent_cutoff]
        
        if not recent_snapshots:
            return
        
        # Calculate KPIs
        self.trading_kpis = {
            "sentiment_analysis_capacity": {
                "express_requests_per_minute": sum(s.express_requests_per_minute for s in recent_snapshots) / len(recent_snapshots),
                "total_requests_per_minute": sum(s.express_requests_per_minute + s.standard_requests_per_minute + s.batch_requests_per_minute for s in recent_snapshots) / len(recent_snapshots),
                "capacity_utilization": sum(s.queue_utilization for s in recent_snapshots) / len(recent_snapshots)
            },
            "performance_quality": {
                "avg_express_latency_ms": sum(s.avg_express_latency_ms for s in recent_snapshots) / len(recent_snapshots),
                "avg_standard_latency_ms": sum(s.avg_standard_latency_ms for s in recent_snapshots) / len(recent_snapshots),
                "trading_cycle_impact_ms": sum(s.trading_cycle_impact_ms for s in recent_snapshots) / len(recent_snapshots)
            },
            "cache_efficiency": {
                "l1_hit_rate": sum(s.l1_cache_hit_rate for s in recent_snapshots) / len(recent_snapshots),
                "l2_hit_rate": sum(s.l2_cache_hit_rate for s in recent_snapshots) / len(recent_snapshots),
                "combined_hit_rate": (sum(s.l1_cache_hit_rate + s.l2_cache_hit_rate for s in recent_snapshots) / len(recent_snapshots)) / 2
            },
            "system_health": {
                "circuit_breaker_healthy": all(s.circuit_breaker_state == "CLOSED" for s in recent_snapshots[-3:]),  # Last 3 snapshots
                "queue_under_control": all(s.queue_utilization < 0.8 for s in recent_snapshots),
                "memory_usage_stable": sum(s.memory_usage_mb for s in recent_snapshots) / len(recent_snapshots) < 1000
            }
        }
    
    def get_current_dashboard(self) -> Dict[str, Any]:
        """Get current dashboard state."""
        with self._lock:
            current_snapshot = self.snapshots[-1] if self.snapshots else None
        
        if not current_snapshot:
            return {"error": "No performance data available"}
        
        # Generate alerts
        alerts = self._generate_performance_alerts(current_snapshot)
        
        # Get trend analysis
        trend_analysis = self._analyze_performance_trends()
        
        # Performance recommendations
        recommendations = self._generate_performance_recommendations()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_performance": {
                "express_lane": {
                    "requests_per_minute": current_snapshot.express_requests_per_minute,
                    "avg_latency_ms": current_snapshot.avg_express_latency_ms,
                    "status": "healthy" if current_snapshot.avg_express_latency_ms < self.thresholds["express_latency_warning_ms"] else "warning"
                },
                "standard_processing": {
                    "requests_per_minute": current_snapshot.standard_requests_per_minute,
                    "avg_latency_ms": current_snapshot.avg_standard_latency_ms,
                    "status": "healthy" if current_snapshot.avg_standard_latency_ms < self.thresholds["standard_latency_warning_ms"] else "warning"
                },
                "cache_performance": {
                    "l1_hit_rate": f"{current_snapshot.l1_cache_hit_rate:.1%}",
                    "l2_hit_rate": f"{current_snapshot.l2_cache_hit_rate:.1%}",
                    "status": "healthy" if current_snapshot.l1_cache_hit_rate > self.thresholds["cache_hit_rate_warning"] else "warning"
                },
                "system_status": {
                    "queue_utilization": f"{current_snapshot.queue_utilization:.1%}",
                    "circuit_breaker": current_snapshot.circuit_breaker_state,
                    "active_workers": current_snapshot.active_workers,
                    "trading_cycle_impact_ms": current_snapshot.trading_cycle_impact_ms
                }
            },
            "trading_kpis": self.trading_kpis,
            "alerts": alerts,
            "trend_analysis": trend_analysis,
            "recommendations": recommendations,
            "data_points": len(self.snapshots),
            "monitoring_duration_minutes": len(self.snapshots) / 6 if self.snapshots else 0
        }
    
    def _generate_performance_alerts(self, snapshot: PerformanceSnapshot) -> List[Dict[str, Any]]:
        """Generate performance alerts based on thresholds."""
        alerts = []
        
        # Express latency alerts
        if snapshot.avg_express_latency_ms > self.thresholds["express_latency_critical_ms"]:
            alerts.append({
                "level": "critical",
                "metric": "express_latency",
                "message": f"Express lane latency ({snapshot.avg_express_latency_ms:.0f}ms) exceeds critical threshold",
                "impact": "Trading decisions may be delayed"
            })
        elif snapshot.avg_express_latency_ms > self.thresholds["express_latency_warning_ms"]:
            alerts.append({
                "level": "warning",
                "metric": "express_latency",
                "message": f"Express lane latency ({snapshot.avg_express_latency_ms:.0f}ms) above optimal",
                "impact": "Consider reducing express lane load"
            })
        
        # Queue utilization alerts
        if snapshot.queue_utilization > self.thresholds["queue_utilization_critical"]:
            alerts.append({
                "level": "critical",
                "metric": "queue_utilization",
                "message": f"Queue utilization ({snapshot.queue_utilization:.1%}) critically high",
                "impact": "Request processing severely degraded"
            })
        elif snapshot.queue_utilization > self.thresholds["queue_utilization_warning"]:
            alerts.append({
                "level": "warning",
                "metric": "queue_utilization",
                "message": f"Queue utilization ({snapshot.queue_utilization:.1%}) high",
                "impact": "Consider increasing worker count"
            })
        
        # Circuit breaker alerts
        if snapshot.circuit_breaker_state == "OPEN":
            alerts.append({
                "level": "critical",
                "metric": "circuit_breaker",
                "message": "Circuit breaker is OPEN",
                "impact": "LLM sentiment analysis degraded to fallback mode"
            })
        elif snapshot.circuit_breaker_state == "HALF_OPEN":
            alerts.append({
                "level": "warning",
                "metric": "circuit_breaker",
                "message": "Circuit breaker in recovery mode",
                "impact": "System attempting to recover from failures"
            })
        
        # Cache performance alerts
        if snapshot.l1_cache_hit_rate < self.thresholds["cache_hit_rate_warning"]:
            alerts.append({
                "level": "warning",
                "metric": "cache_performance",
                "message": f"L1 cache hit rate ({snapshot.l1_cache_hit_rate:.1%}) below optimal",
                "impact": "Consider increasing cache size or adjusting TTL"
            })
        
        # Trading cycle impact alerts
        if snapshot.trading_cycle_impact_ms > self.thresholds["trading_cycle_impact_warning_ms"]:
            alerts.append({
                "level": "warning",
                "metric": "trading_cycle_impact",
                "message": f"Trading cycle impact ({snapshot.trading_cycle_impact_ms:.1f}ms) above target",
                "impact": "LLM processing may affect trading performance"
            })
        
        return alerts
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self.snapshots) < 10:  # Need at least 10 data points
            return {"status": "insufficient_data"}
        
        # Calculate trends for key metrics
        trend_analysis = {}
        
        for metric_name, trend_data in self.performance_trends.items():
            if len(trend_data) < 5:
                continue
            
            # Get recent values (last 5 minutes)
            recent_values = [d["value"] for d in trend_data[-30:]]  # Last 30 data points
            
            if len(recent_values) >= 2:
                # Calculate simple trend (improvement/degradation)
                first_half = recent_values[:len(recent_values)//2]
                second_half = recent_values[len(recent_values)//2:]
                
                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)
                
                # Determine trend direction
                if metric_name in ["express_latency", "standard_latency", "queue_utilization", "trading_cycle_impact"]:
                    # Lower is better for these metrics
                    trend = "improving" if second_avg < first_avg else "degrading" if second_avg > first_avg else "stable"
                else:
                    # Higher is better for cache hit rates
                    trend = "improving" if second_avg > first_avg else "degrading" if second_avg < first_avg else "stable"
                
                trend_analysis[metric_name] = {
                    "direction": trend,
                    "current_value": recent_values[-1],
                    "trend_strength": abs(second_avg - first_avg) / max(first_avg, 0.001)
                }
        
        return trend_analysis
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if not self.snapshots:
            return recommendations
        
        recent_snapshot = self.snapshots[-1]
        
        # Express lane recommendations
        if recent_snapshot.avg_express_latency_ms > 1000:
            recommendations.append("Consider using quantized models (llama3:8b-q4_0) for express lane to reduce latency")
        
        # Cache recommendations
        if recent_snapshot.l1_cache_hit_rate < 0.4:
            recommendations.append("Increase L1 cache size or TTL to improve express lane cache hits")
        
        if recent_snapshot.l2_cache_hit_rate < 0.5:
            recommendations.append("Optimize prompt normalization to improve cache efficiency")
        
        # Queue management recommendations
        if recent_snapshot.queue_utilization > 0.7:
            recommendations.append("Consider increasing worker count or queue size to handle load")
        
        # Trading impact recommendations
        if recent_snapshot.trading_cycle_impact_ms > 5:
            recommendations.append("Optimize request prioritization to minimize trading cycle impact")
        
        # System optimization recommendations
        if recent_snapshot.active_workers < 4:
            recommendations.append("Consider increasing worker count for better throughput")
        
        return recommendations
    
    def export_performance_report(self, filename: Optional[str] = None) -> str:
        """Export comprehensive performance report."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_ollama_performance_report_{timestamp}.json"
        
        # Get current dashboard data
        dashboard_data = self.get_current_dashboard()
        
        # Add historical data
        historical_data = []
        for snapshot in list(self.snapshots):
            historical_data.append({
                "timestamp": snapshot.timestamp.isoformat(),
                "express_requests_per_minute": snapshot.express_requests_per_minute,
                "avg_express_latency_ms": snapshot.avg_express_latency_ms,
                "l1_cache_hit_rate": snapshot.l1_cache_hit_rate,
                "queue_utilization": snapshot.queue_utilization,
                "circuit_breaker_state": snapshot.circuit_breaker_state
            })
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "monitoring_summary": {
                "duration_minutes": len(self.snapshots) / 6,
                "data_points": len(self.snapshots),
                "monitoring_active": self.dashboard_active
            },
            "current_status": dashboard_data,
            "historical_data": historical_data,
            "performance_trends": dict(self.performance_trends)
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report exported to {filename}")
        return filename
    
    def print_dashboard(self):
        """Print formatted dashboard to console."""
        dashboard = self.get_current_dashboard()
        
        print("\n" + "="*80)
        print("🎛️  ENHANCED OLLAMA PERFORMANCE DASHBOARD")
        print("="*80)
        
        if "error" in dashboard:
            print(f"❌ {dashboard['error']}")
            return
        
        # Current performance
        current = dashboard["current_performance"]
        
        print("\n📊 CURRENT PERFORMANCE")
        print("-" * 40)
        
        express = current["express_lane"]
        print(f"Express Lane:      {express['requests_per_minute']:.1f} req/min, {express['avg_latency_ms']:.0f}ms avg")
        
        standard = current["standard_processing"]
        print(f"Standard:          {standard['requests_per_minute']:.1f} req/min, {standard['avg_latency_ms']:.0f}ms avg")
        
        cache = current["cache_performance"]
        print(f"Cache Hit Rates:   L1: {cache['l1_hit_rate']}, L2: {cache['l2_hit_rate']}")
        
        system = current["system_status"]
        print(f"Queue Utilization: {system['queue_utilization']}")
        print(f"Circuit Breaker:   {system['circuit_breaker']}")
        print(f"Active Workers:    {system['active_workers']}")
        
        # Alerts
        alerts = dashboard.get("alerts", [])
        if alerts:
            print("\n🚨 ALERTS")
            print("-" * 40)
            for alert in alerts:
                level_icon = "🔴" if alert["level"] == "critical" else "🟡"
                print(f"{level_icon} {alert['message']}")
        
        # Trading KPIs
        trading_kpis = dashboard.get("trading_kpis", {})
        if trading_kpis:
            print("\n📈 TRADING KPIs")
            print("-" * 40)
            
            capacity = trading_kpis.get("sentiment_analysis_capacity", {})
            print(f"Analysis Capacity: {capacity.get('total_requests_per_minute', 0):.1f} req/min")
            
            quality = trading_kpis.get("performance_quality", {})
            print(f"Trading Impact:    {quality.get('trading_cycle_impact_ms', 0):.1f}ms")
            
            health = trading_kpis.get("system_health", {})
            health_status = "✅ Healthy" if all(health.values()) else "⚠️  Degraded"
            print(f"System Health:     {health_status}")
        
        # Recommendations
        recommendations = dashboard.get("recommendations", [])
        if recommendations:
            print("\n💡 RECOMMENDATIONS")
            print("-" * 40)
            for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                print(f"{i}. {rec}")
        
        print(f"\n⏱️  Monitoring: {dashboard.get('monitoring_duration_minutes', 0):.1f} minutes ({dashboard.get('data_points', 0)} data points)")
        print("="*80)


# Global dashboard instance
_dashboard: Optional[EnhancedOllamaPerformanceDashboard] = None


async def get_performance_dashboard() -> EnhancedOllamaPerformanceDashboard:
    """Get or create global performance dashboard."""
    global _dashboard
    
    if _dashboard is None:
        _dashboard = EnhancedOllamaPerformanceDashboard()
        logger.info("Enhanced Ollama performance dashboard initialized")
    
    return _dashboard


async def start_dashboard_monitoring(update_interval: float = 10.0):
    """Start dashboard monitoring in background."""
    dashboard = await get_performance_dashboard()
    
    monitoring_task = asyncio.create_task(dashboard.start_monitoring(update_interval))
    logger.info("Enhanced Ollama dashboard monitoring started")
    
    return monitoring_task


if __name__ == "__main__":
    async def run_dashboard():
        """Run interactive dashboard."""
        print("🎛️  Starting Enhanced Ollama Performance Dashboard")
        print("=" * 60)
        
        dashboard = await get_performance_dashboard()
        
        # Start monitoring
        monitoring_task = asyncio.create_task(dashboard.start_monitoring(10.0))
        
        try:
            # Interactive loop
            while True:
                await asyncio.sleep(30)  # Update every 30 seconds
                dashboard.print_dashboard()
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping dashboard...")
            dashboard.stop_monitoring()
            monitoring_task.cancel()
            
            # Export final report
            report_file = dashboard.export_performance_report()
            print(f"📊 Final performance report: {report_file}")
    
    asyncio.run(run_dashboard())