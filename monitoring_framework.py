#!/usr/bin/env python3
"""
Production Monitoring Framework for RL Trading System
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
import psutil
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert definition."""
    alert_id: str
    severity: str  # "info", "warning", "critical"
    message: str
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = None

@dataclass
class MetricSnapshot:
    """Point-in-time metric snapshot."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None

class ProductionMonitor:
    """Comprehensive production monitoring for RL trading system."""
    
    def __init__(self, alert_webhook_url: Optional[str] = None):
        self.alert_webhook = alert_webhook_url
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        
        # Performance thresholds
        self.thresholds = {
            'inference_latency_p99_ms': 100.0,
            'memory_usage_percent': 85.0,
            'cpu_usage_percent': 80.0,
            'error_rate_percent': 5.0,
            'model_drift_score': 0.15,
            'portfolio_drawdown_percent': 10.0,
            'consecutive_failures': 3
        }
        
        # System health tracking
        self.health_checks = {
            'alpaca_connection': False,
            'ollama_service': False,
            'redis_connection': False,
            'model_loaded': False,
            'data_pipeline': False
        }
        
        # Start background monitoring
        asyncio.create_task(self._background_monitoring())
    
    async def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric with optional tags."""
        
        snapshot = MetricSnapshot(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        self.metrics_buffer[name].append(snapshot)
        
        # Check for threshold violations
        await self._check_thresholds(name, value)
    
    async def _check_thresholds(self, metric_name: str, value: float):
        """Check if metric violates thresholds and create alerts."""
        
        threshold_key = f"{metric_name}"
        if threshold_key in self.thresholds:
            threshold = self.thresholds[threshold_key]
            
            if value > threshold:
                alert_id = f"threshold_{metric_name}_{int(time.time())}"
                
                # Don't duplicate alerts
                if alert_id not in self.active_alerts:
                    alert = Alert(
                        alert_id=alert_id,
                        severity="warning" if value < threshold * 1.2 else "critical",
                        message=f"{metric_name} exceeded threshold: {value:.2f} > {threshold:.2f}",
                        timestamp=datetime.now(),
                        metadata={
                            'metric': metric_name,
                            'value': value,
                            'threshold': threshold
                        }
                    )
                    
                    await self._fire_alert(alert)
    
    async def _fire_alert(self, alert: Alert):
        """Fire an alert through configured channels."""
        
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Log alert
        log_level = logging.CRITICAL if alert.severity == "critical" else logging.WARNING
        logger.log(log_level, f"🚨 ALERT: {alert.message}")
        
        # Send to webhook if configured
        if self.alert_webhook:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    payload = {
                        'alert_id': alert.alert_id,
                        'severity': alert.severity,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'system': 'rl_trading_system'
                    }
                    await session.post(self.alert_webhook, json=payload, timeout=5)
            except Exception as e:
                logger.error(f"Failed to send alert webhook: {e}")
    
    async def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            del self.active_alerts[alert_id]
            logger.info(f"✅ Alert resolved: {alert_id}")
    
    def get_metric_stats(self, name: str, window_minutes: int = 60) -> Dict[str, float]:
        """Get statistics for a metric over a time window."""
        
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_values = [
            snapshot.value for snapshot in self.metrics_buffer[name]
            if snapshot.timestamp >= cutoff
        ]
        
        if not recent_values:
            return {'count': 0}
        
        return {
            'count': len(recent_values),
            'mean': np.mean(recent_values),
            'median': np.median(recent_values),
            'p95': np.percentile(recent_values, 95),
            'p99': np.percentile(recent_values, 99),
            'min': np.min(recent_values),
            'max': np.max(recent_values),
            'std': np.std(recent_values)
        }
    
    async def _background_monitoring(self):
        """Background task for continuous system monitoring."""
        
        while True:
            try:
                # System resource monitoring
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                
                await self.record_metric('cpu_usage_percent', cpu_percent)
                await self.record_metric('memory_usage_percent', memory_info.percent)
                await self.record_metric('memory_available_gb', memory_info.available / (1024**3))
                
                # Health checks
                await self._perform_health_checks()
                
                # Sleep until next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_checks(self):
        """Perform health checks on system components."""
        
        try:
            # Check Alpaca connection
            from tools.alpaca_client import alpaca_client
            account = alpaca_client.get_account_info()
            self.health_checks['alpaca_connection'] = bool(account)
            
        except Exception:
            self.health_checks['alpaca_connection'] = False
        
        try:
            # Check Ollama service (simplified check)
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:11434/api/tags', timeout=5) as resp:
                    self.health_checks['ollama_service'] = resp.status == 200
                    
        except Exception:
            self.health_checks['ollama_service'] = False
        
        # Record health check metrics
        for service, healthy in self.health_checks.items():
            await self.record_metric(f'health_check_{service}', 1.0 if healthy else 0.0)
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': self._calculate_overall_health(),
            'active_alerts': len(self.active_alerts),
            'critical_alerts': len([a for a in self.active_alerts.values() if a.severity == 'critical']),
            'health_checks': self.health_checks.copy(),
            'recent_metrics': {
                name: self.get_metric_stats(name, window_minutes=10)
                for name in ['cpu_usage_percent', 'memory_usage_percent', 'inference_latency_p99_ms']
                if name in self.metrics_buffer
            },
            'alert_history_24h': len([
                a for a in self.alert_history
                if datetime.now() - a.timestamp < timedelta(hours=24)
            ])
        }
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall system health status."""
        
        # Critical if any critical alerts
        if any(a.severity == 'critical' for a in self.active_alerts.values()):
            return 'critical'
        
        # Warning if any warnings or health checks failing
        if (len(self.active_alerts) > 0 or 
            not all(self.health_checks.values())):
            return 'warning'
        
        return 'healthy'
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Generate performance dashboard data."""
        
        dashboard = {
            'system_overview': {
                'health_status': self._calculate_overall_health(),
                'uptime_hours': self._get_uptime_hours(),
                'active_alerts': len(self.active_alerts)
            },
            'performance_metrics': {},
            'recent_alerts': [
                {
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in list(self.alert_history)[-10:]
            ]
        }
        
        # Add performance metrics if available
        key_metrics = ['inference_latency_p99_ms', 'cpu_usage_percent', 'memory_usage_percent']
        for metric in key_metrics:
            if metric in self.metrics_buffer:
                stats = self.get_metric_stats(metric, window_minutes=60)
                if stats['count'] > 0:
                    dashboard['performance_metrics'][metric] = {
                        'current': stats['mean'],
                        'p95': stats['p95'],
                        'threshold': self.thresholds.get(metric, 'N/A'),
                        'status': 'ok' if stats['p95'] < self.thresholds.get(metric, float('inf')) else 'warning'
                    }
        
        return dashboard
    
    def _get_uptime_hours(self) -> float:
        """Get system uptime in hours."""
        # This would track actual startup time in production
        return psutil.boot_time() / 3600  # Simplified

class PerformanceProfiler:
    """Performance profiling for RL system components."""
    
    def __init__(self, monitor: ProductionMonitor):
        self.monitor = monitor
        self.active_profiles = {}
    
    async def profile_rl_inference(self, func, *args, **kwargs):
        """Profile RL inference performance."""
        
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            logger.error(f"RL inference failed: {e}")
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        # Record metrics
        await self.monitor.record_metric('inference_latency_p99_ms', latency_ms)
        await self.monitor.record_metric('inference_success_rate', 1.0 if success else 0.0)
        
        return result
    
    async def profile_data_fetch(self, func, symbol: str, *args, **kwargs):
        """Profile data fetching performance."""
        
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        # Record metrics with symbol tag
        await self.monitor.record_metric('data_fetch_latency_ms', latency_ms, {'symbol': symbol})
        await self.monitor.record_metric('data_fetch_success_rate', 1.0 if success else 0.0, {'symbol': symbol})
        
        return result

# Global monitoring instance
production_monitor = ProductionMonitor()
performance_profiler = PerformanceProfiler(production_monitor)

async def start_monitoring(webhook_url: Optional[str] = None):
    """Start production monitoring system."""
    global production_monitor
    production_monitor = ProductionMonitor(webhook_url)
    logger.info("🔍 Production monitoring started")
    return production_monitor

def get_health_report() -> Dict[str, Any]:
    """Get current system health report."""
    return production_monitor.get_system_health_report()

def get_performance_dashboard() -> Dict[str, Any]:
    """Get performance dashboard data."""
    return production_monitor.get_performance_dashboard()

if __name__ == "__main__":
    async def test_monitoring():
        monitor = await start_monitoring()
        
        # Test metrics
        await monitor.record_metric('inference_latency_p99_ms', 45.2)
        await monitor.record_metric('cpu_usage_percent', 67.8)
        await monitor.record_metric('memory_usage_percent', 82.1)
        
        # Generate test alert
        await monitor.record_metric('inference_latency_p99_ms', 150.0)  # Above threshold
        
        # Get reports
        health = monitor.get_system_health_report()
        dashboard = monitor.get_performance_dashboard()
        
        print("🔍 Health Report:", json.dumps(health, indent=2, default=str))
        print("📊 Dashboard:", json.dumps(dashboard, indent=2, default=str))
    
    asyncio.run(test_monitoring())