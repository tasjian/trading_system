#!/usr/bin/env python3
"""
Comprehensive MLOps Integration for Ollama Trading System

Integrates all MLOps components:
- Advanced async bridge with connection pooling
- Performance monitoring and metrics collection
- Error handling with circuit breakers
- Health checks and load balancing
- Structured logging and debugging
- Load testing and validation

This module provides a unified interface for production deployment.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Import all MLOps components
from .ollama_async_bridge import OllamaAsyncBridge, get_ollama_bridge, shutdown_ollama_bridge
from .ollama_performance_metrics import (
    get_metrics_collector, get_trading_observer, 
    start_metrics_background_collection
)
from .ollama_error_handling import get_retry_handler
from .ollama_health_system import (
    get_health_checker, get_load_balancer, setup_ollama_cluster
)
from .ollama_logging import (
    get_logger, LogLevel, create_correlation_context,
    RequestTracker, OllamaDebugLogger
)
from .ollama_load_testing import run_comprehensive_load_test

logger = logging.getLogger(__name__)


@dataclass
class MLOpsSystemStatus:
    """Status of the entire MLOps system."""
    async_bridge_status: str
    health_system_status: str
    monitoring_status: str
    error_handling_status: str
    logging_status: str
    load_balancer_status: str
    overall_health: str
    initialization_time: datetime
    components_initialized: List[str]
    components_failed: List[str]


class OllamaMLOpsManager:
    """Central manager for all Ollama MLOps components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_default_config()
        self.status = None
        self.components = {
            "async_bridge": None,
            "health_checker": None,
            "load_balancer": None,
            "metrics_collector": None,
            "trading_observer": None,
            "retry_handler": None,
            "logger": None,
            "request_tracker": None,
            "debug_logger": None
        }
        self.background_tasks = []
        self.initialization_start = None
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from settings."""
        try:
            from config.settings import settings
            return {
                "ollama_base_url": settings.ollama_base_url,
                "max_workers": settings.ollama_max_workers,
                "queue_size": settings.ollama_queue_size,
                "cache_size": settings.ollama_cache_size,
                "cache_ttl": settings.ollama_cache_ttl,
                "default_timeout": settings.ollama_default_timeout,
                "circuit_breaker_threshold": settings.ollama_circuit_breaker_threshold,
                "max_connections": settings.ollama_max_connections,
                "max_connections_per_host": settings.ollama_max_connections_per_host,
                "enable_http2": settings.ollama_enable_http2,
                "enable_performance_monitoring": settings.enable_performance_monitoring,
                "metrics_collection_interval": settings.metrics_collection_interval,
                "enable_health_checks": settings.enable_health_checks,
                "health_check_interval": settings.health_check_interval,
                "enable_load_balancing": settings.enable_load_balancing,
                "load_balancing_strategy": settings.load_balancing_strategy,
                "ollama_instances": settings.ollama_instances,
                "enable_structured_logging": settings.enable_structured_logging,
                "log_level": settings.log_level,
                "enable_model_warmup": settings.enable_model_warmup,
                "log_debug_requests": settings.log_debug_requests,
                "performance_alert_thresholds": settings.performance_alert_thresholds
            }
        except ImportError:
            logger.warning("Settings not available, using default configuration")
            return {
                "ollama_base_url": "http://localhost:11434",
                "max_workers": 8,
                "queue_size": 200,
                "cache_size": 2000,
                "cache_ttl": 300,
                "default_timeout": 30.0,
                "circuit_breaker_threshold": 5,
                "max_connections": 100,
                "max_connections_per_host": 30,
                "enable_http2": True,
                "enable_performance_monitoring": True,
                "metrics_collection_interval": 10,
                "enable_health_checks": True,
                "health_check_interval": 30,
                "enable_load_balancing": True,
                "load_balancing_strategy": "health_weighted",
                "ollama_instances": [
                    {"url": "http://localhost:11434", "name": "ollama-primary", "weight": 2.0, "priority": 1}
                ],
                "enable_structured_logging": True,
                "log_level": "INFO",
                "enable_model_warmup": True,
                "log_debug_requests": False,
                "performance_alert_thresholds": {
                    "p95_latency_ms": 5000,
                    "error_rate_percent": 5.0,
                    "memory_growth_mb": 500,
                    "cpu_usage_percent": 80.0,
                    "queue_depth": 50,
                    "throughput_drop_percent": 50.0
                }
            }
    
    async def initialize_system(self) -> MLOpsSystemStatus:
        """Initialize the complete MLOps system."""
        self.initialization_start = datetime.now()
        initialized_components = []
        failed_components = []
        
        logger.info("🚀 Initializing Ollama MLOps System...")
        
        # 1. Initialize structured logging
        try:
            log_level = LogLevel[self.config.get("log_level", "INFO").upper()]
            self.components["logger"] = get_logger("ollama_mlops", log_level)
            self.components["request_tracker"] = RequestTracker(self.components["logger"])
            self.components["debug_logger"] = OllamaDebugLogger(
                self.components["logger"], 
                self.config.get("log_debug_requests", False)
            )
            initialized_components.append("structured_logging")
            logger.info("✅ Structured logging initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize structured logging: {e}")
            failed_components.append("structured_logging")
        
        # 2. Initialize async bridge
        try:
            bridge = await get_ollama_bridge()
            self.components["async_bridge"] = bridge
            initialized_components.append("async_bridge")
            logger.info("✅ Async bridge initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize async bridge: {e}")
            failed_components.append("async_bridge")
        
        # 3. Initialize error handling
        try:
            self.components["retry_handler"] = get_retry_handler()
            initialized_components.append("error_handling")
            logger.info("✅ Error handling initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize error handling: {e}")
            failed_components.append("error_handling")
        
        # 4. Initialize health system and load balancing
        if self.config.get("enable_health_checks", True):
            try:
                if self.config.get("enable_load_balancing", True):
                    await setup_ollama_cluster(self.config.get("ollama_instances", []))
                    self.components["load_balancer"] = get_load_balancer()
                
                self.components["health_checker"] = await get_health_checker()
                initialized_components.append("health_system")
                logger.info("✅ Health system and load balancing initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize health system: {e}")
                failed_components.append("health_system")
        
        # 5. Initialize performance monitoring
        if self.config.get("enable_performance_monitoring", True):
            try:
                self.components["metrics_collector"] = get_metrics_collector()
                self.components["trading_observer"] = get_trading_observer()
                
                # Start background metrics collection
                metrics_task = asyncio.create_task(
                    start_metrics_background_collection(
                        self.config.get("metrics_collection_interval", 10)
                    )
                )
                self.background_tasks.append(metrics_task)
                
                initialized_components.append("performance_monitoring")
                logger.info("✅ Performance monitoring initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize performance monitoring: {e}")
                failed_components.append("performance_monitoring")
        
        # 6. Start health check background task
        if self.config.get("enable_health_checks", True) and self.components["health_checker"]:
            try:
                health_task = asyncio.create_task(
                    self._health_check_loop(self.config.get("health_check_interval", 30))
                )
                self.background_tasks.append(health_task)
                logger.info("✅ Health check monitoring started")
            except Exception as e:
                logger.error(f"❌ Failed to start health monitoring: {e}")
        
        # Determine overall health
        overall_health = "healthy" if not failed_components else "degraded" if initialized_components else "unhealthy"
        
        # Create status
        self.status = MLOpsSystemStatus(
            async_bridge_status="healthy" if "async_bridge" in initialized_components else "failed",
            health_system_status="healthy" if "health_system" in initialized_components else "failed",
            monitoring_status="healthy" if "performance_monitoring" in initialized_components else "failed",
            error_handling_status="healthy" if "error_handling" in initialized_components else "failed",
            logging_status="healthy" if "structured_logging" in initialized_components else "failed",
            load_balancer_status="healthy" if self.components["load_balancer"] else "disabled",
            overall_health=overall_health,
            initialization_time=self.initialization_start,
            components_initialized=initialized_components,
            components_failed=failed_components
        )
        
        initialization_duration = (datetime.now() - self.initialization_start).total_seconds()
        
        logger.info(f"🎉 MLOps system initialization completed in {initialization_duration:.2f}s")
        logger.info(f"   Components initialized: {len(initialized_components)}")
        logger.info(f"   Components failed: {len(failed_components)}")
        logger.info(f"   Overall health: {overall_health}")
        
        return self.status
    
    async def _health_check_loop(self, interval: int):
        """Background health check loop."""
        health_checker = self.components["health_checker"]
        load_balancer = self.components["load_balancer"]
        
        while True:
            try:
                if load_balancer:
                    # Check health of all instances
                    for instance in load_balancer.instances:
                        try:
                            metrics = await health_checker.perform_intermediate_health_check(
                                instance.url, timeout=10.0
                            )
                            instance.health_metrics = metrics
                            health_checker.update_health_history(instance.url, metrics)
                            
                            # Log health status changes
                            if self.components["logger"]:
                                self.components["logger"].log_health_check(
                                    instance.name,
                                    metrics.status.value,
                                    metrics.response_time,
                                    model_availability=metrics.model_availability
                                )
                        
                        except Exception as e:
                            logger.warning(f"Health check failed for {instance.name}: {e}")
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)
    
    async def shutdown_system(self):
        """Shutdown the MLOps system gracefully."""
        logger.info("🔄 Shutting down MLOps system...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown async bridge
        if self.components["async_bridge"]:
            await shutdown_ollama_bridge()
        
        # Close health checker session
        if self.components["health_checker"] and hasattr(self.components["health_checker"], 'session'):
            session = self.components["health_checker"].session
            if session and not session.closed:
                await session.close()
        
        logger.info("✅ MLOps system shutdown completed")
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system_status": asdict(self.status) if self.status else None,
            "uptime_seconds": (datetime.now() - self.initialization_start).total_seconds() if self.initialization_start else 0
        }
        
        # Async bridge metrics
        if self.components["async_bridge"]:
            try:
                bridge_metrics = await self.components["async_bridge"].get_metrics()
                metrics["async_bridge"] = bridge_metrics
            except Exception as e:
                metrics["async_bridge"] = {"error": str(e)}
        
        # Performance metrics
        if self.components["metrics_collector"]:
            try:
                perf_snapshot = self.components["metrics_collector"].get_current_snapshot()
                metrics["performance"] = asdict(perf_snapshot)
            except Exception as e:
                metrics["performance"] = {"error": str(e)}
        
        # Health metrics
        if self.components["load_balancer"]:
            try:
                lb_stats = self.components["load_balancer"].get_load_balancer_stats()
                metrics["load_balancer"] = lb_stats
            except Exception as e:
                metrics["load_balancer"] = {"error": str(e)}
        
        # Error handling metrics
        if self.components["retry_handler"]:
            try:
                error_stats = self.components["retry_handler"].get_error_statistics()
                metrics["error_handling"] = error_stats
            except Exception as e:
                metrics["error_handling"] = {"error": str(e)}
        
        return metrics
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        health_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown"
        }
        
        # Test async bridge
        if self.components["async_bridge"]:
            try:
                bridge_health = await self.components["async_bridge"].health_check()
                health_results["async_bridge"] = bridge_health
            except Exception as e:
                health_results["async_bridge"] = {"healthy": False, "error": str(e)}
        
        # Test load balancer instances
        if self.components["load_balancer"]:
            try:
                instance_health = []
                for instance in self.components["load_balancer"].instances:
                    if instance.health_metrics:
                        instance_health.append({
                            "name": instance.name,
                            "url": instance.url,
                            "status": instance.health_metrics.status.value,
                            "response_time": instance.health_metrics.response_time,
                            "enabled": instance.enabled
                        })
                
                health_results["instances"] = instance_health
            except Exception as e:
                health_results["instances"] = {"error": str(e)}
        
        # Determine overall status
        bridge_healthy = health_results.get("async_bridge", {}).get("healthy", False)
        instances_healthy = any(
            inst.get("status") == "healthy" 
            for inst in health_results.get("instances", [])
        ) if health_results.get("instances") else False
        
        if bridge_healthy and instances_healthy:
            health_results["overall_status"] = "healthy"
        elif bridge_healthy or instances_healthy:
            health_results["overall_status"] = "degraded"
        else:
            health_results["overall_status"] = "unhealthy"
        
        return health_results
    
    async def run_load_test(self, test_duration: int = 60) -> Dict[str, Any]:
        """Run load test against the system."""
        if not self.components["async_bridge"]:
            return {"error": "Async bridge not initialized"}
        
        logger.info(f"🔬 Running load test for {test_duration} seconds...")
        
        try:
            # Use first available instance for testing
            test_url = self.config.get("ollama_base_url", "http://localhost:11434")
            
            if self.components["load_balancer"] and self.components["load_balancer"].instances:
                healthy_instances = self.components["load_balancer"].get_healthy_instances()
                if healthy_instances:
                    test_url = healthy_instances[0].url
            
            # Run load test
            results = await run_comprehensive_load_test(test_url)
            
            logger.info("✅ Load test completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Load test failed: {e}")
            return {"error": str(e)}
    
    def get_recommendations(self) -> List[str]:
        """Get system optimization recommendations."""
        recommendations = []
        
        if not self.status:
            return ["System not initialized"]
        
        if self.status.components_failed:
            recommendations.append(f"Fix failed components: {', '.join(self.status.components_failed)}")
        
        if self.components["retry_handler"]:
            error_recommendations = self.components["retry_handler"].get_recommendations()
            recommendations.extend(error_recommendations)
        
        # Performance recommendations
        if self.components["metrics_collector"]:
            try:
                snapshot = self.components["metrics_collector"].get_current_snapshot()
                
                if snapshot.p95_latency > 5.0:
                    recommendations.append("High P95 latency detected - consider scaling or optimizing models")
                
                if snapshot.error_rate > 5.0:
                    recommendations.append("High error rate detected - check system stability")
                
                if snapshot.cache_hit_rate < 30:
                    recommendations.append("Low cache hit rate - consider increasing cache size or TTL")
                
                if snapshot.queue_depth > 50:
                    recommendations.append("High queue depth - consider increasing worker count")
                
            except Exception as e:
                recommendations.append(f"Unable to analyze performance metrics: {e}")
        
        return recommendations or ["System is operating optimally"]


# Global MLOps manager instance
_mlops_manager: Optional[OllamaMLOpsManager] = None


async def get_mlops_manager(config: Optional[Dict[str, Any]] = None) -> OllamaMLOpsManager:
    """Get or create global MLOps manager."""
    global _mlops_manager
    
    if _mlops_manager is None:
        _mlops_manager = OllamaMLOpsManager(config)
        await _mlops_manager.initialize_system()
    
    return _mlops_manager


async def initialize_ollama_mlops(config: Optional[Dict[str, Any]] = None) -> MLOpsSystemStatus:
    """Initialize the complete Ollama MLOps system."""
    manager = await get_mlops_manager(config)
    return manager.status


async def shutdown_ollama_mlops():
    """Shutdown the Ollama MLOps system."""
    global _mlops_manager
    
    if _mlops_manager:
        await _mlops_manager.shutdown_system()
        _mlops_manager = None


# Convenience functions for common operations
async def perform_sentiment_analysis(text: str, symbol: str, priority: str = "NORMAL") -> Dict[str, Any]:
    """Perform sentiment analysis using the MLOps system."""
    manager = await get_mlops_manager()
    
    if not manager.components["async_bridge"]:
        return {"error": "Async bridge not available"}
    
    try:
        # Use request tracking
        if manager.components["request_tracker"]:
            async with manager.components["request_tracker"].track_async_request(
                "sentiment_analysis", symbol=symbol
            ) as context:
                from .ollama_async_bridge import query_ollama_sentiment, RequestPriority
                
                priority_map = {
                    "CRITICAL": RequestPriority.CRITICAL,
                    "HIGH": RequestPriority.HIGH,
                    "NORMAL": RequestPriority.NORMAL,
                    "LOW": RequestPriority.LOW
                }
                
                response = await query_ollama_sentiment(
                    text=text,
                    symbol=symbol,
                    priority=priority_map.get(priority.upper(), RequestPriority.NORMAL),
                    timeout=30.0
                )
                
                # Record metrics
                if manager.components["trading_observer"]:
                    manager.components["trading_observer"].record_sentiment_analysis(
                        symbol, response.processing_time, response.success, priority
                    )
                
                return {
                    "success": response.success,
                    "content": response.content,
                    "processing_time": response.processing_time,
                    "cached": response.cached,
                    "error": response.error
                }
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Test the complete MLOps system
    async def test_mlops_system():
        print("🚀 Testing Complete Ollama MLOps System")
        print("=" * 60)
        
        try:
            # Initialize system
            status = await initialize_ollama_mlops()
            
            print(f"✅ System Status: {status.overall_health}")
            print(f"   Components initialized: {len(status.components_initialized)}")
            print(f"   Components failed: {len(status.components_failed)}")
            
            # Get system metrics
            manager = await get_mlops_manager()
            metrics = await manager.get_system_metrics()
            
            print(f"\n📊 System Metrics:")
            print(f"   Uptime: {metrics.get('uptime_seconds', 0):.1f}s")
            
            # Test sentiment analysis
            print(f"\n🧠 Testing Sentiment Analysis:")
            result = await perform_sentiment_analysis(
                "Apple reports strong quarterly earnings with 15% revenue growth",
                "AAPL",
                "HIGH"
            )
            
            if result.get("success"):
                print(f"   ✅ Analysis completed in {result.get('processing_time', 0):.2f}s")
                print(f"   Cached: {result.get('cached', False)}")
            else:
                print(f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}")
            
            # Run health check
            print(f"\n🏥 Health Check:")
            health = await manager.run_health_check()
            print(f"   Overall status: {health.get('overall_status', 'unknown')}")
            
            # Get recommendations
            print(f"\n💡 Recommendations:")
            recommendations = manager.get_recommendations()
            for rec in recommendations[:3]:  # Show first 3
                print(f"   • {rec}")
            
        except Exception as e:
            print(f"❌ MLOps system test failed: {e}")
        
        finally:
            # Shutdown
            await shutdown_ollama_mlops()
            print(f"\n🔄 System shutdown completed")
    
    asyncio.run(test_mlops_system())