#!/usr/bin/env python3
"""
Comprehensive Health Check and Load Balancing System for Ollama Trading System

Implements:
- Multi-tier health checking (basic, intermediate, comprehensive)
- Model warming and preloading strategies
- Intelligent load balancing across multiple Ollama instances
- Health-based failover and recovery
- Performance-based routing decisions
"""

import asyncio
import logging
import json
import time
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import statistics
import random
from urllib.parse import urljoin
import weakref

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheckType(Enum):
    """Types of health checks."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    COMPREHENSIVE = "comprehensive"
    MODEL_SPECIFIC = "model_specific"


@dataclass
class HealthMetrics:
    """Health metrics for an Ollama instance."""
    status: HealthStatus
    response_time: float
    model_availability: Dict[str, bool]
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    queue_depth: int = 0
    error_rate: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    consecutive_successes: int = 0


@dataclass
class OllamaInstance:
    """Ollama instance configuration and state."""
    url: str
    name: str
    weight: float = 1.0
    priority: int = 1
    enabled: bool = True
    health_metrics: Optional[HealthMetrics] = None
    last_request_time: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0


@dataclass
class ModelWarmupConfig:
    """Configuration for model warming."""
    model_name: str
    warmup_prompts: List[str]
    max_tokens: int = 100
    timeout: float = 30.0
    priority: int = 1


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESPONSE_TIME = "response_time"
    HEALTH_WEIGHTED = "health_weighted"


class ModelPreloader:
    """Model preloading and warming system."""
    
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.warmup_configs = self._create_default_warmup_configs()
        self.preloaded_models: Dict[str, Set[str]] = {}  # instance_url -> set of models
        
    def _create_default_warmup_configs(self) -> List[ModelWarmupConfig]:
        """Create default warmup configurations for common models."""
        return [
            ModelWarmupConfig(
                model_name="llama3.1",
                warmup_prompts=[
                    "Hello, please respond with 'ready'",
                    "What is 2+2?",
                    "Analyze sentiment: The market is performing well today."
                ],
                max_tokens=50,
                timeout=15.0,
                priority=1
            ),
            ModelWarmupConfig(
                model_name="llama3:8b",
                warmup_prompts=[
                    "Test response",
                    "Quick math: 5*5",
                    "Brief sentiment analysis test"
                ],
                max_tokens=30,
                timeout=10.0,
                priority=2
            )
        ]
    
    async def warm_up_model(self, instance_url: str, config: ModelWarmupConfig) -> bool:
        """Warm up a specific model on an instance."""
        logger.info(f"Warming up model {config.model_name} on {instance_url}")
        
        success_count = 0
        total_prompts = len(config.warmup_prompts)
        
        for i, prompt in enumerate(config.warmup_prompts):
            try:
                start_time = time.time()
                
                url = urljoin(instance_url, "/api/generate")
                payload = {
                    "model": config.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "max_tokens": config.max_tokens,
                        "temperature": 0.1
                    }
                }
                
                async with self.session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.timeout)
                ) as response:
                    
                    if response.status == 200:
                        await response.json()  # Consume response
                        response_time = time.time() - start_time
                        success_count += 1
                        
                        logger.debug(f"Warmup prompt {i+1}/{total_prompts} successful "
                                   f"({response_time:.2f}s) for {config.model_name}")
                    else:
                        logger.warning(f"Warmup prompt {i+1} failed with status {response.status}")
                        
            except Exception as e:
                logger.warning(f"Warmup prompt {i+1} failed for {config.model_name}: {e}")
                
        success_rate = success_count / total_prompts
        is_successful = success_rate >= 0.7  # At least 70% success rate
        
        if is_successful:
            # Mark model as preloaded
            if instance_url not in self.preloaded_models:
                self.preloaded_models[instance_url] = set()
            self.preloaded_models[instance_url].add(config.model_name)
            
            logger.info(f"Model {config.model_name} warmed up successfully on {instance_url} "
                       f"({success_count}/{total_prompts} prompts succeeded)")
        else:
            logger.warning(f"Model {config.model_name} warmup failed on {instance_url} "
                          f"({success_count}/{total_prompts} prompts succeeded)")
        
        return is_successful
    
    async def warm_up_all_models(self, instance_url: str) -> Dict[str, bool]:
        """Warm up all configured models on an instance."""
        results = {}
        
        # Sort by priority (lower number = higher priority)
        sorted_configs = sorted(self.warmup_configs, key=lambda c: c.priority)
        
        for config in sorted_configs:
            try:
                results[config.model_name] = await self.warm_up_model(instance_url, config)
                
                # Brief pause between models to avoid overwhelming the instance
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Failed to warm up {config.model_name} on {instance_url}: {e}")
                results[config.model_name] = False
        
        return results
    
    def is_model_preloaded(self, instance_url: str, model_name: str) -> bool:
        """Check if a model is preloaded on an instance."""
        return (instance_url in self.preloaded_models and 
                model_name in self.preloaded_models[instance_url])


class AdvancedHealthChecker:
    """Advanced health checking system with multiple check types."""
    
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.model_preloader = ModelPreloader(session)
        self.check_history: Dict[str, List[HealthMetrics]] = {}
        self.baseline_performance: Dict[str, float] = {}
        
    async def perform_basic_health_check(self, instance_url: str, timeout: float = 5.0) -> HealthMetrics:
        """Basic health check - simple connectivity test."""
        start_time = time.time()
        
        try:
            # Test basic connectivity
            url = urljoin(instance_url, "/api/version")
            
            async with self.session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                response_time = time.time() - start_time
                
                if response.status == 200:
                    return HealthMetrics(
                        status=HealthStatus.HEALTHY,
                        response_time=response_time,
                        model_availability={},
                        last_check=datetime.now(),
                        consecutive_successes=1
                    )
                else:
                    return HealthMetrics(
                        status=HealthStatus.UNHEALTHY,
                        response_time=response_time,
                        model_availability={},
                        last_check=datetime.now(),
                        consecutive_failures=1
                    )
                    
        except Exception as e:
            response_time = time.time() - start_time
            logger.debug(f"Basic health check failed for {instance_url}: {e}")
            
            return HealthMetrics(
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                model_availability={},
                last_check=datetime.now(),
                consecutive_failures=1
            )
    
    async def perform_intermediate_health_check(self, instance_url: str, timeout: float = 10.0) -> HealthMetrics:
        """Intermediate health check - test model availability."""
        basic_metrics = await self.perform_basic_health_check(instance_url, timeout / 2)
        
        if basic_metrics.status == HealthStatus.UNHEALTHY:
            return basic_metrics
        
        # Test model availability
        models_to_test = ["llama3.1", "llama3:8b"]
        model_availability = {}
        
        for model in models_to_test:
            try:
                url = urljoin(instance_url, "/api/generate")
                payload = {
                    "model": model,
                    "prompt": "test",
                    "stream": False,
                    "options": {"max_tokens": 1}
                }
                
                async with self.session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout / len(models_to_test))
                ) as response:
                    
                    model_availability[model] = response.status == 200
                    
            except Exception:
                model_availability[model] = False
        
        # Determine overall status based on model availability
        available_models = sum(model_availability.values())
        total_models = len(model_availability)
        
        if available_models == 0:
            status = HealthStatus.UNHEALTHY
        elif available_models < total_models:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        return HealthMetrics(
            status=status,
            response_time=basic_metrics.response_time,
            model_availability=model_availability,
            last_check=datetime.now(),
            consecutive_successes=1 if status != HealthStatus.UNHEALTHY else 0,
            consecutive_failures=1 if status == HealthStatus.UNHEALTHY else 0
        )
    
    async def perform_comprehensive_health_check(self, instance_url: str, timeout: float = 30.0) -> HealthMetrics:
        """Comprehensive health check - full performance evaluation."""
        intermediate_metrics = await self.perform_intermediate_health_check(instance_url, timeout / 2)
        
        if intermediate_metrics.status == HealthStatus.UNHEALTHY:
            return intermediate_metrics
        
        # Perform performance tests
        test_prompts = [
            "What is machine learning?",
            "Analyze the sentiment: The stock market is bullish today.",
            "Calculate the return on investment for a stock that increased from $100 to $150."
        ]
        
        response_times = []
        success_count = 0
        
        for prompt in test_prompts:
            try:
                start_time = time.time()
                
                url = urljoin(instance_url, "/api/generate")
                payload = {
                    "model": "llama3.1",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "max_tokens": 100,
                        "temperature": 0.3
                    }
                }
                
                async with self.session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout / len(test_prompts))
                ) as response:
                    
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    
                    if response.status == 200:
                        result = await response.json()
                        if result.get("response", "").strip():
                            success_count += 1
                            
            except Exception:
                response_times.append(timeout / len(test_prompts))  # Max time on failure
        
        # Calculate performance metrics
        avg_response_time = statistics.mean(response_times) if response_times else timeout
        error_rate = 1.0 - (success_count / len(test_prompts))
        
        # Determine status based on performance
        if error_rate > 0.5:  # More than 50% errors
            status = HealthStatus.UNHEALTHY
        elif error_rate > 0.2 or avg_response_time > 10.0:  # Some errors or slow response
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        # Get system metrics if available
        memory_usage, cpu_usage = await self._get_system_metrics(instance_url)
        
        return HealthMetrics(
            status=status,
            response_time=avg_response_time,
            model_availability=intermediate_metrics.model_availability,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_rate=error_rate,
            last_check=datetime.now(),
            consecutive_successes=1 if status != HealthStatus.UNHEALTHY else 0,
            consecutive_failures=1 if status == HealthStatus.UNHEALTHY else 0
        )
    
    async def _get_system_metrics(self, instance_url: str) -> Tuple[Optional[float], Optional[float]]:
        """Get system metrics from Ollama instance if available."""
        try:
            # This would depend on Ollama exposing system metrics
            # For now, return None values
            return None, None
        except Exception:
            return None, None
    
    def update_health_history(self, instance_url: str, metrics: HealthMetrics):
        """Update health check history for an instance."""
        if instance_url not in self.check_history:
            self.check_history[instance_url] = []
        
        self.check_history[instance_url].append(metrics)
        
        # Keep only last 100 checks
        if len(self.check_history[instance_url]) > 100:
            self.check_history[instance_url] = self.check_history[instance_url][-100:]
    
    def get_health_trend(self, instance_url: str, lookback_checks: int = 10) -> Dict[str, Any]:
        """Analyze health trend for an instance."""
        if instance_url not in self.check_history:
            return {"trend": "unknown", "data_points": 0}
        
        recent_checks = self.check_history[instance_url][-lookback_checks:]
        
        if len(recent_checks) < 3:
            return {"trend": "insufficient_data", "data_points": len(recent_checks)}
        
        # Analyze response time trend
        response_times = [check.response_time for check in recent_checks]
        first_half = response_times[:len(response_times)//2]
        second_half = response_times[len(response_times)//2:]
        
        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)
        
        # Analyze health status trend
        healthy_checks = sum(1 for check in recent_checks if check.status == HealthStatus.HEALTHY)
        health_percentage = healthy_checks / len(recent_checks)
        
        # Determine overall trend
        if avg_second > avg_first * 1.2:
            response_trend = "degrading"
        elif avg_second < avg_first * 0.8:
            response_trend = "improving"
        else:
            response_trend = "stable"
        
        return {
            "trend": response_trend,
            "data_points": len(recent_checks),
            "health_percentage": health_percentage,
            "avg_response_time": statistics.mean(response_times),
            "response_time_std": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            "recent_failures": sum(1 for check in recent_checks if check.status == HealthStatus.UNHEALTHY)
        }


class IntelligentLoadBalancer:
    """Intelligent load balancer with health-aware routing."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.HEALTH_WEIGHTED):
        self.strategy = strategy
        self.instances: List[OllamaInstance] = []
        self.round_robin_index = 0
        self.health_checker = None  # Will be set externally
        
    def add_instance(self, url: str, name: str, weight: float = 1.0, priority: int = 1):
        """Add an Ollama instance to the load balancer."""
        instance = OllamaInstance(
            url=url,
            name=name,
            weight=weight,
            priority=priority,
            enabled=True
        )
        self.instances.append(instance)
        logger.info(f"Added Ollama instance: {name} ({url}) with weight {weight}")
    
    def remove_instance(self, name: str):
        """Remove an instance from the load balancer."""
        self.instances = [inst for inst in self.instances if inst.name != name]
        logger.info(f"Removed Ollama instance: {name}")
    
    def get_healthy_instances(self) -> List[OllamaInstance]:
        """Get list of healthy instances."""
        healthy = []
        for instance in self.instances:
            if (instance.enabled and 
                instance.health_metrics and 
                instance.health_metrics.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]):
                healthy.append(instance)
        return healthy
    
    def select_instance(self, model_name: Optional[str] = None) -> Optional[OllamaInstance]:
        """Select the best instance based on the current strategy."""
        healthy_instances = self.get_healthy_instances()
        
        if not healthy_instances:
            logger.warning("No healthy instances available")
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
            return self._response_time_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.HEALTH_WEIGHTED:
            return self._health_weighted_selection(healthy_instances, model_name)
        else:
            # Default to round robin
            return self._round_robin_selection(healthy_instances)
    
    def _round_robin_selection(self, instances: List[OllamaInstance]) -> OllamaInstance:
        """Simple round-robin selection."""
        selected = instances[self.round_robin_index % len(instances)]
        self.round_robin_index += 1
        return selected
    
    def _weighted_round_robin_selection(self, instances: List[OllamaInstance]) -> OllamaInstance:
        """Weighted round-robin selection."""
        total_weight = sum(inst.weight for inst in instances)
        if total_weight == 0:
            return random.choice(instances)
        
        # Create weighted list
        weighted_instances = []
        for instance in instances:
            count = max(1, int(instance.weight * 10))  # Scale weights
            weighted_instances.extend([instance] * count)
        
        return weighted_instances[self.round_robin_index % len(weighted_instances)]
    
    def _response_time_selection(self, instances: List[OllamaInstance]) -> OllamaInstance:
        """Select instance with best response time."""
        return min(instances, 
                  key=lambda inst: inst.health_metrics.response_time if inst.health_metrics else float('inf'))
    
    def _health_weighted_selection(self, instances: List[OllamaInstance], model_name: Optional[str]) -> OllamaInstance:
        """Select instance based on health score and model availability."""
        scored_instances = []
        
        for instance in instances:
            if not instance.health_metrics:
                continue
            
            score = 0
            metrics = instance.health_metrics
            
            # Base score from health status
            if metrics.status == HealthStatus.HEALTHY:
                score += 100
            elif metrics.status == HealthStatus.DEGRADED:
                score += 50
            
            # Response time factor (lower is better)
            if metrics.response_time > 0:
                score -= min(metrics.response_time * 10, 50)
            
            # Error rate factor
            score -= metrics.error_rate * 30
            
            # Model availability bonus
            if model_name and model_name in metrics.model_availability:
                if metrics.model_availability[model_name]:
                    score += 20
                else:
                    score -= 50  # Heavy penalty for unavailable model
            
            # Preloaded model bonus
            if (model_name and self.health_checker and 
                hasattr(self.health_checker, 'model_preloader') and
                self.health_checker.model_preloader.is_model_preloaded(instance.url, model_name)):
                score += 15
            
            # Weight factor
            score *= instance.weight
            
            scored_instances.append((score, instance))
        
        if not scored_instances:
            return random.choice(instances)
        
        # Select instance with highest score
        scored_instances.sort(key=lambda x: x[0], reverse=True)
        return scored_instances[0][1]
    
    def record_request_result(self, instance: OllamaInstance, success: bool, response_time: float):
        """Record request result for learning and adjustment."""
        instance.total_requests += 1
        instance.last_request_time = datetime.now()
        
        if success:
            instance.successful_requests += 1
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        total_requests = sum(inst.total_requests for inst in self.instances)
        
        instance_stats = []
        for instance in self.instances:
            success_rate = (instance.successful_requests / instance.total_requests 
                          if instance.total_requests > 0 else 0)
            
            instance_stats.append({
                "name": instance.name,
                "url": instance.url,
                "enabled": instance.enabled,
                "total_requests": instance.total_requests,
                "success_rate": success_rate,
                "health_status": instance.health_metrics.status.value if instance.health_metrics else "unknown",
                "response_time": instance.health_metrics.response_time if instance.health_metrics else None,
                "weight": instance.weight,
                "priority": instance.priority
            })
        
        return {
            "strategy": self.strategy.value,
            "total_instances": len(self.instances),
            "healthy_instances": len(self.get_healthy_instances()),
            "total_requests": total_requests,
            "instance_stats": instance_stats
        }


# Global health system components
_health_checker: Optional[AdvancedHealthChecker] = None
_load_balancer: Optional[IntelligentLoadBalancer] = None


async def get_health_checker() -> AdvancedHealthChecker:
    """Get or create global health checker."""
    global _health_checker
    
    if _health_checker is None:
        # Create aiohttp session
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        
        _health_checker = AdvancedHealthChecker(session)
        logger.info("Initialized advanced health checker")
    
    return _health_checker


def get_load_balancer() -> IntelligentLoadBalancer:
    """Get or create global load balancer."""
    global _load_balancer
    
    if _load_balancer is None:
        _load_balancer = IntelligentLoadBalancer(LoadBalancingStrategy.HEALTH_WEIGHTED)
        logger.info("Initialized intelligent load balancer")
    
    return _load_balancer


async def setup_ollama_cluster(instance_configs: List[Dict[str, Any]]):
    """Set up Ollama cluster with health checking and load balancing."""
    health_checker = await get_health_checker()
    load_balancer = get_load_balancer()
    
    # Connect health checker to load balancer
    load_balancer.health_checker = health_checker
    
    # Add instances to load balancer
    for config in instance_configs:
        load_balancer.add_instance(
            url=config["url"],
            name=config["name"],
            weight=config.get("weight", 1.0),
            priority=config.get("priority", 1)
        )
    
    # Perform initial health checks and model warming
    for instance in load_balancer.instances:
        try:
            # Comprehensive health check
            metrics = await health_checker.perform_comprehensive_health_check(instance.url)
            instance.health_metrics = metrics
            health_checker.update_health_history(instance.url, metrics)
            
            # Warm up models if instance is healthy
            if metrics.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                warmup_results = await health_checker.model_preloader.warm_up_all_models(instance.url)
                logger.info(f"Model warmup results for {instance.name}: {warmup_results}")
            
        except Exception as e:
            logger.error(f"Failed to initialize instance {instance.name}: {e}")
            instance.enabled = False
    
    logger.info(f"Ollama cluster initialized with {len(load_balancer.instances)} instances")


if __name__ == "__main__":
    # Test the health system
    async def test_health_system():
        print("🏥 Testing Advanced Health System")
        print("=" * 50)
        
        # Example cluster configuration
        cluster_config = [
            {"url": "http://localhost:11434", "name": "ollama-primary", "weight": 2.0, "priority": 1},
            {"url": "http://localhost:11435", "name": "ollama-secondary", "weight": 1.0, "priority": 2},
        ]
        
        try:
            await setup_ollama_cluster(cluster_config)
            
            load_balancer = get_load_balancer()
            health_checker = await get_health_checker()
            
            # Test instance selection
            selected = load_balancer.select_instance("llama3.1")
            if selected:
                print(f"Selected instance: {selected.name} ({selected.url})")
                print(f"Health status: {selected.health_metrics.status.value}")
                print(f"Response time: {selected.health_metrics.response_time:.3f}s")
            else:
                print("No healthy instances available")
            
            # Get statistics
            stats = load_balancer.get_load_balancer_stats()
            print(f"\nLoad Balancer Stats:")
            print(json.dumps(stats, indent=2, default=str))
            
        except Exception as e:
            print(f"Health system test failed: {e}")
    
    asyncio.run(test_health_system())