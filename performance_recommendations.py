#!/usr/bin/env python3
"""
Production Performance Optimizations for RL Trading System
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Production performance optimizations."""
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
    async def optimize_data_pipeline(self):
        """Optimize the data ingestion pipeline."""
        
        # 1. Batch Market Data Fetching
        async def batch_fetch_market_data(symbols: List[str]) -> Dict[str, Any]:
            async with aiohttp.ClientSession() as session:
                tasks = []
                for batch in self._create_batches(symbols, batch_size=10):
                    task = self._fetch_batch_data(session, batch)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return self._combine_results(results)
        
        # 2. Parallel Sentiment Analysis
        async def parallel_sentiment_analysis(symbols: List[str]) -> Dict[str, float]:
            semaphore = asyncio.Semaphore(3)  # Limit concurrent requests
            
            async def analyze_symbol(symbol: str):
                async with semaphore:
                    return await self._analyze_sentiment(symbol)
            
            tasks = [analyze_symbol(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return dict(zip(symbols, results))
    
    def optimize_rl_inference(self, rl_system):
        """Optimize RL model inference performance."""
        
        # 1. Model Quantization for Production
        if hasattr(rl_system.dual_agent, 'stable_policy'):
            policy = rl_system.dual_agent.stable_policy
            if torch.cuda.is_available():
                # Use mixed precision for faster inference
                policy.half()  # Convert to FP16
                logger.info("✅ Applied FP16 quantization to stable policy")
        
        # 2. Batch Inference Optimization
        @torch.no_grad()
        def batch_inference(observations: np.ndarray):
            # Convert to tensor once
            obs_tensor = torch.FloatTensor(observations)
            if torch.cuda.is_available():
                obs_tensor = obs_tensor.cuda().half()
            
            # Single forward pass for batch
            actions = policy(obs_tensor)
            return actions.cpu().numpy()
        
        return batch_inference
    
    def optimize_memory_usage(self):
        """Production memory optimizations."""
        
        return {
            'connection_pooling': {
                'max_connections': 20,
                'max_keepalive_connections': 10,
                'keepalive_expiry': 30
            },
            'caching': {
                'sentiment_cache_ttl': 3600,  # 1 hour
                'market_data_cache_ttl': 60,  # 1 minute
                'model_cache_size': 5  # Keep last 5 models
            },
            'garbage_collection': {
                'enable_gc': True,
                'gc_threshold': (700, 10, 10),
                'explicit_gc_interval': 300  # 5 minutes
            }
        }
    
    def _create_batches(self, items: List, batch_size: int):
        """Create batches from list of items."""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
    
    async def _fetch_batch_data(self, session: aiohttp.ClientSession, symbols: List[str]):
        """Fetch market data for a batch of symbols."""
        # Implementation would depend on specific API
        pass
    
    def _combine_results(self, results: List) -> Dict[str, Any]:
        """Combine batch results into single dictionary."""
        combined = {}
        for result in results:
            if not isinstance(result, Exception):
                combined.update(result)
        return combined
    
    async def _analyze_sentiment(self, symbol: str) -> float:
        """Analyze sentiment for a single symbol."""
        # Implementation would integrate with sentiment engine
        return 0.0

class ProductionMetrics:
    """Production monitoring and metrics."""
    
    def __init__(self):
        self.metrics = {
            'inference_latency_p99': [],
            'data_fetch_latency': [],
            'memory_usage': [],
            'model_accuracy': [],
            'safety_violations': []
        }
    
    def record_inference_latency(self, latency_ms: float):
        """Record RL inference latency."""
        self.metrics['inference_latency_p99'].append(latency_ms)
        
        # Alert if latency too high
        if latency_ms > 100:  # 100ms threshold
            logger.warning(f"🚨 High inference latency: {latency_ms:.2f}ms")
    
    def record_safety_violation(self, violation_type: str, severity: str):
        """Record safety violation for monitoring."""
        self.metrics['safety_violations'].append({
            'type': violation_type,
            'severity': severity,
            'timestamp': asyncio.get_event_loop().time()
        })
        
        logger.error(f"🛡️ Safety violation: {violation_type} ({severity})")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate system health report."""
        
        if not self.metrics['inference_latency_p99']:
            return {'status': 'no_data'}
        
        latencies = self.metrics['inference_latency_p99'][-100:]  # Last 100
        
        return {
            'inference_performance': {
                'p50_latency_ms': np.percentile(latencies, 50),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'avg_latency_ms': np.mean(latencies)
            },
            'safety_status': {
                'violations_last_hour': len([
                    v for v in self.metrics['safety_violations'] 
                    if asyncio.get_event_loop().time() - v['timestamp'] < 3600
                ]),
                'critical_violations': len([
                    v for v in self.metrics['safety_violations'] 
                    if v['severity'] == 'critical'
                ])
            },
            'system_health': 'healthy' if np.mean(latencies) < 50 else 'degraded'
        }

# Production Configuration
PRODUCTION_CONFIG = {
    'rl_system': {
        'inference_timeout_ms': 50,
        'max_batch_size': 32,
        'enable_gpu_inference': True,
        'model_quantization': True
    },
    'data_pipeline': {
        'max_concurrent_requests': 10,
        'request_timeout_sec': 5,
        'retry_attempts': 2,
        'circuit_breaker_threshold': 5
    },
    'monitoring': {
        'enable_detailed_logging': False,  # Reduce I/O in production
        'metric_aggregation_interval': 60,
        'alert_thresholds': {
            'inference_latency_p99': 100,
            'memory_usage_percent': 80,
            'error_rate_percent': 5
        }
    }
}

if __name__ == "__main__":
    optimizer = PerformanceOptimizer()
    metrics = ProductionMetrics()
    
    print("🚀 Production Performance Optimizations")
    print(f"Recommended configuration: {PRODUCTION_CONFIG}")