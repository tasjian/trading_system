#!/usr/bin/env python3
"""
Ollama Performance Benchmark for Trading System

Comprehensive performance testing and comparison between:
- Original blocking implementation
- Standard async bridge
- Enhanced async bridge with trading optimizations

Tests latency, throughput, and concurrency under trading workloads.
"""

import asyncio
import time
import logging
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import as_completed
import threading
import psutil
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    test_name: str
    implementation: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    throughput_rps: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: Optional[float] = None
    error_details: List[str] = None


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self) -> Dict[str, List[str]]:
        """Generate realistic test data for sentiment analysis."""
        return {
            "positive_news": [
                "Company reports record quarterly earnings with 25% revenue growth",
                "Major breakthrough in renewable energy technology announced",
                "Stock price surges on strong guidance and positive outlook",
                "New product launch exceeds expectations with high demand",
                "Partnership with industry leader drives expansion opportunities"
            ],
            "negative_news": [
                "Company misses earnings expectations due to supply chain issues",
                "Regulatory investigation announced causing investor concerns",
                "Major product recall affects quarterly revenue projections",
                "CEO departure creates uncertainty about future direction",
                "Market share loss to competitors impacts growth prospects"
            ],
            "neutral_news": [
                "Company maintains steady performance in line with expectations",
                "Regular quarterly dividend payment announced to shareholders",
                "Standard operational update provided during earnings call",
                "Routine leadership transition follows planned succession",
                "Market conditions remain stable with modest trading volumes"
            ],
            "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NFLX", "NVDA", "CRM", "ORCL"]
        }
    
    async def benchmark_single_request_latency(self, implementation: str, num_requests: int = 50) -> BenchmarkResult:
        """Benchmark single request latency."""
        logger.info(f"Testing single request latency - {implementation} ({num_requests} requests)")
        
        # Get bridge instance
        if implementation == "enhanced":
            from tools.enhanced_ollama_bridge import get_enhanced_ollama_bridge, analyze_sentiment_express
            bridge = await get_enhanced_ollama_bridge()
            test_func = self._test_enhanced_single_request
        elif implementation == "standard":
            from tools.ollama_async_bridge import get_ollama_bridge, query_ollama_sentiment, RequestPriority
            bridge = await get_ollama_bridge()
            test_func = self._test_standard_single_request
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        # Warm up
        await test_func(bridge, "Warm up request", "TEST")
        
        # Measure performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()
        
        latencies = []
        successful = 0
        failed = 0
        errors = []
        
        for i in range(num_requests):
            # Select random test text
            category = ["positive_news", "negative_news", "neutral_news"][i % 3]
            text = self.test_data[category][i % len(self.test_data[category])]
            symbol = self.test_data["symbols"][i % len(self.test_data["symbols"])]
            
            request_start = time.time()
            try:
                response = await test_func(bridge, text, symbol)
                request_time = time.time() - request_start
                latencies.append(request_time)
                
                if hasattr(response, 'success') and response.success:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                request_time = time.time() - request_start
                latencies.append(request_time)
                failed += 1
                errors.append(str(e))
        
        total_time = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies) if latencies else 0
        median_latency = statistics.median(latencies) if latencies else 0
        p95_latency = self._percentile(latencies, 0.95) if latencies else 0
        p99_latency = self._percentile(latencies, 0.99) if latencies else 0
        throughput = successful / total_time if total_time > 0 else 0
        
        # Get cache hit rate if available
        cache_hit_rate = None
        try:
            if implementation == "enhanced":
                metrics = await bridge.get_enhanced_metrics()
                l1_stats = metrics.get("trading_enhancements", {}).get("cache_tiers", {}).get("l1_stats", {})
                cache_hit_rate = l1_stats.get("hit_rate", 0)
            else:
                metrics = await bridge.get_metrics()
                cache_hit_rate = metrics.get("cache_hit_rate", 0)
        except:
            pass
        
        return BenchmarkResult(
            test_name="single_request_latency",
            implementation=implementation,
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
            avg_latency=avg_latency,
            median_latency=median_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            throughput_rps=throughput,
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_percent=end_cpu - start_cpu,
            cache_hit_rate=cache_hit_rate,
            error_details=errors[:5]  # Keep first 5 errors
        )
    
    async def benchmark_concurrent_throughput(self, implementation: str, concurrent_requests: int = 20, total_requests: int = 100) -> BenchmarkResult:
        """Benchmark concurrent request throughput."""
        logger.info(f"Testing concurrent throughput - {implementation} ({concurrent_requests} concurrent, {total_requests} total)")
        
        # Get bridge instance
        if implementation == "enhanced":
            from tools.enhanced_ollama_bridge import get_enhanced_ollama_bridge
            bridge = await get_enhanced_ollama_bridge()
            test_func = self._test_enhanced_single_request
        elif implementation == "standard":
            from tools.ollama_async_bridge import get_ollama_bridge
            bridge = await get_ollama_bridge()
            test_func = self._test_standard_single_request
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def limited_request(i: int) -> tuple:
            async with semaphore:
                category = ["positive_news", "negative_news", "neutral_news"][i % 3]
                text = self.test_data[category][i % len(self.test_data[category])]
                symbol = self.test_data["symbols"][i % len(self.test_data["symbols"])]
                
                request_start = time.time()
                try:
                    response = await test_func(bridge, text, symbol)
                    request_time = time.time() - request_start
                    success = hasattr(response, 'success') and response.success
                    return (request_time, success, None)
                except Exception as e:
                    request_time = time.time() - request_start
                    return (request_time, False, str(e))
        
        # Measure performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()
        
        # Execute all requests concurrently
        tasks = [limited_request(i) for i in range(total_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        # Process results
        latencies = []
        successful = 0
        failed = 0
        errors = []
        
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                errors.append(str(result))
                continue
            
            latency, success, error = result
            latencies.append(latency)
            
            if success:
                successful += 1
            else:
                failed += 1
                if error:
                    errors.append(error)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies) if latencies else 0
        median_latency = statistics.median(latencies) if latencies else 0
        p95_latency = self._percentile(latencies, 0.95) if latencies else 0
        p99_latency = self._percentile(latencies, 0.99) if latencies else 0
        throughput = successful / total_time if total_time > 0 else 0
        
        # Get cache hit rate
        cache_hit_rate = None
        try:
            if implementation == "enhanced":
                metrics = await bridge.get_enhanced_metrics()
                l1_stats = metrics.get("trading_enhancements", {}).get("cache_tiers", {}).get("l1_stats", {})
                cache_hit_rate = l1_stats.get("hit_rate", 0)
            else:
                metrics = await bridge.get_metrics()
                cache_hit_rate = metrics.get("cache_hit_rate", 0)
        except:
            pass
        
        return BenchmarkResult(
            test_name="concurrent_throughput",
            implementation=implementation,
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
            avg_latency=avg_latency,
            median_latency=median_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            throughput_rps=throughput,
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_percent=end_cpu - start_cpu,
            cache_hit_rate=cache_hit_rate,
            error_details=errors[:5]
        )
    
    async def benchmark_express_lane(self, num_requests: int = 30) -> BenchmarkResult:
        """Benchmark enhanced express lane performance."""
        logger.info(f"Testing express lane performance ({num_requests} requests)")
        
        from tools.enhanced_ollama_bridge import get_enhanced_ollama_bridge, analyze_sentiment_express
        
        bridge = await get_enhanced_ollama_bridge()
        
        # Warm up
        await analyze_sentiment_express("Warm up", "TEST", timeout=1.0)
        
        # Measure performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        latencies = []
        successful = 0
        failed = 0
        errors = []
        
        for i in range(num_requests):
            text = self.test_data["positive_news"][i % len(self.test_data["positive_news"])]
            symbol = self.test_data["symbols"][i % len(self.test_data["symbols"])]
            
            request_start = time.time()
            try:
                response = await analyze_sentiment_express(text, symbol, timeout=2.0)
                request_time = time.time() - request_start
                latencies.append(request_time)
                
                if response.success:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                request_time = time.time() - request_start
                latencies.append(request_time)
                failed += 1
                errors.append(str(e))
        
        total_time = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies) if latencies else 0
        median_latency = statistics.median(latencies) if latencies else 0
        p95_latency = self._percentile(latencies, 0.95) if latencies else 0
        p99_latency = self._percentile(latencies, 0.99) if latencies else 0
        throughput = successful / total_time if total_time > 0 else 0
        
        # Get metrics
        cache_hit_rate = None
        try:
            metrics = await bridge.get_enhanced_metrics()
            l1_stats = metrics.get("trading_enhancements", {}).get("cache_tiers", {}).get("l1_stats", {})
            cache_hit_rate = l1_stats.get("hit_rate", 0)
        except:
            pass
        
        return BenchmarkResult(
            test_name="express_lane",
            implementation="enhanced_express",
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
            avg_latency=avg_latency,
            median_latency=median_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            throughput_rps=throughput,
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_percent=0,  # Not measured for this test
            cache_hit_rate=cache_hit_rate,
            error_details=errors[:5]
        )
    
    async def benchmark_batch_processing(self) -> BenchmarkResult:
        """Benchmark batch processing performance."""
        logger.info("Testing batch processing performance")
        
        from tools.enhanced_ollama_bridge import get_enhanced_ollama_bridge, batch_multi_symbol_sentiment, RequestPriority
        
        bridge = await get_enhanced_ollama_bridge()
        
        # Create batch data
        symbol_texts = {}
        for i in range(10):  # 10 symbols
            symbol = self.test_data["symbols"][i]
            text = self.test_data["positive_news"][i % len(self.test_data["positive_news"])]
            symbol_texts[symbol] = text
        
        # Measure performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            results = await batch_multi_symbol_sentiment(symbol_texts, RequestPriority.NORMAL)
            
            total_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            successful = len([r for r in results.values() if r.success])
            failed = len([r for r in results.values() if not r.success])
            
            # Calculate average latency across batch
            avg_latency = total_time / len(symbol_texts) if symbol_texts else 0
            throughput = successful / total_time if total_time > 0 else 0
            
            return BenchmarkResult(
                test_name="batch_processing",
                implementation="enhanced_batch",
                total_requests=len(symbol_texts),
                successful_requests=successful,
                failed_requests=failed,
                total_time=total_time,
                avg_latency=avg_latency,
                median_latency=avg_latency,
                p95_latency=avg_latency,
                p99_latency=avg_latency,
                throughput_rps=throughput,
                memory_usage_mb=end_memory - start_memory,
                cpu_usage_percent=0,
                cache_hit_rate=None,
                error_details=[]
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="batch_processing",
                implementation="enhanced_batch",
                total_requests=len(symbol_texts),
                successful_requests=0,
                failed_requests=len(symbol_texts),
                total_time=time.time() - start_time,
                avg_latency=0,
                median_latency=0,
                p95_latency=0,
                p99_latency=0,
                throughput_rps=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                cache_hit_rate=None,
                error_details=[str(e)]
            )
    
    async def _test_enhanced_single_request(self, bridge, text: str, symbol: str):
        """Test single request with enhanced bridge."""
        from tools.enhanced_ollama_bridge import RequestPriority
        
        response = await bridge.query_async(
            prompt=f"Analyze sentiment for {symbol}: {text}",
            priority=RequestPriority.NORMAL,
            context="sentiment_analysis",
            timeout=30.0,
            metadata={"symbol": symbol}
        )
        return response
    
    async def _test_standard_single_request(self, bridge, text: str, symbol: str):
        """Test single request with standard bridge."""
        from tools.ollama_async_bridge import query_ollama_sentiment, RequestPriority
        
        response = await query_ollama_sentiment(
            text=text,
            symbol=symbol,
            priority=RequestPriority.NORMAL,
            timeout=30.0
        )
        return response
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * percentile
        f = int(k)
        c = k - f
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    
    def print_results(self):
        """Print formatted benchmark results."""
        print("\n" + "="*100)
        print("OLLAMA PERFORMANCE BENCHMARK RESULTS")
        print("="*100)
        
        for result in self.results:
            print(f"\n{result.test_name.upper()} - {result.implementation}")
            print("-" * 60)
            print(f"Total Requests:     {result.total_requests}")
            print(f"Successful:         {result.successful_requests} ({result.successful_requests/result.total_requests*100:.1f}%)")
            print(f"Failed:             {result.failed_requests} ({result.failed_requests/result.total_requests*100:.1f}%)")
            print(f"Total Time:         {result.total_time:.2f}s")
            print(f"Average Latency:    {result.avg_latency*1000:.1f}ms")
            print(f"Median Latency:     {result.median_latency*1000:.1f}ms")
            print(f"95th Percentile:    {result.p95_latency*1000:.1f}ms")
            print(f"99th Percentile:    {result.p99_latency*1000:.1f}ms")
            print(f"Throughput:         {result.throughput_rps:.1f} req/s")
            print(f"Memory Usage:       {result.memory_usage_mb:.1f} MB")
            
            if result.cache_hit_rate is not None:
                print(f"Cache Hit Rate:     {result.cache_hit_rate*100:.1f}%")
            
            if result.error_details:
                print(f"Sample Errors:      {', '.join(result.error_details[:3])}")
    
    def export_results(self, filename: Optional[str] = None) -> str:
        """Export results to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ollama_benchmark_results_{timestamp}.json"
        
        export_data = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}"
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "implementation": r.implementation,
                    "total_requests": r.total_requests,
                    "successful_requests": r.successful_requests,
                    "failed_requests": r.failed_requests,
                    "total_time": r.total_time,
                    "avg_latency_ms": r.avg_latency * 1000,
                    "median_latency_ms": r.median_latency * 1000,
                    "p95_latency_ms": r.p95_latency * 1000,
                    "p99_latency_ms": r.p99_latency * 1000,
                    "throughput_rps": r.throughput_rps,
                    "memory_usage_mb": r.memory_usage_mb,
                    "cpu_usage_percent": r.cpu_usage_percent,
                    "cache_hit_rate": r.cache_hit_rate,
                    "error_details": r.error_details
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Benchmark results exported to {filename}")
        return filename
    
    async def run_full_benchmark(self) -> List[BenchmarkResult]:
        """Run complete benchmark suite."""
        logger.info("Starting full Ollama performance benchmark suite")
        
        # Clear any existing results
        self.results.clear()
        gc.collect()  # Clean up memory before testing
        
        try:
            # Test 1: Standard bridge single request latency
            logger.info("Running standard bridge single request benchmark...")
            result = await self.benchmark_single_request_latency("standard", 30)
            self.results.append(result)
            
            # Small delay between tests
            await asyncio.sleep(2)
            
            # Test 2: Enhanced bridge single request latency
            logger.info("Running enhanced bridge single request benchmark...")
            result = await self.benchmark_single_request_latency("enhanced", 30)
            self.results.append(result)
            
            await asyncio.sleep(2)
            
            # Test 3: Standard bridge concurrent throughput
            logger.info("Running standard bridge concurrent throughput benchmark...")
            result = await self.benchmark_concurrent_throughput("standard", 10, 50)
            self.results.append(result)
            
            await asyncio.sleep(2)
            
            # Test 4: Enhanced bridge concurrent throughput
            logger.info("Running enhanced bridge concurrent throughput benchmark...")
            result = await self.benchmark_concurrent_throughput("enhanced", 10, 50)
            self.results.append(result)
            
            await asyncio.sleep(2)
            
            # Test 5: Express lane performance (enhanced only)
            logger.info("Running express lane benchmark...")
            result = await self.benchmark_express_lane(20)
            self.results.append(result)
            
            await asyncio.sleep(2)
            
            # Test 6: Batch processing performance (enhanced only)
            logger.info("Running batch processing benchmark...")
            result = await self.benchmark_batch_processing()
            self.results.append(result)
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise
        
        logger.info("Full benchmark suite completed")
        return self.results


async def main():
    """Run the performance benchmark."""
    benchmark = PerformanceBenchmark()
    
    try:
        # Run full benchmark
        results = await benchmark.run_full_benchmark()
        
        # Print results
        benchmark.print_results()
        
        # Export results
        filename = benchmark.export_results()
        
        # Print summary comparison
        print("\n" + "="*100)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*100)
        
        # Find comparable tests
        standard_single = next((r for r in results if r.test_name == "single_request_latency" and r.implementation == "standard"), None)
        enhanced_single = next((r for r in results if r.test_name == "single_request_latency" and r.implementation == "enhanced"), None)
        
        if standard_single and enhanced_single:
            latency_improvement = (standard_single.avg_latency - enhanced_single.avg_latency) / standard_single.avg_latency * 100
            throughput_improvement = (enhanced_single.throughput_rps - standard_single.throughput_rps) / standard_single.throughput_rps * 100
            
            print(f"Single Request Latency Improvement: {latency_improvement:+.1f}%")
            print(f"Single Request Throughput Improvement: {throughput_improvement:+.1f}%")
        
        # Express lane performance
        express_result = next((r for r in results if r.test_name == "express_lane"), None)
        if express_result:
            print(f"Express Lane Average Latency: {express_result.avg_latency*1000:.1f}ms")
            print(f"Express Lane 95th Percentile: {express_result.p95_latency*1000:.1f}ms")
        
        print(f"\nDetailed results exported to: {filename}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)