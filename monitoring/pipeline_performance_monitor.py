#!/usr/bin/env python3
"""
ML Pipeline Performance Monitor
Real-time monitoring and optimization of the trading system ML pipeline.
Provides actionable insights and automatic performance tuning.
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from collections import deque, defaultdict
import psutil
import os

logger = logging.getLogger(__name__)

class PerformanceLevel(Enum):
    """System performance levels."""
    EXCELLENT = "excellent"    # >95% of targets
    GOOD = "good"             # 80-95% of targets  
    DEGRADED = "degraded"     # 60-80% of targets
    POOR = "poor"             # <60% of targets
    CRITICAL = "critical"     # System issues detected

@dataclass
class PipelineStageMetrics:
    """Metrics for a single pipeline stage."""
    stage_name: str
    avg_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    success_rate: float = 100.0
    error_count: int = 0
    total_runs: int = 0
    last_run_time: Optional[float] = None
    target_duration: float = 30.0  # Default 30s target
    
    @property
    def performance_ratio(self) -> float:
        """Performance vs target (1.0 = meeting target, >1.0 = better than target)."""
        if self.avg_duration == 0:
            return 1.0
        return self.target_duration / self.avg_duration
    
    @property
    def performance_level(self) -> PerformanceLevel:
        """Determine performance level."""
        ratio = self.performance_ratio
        success = self.success_rate
        
        if ratio >= 0.95 and success >= 98:
            return PerformanceLevel.EXCELLENT
        elif ratio >= 0.80 and success >= 95:
            return PerformanceLevel.GOOD
        elif ratio >= 0.60 and success >= 90:
            return PerformanceLevel.DEGRADED
        elif success >= 80:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL

@dataclass 
class SystemResourceMetrics:
    """System resource utilization metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def resource_pressure(self) -> PerformanceLevel:
        """Overall resource pressure level."""
        if self.cpu_percent > 90 or self.memory_percent > 85:
            return PerformanceLevel.CRITICAL
        elif self.cpu_percent > 80 or self.memory_percent > 75:
            return PerformanceLevel.POOR
        elif self.cpu_percent > 70 or self.memory_percent > 65:
            return PerformanceLevel.DEGRADED
        elif self.cpu_percent > 50 or self.memory_percent > 50:
            return PerformanceLevel.GOOD
        else:
            return PerformanceLevel.EXCELLENT

@dataclass
class OptimizationSuggestion:
    """Actionable optimization suggestion."""
    category: str  # 'caching', 'parallelization', 'resource_usage', 'api_optimization'
    priority: str  # 'critical', 'high', 'medium', 'low'
    title: str
    description: str
    expected_improvement: str  # e.g., "20-30% faster", "50% fewer API calls"
    implementation_complexity: str  # 'low', 'medium', 'high'
    estimated_impact_score: float  # 0-10 scale

class PipelinePerformanceMonitor:
    """
    Comprehensive ML pipeline performance monitor with real-time optimization.
    Tracks performance, identifies bottlenecks, and provides actionable improvements.
    """
    
    def __init__(self):
        # Performance targets for each stage
        self.performance_targets = {
            'market_monitor': 5.0,        # 5 seconds
            'universe_filter': 8.0,       # 8 seconds
            'sentiment_analysis': 15.0,    # 15 seconds (down from 30s)
            'risk_assessment': 3.0,        # 3 seconds
            'hybrid_portfolio_decision': 10.0,  # 10 seconds
            'signal_generation': 5.0,      # 5 seconds
            'strategy_optimization': 3.0,   # 3 seconds
            'order_management': 8.0,       # 8 seconds
            'portfolio_tracking': 2.0,     # 2 seconds
            'total_pipeline': 45.0         # Total target: 45 seconds (down from 78s)
        }
        
        # Metrics tracking
        self.stage_metrics: Dict[str, PipelineStageMetrics] = {}
        self.resource_metrics: deque = deque(maxlen=100)  # Last 100 resource samples
        self.pipeline_run_history: deque = deque(maxlen=50)  # Last 50 pipeline runs
        
        # Performance optimization tracking
        self.bottleneck_history: deque = deque(maxlen=20)
        self.optimization_suggestions: List[OptimizationSuggestion] = []
        
        # Monitoring control
        self.monitoring_enabled = True
        self.resource_monitoring_task = None
        self.alert_thresholds = {
            'pipeline_duration_multiplier': 1.5,  # Alert if >1.5x target
            'success_rate_threshold': 90.0,       # Alert if <90% success
            'resource_usage_threshold': 80.0      # Alert if >80% resource usage
        }
        
        # Initialize stage metrics
        for stage_name, target in self.performance_targets.items():
            self.stage_metrics[stage_name] = PipelineStageMetrics(
                stage_name=stage_name, 
                target_duration=target
            )
        
        logger.info("🎯 PipelinePerformanceMonitor initialized with optimized targets")
        
    async def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self.resource_monitoring_task:
            return
            
        self.resource_monitoring_task = asyncio.create_task(self._resource_monitoring_loop())
        logger.info("📊 Started continuous pipeline performance monitoring")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if self.resource_monitoring_task:
            self.resource_monitoring_task.cancel()
            self.resource_monitoring_task = None
            logger.info("📊 Stopped pipeline performance monitoring")
    
    def record_stage_performance(self, stage_name: str, duration: float, success: bool = True):
        """Record performance metrics for a pipeline stage."""
        if stage_name not in self.stage_metrics:
            self.stage_metrics[stage_name] = PipelineStageMetrics(
                stage_name=stage_name,
                target_duration=self.performance_targets.get(stage_name, 30.0)
            )
        
        metrics = self.stage_metrics[stage_name]
        
        # Update duration statistics
        metrics.total_runs += 1
        metrics.last_run_time = time.time()
        
        if success:
            # Update duration stats only for successful runs
            if metrics.avg_duration == 0:
                metrics.avg_duration = duration
            else:
                # Exponential moving average with more weight on recent values
                alpha = 0.3
                metrics.avg_duration = alpha * duration + (1 - alpha) * metrics.avg_duration
            
            metrics.min_duration = min(metrics.min_duration, duration)
            metrics.max_duration = max(metrics.max_duration, duration)
        else:
            metrics.error_count += 1
        
        # Update success rate
        metrics.success_rate = ((metrics.total_runs - metrics.error_count) / metrics.total_runs) * 100
        
        # Check for performance issues
        if duration > metrics.target_duration * self.alert_thresholds['pipeline_duration_multiplier']:
            self._record_performance_issue(stage_name, duration, metrics.target_duration)
        
        logger.debug(f"📊 {stage_name}: {duration:.2f}s (target: {metrics.target_duration:.2f}s, "
                    f"avg: {metrics.avg_duration:.2f}s, success: {metrics.success_rate:.1f}%)")
    
    def record_pipeline_run(self, total_duration: float, stages_data: Dict[str, float], 
                          signals_generated: int, orders_executed: int, success: bool = True):
        """Record complete pipeline run metrics."""
        
        # Record individual stage metrics
        for stage_name, duration in stages_data.items():
            self.record_stage_performance(stage_name, duration, success)
        
        # Record total pipeline performance
        self.record_stage_performance('total_pipeline', total_duration, success)
        
        # Store pipeline run summary
        run_data = {
            'timestamp': time.time(),
            'total_duration': total_duration,
            'stages': stages_data.copy(),
            'signals_generated': signals_generated,
            'orders_executed': orders_executed,
            'signal_conversion_rate': (orders_executed / max(signals_generated, 1)) * 100,
            'success': success
        }
        self.pipeline_run_history.append(run_data)
        
        # Analyze performance and update optimization suggestions
        self._analyze_performance_and_suggest_optimizations()
        
        # Log performance summary
        conversion_rate = run_data['signal_conversion_rate']
        target_duration = self.performance_targets['total_pipeline']
        performance_vs_target = (target_duration / total_duration) * 100
        
        logger.info(f"🎯 Pipeline run complete: {total_duration:.1f}s "
                   f"({performance_vs_target:.1f}% of target), "
                   f"{signals_generated} signals → {orders_executed} orders "
                   f"({conversion_rate:.1f}% conversion)")
        
        if total_duration > target_duration * 1.2:
            logger.warning(f"⚠️ Pipeline running {((total_duration / target_duration - 1) * 100):.1f}% "
                          f"slower than target")
    
    def _record_performance_issue(self, stage_name: str, actual_duration: float, target_duration: float):
        """Record a performance issue for analysis."""
        issue = {
            'timestamp': time.time(),
            'stage': stage_name,
            'actual_duration': actual_duration,
            'target_duration': target_duration,
            'slowdown_factor': actual_duration / target_duration
        }
        self.bottleneck_history.append(issue)
        
        logger.debug(f"🐌 Performance issue: {stage_name} took {actual_duration:.2f}s "
                    f"(target: {target_duration:.2f}s, {issue['slowdown_factor']:.1f}x slower)")
    
    async def _resource_monitoring_loop(self):
        """Background loop for resource monitoring."""
        while self.monitoring_enabled:
            try:
                # Collect system resource metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Network I/O (simplified)
                net_io = psutil.net_io_counters()
                network_mb = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)
                
                resource_metrics = SystemResourceMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / (1024 * 1024),
                    disk_usage_percent=disk.percent,
                    network_io_mb=network_mb
                )
                
                self.resource_metrics.append(resource_metrics)
                
                # Check for resource pressure
                if resource_metrics.resource_pressure in [PerformanceLevel.CRITICAL, PerformanceLevel.POOR]:
                    logger.warning(f"🚨 High resource pressure detected: "
                                 f"CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%")
                
                await asyncio.sleep(10)  # Sample every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(30)
    
    def _analyze_performance_and_suggest_optimizations(self):
        """Analyze recent performance and generate optimization suggestions."""
        if len(self.pipeline_run_history) < 5:
            return  # Need more data
        
        # Clear old suggestions
        self.optimization_suggestions.clear()
        
        # Analyze bottlenecks
        recent_runs = list(self.pipeline_run_history)[-10:]  # Last 10 runs
        stage_performance = defaultdict(list)
        
        for run in recent_runs:
            for stage, duration in run['stages'].items():
                stage_performance[stage].append(duration)
        
        # Identify consistently slow stages
        slow_stages = []
        for stage, durations in stage_performance.items():
            if stage in self.performance_targets:
                avg_duration = np.mean(durations)
                target = self.performance_targets[stage]
                if avg_duration > target * 1.3:  # >30% slower than target
                    slow_stages.append((stage, avg_duration, target))
        
        # Generate specific optimization suggestions
        if slow_stages:
            slow_stages.sort(key=lambda x: x[1] / x[2], reverse=True)  # Sort by slowdown factor
            
            for stage, avg_duration, target in slow_stages[:3]:  # Top 3 bottlenecks
                suggestions = self._generate_stage_specific_optimizations(stage, avg_duration, target)
                self.optimization_suggestions.extend(suggestions)
        
        # Analyze signal conversion efficiency
        recent_conversion_rates = [run['signal_conversion_rate'] for run in recent_runs]
        avg_conversion = np.mean(recent_conversion_rates)
        
        if avg_conversion < 30:  # <30% conversion rate
            self.optimization_suggestions.append(OptimizationSuggestion(
                category='signal_processing',
                priority='high',
                title='Improve Signal-to-Order Conversion Rate',
                description=f'Current conversion rate is {avg_conversion:.1f}%. '
                           'Consider refining signal quality thresholds and position sizing logic.',
                expected_improvement='Increase conversion rate to 35-45%',
                implementation_complexity='medium',
                estimated_impact_score=7.5
            ))
    
    def _generate_stage_specific_optimizations(self, stage: str, avg_duration: float, 
                                              target: float) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions for specific pipeline stages."""
        suggestions = []
        slowdown_factor = avg_duration / target
        
        if stage == 'sentiment_analysis':
            suggestions.append(OptimizationSuggestion(
                category='caching',
                priority='critical' if slowdown_factor > 2.0 else 'high',
                title='Optimize Sentiment Analysis Caching',
                description=f'Sentiment analysis is {slowdown_factor:.1f}x slower than target. '
                           'Implement intelligent caching and batch processing for GPT-5-nano calls.',
                expected_improvement='60-70% faster sentiment analysis',
                implementation_complexity='medium',
                estimated_impact_score=9.0
            ))
            
            if slowdown_factor > 1.8:
                suggestions.append(OptimizationSuggestion(
                    category='api_optimization',
                    priority='high',
                    title='Implement Ollama Load Balancing',
                    description='Set up multiple Ollama instances with intelligent load balancing '
                               'to reduce sentiment analysis bottlenecks.',
                    expected_improvement='40-50% faster processing',
                    implementation_complexity='high',
                    estimated_impact_score=8.0
                ))
        
        elif stage == 'universe_filter':
            suggestions.append(OptimizationSuggestion(
                category='api_optimization',
                priority='high',
                title='Implement Market Data Circuit Breakers',
                description=f'Universe filtering is {slowdown_factor:.1f}x slower than target. '
                           'YFinance rate limiting is likely the cause. Implement circuit breakers '
                           'and enhanced caching.',
                expected_improvement='50-60% faster universe filtering',
                implementation_complexity='medium',
                estimated_impact_score=8.5
            ))
        
        elif stage == 'hybrid_portfolio_decision':
            suggestions.append(OptimizationSuggestion(
                category='parallelization',
                priority='medium',
                title='Parallelize RL Portfolio Computations',
                description=f'Portfolio decision stage is {slowdown_factor:.1f}x slower than target. '
                           'Implement parallel processing for RL model inference.',
                expected_improvement='30-40% faster RL processing',
                implementation_complexity='medium',
                estimated_impact_score=7.0
            ))
        
        return suggestions
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        # Overall system performance
        total_metrics = self.stage_metrics.get('total_pipeline')
        overall_performance = total_metrics.performance_level if total_metrics else PerformanceLevel.GOOD
        
        # Recent resource metrics
        recent_resources = list(self.resource_metrics)[-10:] if self.resource_metrics else []
        avg_cpu = np.mean([r.cpu_percent for r in recent_resources]) if recent_resources else 0
        avg_memory = np.mean([r.memory_percent for r in recent_resources]) if recent_resources else 0
        
        # Pipeline efficiency metrics
        recent_runs = list(self.pipeline_run_history)[-10:] if self.pipeline_run_history else []
        if recent_runs:
            avg_duration = np.mean([r['total_duration'] for r in recent_runs])
            avg_conversion = np.mean([r['signal_conversion_rate'] for r in recent_runs])
            success_rate = np.mean([r['success'] for r in recent_runs]) * 100
        else:
            avg_duration = avg_conversion = success_rate = 0
        
        # Top bottlenecks
        bottlenecks = []
        for stage_name, metrics in self.stage_metrics.items():
            if metrics.total_runs > 0 and metrics.performance_ratio < 0.8:
                bottlenecks.append({
                    'stage': stage_name,
                    'performance_ratio': metrics.performance_ratio,
                    'avg_duration': metrics.avg_duration,
                    'target_duration': metrics.target_duration
                })
        bottlenecks.sort(key=lambda x: x['performance_ratio'])
        
        return {
            'overall_performance': overall_performance.value,
            'pipeline_metrics': {
                'avg_duration_seconds': avg_duration,
                'target_duration_seconds': self.performance_targets['total_pipeline'],
                'performance_vs_target_percent': (self.performance_targets['total_pipeline'] / max(avg_duration, 1)) * 100,
                'success_rate_percent': success_rate,
                'avg_signal_conversion_percent': avg_conversion
            },
            'resource_utilization': {
                'avg_cpu_percent': avg_cpu,
                'avg_memory_percent': avg_memory,
                'resource_pressure': 'high' if avg_cpu > 70 or avg_memory > 70 else 'normal'
            },
            'stage_performance': [asdict(metrics) for metrics in self.stage_metrics.values()],
            'top_bottlenecks': bottlenecks[:5],
            'optimization_suggestions': [asdict(s) for s in self.optimization_suggestions[:10]],
            'recent_runs_count': len(self.pipeline_run_history),
            'monitoring_duration_hours': (time.time() - (self.resource_metrics[0].timestamp if self.resource_metrics else time.time())) / 3600
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate detailed optimization report with actionable recommendations."""
        dashboard = self.get_performance_dashboard()
        
        # Prioritize optimizations by impact
        high_impact_optimizations = [
            s for s in self.optimization_suggestions 
            if s.estimated_impact_score >= 7.0 and s.priority in ['critical', 'high']
        ]
        
        # Calculate potential performance gains
        total_potential_improvement = sum(s.estimated_impact_score for s in high_impact_optimizations)
        estimated_time_savings = total_potential_improvement * 2  # Rough estimate: 2s per impact point
        
        return {
            'executive_summary': {
                'current_performance_level': dashboard['overall_performance'],
                'pipeline_vs_target_percent': dashboard['pipeline_metrics']['performance_vs_target_percent'],
                'high_impact_optimizations_count': len(high_impact_optimizations),
                'estimated_time_savings_seconds': estimated_time_savings,
                'estimated_efficiency_gain_percent': min(50, total_potential_improvement * 3)
            },
            'detailed_analysis': dashboard,
            'recommended_actions': sorted(
                high_impact_optimizations, 
                key=lambda x: (x.estimated_impact_score, x.priority == 'critical'), 
                reverse=True
            )[:5],
            'generated_at': datetime.now().isoformat()
        }
    
    async def export_performance_data(self, filepath: str):
        """Export performance data for analysis."""
        data = {
            'stage_metrics': {k: asdict(v) for k, v in self.stage_metrics.items()},
            'recent_runs': list(self.pipeline_run_history),
            'resource_metrics': [asdict(r) for r in list(self.resource_metrics)],
            'optimization_suggestions': [asdict(s) for s in self.optimization_suggestions],
            'performance_targets': self.performance_targets,
            'exported_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"📊 Performance data exported to {filepath}")

# Global instance
pipeline_monitor = PipelinePerformanceMonitor()