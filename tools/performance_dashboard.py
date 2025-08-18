#!/usr/bin/env python3
"""
Performance Dashboard CLI
Real-time performance monitoring and optimization dashboard for the ML trading pipeline.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import argparse

from monitoring.pipeline_performance_monitor import pipeline_monitor
from core.optimized_market_data_cache import optimized_cache
from core.enhanced_sentiment_performance import sentiment_performance_manager

class PerformanceDashboard:
    """Interactive performance dashboard for ML pipeline optimization."""
    
    def __init__(self):
        self.running = False
    
    def print_header(self):
        """Print dashboard header."""
        print("\n" + "="*80)
        print("🎯 ML TRADING PIPELINE PERFORMANCE DASHBOARD")
        print("="*80)
        print(f"📊 Real-time monitoring at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def print_pipeline_overview(self, dashboard_data: Dict[str, Any]):
        """Print pipeline performance overview."""
        metrics = dashboard_data['pipeline_metrics']
        overall = dashboard_data['overall_performance']
        
        # Performance status indicator
        status_icon = {
            'excellent': '🟢',
            'good': '🟡', 
            'degraded': '🟠',
            'poor': '🔴',
            'critical': '💥'
        }
        
        print(f"🚀 PIPELINE PERFORMANCE: {status_icon.get(overall, '❓')} {overall.upper()}")
        print(f"   Current Runtime: {metrics['avg_duration_seconds']:.1f}s")
        print(f"   Target Runtime:  {metrics['target_duration_seconds']:.1f}s")
        print(f"   Performance vs Target: {metrics['performance_vs_target_percent']:.1f}%")
        print(f"   Success Rate: {metrics['success_rate_percent']:.1f}%")
        print(f"   Signal Conversion: {metrics['avg_signal_conversion_percent']:.1f}%")
        print()
    
    def print_stage_breakdown(self, dashboard_data: Dict[str, Any]):
        """Print detailed stage performance breakdown."""
        print("📊 STAGE PERFORMANCE BREAKDOWN")
        print("-" * 50)
        
        stages = dashboard_data['stage_performance']
        stages.sort(key=lambda x: x.get('avg_duration', 0), reverse=True)
        
        for stage in stages[:8]:  # Top 8 stages
            name = stage['stage_name']
            avg_duration = stage.get('avg_duration', 0)
            target = stage.get('target_duration', 10)
            performance_ratio = stage.get('performance_ratio', 1.0)
            success_rate = stage.get('success_rate', 100)
            
            # Performance indicator
            if performance_ratio >= 0.95:
                indicator = "🟢"
            elif performance_ratio >= 0.8:
                indicator = "🟡"
            elif performance_ratio >= 0.6:
                indicator = "🟠"
            else:
                indicator = "🔴"
            
            print(f"{indicator} {name:25} {avg_duration:6.2f}s (target: {target:5.1f}s) "
                  f"{performance_ratio*100:6.1f}% {success_rate:6.1f}% success")
        
        print()
    
    def print_bottlenecks_and_optimizations(self, dashboard_data: Dict[str, Any]):
        """Print current bottlenecks and optimization suggestions."""
        bottlenecks = dashboard_data['top_bottlenecks']
        optimizations = dashboard_data['optimization_suggestions']
        
        if bottlenecks:
            print("🐌 TOP PERFORMANCE BOTTLENECKS")
            print("-" * 40)
            for i, bottleneck in enumerate(bottlenecks[:3], 1):
                stage = bottleneck['stage']
                ratio = bottleneck['performance_ratio']
                avg_dur = bottleneck['avg_duration']
                target_dur = bottleneck['target_duration']
                slowdown = avg_dur / target_dur
                
                print(f"{i}. {stage}")
                print(f"   Performance: {ratio*100:.1f}% of target ({slowdown:.1f}x slower)")
                print(f"   Duration: {avg_dur:.2f}s vs {target_dur:.2f}s target")
                print()
        
        if optimizations:
            print("💡 OPTIMIZATION RECOMMENDATIONS")
            print("-" * 40)
            high_impact = [opt for opt in optimizations if opt.get('estimated_impact_score', 0) >= 7.0]
            
            for i, opt in enumerate(high_impact[:3], 1):
                print(f"{i}. [{opt.get('priority', 'medium').upper()}] {opt.get('title', 'N/A')}")
                print(f"   Category: {opt.get('category', 'N/A')}")
                print(f"   Expected: {opt.get('expected_improvement', 'N/A')}")
                print(f"   Impact Score: {opt.get('estimated_impact_score', 0):.1f}/10")
                print(f"   Complexity: {opt.get('implementation_complexity', 'N/A')}")
                print()
    
    def print_resource_utilization(self, dashboard_data: Dict[str, Any]):
        """Print system resource utilization."""
        resources = dashboard_data['resource_utilization']
        
        print("💻 SYSTEM RESOURCE UTILIZATION")
        print("-" * 35)
        print(f"CPU Usage:    {resources['avg_cpu_percent']:6.1f}%")
        print(f"Memory Usage: {resources['avg_memory_percent']:6.1f}%")
        print(f"Pressure:     {resources['resource_pressure'].upper()}")
        print()
    
    def print_cache_performance(self):
        """Print cache performance metrics."""
        try:
            cache_metrics = optimized_cache.get_performance_metrics()
            sentiment_metrics = sentiment_performance_manager.get_performance_metrics()
            
            print("🚀 CACHE PERFORMANCE")
            print("-" * 25)
            print(f"Market Data Cache Hit Rate: {cache_metrics.get('cache_hit_rate_percent', 0):.1f}%")
            print(f"API Calls Saved: {cache_metrics.get('api_calls_saved', 0):,}")
            print(f"Sentiment Cache Hit Rate: {sentiment_metrics.get('cache_hit_rate_percent', 0):.1f}%")
            print(f"Estimated Cost Savings: ${sentiment_metrics.get('cost_savings_usd', 0):.2f}")
            
            # Circuit breaker status
            cb_status = cache_metrics.get('circuit_breaker_status', {})
            if cb_status:
                print(f"Circuit Breakers:")
                for name, status in cb_status.items():
                    icon = "🟢" if status == "CLOSED" else "🔴"
                    print(f"  {icon} {name}: {status}")
            print()
            
        except Exception as e:
            print(f"⚠️ Cache metrics unavailable: {e}")
            print()
    
    async def show_dashboard(self, refresh_interval: int = 30):
        """Show live performance dashboard."""
        print("Starting Performance Dashboard... (Ctrl+C to exit)")
        
        try:
            while True:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H")
                
                # Get current performance data
                dashboard_data = pipeline_monitor.get_performance_dashboard()
                
                # Display dashboard sections
                self.print_header()
                self.print_pipeline_overview(dashboard_data)
                self.print_stage_breakdown(dashboard_data)
                self.print_bottlenecks_and_optimizations(dashboard_data)
                self.print_resource_utilization(dashboard_data)
                self.print_cache_performance()
                
                # Show refresh info
                print(f"📡 Auto-refreshing every {refresh_interval}s | Last updated: {datetime.now().strftime('%H:%M:%S')}")
                print("Press Ctrl+C to exit")
                
                await asyncio.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped by user")
    
    async def show_optimization_report(self):
        """Show detailed optimization report."""
        print("Generating Performance Optimization Report...")
        
        report = pipeline_monitor.get_optimization_report()
        
        self.print_header()
        
        # Executive summary
        summary = report['executive_summary']
        print("📋 EXECUTIVE SUMMARY")
        print("-" * 30)
        print(f"Current Performance Level: {summary['current_performance_level'].upper()}")
        print(f"Pipeline vs Target: {summary['pipeline_vs_target_percent']:.1f}%")
        print(f"High-Impact Optimizations Available: {summary['high_impact_optimizations_count']}")
        print(f"Estimated Time Savings: {summary['estimated_time_savings_seconds']:.1f}s per cycle")
        print(f"Estimated Efficiency Gain: {summary['estimated_efficiency_gain_percent']:.1f}%")
        print()
        
        # Recommended actions
        actions = report['recommended_actions']
        if actions:
            print("🎯 RECOMMENDED ACTIONS (High Impact)")
            print("-" * 45)
            for i, action in enumerate(actions, 1):
                print(f"{i}. {action['title']}")
                print(f"   Priority: {action['priority'].upper()}")
                print(f"   Expected Improvement: {action['expected_improvement']}")
                print(f"   Implementation: {action['implementation_complexity']} complexity")
                print(f"   Impact Score: {action['estimated_impact_score']:.1f}/10")
                print(f"   Description: {action['description']}")
                print()
        
        print("📊 For detailed technical analysis, see the dashboard (--dashboard)")
    
    async def warm_caches(self, symbols: List[str] = None):
        """Warm system caches proactively."""
        if not symbols:
            symbols = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLK', 'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
        
        print(f"🔥 Warming caches for {len(symbols)} symbols...")
        
        # Warm market data cache
        market_results = await optimized_cache.warm_cache_for_symbols(symbols)
        market_success = sum(market_results.values())
        
        # Warm sentiment cache
        sentiment_results = await sentiment_performance_manager.warm_sentiment_cache(symbols)
        sentiment_success = sum(sentiment_results.values())
        
        print(f"✅ Cache warming complete:")
        print(f"   Market Data: {market_success}/{len(symbols)} symbols")
        print(f"   Sentiment Data: {sentiment_success}/{len(symbols)} symbols")
    
    async def export_performance_data(self, filepath: str):
        """Export performance data for analysis."""
        print(f"📊 Exporting performance data to {filepath}...")
        await pipeline_monitor.export_performance_data(filepath)
        print("✅ Export complete")

async def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="ML Trading Pipeline Performance Dashboard")
    parser.add_argument("--dashboard", action="store_true", help="Show live performance dashboard")
    parser.add_argument("--report", action="store_true", help="Generate optimization report")
    parser.add_argument("--warm-cache", action="store_true", help="Warm system caches")
    parser.add_argument("--export", type=str, help="Export performance data to file")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols for cache warming")
    parser.add_argument("--refresh", type=int, default=30, help="Dashboard refresh interval (seconds)")
    
    args = parser.parse_args()
    
    dashboard = PerformanceDashboard()
    
    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    if args.dashboard:
        await dashboard.show_dashboard(refresh_interval=args.refresh)
    elif args.report:
        await dashboard.show_optimization_report()
    elif args.warm_cache:
        await dashboard.warm_caches(symbols)
    elif args.export:
        await dashboard.export_performance_data(args.export)
    else:
        # Default: show current status
        print("🎯 ML TRADING PIPELINE PERFORMANCE STATUS")
        print("=" * 50)
        
        # Quick status overview
        dashboard_data = pipeline_monitor.get_performance_dashboard()
        dashboard.print_pipeline_overview(dashboard_data)
        dashboard.print_cache_performance()
        
        print("Use --help for more options:")
        print("  --dashboard     Live performance dashboard")
        print("  --report        Detailed optimization report")
        print("  --warm-cache    Proactively warm caches")
        print("  --export FILE   Export performance data")

if __name__ == "__main__":
    asyncio.run(main())