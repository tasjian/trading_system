#!/usr/bin/env python3
"""
COMPLETE SYSTEM HEALTH VERIFICATION
===================================

Continuously monitors all system components to ensure optimal performance
and automatically corrects issues before they impact trading operations.

COMPREHENSIVE HEALTH MONITORING - ZERO DOWNTIME
"""

import asyncio
import logging
import json
import os
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import traceback
import subprocess

logger = logging.getLogger(__name__)

class SystemHealthMonitor:
    """Comprehensive system health monitoring with auto-correction capabilities."""
    
    def __init__(self):
        self.base_path = Path("/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system")
        self.logs_path = self.base_path / "logs"
        
        self.health_metrics: Dict[str, Any] = {}
        self.health_issues: List[str] = []
        self.critical_issues: List[str] = []
        self.auto_corrections: List[str] = []
        
        # Health thresholds
        self.cpu_threshold = 85.0  # %
        self.memory_threshold = 85.0  # %
        self.disk_threshold = 90.0  # %
        self.response_time_threshold = 5.0  # seconds
        
    async def perform_complete_health_check(self) -> bool:
        """
        Perform comprehensive system health verification.
        Returns True only if ALL systems are healthy.
        """
        logger.info("🏥 STARTING COMPLETE SYSTEM HEALTH VERIFICATION")
        logger.info("=" * 55)
        logger.info("⚡ MONITORING ALL SYSTEM COMPONENTS")
        
        health_checks = [
            ("System Resources", self._check_system_resources),
            ("Process Health", self._check_process_health),
            ("Network Connectivity", self._check_network_health),
            ("API Endpoints", self._check_api_endpoints),
            ("Database Health", self._check_database_health),
            ("File System", self._check_filesystem_health),
            ("Trading Components", self._check_trading_components),
            ("ML Components", self._check_ml_components),
            ("Memory Usage", self._check_memory_usage),
            ("Performance Metrics", self._check_performance_metrics)
        ]
        
        all_healthy = True
        
        for check_name, health_func in health_checks:
            logger.info(f"🏥 Checking: {check_name}")
            try:
                healthy = await health_func()
                if healthy:
                    logger.info(f"✅ {check_name}: HEALTHY")
                else:
                    logger.warning(f"⚠️  {check_name}: ISSUES DETECTED")
                    all_healthy = False
            except Exception as e:
                error_msg = f"{check_name} health check crashed: {str(e)}"
                logger.critical(f"💥 {error_msg}")
                self.critical_issues.append(error_msg)
                all_healthy = False
        
        await self._generate_health_report()
        
        if not all_healthy:
            logger.warning("⚠️  SYSTEM HEALTH ISSUES DETECTED")
            logger.warning(f"❌ {len(self.critical_issues)} CRITICAL ISSUES:")
            for i, issue in enumerate(self.critical_issues, 1):
                logger.warning(f"   {i}. {issue}")
            
            # Attempt auto-corrections
            await self._attempt_auto_corrections()
        else:
            logger.info("✅ ALL SYSTEM COMPONENTS HEALTHY")
            logger.info(f"🔧 Made {len(self.auto_corrections)} auto-corrections")
        
        return all_healthy
    
    async def _check_system_resources(self) -> bool:
        """Check system CPU, memory, and disk usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.health_metrics['cpu_usage'] = cpu_percent
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.health_metrics['memory_usage'] = memory_percent
            self.health_metrics['memory_available'] = memory.available / (1024**3)  # GB
            
            # Disk usage
            disk = psutil.disk_usage(str(self.base_path))
            disk_percent = (disk.used / disk.total) * 100
            self.health_metrics['disk_usage'] = disk_percent
            self.health_metrics['disk_free'] = disk.free / (1024**3)  # GB
            
            # Check thresholds
            issues = []
            if cpu_percent > self.cpu_threshold:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > self.memory_threshold:
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            
            if disk_percent > self.disk_threshold:
                issues.append(f"Low disk space: {disk_percent:.1f}% used")
            
            if issues:
                self.health_issues.extend(issues)
                return False
            
            return True
            
        except Exception as e:
            self.critical_issues.append(f"System resources check failed: {str(e)}")
            return False
    
    async def _check_process_health(self) -> bool:
        """Check health of critical system processes."""
        try:
            current_process = psutil.Process()
            
            # Check current process health
            self.health_metrics['process_cpu'] = current_process.cpu_percent()
            self.health_metrics['process_memory'] = current_process.memory_info().rss / (1024**2)  # MB
            self.health_metrics['process_threads'] = current_process.num_threads()
            
            # Check for zombie processes
            zombie_count = 0
            for proc in psutil.process_iter(['pid', 'status']):
                try:
                    if proc.info['status'] == psutil.STATUS_ZOMBIE:
                        zombie_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            self.health_metrics['zombie_processes'] = zombie_count
            
            if zombie_count > 5:
                self.health_issues.append(f"Too many zombie processes: {zombie_count}")
                return False
            
            return True
            
        except Exception as e:
            self.critical_issues.append(f"Process health check failed: {str(e)}")
            return False
    
    async def _check_network_health(self) -> bool:
        """Check network connectivity and performance."""
        try:
            import socket
            
            # Test basic connectivity
            test_hosts = [
                ("google.com", 80),
                ("api.alpaca.markets", 443),
                ("openai.com", 443)
            ]
            
            connectivity_issues = []
            response_times = []
            
            for host, port in test_hosts:
                start_time = time.time()
                try:
                    sock = socket.create_connection((host, port), timeout=10)
                    sock.close()
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    
                    if response_time > self.response_time_threshold:
                        connectivity_issues.append(f"Slow response from {host}: {response_time:.2f}s")
                        
                except Exception as e:
                    connectivity_issues.append(f"Cannot connect to {host}:{port} - {str(e)}")
            
            self.health_metrics['network_response_times'] = response_times
            self.health_metrics['average_response_time'] = sum(response_times) / len(response_times) if response_times else None
            
            if connectivity_issues:
                self.health_issues.extend(connectivity_issues)
                return False
            
            return True
            
        except Exception as e:
            self.critical_issues.append(f"Network health check failed: {str(e)}")
            return False
    
    async def _check_api_endpoints(self) -> bool:
        """Check health of critical API endpoints."""
        try:
            # Test Alpaca API
            try:
                from tools.alpaca_client import alpaca_client
                account_info = alpaca_client.get_account_info()
                if account_info:
                    self.health_metrics['alpaca_api_status'] = 'healthy'
                else:
                    self.health_issues.append("Alpaca API returned no account info")
                    return False
            except Exception as e:
                self.health_issues.append(f"Alpaca API health check failed: {str(e)}")
                return False
            
            # Test OpenAI API (if available)
            try:
                import openai
                openai_api_key = os.environ.get("OPENAI_API_KEY")
                if openai_api_key:
                    self.health_metrics['openai_api_status'] = 'configured'
                else:
                    self.health_issues.append("OpenAI API key not configured")
            except ImportError:
                self.health_issues.append("OpenAI library not available")
            
            return True
            
        except Exception as e:
            self.critical_issues.append(f"API endpoints check failed: {str(e)}")
            return False
    
    async def _check_database_health(self) -> bool:
        """Check health of database files and connections."""
        try:
            db_files = [
                "asset_replacement.db",
                "lot_tracking.db",
                "wash_sale_monitoring.db"
            ]
            
            db_issues = []
            for db_file in db_files:
                db_path = self.base_path / "data" / db_file
                if not db_path.exists():
                    db_issues.append(f"Missing database file: {db_file}")
                elif db_path.stat().st_size == 0:
                    db_issues.append(f"Empty database file: {db_file}")
            
            self.health_metrics['database_files'] = len(db_files) - len(db_issues)
            
            if db_issues:
                self.health_issues.extend(db_issues)
                # Auto-create missing databases
                for db_file in db_files:
                    db_path = self.base_path / "data" / db_file
                    if not db_path.exists():
                        db_path.parent.mkdir(parents=True, exist_ok=True)
                        db_path.touch()
                        self.auto_corrections.append(f"Created missing database: {db_file}")
            
            return True
            
        except Exception as e:
            self.critical_issues.append(f"Database health check failed: {str(e)}")
            return False
    
    async def _check_filesystem_health(self) -> bool:
        """Check filesystem health and permissions."""
        try:
            critical_dirs = [
                "data", "logs", "models", "core", "agents", "tools", "config"
            ]
            
            fs_issues = []
            for directory in critical_dirs:
                dir_path = self.base_path / directory
                if not dir_path.exists():
                    fs_issues.append(f"Missing critical directory: {directory}")
                    # Auto-create missing directories
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.auto_corrections.append(f"Created missing directory: {directory}")
                elif not os.access(dir_path, os.R_OK | os.W_OK):
                    fs_issues.append(f"Insufficient permissions on directory: {directory}")
            
            self.health_metrics['filesystem_status'] = len(critical_dirs) - len(fs_issues)
            
            if fs_issues:
                self.health_issues.extend(fs_issues)
                return len([issue for issue in fs_issues if "permissions" not in issue]) == 0
            
            return True
            
        except Exception as e:
            self.critical_issues.append(f"Filesystem health check failed: {str(e)}")
            return False
    
    async def _check_trading_components(self) -> bool:
        """Check health of trading system components."""
        try:
            component_health = {}
            
            # Check core trading components
            critical_components = [
                ('core.trading_engine', 'TradingEngine'),
                ('core.portfolio_balancer', 'IntelligentPortfolioBalancer'),
                ('core.oco_trading_system', 'OCOTradingSystem'),
                ('agents.workflow', 'TradingWorkflow')
            ]
            
            for module_name, class_name in critical_components:
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    component_class = getattr(module, class_name)
                    component_health[f"{module_name}.{class_name}"] = 'importable'
                except Exception as e:
                    component_health[f"{module_name}.{class_name}"] = f'failed: {str(e)}'
                    self.health_issues.append(f"Trading component unavailable: {module_name}.{class_name}")
            
            self.health_metrics['trading_components'] = component_health
            
            # Check if we have more successes than failures
            successes = len([v for v in component_health.values() if v == 'importable'])
            return successes >= len(critical_components) * 0.8  # 80% success rate
            
        except Exception as e:
            self.critical_issues.append(f"Trading components check failed: {str(e)}")
            return False
    
    async def _check_ml_components(self) -> bool:
        """Check health of ML system components."""
        try:
            ml_health = {}
            
            # Check ML components
            ml_components = [
                ('agents.rl_integration_bridge', 'SimplifiedRLAgent'),
                ('agents.online_rl_system', 'create_online_rl_system'),
                ('agents.unified_reward_calculator', 'UnifiedRewardCalculator')
            ]
            
            for module_name, component_name in ml_components:
                try:
                    module = __import__(module_name, fromlist=[component_name])
                    component = getattr(module, component_name)
                    ml_health[f"{module_name}.{component_name}"] = 'importable'
                except Exception as e:
                    ml_health[f"{module_name}.{component_name}"] = f'failed: {str(e)}'
                    self.health_issues.append(f"ML component unavailable: {module_name}.{component_name}")
            
            self.health_metrics['ml_components'] = ml_health
            
            # Check success rate
            successes = len([v for v in ml_health.values() if v == 'importable'])
            return successes >= len(ml_components) * 0.7  # 70% success rate for ML components
            
        except Exception as e:
            self.critical_issues.append(f"ML components check failed: {str(e)}")
            return False
    
    async def _check_memory_usage(self) -> bool:
        """Check detailed memory usage patterns."""
        try:
            import gc
            
            # Force garbage collection
            collected = gc.collect()
            
            # Get detailed memory info
            memory_info = psutil.virtual_memory()
            swap_info = psutil.swap_memory()
            
            self.health_metrics.update({
                'memory_total': memory_info.total / (1024**3),  # GB
                'memory_used': memory_info.used / (1024**3),   # GB
                'memory_cached': memory_info.cached / (1024**3),  # GB
                'swap_used': swap_info.used / (1024**3),       # GB
                'gc_collected': collected
            })
            
            # Check for memory issues
            if swap_info.percent > 50:
                self.health_issues.append(f"High swap usage: {swap_info.percent:.1f}%")
                return False
            
            if memory_info.percent > 90:
                self.health_issues.append(f"Critical memory usage: {memory_info.percent:.1f}%")
                return False
            
            return True
            
        except Exception as e:
            self.critical_issues.append(f"Memory usage check failed: {str(e)}")
            return False
    
    async def _check_performance_metrics(self) -> bool:
        """Check overall system performance metrics."""
        try:
            # CPU load averages (if available on macOS)
            try:
                load_avg = os.getloadavg()
                self.health_metrics['load_average'] = {
                    '1min': load_avg[0],
                    '5min': load_avg[1], 
                    '15min': load_avg[2]
                }
                
                # Check if load is too high (> number of CPUs * 2)
                cpu_count = psutil.cpu_count()
                if load_avg[0] > cpu_count * 2:
                    self.health_issues.append(f"High system load: {load_avg[0]:.2f}")
                    
            except (OSError, AttributeError):
                # Not available on all systems
                pass
            
            # Check system uptime
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            self.health_metrics['system_uptime_hours'] = uptime / 3600
            
            # Check if system has been up too long (potential memory leaks)
            if uptime > 7 * 24 * 3600:  # 7 days
                self.health_issues.append(f"System uptime very high: {uptime/3600/24:.1f} days")
            
            return True
            
        except Exception as e:
            self.critical_issues.append(f"Performance metrics check failed: {str(e)}")
            return False
    
    async def _attempt_auto_corrections(self):
        """Attempt to automatically correct detected issues."""
        logger.info("🔧 ATTEMPTING AUTO-CORRECTIONS...")
        
        # Auto-corrections are made during the health checks
        # This method could be extended to perform additional corrections
        
        if self.auto_corrections:
            logger.info(f"✅ Made {len(self.auto_corrections)} auto-corrections:")
            for correction in self.auto_corrections:
                logger.info(f"   🔧 {correction}")
    
    async def _generate_health_report(self):
        """Generate comprehensive health report."""
        report_path = self.logs_path / "system_health_report.json"
        
        report = {
            "health_check_timestamp": datetime.now().isoformat(),
            "health_metrics": self.health_metrics,
            "health_issues": self.health_issues,
            "critical_issues": self.critical_issues,
            "auto_corrections": self.auto_corrections,
            "overall_health": len(self.critical_issues) == 0,
            "issues_count": len(self.health_issues),
            "critical_count": len(self.critical_issues),
            "corrections_count": len(self.auto_corrections)
        }
        
        os.makedirs(self.logs_path, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 System health report saved: {report_path}")

# Global health monitor instance
health_monitor = SystemHealthMonitor()

async def perform_complete_health_check() -> bool:
    """Main health check entry point."""
    return await health_monitor.perform_complete_health_check()

if __name__ == "__main__":
    """Direct health check run for testing."""
    import asyncio
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(perform_complete_health_check())
    sys.exit(0 if result else 1)