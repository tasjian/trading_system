#!/usr/bin/env python3
"""
COMPREHENSIVE DEPENDENCY RESOLVER
=================================

Automatically detects, installs, and resolves all system dependencies
to ensure complete system functionality on every startup.

NO MANUAL INTERVENTION - FULL AUTO-RESOLUTION
"""

import subprocess
import sys
import importlib
import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

class ComprehensiveDependencyResolver:
    """Automatically resolves all system dependencies without user intervention."""
    
    def __init__(self):
        self.base_path = Path("/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system")
        self.resolved_dependencies: List[str] = []
        self.failed_resolutions: List[str] = []
        self.critical_failures: List[str] = []
        
    async def resolve_all_dependencies(self) -> bool:
        """
        Automatically resolve ALL system dependencies.
        Returns True only if ALL dependencies are resolved.
        """
        logger.info("🔧 STARTING COMPREHENSIVE DEPENDENCY RESOLUTION")
        logger.info("=" * 55)
        logger.info("⚡ AUTO-INSTALLING ALL MISSING DEPENDENCIES")
        
        resolution_steps = [
            ("Core Python Modules", self._resolve_python_modules),
            ("Trading APIs", self._resolve_trading_apis),
            ("Machine Learning", self._resolve_ml_dependencies),
            ("Data Processing", self._resolve_data_dependencies),
            ("Async Libraries", self._resolve_async_dependencies),
            ("Networking", self._resolve_networking_dependencies),
            ("Database Libraries", self._resolve_database_dependencies),
            ("System Utilities", self._resolve_system_utilities),
            ("Optional Performance", self._resolve_performance_dependencies)
        ]
        
        all_resolved = True
        
        for step_name, resolution_func in resolution_steps:
            logger.info(f"🔧 Resolving: {step_name}")
            try:
                success = await resolution_func()
                if success:
                    logger.info(f"✅ {step_name}: RESOLVED")
                else:
                    logger.critical(f"❌ {step_name}: FAILED TO RESOLVE")
                    all_resolved = False
            except Exception as e:
                error_msg = f"{step_name} resolution crashed: {str(e)}"
                logger.critical(f"💥 {error_msg}")
                self.critical_failures.append(error_msg)
                all_resolved = False
        
        await self._generate_resolution_report()
        
        if not all_resolved:
            logger.critical("🚫 SOME DEPENDENCIES COULD NOT BE RESOLVED")
            logger.critical(f"❌ {len(self.critical_failures)} CRITICAL FAILURES:")
            for i, failure in enumerate(self.critical_failures, 1):
                logger.critical(f"   {i}. {failure}")
            return False
        
        logger.info("✅ ALL DEPENDENCIES RESOLVED SUCCESSFULLY")
        logger.info(f"🔧 Resolved {len(self.resolved_dependencies)} dependencies")
        return True
    
    async def _resolve_python_modules(self) -> bool:
        """Resolve core Python modules."""
        try:
            core_modules = [
                'numpy>=1.21.0',
                'pandas>=1.3.0', 
                'scipy>=1.7.0',
                'scikit-learn>=1.0.0',
                'matplotlib>=3.4.0',
                'seaborn>=0.11.0',
                'requests>=2.25.0',
                'python-dateutil>=2.8.0',
                'pytz>=2021.1'
            ]
            
            return await self._install_packages(core_modules, "Core Python Modules")
            
        except Exception as e:
            self.critical_failures.append(f"Python modules resolution failed: {str(e)}")
            return False
    
    async def _resolve_trading_apis(self) -> bool:
        """Resolve trading API dependencies."""
        try:
            trading_modules = [
                'alpaca-trade-api>=2.0.0',
                'alpaca-py>=0.5.0',
                'yfinance>=0.1.70',
                'python-binance>=1.0.15'
            ]
            
            return await self._install_packages(trading_modules, "Trading APIs")
            
        except Exception as e:
            self.critical_failures.append(f"Trading API resolution failed: {str(e)}")
            return False
    
    async def _resolve_ml_dependencies(self) -> bool:
        """Resolve machine learning dependencies."""
        try:
            ml_modules = [
                'torch>=1.11.0',
                'transformers>=4.15.0',
                'huggingface-hub>=0.4.0',
                'tokenizers>=0.10.0',
                'datasets>=1.18.0',
                'accelerate>=0.15.0'
            ]
            
            return await self._install_packages(ml_modules, "Machine Learning")
            
        except Exception as e:
            self.critical_failures.append(f"ML dependencies resolution failed: {str(e)}")
            return False
    
    async def _resolve_data_dependencies(self) -> bool:
        """Resolve data processing dependencies."""
        try:
            data_modules = [
                'pydantic>=1.8.0',
                'pydantic-settings>=2.0.0',
                'dataclasses-json>=0.5.0',
                'attrs>=21.0.0',
                'python-dotenv>=0.19.0',
                'configparser>=5.2.0'
            ]
            
            return await self._install_packages(data_modules, "Data Processing")
            
        except Exception as e:
            self.critical_failures.append(f"Data dependencies resolution failed: {str(e)}")
            return False
    
    async def _resolve_async_dependencies(self) -> bool:
        """Resolve async programming dependencies."""
        try:
            async_modules = [
                'aiohttp>=3.8.0',
                'aiofiles>=0.8.0',
                'asyncio-throttle>=1.0.0',
                'websockets>=10.0'
            ]
            
            return await self._install_packages(async_modules, "Async Libraries")
            
        except Exception as e:
            self.critical_failures.append(f"Async dependencies resolution failed: {str(e)}")
            return False
    
    async def _resolve_networking_dependencies(self) -> bool:
        """Resolve networking dependencies."""
        try:
            networking_modules = [
                'urllib3>=1.26.0',
                'certifi>=2021.10.0',
                'httpx>=0.23.0',
                'websocket-client>=1.3.0'
            ]
            
            return await self._install_packages(networking_modules, "Networking")
            
        except Exception as e:
            self.critical_failures.append(f"Networking dependencies resolution failed: {str(e)}")
            return False
    
    async def _resolve_database_dependencies(self) -> bool:
        """Resolve database dependencies."""
        try:
            db_modules = [
                'sqlite3',  # Built-in, but check availability
                'aiosqlite>=0.17.0',
                'sqlalchemy>=1.4.0',
                'redis>=4.0.0'
            ]
            
            # Handle built-in sqlite3 separately
            try:
                import sqlite3
                logger.info("✅ sqlite3: Already available (built-in)")
            except ImportError:
                self.critical_failures.append("sqlite3 not available - Python installation issue")
                return False
            
            # Install other packages
            other_db_modules = [m for m in db_modules if m != 'sqlite3']
            return await self._install_packages(other_db_modules, "Database Libraries")
            
        except Exception as e:
            self.critical_failures.append(f"Database dependencies resolution failed: {str(e)}")
            return False
    
    async def _resolve_system_utilities(self) -> bool:
        """Resolve system utility dependencies."""
        try:
            utility_modules = [
                'psutil>=5.8.0',
                'schedule>=1.1.0',
                'croniter>=1.3.0',
                'click>=8.0.0',
                'rich>=12.0.0',
                'tqdm>=4.62.0'
            ]
            
            return await self._install_packages(utility_modules, "System Utilities")
            
        except Exception as e:
            self.critical_failures.append(f"System utilities resolution failed: {str(e)}")
            return False
    
    async def _resolve_performance_dependencies(self) -> bool:
        """Resolve optional performance dependencies."""
        try:
            performance_modules = [
                'numba>=0.56.0',
                'cython>=0.29.0',
                'fastapi>=0.75.0',
                'uvicorn>=0.17.0'
            ]
            
            # These are optional - don't fail if they can't be installed
            success = await self._install_packages(performance_modules, "Performance Libraries", optional=True)
            # Always return True for optional dependencies
            return True
            
        except Exception as e:
            # Log but don't fail for optional dependencies
            logger.warning(f"Optional performance dependencies couldn't be resolved: {str(e)}")
            return True
    
    async def _install_packages(self, packages: List[str], category: str, optional: bool = False) -> bool:
        """Install a list of packages using pip."""
        try:
            missing_packages = []
            
            # Check which packages are missing
            for package in packages:
                package_name = package.split('>=')[0].split('==')[0].replace('-', '_')
                try:
                    importlib.import_module(package_name)
                    logger.debug(f"✅ {package_name}: Already installed")
                except ImportError:
                    missing_packages.append(package)
            
            if not missing_packages:
                logger.info(f"✅ All {category} packages already installed")
                return True
            
            logger.info(f"📦 Installing {len(missing_packages)} {category} packages...")
            
            # Install missing packages
            for package in missing_packages:
                logger.info(f"   Installing: {package}")
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', package],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per package
                )
                
                if result.returncode == 0:
                    self.resolved_dependencies.append(package)
                    logger.info(f"   ✅ {package}: INSTALLED")
                else:
                    error_msg = f"{package}: Installation failed - {result.stderr}"
                    if optional:
                        logger.warning(f"   ⚠️  {error_msg}")
                        self.failed_resolutions.append(error_msg)
                    else:
                        logger.critical(f"   ❌ {error_msg}")
                        self.critical_failures.append(error_msg)
                        return False
            
            return True
            
        except subprocess.TimeoutExpired:
            error_msg = f"{category}: Package installation timed out"
            if optional:
                logger.warning(f"⚠️  {error_msg}")
                return True
            else:
                self.critical_failures.append(error_msg)
                return False
        except Exception as e:
            error_msg = f"{category}: Package installation crashed - {str(e)}"
            if optional:
                logger.warning(f"⚠️  {error_msg}")
                return True
            else:
                self.critical_failures.append(error_msg)
                return False
    
    async def _generate_resolution_report(self):
        """Generate comprehensive resolution report."""
        report_path = self.base_path / "logs" / "dependency_resolution_report.json"
        
        report = {
            "resolution_timestamp": datetime.now().isoformat(),
            "resolved_dependencies": self.resolved_dependencies,
            "failed_resolutions": self.failed_resolutions,
            "critical_failures": self.critical_failures,
            "total_resolved": len(self.resolved_dependencies),
            "total_failed": len(self.failed_resolutions),
            "critical_count": len(self.critical_failures),
            "success": len(self.critical_failures) == 0
        }
        
        os.makedirs(report_path.parent, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 Dependency resolution report saved: {report_path}")

# Global dependency resolver instance
dependency_resolver = ComprehensiveDependencyResolver()

async def resolve_all_dependencies() -> bool:
    """Main dependency resolution entry point."""
    return await dependency_resolver.resolve_all_dependencies()

if __name__ == "__main__":
    """Direct dependency resolution run for testing."""
    import asyncio
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(resolve_all_dependencies())
    sys.exit(0 if result else 1)