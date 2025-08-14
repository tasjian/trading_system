#!/usr/bin/env python3
"""
Service Manager for Trading System
Manages Redis and Ollama services with health checks and auto-starting capabilities.
"""

import subprocess
import time
import socket
import logging
import asyncio
import aiohttp
import redis
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import psutil
import json

from config.settings import settings

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status enumeration."""
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    ERROR = "error"
    UNKNOWN = "unknown"

@dataclass
class ServiceHealth:
    """Service health status."""
    name: str
    status: ServiceStatus
    pid: Optional[int] = None
    port: Optional[int] = None
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None
    last_check: float = 0.0

class ServiceManager:
    """Manages Redis and Ollama services for the trading system."""
    
    def __init__(self):
        self.redis_host = settings.redis_host
        self.redis_port = settings.redis_port
        self.ollama_base_url = settings.ollama_base_url
        self.ollama_port = 11434  # Default Ollama port
        
        # Service health cache
        self.service_health: Dict[str, ServiceHealth] = {}
        
        logger.info("🔧 Service Manager initialized")
    
    async def check_all_services(self) -> Dict[str, ServiceHealth]:
        """Check health of all required services."""
        logger.info("🔍 Checking health of all services...")
        
        # Check services concurrently
        redis_health, ollama_health = await asyncio.gather(
            self.check_redis_health(),
            self.check_ollama_health(),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(redis_health, Exception):
            redis_health = ServiceHealth("redis", ServiceStatus.ERROR, error_message=str(redis_health))
        if isinstance(ollama_health, Exception):
            ollama_health = ServiceHealth("ollama", ServiceStatus.ERROR, error_message=str(ollama_health))
        
        self.service_health.update({
            "redis": redis_health,
            "ollama": ollama_health
        })
        
        # Log results
        for name, health in self.service_health.items():
            status_emoji = "✅" if health.status == ServiceStatus.RUNNING else "❌"
            logger.info(f"{status_emoji} {name.upper()}: {health.status.value}")
            if health.error_message:
                logger.warning(f"   Error: {health.error_message}")
            if health.response_time_ms:
                logger.info(f"   Response time: {health.response_time_ms:.1f}ms")
        
        return self.service_health
    
    async def check_redis_health(self) -> ServiceHealth:
        """Check Redis service health."""
        start_time = time.time()
        
        try:
            # Try to connect to Redis
            redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                socket_connect_timeout=5,
                socket_timeout=5,
                decode_responses=True
            )
            
            # Test connection with ping
            redis_client.ping()
            
            response_time = (time.time() - start_time) * 1000
            
            # Get Redis process info if possible
            pid = self._get_process_pid_by_port(self.redis_port)
            
            return ServiceHealth(
                name="redis",
                status=ServiceStatus.RUNNING,
                pid=pid,
                port=self.redis_port,
                response_time_ms=response_time,
                last_check=time.time()
            )
            
        except redis.ConnectionError as e:
            return ServiceHealth(
                name="redis",
                status=ServiceStatus.STOPPED,
                port=self.redis_port,
                error_message=f"Connection failed: {e}",
                last_check=time.time()
            )
        except Exception as e:
            return ServiceHealth(
                name="redis",
                status=ServiceStatus.ERROR,
                port=self.redis_port,
                error_message=str(e),
                last_check=time.time()
            )
    
    async def check_ollama_health(self) -> ServiceHealth:
        """Check Ollama service health."""
        start_time = time.time()
        
        try:
            # Check if Ollama is running on the port
            if not self._is_port_open("localhost", self.ollama_port):
                return ServiceHealth(
                    name="ollama",
                    status=ServiceStatus.STOPPED,
                    port=self.ollama_port,
                    error_message="Port not responding",
                    last_check=time.time()
                )
            
            # Test Ollama API endpoint
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{self.ollama_base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        response_time = (time.time() - start_time) * 1000
                        
                        # Get Ollama process info
                        pid = self._get_process_pid_by_name("ollama")
                        
                        # Check if our target model is available
                        models = [model.get('name', '') for model in data.get('models', [])]
                        has_target_model = any(settings.ollama_model in model for model in models)
                        
                        status = ServiceStatus.RUNNING if has_target_model else ServiceStatus.ERROR
                        error_msg = None if has_target_model else f"Target model '{settings.ollama_model}' not found"
                        
                        return ServiceHealth(
                            name="ollama",
                            status=status,
                            pid=pid,
                            port=self.ollama_port,
                            response_time_ms=response_time,
                            error_message=error_msg,
                            last_check=time.time()
                        )
                    else:
                        return ServiceHealth(
                            name="ollama",
                            status=ServiceStatus.ERROR,
                            port=self.ollama_port,
                            error_message=f"HTTP {response.status}",
                            last_check=time.time()
                        )
        
        except asyncio.TimeoutError:
            return ServiceHealth(
                name="ollama",
                status=ServiceStatus.ERROR,
                port=self.ollama_port,
                error_message="Request timeout",
                last_check=time.time()
            )
        except Exception as e:
            return ServiceHealth(
                name="ollama",
                status=ServiceStatus.ERROR,
                port=self.ollama_port,
                error_message=str(e),
                last_check=time.time()
            )
    
    async def ensure_services_running(self) -> bool:
        """Ensure all required services are running, start them if needed."""
        logger.info("🚀 Ensuring all services are running...")
        
        # Check current status
        health_status = await self.check_all_services()
        
        services_to_start = []
        for name, health in health_status.items():
            if health.status != ServiceStatus.RUNNING:
                services_to_start.append(name)
        
        if not services_to_start:
            logger.info("✅ All services are already running")
            return True
        
        # Start services that need starting
        success = True
        for service_name in services_to_start:
            logger.info(f"🔄 Starting {service_name}...")
            
            if service_name == "redis":
                if await self.start_redis():
                    logger.info(f"✅ {service_name} started successfully")
                else:
                    logger.error(f"❌ Failed to start {service_name}")
                    success = False
            
            elif service_name == "ollama":
                if await self.start_ollama():
                    logger.info(f"✅ {service_name} started successfully")
                else:
                    logger.error(f"❌ Failed to start {service_name}")
                    success = False
        
        # Verify all services are now running
        if success:
            await asyncio.sleep(3)  # Give services time to fully start
            final_status = await self.check_all_services()
            
            for name, health in final_status.items():
                if health.status != ServiceStatus.RUNNING:
                    logger.error(f"❌ {name} failed to start properly: {health.error_message}")
                    success = False
        
        return success
    
    async def start_redis(self) -> bool:
        """Start Redis service."""
        try:
            # Try different Redis start commands based on the system
            redis_commands = [
                ["redis-server", "--daemonize", "yes"],  # Standard Redis
                ["brew", "services", "start", "redis"],   # macOS Homebrew
                ["sudo", "systemctl", "start", "redis"], # Linux systemd
                ["sudo", "service", "redis-server", "start"], # Linux sysvinit
            ]
            
            for cmd in redis_commands:
                try:
                    logger.debug(f"Trying Redis start command: {' '.join(cmd)}")
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"Redis start command succeeded: {' '.join(cmd)}")
                        
                        # Wait and verify
                        await asyncio.sleep(2)
                        health = await self.check_redis_health()
                        if health.status == ServiceStatus.RUNNING:
                            return True
                    else:
                        logger.debug(f"Command failed: {result.stderr}")
                        
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
                    logger.debug(f"Command failed: {e}")
                    continue
            
            logger.error("All Redis start commands failed")
            return False
            
        except Exception as e:
            logger.error(f"Error starting Redis: {e}")
            return False
    
    async def start_ollama(self) -> bool:
        """Start Ollama service."""
        try:
            # Check if Ollama is already running
            if self._get_process_pid_by_name("ollama"):
                logger.info("Ollama process already running")
                health = await self.check_ollama_health()
                return health.status == ServiceStatus.RUNNING
            
            # Start Ollama
            logger.info("Starting Ollama service...")
            
            # Try to start Ollama
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give Ollama time to start
            await asyncio.sleep(5)
            
            # Check if it's running
            health = await self.check_ollama_health()
            if health.status == ServiceStatus.RUNNING:
                logger.info("✅ Ollama service started successfully")
                return True
            elif health.status == ServiceStatus.ERROR and "not found" in (health.error_message or ""):
                # Model not found, try to pull it
                logger.info(f"🔄 Pulling required model: {settings.ollama_model}")
                if await self.pull_ollama_model(settings.ollama_model):
                    health = await self.check_ollama_health()
                    return health.status == ServiceStatus.RUNNING
            
            return False
            
        except Exception as e:
            logger.error(f"Error starting Ollama: {e}")
            return False
    
    async def pull_ollama_model(self, model_name: str) -> bool:
        """Pull an Ollama model if it's not available."""
        try:
            logger.info(f"📥 Pulling Ollama model: {model_name}")
            
            process = subprocess.Popen(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for the pull to complete (can take a while)
            stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
            
            if process.returncode == 0:
                logger.info(f"✅ Successfully pulled model: {model_name}")
                return True
            else:
                logger.error(f"❌ Failed to pull model {model_name}: {stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout pulling model {model_name}")
            return False
        except Exception as e:
            logger.error(f"Error pulling Ollama model: {e}")
            return False
    
    def _is_port_open(self, host: str, port: int, timeout: float = 3.0) -> bool:
        """Check if a port is open."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.error, socket.timeout):
            return False
    
    def _get_process_pid_by_port(self, port: int) -> Optional[int]:
        """Get process PID by port number."""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                    return conn.pid
        except (psutil.Error, AttributeError):
            pass
        return None
    
    def _get_process_pid_by_name(self, name: str) -> Optional[int]:
        """Get process PID by name."""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if name.lower() in proc.info['name'].lower():
                    return proc.info['pid']
                # Also check command line for cases where the process name might be different
                cmdline = ' '.join(proc.info.get('cmdline', []))
                if name.lower() in cmdline.lower():
                    return proc.info['pid']
        except (psutil.Error, AttributeError):
            pass
        return None
    
    def get_service_summary(self) -> Dict[str, str]:
        """Get a summary of all service statuses."""
        summary = {}
        for name, health in self.service_health.items():
            if health.status == ServiceStatus.RUNNING:
                summary[name] = f"✅ Running (PID: {health.pid}, {health.response_time_ms:.1f}ms)"
            else:
                summary[name] = f"❌ {health.status.value}: {health.error_message or 'Unknown error'}"
        
        return summary

# Global service manager instance
service_manager = ServiceManager()

# Convenience functions
async def ensure_services_running() -> bool:
    """Ensure all required services are running."""
    return await service_manager.ensure_services_running()

async def check_service_health() -> Dict[str, ServiceHealth]:
    """Check health of all services."""
    return await service_manager.check_all_services()

def get_service_summary() -> Dict[str, str]:
    """Get service status summary."""
    return service_manager.get_service_summary()

if __name__ == "__main__":
    async def test_service_manager():
        """Test the service manager."""
        print("🧪 Testing Service Manager...")
        
        # Check initial status
        health = await service_manager.check_all_services()
        print("\n📊 Current Service Status:")
        for name, status in service_manager.get_service_summary().items():
            print(f"  {name}: {status}")
        
        # Try to ensure services are running
        print("\n🚀 Ensuring services are running...")
        success = await service_manager.ensure_services_running()
        
        if success:
            print("✅ All services are running!")
        else:
            print("❌ Some services failed to start")
        
        # Final status check
        final_health = await service_manager.check_all_services()
        print("\n📊 Final Service Status:")
        for name, status in service_manager.get_service_summary().items():
            print(f"  {name}: {status}")
    
    # Run the test
    asyncio.run(test_service_manager())