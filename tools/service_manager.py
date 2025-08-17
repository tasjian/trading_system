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
        # Ollama removed - using GPT-5-nano for sentiment analysis
        
        # Service health cache
        self.service_health: Dict[str, ServiceHealth] = {}
        
        logger.info("🔧 Service Manager initialized")
    
    async def check_all_services(self) -> Dict[str, ServiceHealth]:
        """Check health of required services (Redis only - GPT-5-nano used for sentiment)."""
        logger.info("🔍 Checking health of required services...")
        
        # Only check Redis - GPT-5-nano handles sentiment analysis
        redis_health = await self.check_redis_health()
        
        # Handle exceptions
        if isinstance(redis_health, Exception):
            error_msg = str(redis_health)
            logger.debug(f"Redis health check exception: {error_msg}")
            redis_health = ServiceHealth("redis", ServiceStatus.ERROR, error_message=error_msg)
        
        self.service_health.update({
            "redis": redis_health
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
    
    # REMOVED: Ollama health check - using GPT-5-nano for sentiment analysis
    async def _removed_check_ollama_health(self) -> ServiceHealth:
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
                        try:
                            pid = self._get_process_pid_by_name("ollama")
                        except Exception as e:
                            logger.debug(f"Error getting Ollama PID: {e}")
                            pid = None
                        
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
            
            # Ollama removed - using GPT-5-nano for sentiment analysis
        
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
        """Start Redis service with enhanced reliability."""
        try:
            # Check if Redis is already running first
            health = await self.check_redis_health()
            if health.status == ServiceStatus.RUNNING:
                logger.info("Redis is already running")
                return True
            
            # Check if there's a Redis process that's not responding
            redis_pid = self._get_process_pid_by_name("redis-server")
            if redis_pid:
                logger.info(f"Found non-responsive Redis process (PID: {redis_pid}), attempting restart")
                try:
                    subprocess.run(["kill", str(redis_pid)], timeout=5)
                    await asyncio.sleep(2)
                except Exception as e:
                    logger.debug(f"Failed to kill existing Redis process: {e}")
            
            # Try different Redis start commands based on the system
            import platform
            system = platform.system().lower()
            
            redis_commands = []
            if system == "darwin":  # macOS
                redis_commands = [
                    # Try Homebrew service management first
                    ["brew", "services", "restart", "redis"],  # Restart ensures fresh start
                    ["brew", "services", "start", "redis"],   # Start if not running
                    # Try direct Redis server with optimal settings
                    ["redis-server", "--daemonize", "yes", "--port", "6379", "--bind", "127.0.0.1"],
                    ["/opt/homebrew/bin/redis-server", "--daemonize", "yes", "--port", "6379", "--bind", "127.0.0.1"],  # M1 Mac
                    ["/usr/local/bin/redis-server", "--daemonize", "yes", "--port", "6379", "--bind", "127.0.0.1"],     # Intel Mac
                ]
            else:  # Linux
                redis_commands = [
                    ["systemctl", "restart", "redis"],         # Restart for fresh start
                    ["systemctl", "start", "redis"],           # systemd (no sudo needed if user has perms)
                    ["sudo", "systemctl", "restart", "redis"], # systemd with sudo restart
                    ["sudo", "systemctl", "start", "redis"],   # systemd with sudo
                    ["service", "redis-server", "restart"],    # sysvinit restart
                    ["service", "redis-server", "start"],      # sysvinit
                    ["sudo", "service", "redis-server", "restart"], # sysvinit with sudo restart
                    ["sudo", "service", "redis-server", "start"], # sysvinit with sudo
                    ["redis-server", "--daemonize", "yes", "--port", "6379", "--bind", "127.0.0.1"],    # Direct Redis
                ]
            
            for cmd in redis_commands:
                try:
                    # Ensure cmd is a list and all elements are strings
                    if not isinstance(cmd, (list, tuple)) or not cmd:
                        logger.debug(f"Skipping invalid command: {cmd}")
                        continue
                    
                    # Convert all command parts to strings and filter out empty ones
                    cmd_safe = [str(part) for part in cmd if part]
                    if not cmd_safe:
                        logger.debug("Skipping empty command after filtering")
                        continue
                    
                    logger.info(f"Trying Redis start command: {' '.join(cmd_safe)}")
                    result = subprocess.run(
                        cmd_safe,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    # Log command result for debugging
                    logger.debug(f"Command '{' '.join(cmd_safe)}' returned {result.returncode}")
                    if result.stdout:
                        logger.debug(f"stdout: {result.stdout.strip()}")
                    if result.stderr:
                        logger.debug(f"stderr: {result.stderr.strip()}")
                    
                    # Wait longer for Redis to fully start
                    await asyncio.sleep(5)
                    health = await self.check_redis_health()
                    if health.status == ServiceStatus.RUNNING:
                        logger.info(f"✅ Redis started successfully with: {' '.join(cmd_safe)}")
                        return True
                        
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
                    logger.debug(f"Command '{' '.join(cmd_safe)}' failed: {e}")
                    continue
                except Exception as e:
                    logger.debug(f"Unexpected error with command '{cmd}': {e}")
                    continue
            
            # Try Docker Redis as last resort
            logger.info("Trying Docker Redis as fallback...")
            try:
                # Check if Docker is available
                docker_check = subprocess.run(["docker", "--version"], capture_output=True, timeout=10)
                if docker_check.returncode == 0:
                    # Try to start Redis in Docker
                    docker_cmd = [
                        "docker", "run", "-d", "--name", "redis-trading", 
                        "-p", "6379:6379", "redis:alpine", "redis-server", "--appendonly", "yes"
                    ]
                    
                    # Remove any existing container first
                    subprocess.run(["docker", "rm", "-f", "redis-trading"], capture_output=True)
                    
                    result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=60)
                    if result.returncode == 0:
                        logger.info("Started Redis in Docker container")
                        await asyncio.sleep(5)
                        health = await self.check_redis_health()
                        if health.status == ServiceStatus.RUNNING:
                            logger.info("✅ Redis started successfully in Docker")
                            return True
            except Exception as e:
                logger.debug(f"Docker Redis startup failed: {e}")
            
            logger.warning("⚠️ Could not start Redis - continuing without cache (will use fallback providers)")
            
            # Log helpful diagnostic information
            logger.info("Redis troubleshooting suggestions:")
            logger.info("  • Install Redis: brew install redis (macOS) or apt-get install redis-server (Ubuntu)")
            logger.info("  • Check if Redis is already running: ps aux | grep redis")
            logger.info("  • Manual start: redis-server /usr/local/etc/redis.conf")
            logger.info("  • System will continue with degraded performance (no caching)")
            return False
            
        except Exception as e:
            logger.error(f"Error starting Redis: {e}")
            return False
    
    async def start_ollama(self) -> bool:
        """Start Ollama service with enhanced reliability."""
        try:
            # Check if Ollama is already running and healthy
            health = await self.check_ollama_health()
            if health.status == ServiceStatus.RUNNING:
                logger.info("Ollama service is already running and healthy")
                return True
            
            # Check for existing Ollama process that might be unhealthy
            try:
                existing_pid = self._get_process_pid_by_name("ollama")
                if existing_pid:
                    logger.info(f"Found existing Ollama process (PID: {existing_pid}) but it's not responding properly")
                    try:
                        subprocess.run(["kill", "-TERM", str(existing_pid)], timeout=5)
                        await asyncio.sleep(3)
                        # Force kill if still running
                        if self._get_process_pid_by_name("ollama"):
                            subprocess.run(["kill", "-KILL", str(existing_pid)], timeout=5)
                            await asyncio.sleep(2)
                    except Exception as e:
                        logger.debug(f"Failed to terminate existing Ollama process: {e}")
            except Exception as e:
                logger.debug(f"Error checking for existing Ollama process: {e}")
                # Continue anyway - don't let this block startup
            
            # Ensure Ollama is installed
            try:
                version_check = subprocess.run(["ollama", "--version"], capture_output=True, timeout=10)
                if version_check.returncode != 0:
                    logger.error("Ollama not installed. Install from: https://ollama.com/download")
                    return False
                else:
                    logger.info(f"Ollama version: {version_check.stdout.decode().strip()}")
            except FileNotFoundError:
                logger.error("Ollama not found in PATH. Install from: https://ollama.com/download")
                return False
            except Exception as e:
                logger.error(f"Ollama version check failed: {e}")
                return False
            
            # Start Ollama with proper environment
            logger.info("Starting Ollama service...")
            
            # Set up environment for Ollama
            env = {
                **dict(subprocess.os.environ),
                "OLLAMA_HOST": "127.0.0.1:11434",
                "OLLAMA_ORIGINS": "*"
            }
            
            # Try to start Ollama server
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            logger.info(f"Ollama server process started with PID: {process.pid}")
            
            # Give Ollama more time to start up properly
            startup_timeout = 15
            for i in range(startup_timeout):
                await asyncio.sleep(1)
                health = await self.check_ollama_health()
                if health.status == ServiceStatus.RUNNING:
                    logger.info(f"✅ Ollama service started successfully after {i+1} seconds")
                    break
                elif i < startup_timeout - 1:
                    logger.debug(f"Waiting for Ollama to start... ({i+1}/{startup_timeout})")
            
            # Final health check
            health = await self.check_ollama_health()
            if health.status == ServiceStatus.RUNNING:
                logger.info("✅ Ollama service is running and healthy")
                return True
            elif health.status == ServiceStatus.ERROR and "not found" in (health.error_message or ""):
                # Model not found, try to pull it
                logger.info(f"🔄 Required model '{settings.ollama_model}' not found, attempting to pull...")
                if await self.pull_ollama_model(settings.ollama_model):
                    # Check again after model pull
                    health = await self.check_ollama_health()
                    if health.status == ServiceStatus.RUNNING:
                        logger.info("✅ Ollama service is running with required model")
                        return True
                else:
                    logger.warning(f"Failed to pull model '{settings.ollama_model}'")
            
            # If we get here, Ollama didn't start properly
            logger.warning(f"Ollama service failed to start properly. Status: {health.status.value}")
            if health.error_message:
                logger.warning(f"Error: {health.error_message}")
            
            # Log helpful troubleshooting information
            logger.info("Ollama troubleshooting suggestions:")
            logger.info("  • Install Ollama: https://ollama.com/download")
            logger.info("  • Check if port 11434 is available: lsof -i :11434")
            logger.info("  • Manual start: ollama serve")
            logger.info("  • System will continue without LLM capabilities")
            
            return False
            
        except Exception as e:
            logger.error(f"Error starting Ollama: {e}")
            return False
    
    async def pull_ollama_model(self, model_name: str) -> bool:
        """Pull an Ollama model with progress tracking."""
        try:
            logger.info(f"📥 Pulling Ollama model: {model_name} (this may take several minutes for large models)")
            
            # Check if model already exists
            try:
                list_result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
                if list_result.returncode == 0 and model_name in list_result.stdout:
                    logger.info(f"Model {model_name} already exists locally")
                    return True
            except Exception as e:
                logger.debug(f"Could not check existing models: {e}")
            
            # Pull the model with timeout based on model size
            pull_timeout = 600  # 10 minutes for large models
            if "7b" in model_name.lower():
                pull_timeout = 300  # 5 minutes for 7B models
            elif "13b" in model_name.lower():
                pull_timeout = 900  # 15 minutes for 13B models
            elif "70b" in model_name.lower():
                pull_timeout = 1800  # 30 minutes for 70B models
            
            logger.info(f"Starting model pull with {pull_timeout//60} minute timeout...")
            
            process = subprocess.Popen(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout for progress tracking
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Monitor progress
            output_lines = []
            try:
                stdout, stderr = process.communicate(timeout=pull_timeout)
                output_lines.append(stdout if stdout else "")
                
                if process.returncode == 0:
                    logger.info(f"✅ Successfully pulled model: {model_name}")
                    
                    # Verify model is available
                    verify_result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
                    if verify_result.returncode == 0 and model_name in verify_result.stdout:
                        logger.info(f"Model {model_name} verified and ready for use")
                        return True
                    else:
                        logger.warning(f"Model {model_name} pull completed but not found in list")
                        return False
                else:
                    error_output = "\n".join(output_lines)
                    logger.error(f"❌ Failed to pull model {model_name}. Return code: {process.returncode}")
                    logger.error(f"Output: {error_output}")
                    return False
                    
            except subprocess.TimeoutExpired:
                logger.error(f"❌ Timeout pulling model {model_name} after {pull_timeout//60} minutes")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                return False
                
        except Exception as e:
            logger.error(f"Error pulling Ollama model {model_name}: {e}")
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
                try:
                    proc_info = proc.info
                    if proc_info and proc_info.get('name'):
                        proc_name = proc_info['name']
                        if proc_name and name.lower() in proc_name.lower():
                            return proc_info['pid']
                    
                    # Also check command line for cases where the process name might be different
                    cmdline = proc_info.get('cmdline') if proc_info else None
                    if cmdline and isinstance(cmdline, (list, tuple)) and len(cmdline) > 0:
                        try:
                            # Filter out None values and convert to strings safely
                            cmdline_parts = [str(arg) for arg in cmdline if arg is not None]
                            if cmdline_parts:  # Only join if we have valid parts
                                cmdline_str = ' '.join(cmdline_parts)
                                if cmdline_str and name.lower() in cmdline_str.lower():
                                    return proc_info['pid']
                        except (TypeError, ValueError) as e:
                            logger.debug(f"Error processing cmdline for process {proc_info.get('pid', 'unknown')}: {e}")
                            continue
                            
                except (psutil.Error, AttributeError, TypeError, KeyError) as e:
                    logger.debug(f"Error accessing process info: {e}")
                    continue
        except (psutil.Error, AttributeError) as e:
            logger.debug(f"Error iterating processes: {e}")
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