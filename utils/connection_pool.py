"""
Connection Pool Manager for Trading System

Provides centralized connection pooling and session management to prevent memory leaks
and optimize resource usage across all HTTP clients in the trading system.
"""

import asyncio
import logging
import weakref
from contextlib import asynccontextmanager
from typing import Dict, Optional, Set, Any
import aiohttp
import requests
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

class ConnectionPoolManager:
    """Centralized connection pool manager to prevent memory leaks."""
    
    def __init__(self):
        self._async_sessions: Dict[str, aiohttp.ClientSession] = {}
        self._sync_sessions: Dict[str, requests.Session] = {}
        self._session_lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
        self._cleanup_registered = False
        self._active_connections: Set[Any] = weakref.WeakSet()
        
        # Default timeouts and limits
        self._default_timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self._default_connector_limit = 100
        self._default_connector_limit_per_host = 30
    
    async def get_async_session(self, 
                               name: str = "default",
                               timeout: Optional[aiohttp.ClientTimeout] = None,
                               connector_limit: int = None,
                               headers: Dict[str, str] = None) -> aiohttp.ClientSession:
        """Get or create an async HTTP session with connection pooling."""
        
        async with self._session_lock:
            if name not in self._async_sessions or self._async_sessions[name].closed:
                
                # Configure connector with connection limits
                connector = aiohttp.TCPConnector(
                    limit=connector_limit or self._default_connector_limit,
                    limit_per_host=connector_limit or self._default_connector_limit_per_host,
                    enable_cleanup_closed=True,
                    keepalive_timeout=30,
                    ttl_dns_cache=300
                )
                
                session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout or self._default_timeout,
                    headers=headers
                )
                
                self._async_sessions[name] = session
                self._active_connections.add(session)
                self._register_cleanup()
                
                logger.debug(f"Created async session '{name}' with connection pooling")
            
            return self._async_sessions[name]
    
    def get_sync_session(self, 
                        name: str = "default",
                        timeout: int = 30,
                        max_retries: int = 3,
                        headers: Dict[str, str] = None) -> requests.Session:
        """Get or create a sync HTTP session with connection pooling."""
        
        with self._sync_lock:
            if name not in self._sync_sessions:
                session = requests.Session()
                
                # Configure connection pooling
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=20,
                    pool_maxsize=100,
                    max_retries=max_retries,
                    pool_block=False
                )
                
                session.mount('http://', adapter)
                session.mount('https://', adapter)
                
                # Set default headers
                if headers:
                    session.headers.update(headers)
                
                # Set timeout
                session.timeout = timeout
                
                self._sync_sessions[name] = session
                self._active_connections.add(session)
                self._register_cleanup()
                
                logger.debug(f"Created sync session '{name}' with connection pooling")
            
            return self._sync_sessions[name]
    
    @asynccontextmanager
    async def get_temp_session(self, **kwargs):
        """Get a temporary session that auto-closes after use."""
        session = None
        try:
            connector = aiohttp.TCPConnector(
                limit=20,
                limit_per_host=10,
                enable_cleanup_closed=True
            )
            
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=kwargs.get('timeout', self._default_timeout),
                headers=kwargs.get('headers')
            )
            
            yield session
            
        finally:
            if session and not session.closed:
                await session.close()
    
    async def close_session(self, name: str):
        """Close a specific session."""
        async with self._session_lock:
            if name in self._async_sessions:
                session = self._async_sessions[name]
                if not session.closed:
                    await session.close()
                del self._async_sessions[name]
                logger.debug(f"Closed async session '{name}'")
        
        with self._sync_lock:
            if name in self._sync_sessions:
                session = self._sync_sessions[name]
                session.close()
                del self._sync_sessions[name]
                logger.debug(f"Closed sync session '{name}'")
    
    async def cleanup_all_async(self):
        """Clean up all async sessions."""
        async with self._session_lock:
            for name, session in list(self._async_sessions.items()):
                try:
                    if not session.closed:
                        await session.close()
                        logger.debug(f"Cleaned up async session '{name}'")
                except Exception as e:
                    logger.warning(f"Error cleaning up async session '{name}': {e}")
            
            self._async_sessions.clear()
    
    def cleanup_all_sync(self):
        """Clean up all sync sessions."""
        with self._sync_lock:
            for name, session in list(self._sync_sessions.items()):
                try:
                    session.close()
                    logger.debug(f"Cleaned up sync session '{name}'")
                except Exception as e:
                    logger.warning(f"Error cleaning up sync session '{name}': {e}")
            
            self._sync_sessions.clear()
    
    def cleanup_all(self):
        """Clean up all sessions (sync version for atexit)."""
        self.cleanup_all_sync()
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup_all_async())
            else:
                loop.run_until_complete(self.cleanup_all_async())
        except Exception as e:
            logger.debug(f"Error in async cleanup: {e}")
    
    def _register_cleanup(self):
        """Register cleanup at exit."""
        if not self._cleanup_registered:
            import atexit
            atexit.register(self.cleanup_all)
            self._cleanup_registered = True
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "async_sessions": len(self._async_sessions),
            "sync_sessions": len(self._sync_sessions),
            "active_connections": len(self._active_connections),
            "session_names": {
                "async": list(self._async_sessions.keys()),
                "sync": list(self._sync_sessions.keys())
            }
        }

# Global connection pool manager
connection_pool = ConnectionPoolManager()

# Convenience functions for backward compatibility
async def get_async_session(name: str = "default", **kwargs) -> aiohttp.ClientSession:
    """Get async session from global pool."""
    return await connection_pool.get_async_session(name, **kwargs)

def get_sync_session(name: str = "default", **kwargs) -> requests.Session:
    """Get sync session from global pool."""
    return connection_pool.get_sync_session(name, **kwargs)

@asynccontextmanager
async def temp_session(**kwargs):
    """Context manager for temporary sessions."""
    async with connection_pool.get_temp_session(**kwargs) as session:
        yield session