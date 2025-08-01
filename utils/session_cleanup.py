"""
Session Cleanup Utility
Provides global session cleanup to prevent asyncio warnings.
"""

import asyncio
import logging
import weakref
from typing import Set
import aiohttp

logger = logging.getLogger(__name__)

class GlobalSessionManager:
    """Global manager for all aiohttp sessions to ensure proper cleanup."""
    
    def __init__(self):
        self.sessions: Set[aiohttp.ClientSession] = weakref.WeakSet()
        self.cleanup_registered = False
    
    def register_session(self, session: aiohttp.ClientSession):
        """Register a session for global cleanup."""
        self.sessions.add(session)
        
        if not self.cleanup_registered:
            # Register cleanup at exit
            import atexit
            atexit.register(self.cleanup_all_sync)
            self.cleanup_registered = True
    
    async def cleanup_all(self):
        """Clean up all registered sessions."""
        for session in list(self.sessions):
            try:
                if not session.closed:
                    await session.close()
            except Exception as e:
                logger.debug(f"Session cleanup error: {e}")
    
    def cleanup_all_sync(self):
        """Synchronous cleanup for atexit."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule cleanup
                loop.create_task(self.cleanup_all())
            else:
                # Run cleanup
                loop.run_until_complete(self.cleanup_all())
        except Exception as e:
            logger.debug(f"Global session cleanup error: {e}")

# Global instance
global_session_manager = GlobalSessionManager()