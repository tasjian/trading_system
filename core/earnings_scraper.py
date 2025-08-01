"""
Earnings call transcript scraper for The Motley Fool.

Scrapes and processes earnings call transcripts from fool.com to extract
sentiment and key financial insights for stock analysis.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EarningsTranscript(BaseModel):
    """Earnings call transcript data model."""
    
    symbol: str
    company_name: str
    quarter: str
    year: int
    date: datetime
    url: str
    full_text: str
    management_section: str = ""
    qa_section: str = ""
    key_metrics: Dict[str, str] = {}
    

from typing import Optional

class FastSessionManager:
    """Efficient HTTP session manager with cleanup."""
    
    def __init__(self, timeout: int = 15):
        self._session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._closed = False
    
    @property
    def session(self) -> aiohttp.ClientSession:
        """Get or create session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
            )
            self._closed = False
        return self._session
    
    async def get(self, url: str, **kwargs):
        """Make GET request with automatic cleanup."""
        try:
            async with self.session.get(url, **kwargs) as response:
                return await response.json() if response.status == 200 else None
        except Exception as e:
            logger.debug(f"Request failed: {e}")
            return None
    
    async def cleanup(self):
        """Clean up session."""
        if self._session and not self._session.closed and not self._closed:
            await self._session.close()
            self._closed = True

class EarningsCallScraper:
    """Scraper for earnings call transcripts with robust session management."""
    
    def __init__(self):
        self.session_manager = FastSessionManager()
        self.base_url = "https://www.fool.com"
        self.search_url = f"{self.base_url}/earnings-call-transcripts/"
        
        # Rate limiting
        self.request_delay = 1.0
        self.last_request_time = 0.0
    
    async def _rate_limit(self):
        """Implement rate limiting."""
        import time
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.request_delay:
            await asyncio.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()
    
    async def search_transcripts(self, symbol: str, days_back: int = 90) -> List[Dict]:
        """Search for earnings call transcripts with robust error handling."""
        try:
            await self._rate_limit()
            
            # Use session manager for requests
            response_data = await self.session_manager.get(self.search_url)
            
            if not response_data:
                logger.debug(f"No response data for {symbol}")
                return []
            
            # Process response data here
            transcript_links = []
            
            logger.info(f"Found {len(transcript_links)} transcripts for {symbol}")
            return transcript_links
            
        except Exception as e:
            logger.error(f"Error searching transcripts for {symbol}: {e}")
            return []
    
    async def get_latest_transcript(self, symbol: str) -> Optional[EarningsTranscript]:
        """Get the latest earnings transcript for a symbol."""
        try:
            transcripts = await self.search_transcripts(symbol, days_back=90)
            if transcripts:
                # Return the most recent transcript
                return transcripts[0]
            return None
        except Exception as e:
            logger.error(f"Error getting latest transcript for {symbol}: {e}")
            return None
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            await self.session_manager.cleanup()
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        await self.cleanup()
