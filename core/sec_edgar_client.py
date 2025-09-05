"""
SEC EDGAR API Client for financial data extraction and sentiment analysis.

Provides access to SEC filings, company facts, and real-time filing feeds
with proper rate limiting and comprehensive data parsing.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from utils.connection_pool import connection_pool

logger = logging.getLogger(__name__)


class CompanyInfo(BaseModel):
    """Company information from SEC CIK mapping."""
    cik: str
    ticker: str
    title: str
    exchange: Optional[str] = None


class SECFiling(BaseModel):
    """SEC filing information."""
    accession_number: str
    filing_type: str
    filing_date: str
    report_date: Optional[str] = None
    company_name: str
    cik: str
    ticker: str
    form_description: str
    size: Optional[int] = None
    document_count: Optional[int] = None
    period_of_report: Optional[str] = None
    primary_document: Optional[str] = None
    items: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    content: Optional[str] = None
    key_sections: Dict[str, str] = Field(default_factory=dict)
    sentiment_signals: List[str] = Field(default_factory=list)


class CompanyFacts(BaseModel):
    """Company fundamental facts from SEC."""
    cik: str
    entity_name: str
    facts: Dict[str, Any] = Field(default_factory=dict)
    units: Dict[str, List[Dict]] = Field(default_factory=dict)


class SECRateLimiter:
    """Rate limiter for SEC EDGAR API (10 requests per second max)."""
    
    def __init__(self, max_requests_per_second: float = 9.5):
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire rate limit token."""
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                await asyncio.sleep(sleep_time)
            
            self.last_request_time = time.time()


class SECEdgarClient:
    """SEC EDGAR API client with comprehensive filing access."""
    
    def __init__(self, user_agent: str = None):
        self.base_url = "https://www.sec.gov"
        
        # SEC requires specific User-Agent format as of 2024: "Company Name email@domain.com"
        # Based on successful patterns from working libraries
        if user_agent is None:
            self.user_agent = "Tasjian ztaschdjian@gmail.com"
        else:
            self.user_agent = user_agent
            
        self.rate_limiter = SECRateLimiter()
        self._session: Optional[aiohttp.ClientSession] = None
        self._company_tickers: Optional[Dict[str, CompanyInfo]] = None
        self._cik_to_ticker: Optional[Dict[str, str]] = None
        
        # Headers required by SEC (2024 enforcement) - simplified for compatibility
        self.headers = {
            'User-Agent': self.user_agent
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get session from connection pool."""
        return await connection_pool.get_async_session(
            name="sec_edgar_client",
            headers=self.headers,
            connector_limit=10
        )
    
    async def _make_request(self, url: str, **kwargs) -> Optional[Dict]:
        """Make rate-limited request to SEC API with enhanced error handling and retry logic."""
        await self.rate_limiter.acquire()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                session = await self._get_session()
                # Enhanced timeout configuration for SEC.gov reliability issues
                timeout = aiohttp.ClientTimeout(
                    total=120,        # Total timeout increased from 60s
                    connect=30,       # Connection timeout
                    sock_read=90      # Socket read timeout
                )
                
                async with session.get(url, timeout=timeout, **kwargs) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '')
                        if 'application/json' in content_type:
                            return await response.json()
                        else:
                            text = await response.text()
                            return {'content': text, 'content_type': content_type}
                    elif response.status == 429:
                        # Rate limited - wait and retry once
                        logger.warning("Rate limited by SEC, waiting...")
                        await asyncio.sleep(2)
                        return await self._make_request(url, **kwargs)
                    elif response.status == 404:
                        # Not found - could be wrong CIK, no filings, or incorrect endpoint
                        logger.warning(f"SEC API 404 Not Found: {url}")
                        logger.warning(f"This could mean: 1) CIK not found, 2) No recent filings, 3) Wrong endpoint")
                        return None
                    elif response.status == 403:
                        # Forbidden - likely user agent issue (2024 SEC requirements)
                        error_text = await response.text()
                        logger.warning(f"SEC API 403 Forbidden: {url}")
                        logger.warning(f"User-Agent issue with SEC 2024 requirements. Current UA: {self.user_agent}")
                        logger.warning(f"SEC endpoint temporarily unavailable - graceful degradation in effect")
                        logger.debug(f"SEC Response: {error_text[:200]}...")
                        return None
                    else:
                        logger.warning(f"SEC API request failed: {response.status} for {url}")
                        if attempt < max_retries - 1 and response.status >= 500:
                            # Server error - retry after delay
                            backoff_time = min(10 * (attempt + 1), 30)
                            logger.warning(f"SEC API server error. Retrying in {backoff_time}s (attempt {attempt + 1})")
                            await asyncio.sleep(backoff_time)
                            continue
                        return None
                    
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    backoff_time = min(10 * (attempt + 1), 30)  # Reduced backoff time
                    logger.warning(f"SEC API timeout. Retrying in {backoff_time}s (attempt {attempt + 1})")
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    logger.warning(f"SEC API timeout after {max_retries} attempts for {url} - continuing with degraded data")
                    return None
                    
            except aiohttp.ClientConnectorError as e:
                if attempt < max_retries - 1:
                    backoff_time = min(15 * (attempt + 1), 45)  # Reduced backoff time
                    logger.warning(f"SEC API connection error: {e}. Retrying in {backoff_time}s (attempt {attempt + 1})")
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    logger.warning(f"SEC API connection failed after {max_retries} attempts: {e} - continuing with degraded data")
                    return None
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    backoff_time = min(10 * (attempt + 1), 30)
                    logger.warning(f"SEC API request error: {e}. Retrying in {backoff_time}s (attempt {attempt + 1})")
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    logger.error(f"SEC API request error after {max_retries} attempts: {str(e)}")
                    logger.debug(f"Failed URL: {url}")
                    logger.debug(f"Exception type: {type(e).__name__}")
                    return None
        
        return None  # All retries exhausted
    
    def normalize_cik(self, cik: int or str) -> str:
        """Normalize CIK to 10-digit zero-padded format (SEC requirement)."""
        return str(cik).zfill(10)
    
    async def load_company_tickers(self) -> bool:
        """Load company ticker to CIK mapping using SEC's official mapping."""
        try:
            # Use SEC's latest official mapping (updated weekly)
            url = f"{self.base_url}/files/company_tickers.json"
            data = await self._make_request(url)
            
            if not data:
                # Fallback to exchange mapping
                logger.warning("Standard mapping failed, trying exchange mapping...")
                url = f"{self.base_url}/files/company_tickers_exchange.json"
                data = await self._make_request(url)
                
            if not data:
                logger.error("Both standard and exchange mappings failed - SEC API may be temporarily unavailable")
                logger.warning("Operating with degraded functionality - some SEC features disabled")
                return False
            
            self._company_tickers = {}
            self._cik_to_ticker = {}
            
            # Handle both old and new SEC API formats
            if 'fields' in data and 'data' in data:
                # New format: company_tickers_exchange.json
                fields = data['fields']
                cik_idx = fields.index('cik')
                name_idx = fields.index('name') 
                ticker_idx = fields.index('ticker')
                exchange_idx = fields.index('exchange') if 'exchange' in fields else None
                
                for row in data['data']:
                    cik = self.normalize_cik(row[cik_idx])  # Always normalize CIK
                    ticker = row[ticker_idx].upper()
                    title = row[name_idx]
                    exchange = row[exchange_idx] if exchange_idx is not None else None
                    
                    company_info = CompanyInfo(
                        cik=cik,
                        ticker=ticker,
                        title=title,
                        exchange=exchange
                    )
                    
                    self._company_tickers[ticker] = company_info
                    self._cik_to_ticker[cik] = ticker
            else:
                # Standard format: company_tickers.json
                for item in data.values():
                    cik = self.normalize_cik(item['cik_str'])  # Always normalize CIK
                    ticker = item['ticker'].upper()
                    title = item['title']
                    
                    company_info = CompanyInfo(
                        cik=cik,
                        ticker=ticker,
                        title=title
                    )
                    
                    self._company_tickers[ticker] = company_info
                    self._cik_to_ticker[cik] = ticker
            
            logger.info(f"Loaded {len(self._company_tickers)} company tickers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load company tickers: {str(e)}")
            logger.warning("SEC Edgar integration will operate with reduced functionality")
            # Initialize empty mappings to prevent crashes
            self._company_tickers = {}
            self._cik_to_ticker = {}
            return False
    
    def get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """Convert ticker symbol to normalized 10-digit CIK."""
        if not self._company_tickers:
            logger.warning("Company tickers not loaded. Call load_company_tickers() first.")
            return None
        
        ticker = ticker.upper().strip()
        company_info = self._company_tickers.get(ticker)
        if not company_info:
            logger.debug(f"CIK not found for ticker {ticker}")
            return None
            
        return company_info.cik
    
    def get_ticker_from_cik(self, cik: str or int) -> Optional[str]:
        """Convert CIK to ticker symbol."""
        if not self._cik_to_ticker:
            logger.warning("Company tickers not loaded. Call load_company_tickers() first.")
            return None
        
        normalized_cik = self.normalize_cik(cik)
        ticker = self._cik_to_ticker.get(normalized_cik)
        if not ticker:
            logger.debug(f"Ticker not found for CIK {normalized_cik}")
        return ticker
    
    async def get_company_facts(self, ticker: str) -> Optional[CompanyFacts]:
        """Get company facts and fundamentals using proper CIK format."""
        cik = self.get_cik_from_ticker(ticker)
        if not cik:
            logger.warning(f"CIK not found for ticker {ticker}. Ensure company_tickers are loaded.")
            return None
        
        try:
            # Use correct EDGAR API endpoint with normalized CIK
            # Note: data.sec.gov requires exact formatting - tested and working
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            logger.debug(f"Fetching company facts from: {url}")
            data = await self._make_request(url)
            
            if not data:
                logger.warning(f"No company facts data returned for {ticker} (CIK: {cik})")
                return None
            
            return CompanyFacts(
                cik=cik,
                entity_name=data.get('entityName', ''),
                facts=data.get('facts', {}),
                units=self._extract_units_data(data.get('facts', {}))
            )
            
        except Exception as e:
            logger.error(f"Failed to get company facts for {ticker}: {e}")
            return None
    
    def _extract_units_data(self, facts: Dict) -> Dict[str, List[Dict]]:
        """Extract and organize units data from facts."""
        units_data = {}
        
        for taxonomy in facts.values():
            for concept, concept_data in taxonomy.items():
                if 'units' in concept_data:
                    for unit_type, values in concept_data['units'].items():
                        key = f"{concept}_{unit_type}"
                        units_data[key] = values
        
        return units_data
    
    async def get_company_submissions(self, ticker: str, limit: int = 50) -> List[SECFiling]:
        """Get company's recent SEC submissions using proper CIK format."""
        cik = self.get_cik_from_ticker(ticker)
        if not cik:
            logger.warning(f"CIK not found for ticker {ticker}. Check ticker symbol or load company_tickers.")
            return []
        
        try:
            # Use correct EDGAR API endpoint with normalized CIK
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            logger.debug(f"Fetching submissions from: {url}")
            data = await self._make_request(url)
            
            if not data:
                logger.warning(f"No submissions data returned for {ticker} (CIK: {cik}). Company may have no recent filings.")
                return []
                
            if 'filings' not in data:
                logger.warning(f"No filings section found for {ticker} (CIK: {cik})")
                return []
            
            filings_data = data['filings']['recent']
            filings = []
            
            # Process filings
            for i in range(min(limit, len(filings_data.get('form', [])))):
                filing = SECFiling(
                    accession_number=filings_data['accessionNumber'][i],
                    filing_type=filings_data['form'][i],
                    filing_date=filings_data['filingDate'][i],
                    report_date=filings_data.get('reportDate', [None] * (i + 1))[i],
                    company_name=data.get('name', ''),
                    cik=cik,
                    ticker=ticker.upper(),
                    form_description=filings_data.get('primaryDocDescription', [''] * (i + 1))[i],
                    size=filings_data.get('size', [None] * (i + 1))[i],
                    document_count=filings_data.get('documentCount', [None] * (i + 1))[i],
                    period_of_report=filings_data.get('periodOfReport', [None] * (i + 1))[i],
                    primary_document=filings_data.get('primaryDocument', [None] * (i + 1))[i]
                )
                
                # Build document URL
                acc_no_clean = filing.accession_number.replace('-', '')
                filing.url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_clean}/{filing.primary_document}"
                
                filings.append(filing)
            
            logger.info(f"Retrieved {len(filings)} filings for {ticker}")
            return filings
            
        except Exception as e:
            logger.error(f"Failed to get submissions for {ticker}: {e}")
            return []
    
    async def search_filings(self, 
                           ticker: str, 
                           form_types: List[str] = None,
                           days_back: int = 90,
                           keywords: List[str] = None) -> List[SECFiling]:
        """Search for specific filings with optional keyword filtering."""
        if form_types is None:
            form_types = ['10-K', '10-Q', '8-K', '13D', '13G', 'DEF 14A', 'Form 4']
        
        filings = await self.get_company_submissions(ticker, limit=100)
        
        # Filter by form types and date
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        filtered_filings = []
        
        for filing in filings:
            # Date filter
            if filing.filing_date < cutoff_date:
                continue
            
            # Form type filter
            if form_types and filing.filing_type not in form_types:
                continue
            
            # Keyword filter (if content is available)
            if keywords and filing.content:
                content_lower = filing.content.lower()
                if not any(keyword.lower() in content_lower for keyword in keywords):
                    continue
            
            filtered_filings.append(filing)
        
        return filtered_filings
    
    async def get_filing_content(self, filing: SECFiling) -> Optional[SECFiling]:
        """Download and parse full filing content."""
        if not filing.url:
            return None
        
        try:
            # Use direct request for filing content
            await self.rate_limiter.acquire()
            session = await self._get_session()
            
            async with session.get(filing.url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to download filing: {response.status}")
                    return None
                
                content = await response.text()
                filing.content = content
                
                # Parse key sections
                filing.key_sections = self._parse_filing_sections(content, filing.filing_type)
                
                # Extract sentiment signals
                filing.sentiment_signals = self._extract_sentiment_signals(content, filing.filing_type)
                
                return filing
                
        except Exception as e:
            logger.error(f"Failed to get filing content: {e}")
            return None
    
    def _parse_filing_sections(self, content: str, filing_type: str) -> Dict[str, str]:
        """Parse key sections from filing content."""
        sections = {}
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text() if soup else content
            
            if filing_type in ['10-K', '10-Q']:
                # Standard 10-K/10-Q sections
                patterns = {
                    'business': r'item\s+1\b.*?business.*?(?=item\s+1a|item\s+2)',
                    'risk_factors': r'item\s+1a.*?risk\s+factors.*?(?=item\s+1b|item\s+2)',
                    'legal_proceedings': r'item\s+3.*?legal\s+proceedings.*?(?=item\s+4)',
                    'mda': r'item\s+7.*?management.*?discussion.*?(?=item\s+7a|item\s+8)',
                    'controls': r'item\s+9a.*?controls\s+and\s+procedures.*?(?=item\s+9b|item\s+10)',
                }
                
                for section_name, pattern in patterns.items():
                    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if match:
                        sections[section_name] = match.group(0)[:5000]  # Limit size
            
            elif filing_type == '8-K':
                # 8-K items
                item_patterns = {
                    'item_1_01': r'item\s+1\.01.*?(?=item\s+\d|$)',
                    'item_2_02': r'item\s+2\.02.*?(?=item\s+\d|$)',
                    'item_5_02': r'item\s+5\.02.*?(?=item\s+\d|$)',
                    'item_8_01': r'item\s+8\.01.*?(?=item\s+\d|$)',
                }
                
                for item_name, pattern in item_patterns.items():
                    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if match:
                        sections[item_name] = match.group(0)[:3000]
            
            return sections
            
        except Exception as e:
            logger.error(f"Failed to parse filing sections: {e}")
            return {}
    
    def _extract_sentiment_signals(self, content: str, filing_type: str) -> List[str]:
        """Extract sentiment signals from filing content."""
        signals = []
        
        try:
            content_lower = content.lower()
            
            # CEO/Leadership changes
            leadership_signals = [
                'chief executive officer', 'ceo', 'resignation', 'departure',
                'appointed', 'elected', 'transition', 'interim'
            ]
            
            if any(signal in content_lower for signal in leadership_signals):
                if 'resignation' in content_lower or 'departure' in content_lower:
                    signals.append('executive_departure')
                if 'appointed' in content_lower or 'elected' in content_lower:
                    signals.append('executive_appointment')
            
            # Financial stress signals
            stress_signals = [
                'material weakness', 'going concern', 'substantial doubt',
                'covenant violation', 'default', 'bankruptcy', 'chapter 11'
            ]
            
            for signal in stress_signals:
                if signal in content_lower:
                    signals.append(f'financial_stress_{signal.replace(" ", "_")}')
            
            # Positive signals
            positive_signals = [
                'record revenue', 'record earnings', 'exceeded expectations',
                'strong performance', 'growth in', 'expansion'
            ]
            
            for signal in positive_signals:
                if signal in content_lower:
                    signals.append(f'positive_{signal.replace(" ", "_")}')
            
            # M&A signals
            ma_signals = ['acquisition', 'merger', 'divestiture', 'spin-off']
            for signal in ma_signals:
                if signal in content_lower:
                    signals.append(f'ma_{signal}')
            
            # Delayed filings
            if filing_type in ['10-K', '10-Q'] and 'delay' in content_lower:
                signals.append('delayed_filing')
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to extract sentiment signals: {e}")
            return []
    
    async def get_recent_filings_feed(self, hours_back: int = 24) -> List[SECFiling]:
        """Get recent filings from SEC RSS feed."""
        try:
            # SEC RSS feed for recent filings
            url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=&company=&dateb=&owner=include&start=0&count=100&output=atom"
            
            response_data = await self._make_request(url)
            if not response_data or 'content' not in response_data:
                return []
            
            # Parse XML feed
            root = ET.fromstring(response_data['content'])
            filings = []
            
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                try:
                    title_elem = entry.find('.//{http://www.w3.org/2005/Atom}title')
                    updated_elem = entry.find('.//{http://www.w3.org/2005/Atom}updated')
                    link_elem = entry.find('.//{http://www.w3.org/2005/Atom}link')
                    
                    if None in [title_elem, updated_elem, link_elem]:
                        continue
                    
                    # Parse filing info from title
                    title = title_elem.text
                    filing_info = self._parse_feed_title(title)
                    
                    if not filing_info:
                        continue
                    
                    # Check if recent enough
                    updated_time = datetime.fromisoformat(updated_elem.text.replace('Z', '+00:00'))
                    if updated_time.replace(tzinfo=None) < cutoff_time:
                        continue
                    
                    filing = SECFiling(
                        accession_number=filing_info.get('accession_number', ''),
                        filing_type=filing_info.get('form_type', ''),
                        filing_date=updated_time.strftime('%Y-%m-%d'),
                        company_name=filing_info.get('company_name', ''),
                        cik=filing_info.get('cik', ''),
                        ticker=filing_info.get('ticker', ''),
                        form_description=title,
                        url=link_elem.get('href')
                    )
                    
                    filings.append(filing)
                    
                except Exception as e:
                    logger.debug(f"Failed to parse feed entry: {e}")
                    continue
            
            logger.info(f"Retrieved {len(filings)} recent filings from feed")
            return filings
            
        except Exception as e:
            logger.error(f"Failed to get recent filings feed: {e}")
            return []
    
    def _parse_feed_title(self, title: str) -> Optional[Dict[str, str]]:
        """Parse filing information from RSS feed title."""
        try:
            # Example: "10-K - APPLE INC (0000320193) (Filer)"
            pattern = r'(\S+)\s*-\s*(.+?)\s*\((\d+)\)'
            match = re.search(pattern, title)
            
            if match:
                form_type = match.group(1)
                company_name = match.group(2).strip()
                cik = match.group(3).zfill(10)
                ticker = self.get_ticker_from_cik(cik) or ''
                
                return {
                    'form_type': form_type,
                    'company_name': company_name,
                    'cik': cik,
                    'ticker': ticker,
                    'accession_number': ''  # Not available in feed title
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to parse feed title: {e}")
            return None
    
    async def analyze_filing_sentiment(self, filing: SECFiling) -> Dict[str, float]:
        """Analyze sentiment of filing content."""
        if not filing.content:
            return {}
        
        try:
            sentiment_scores = {}
            
            # Basic keyword sentiment analysis
            positive_keywords = [
                'growth', 'strong', 'increased', 'improved', 'expansion',
                'successful', 'profitable', 'exceeded', 'record'
            ]
            
            negative_keywords = [
                'decline', 'decreased', 'loss', 'weak', 'challenging',
                'difficult', 'uncertainty', 'risk', 'impairment', 'restructuring'
            ]
            
            content_lower = filing.content.lower()
            content_words = content_lower.split()
            
            positive_count = sum(1 for word in content_words if any(pos in word for pos in positive_keywords))
            negative_count = sum(1 for word in content_words if any(neg in word for neg in negative_keywords))
            
            total_words = len(content_words)
            if total_words > 0:
                sentiment_scores['positive_ratio'] = positive_count / total_words
                sentiment_scores['negative_ratio'] = negative_count / total_words
                sentiment_scores['net_sentiment'] = (positive_count - negative_count) / total_words
            
            # Signal-based sentiment
            signal_sentiment = 0.0
            for signal in filing.sentiment_signals:
                if 'positive' in signal:
                    signal_sentiment += 0.1
                elif any(neg in signal for neg in ['financial_stress', 'departure', 'delayed']):
                    signal_sentiment -= 0.1
            
            sentiment_scores['signal_sentiment'] = signal_sentiment
            
            return sentiment_scores
            
        except Exception as e:
            logger.error(f"Failed to analyze filing sentiment: {e}")
            return {}
    
    async def cleanup(self):
        """Clean up resources via connection pool."""
        await connection_pool.close_session("sec_edgar_client")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.load_company_tickers()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()