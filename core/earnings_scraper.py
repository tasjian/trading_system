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
    

class EarningsCallScraper:
    """Scraper for earnings call transcripts from The Motley Fool."""
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.session = session
        self.base_url = "https://www.fool.com"
        self.search_url = f"{self.base_url}/earnings-call-transcripts/"
        
        # Rate limiting
        self.request_delay = 1.0  # seconds between requests
        self.last_request_time = 0.0
        
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                },
                timeout=aiohttp.ClientTimeout(total=30)
            )
    
    async def _rate_limit(self):
        """Implement rate limiting."""
        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.request_delay:
            await asyncio.sleep(self.request_delay - elapsed)
        self.last_request_time = asyncio.get_event_loop().time()
    
    async def search_transcripts(self, symbol: str, days_back: int = 90) -> List[Dict]:
        """Search for earnings call transcripts for a symbol."""
        await self._ensure_session()
        await self._rate_limit()
        
        try:
            transcript_links = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Get all current earnings transcript links from main page
            async with self.session.get(self.search_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Look for earnings call transcript links using the proven pattern
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '')
                        link_text = link.get_text(strip=True)
                        
                        # Check for earnings call transcript pattern
                        if '/earnings/call-transcripts/' in href and link_text:
                            # Extract symbol from link text (format: "Company (SYMBOL) Q# YYYY Earnings...")
                            symbol_match = re.search(r'\(([A-Z]{2,5})\)', link_text)
                            found_symbol = symbol_match.group(1) if symbol_match else None
                            
                            # Check if this is the symbol we're looking for
                            if found_symbol and found_symbol.upper() == symbol.upper():
                                full_url = urljoin(self.base_url, href)
                                
                                # Extract date from URL pattern: /YYYY/MM/DD/
                                date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', href)
                                transcript_date = None
                                
                                if date_match:
                                    try:
                                        transcript_date = datetime(
                                            int(date_match.group(1)),
                                            int(date_match.group(2)),
                                            int(date_match.group(3))
                                        )
                                    except ValueError:
                                        pass
                                
                                # If no date from URL, try to extract from text
                                if not transcript_date:
                                    # Look for quarter and year in text (e.g., "Q2 2025")
                                    quarter_match = re.search(r'Q([1-4])\s+(\d{4})', link_text)
                                    if quarter_match:
                                        quarter = int(quarter_match.group(1))
                                        year = int(quarter_match.group(2))
                                        # Estimate quarter start month
                                        month = (quarter - 1) * 3 + 1
                                        transcript_date = datetime(year, month, 1)
                                    else:
                                        # Default to recent if no date found
                                        transcript_date = datetime.now() - timedelta(days=30)
                                
                                # Check if within date range
                                if transcript_date >= cutoff_date:
                                    transcript_links.append({
                                        'url': full_url,
                                        'date': transcript_date,
                                        'title': link_text,
                                        'symbol': found_symbol
                                    })
            
            # Sort by date, most recent first
            transcript_links.sort(key=lambda x: x['date'], reverse=True)
            
            logger.info(f"Found {len(transcript_links)} transcripts for {symbol}")
            return transcript_links
                
        except Exception as e:
            logger.error(f"Error searching transcripts for {symbol}: {e}")
            return []
    
    async def scrape_transcript(self, url: str, symbol: str) -> Optional[EarningsTranscript]:
        """Scrape a single earnings call transcript."""
        await self._ensure_session()
        await self._rate_limit()
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch transcript: {response.status}")
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract transcript content
                transcript_content = self._extract_transcript_content(soup)
                if not transcript_content:
                    logger.warning(f"No transcript content found at {url}")
                    return None
                
                # Extract metadata
                metadata = self._extract_metadata(soup, url, symbol)
                
                # Parse transcript sections
                management_section, qa_section = self._parse_transcript_sections(transcript_content)
                
                # Extract key metrics
                key_metrics = self._extract_key_metrics(transcript_content)
                
                return EarningsTranscript(
                    symbol=symbol,
                    company_name=metadata.get('company_name', ''),
                    quarter=metadata.get('quarter', ''),
                    year=metadata.get('year', datetime.now().year),
                    date=metadata.get('date', datetime.now()),
                    url=url,
                    full_text=transcript_content,
                    management_section=management_section,
                    qa_section=qa_section,
                    key_metrics=key_metrics
                )
                
        except Exception as e:
            logger.error(f"Error scraping transcript from {url}: {e}")
            return None
    
    def _extract_transcript_content(self, soup: BeautifulSoup) -> str:
        """Extract the main transcript content from the page."""
        # Use the proven selector first - this works for current Motley Fool structure
        content_div = soup.select_one('.article-body')
        if content_div:
            # Remove unwanted elements
            for unwanted in content_div.find_all(['script', 'style', 'noscript', 'nav', 'header', 'footer']):
                unwanted.decompose()
            
            content = content_div.get_text(separator='\n', strip=True)
            if len(content) > 1000:  # Ensure substantial content
                logger.info(f"Successfully extracted content using .article-body: {len(content):,} characters")
                return content
        
        # Fallback selectors if the main one fails
        fallback_selectors = [
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content-body',
            '[data-module="ArticleBody"]',
            '.transcript-content'
        ]
        
        for selector in fallback_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                # Remove unwanted elements
                for unwanted in content_div.find_all(['script', 'style', 'noscript', 'nav', 'header', 'footer']):
                    unwanted.decompose()
                
                content = content_div.get_text(separator='\n', strip=True)
                if len(content) > 1000:
                    logger.info(f"Successfully extracted content using {selector}: {len(content):,} characters")
                    return content
        
        # Look for any article tag
        article = soup.find('article')
        if article:
            for unwanted in article.find_all(['script', 'style', 'noscript', 'nav', 'header', 'footer']):
                unwanted.decompose()
            content = article.get_text(separator='\n', strip=True)
            if len(content) > 1000:
                logger.info(f"Successfully extracted content using article tag: {len(content):,} characters")
                return content
        
        # Last resort: look for the largest text block
        text_blocks = soup.find_all(['div', 'section', 'main'])
        content_candidates = []
        
        for block in text_blocks:
            # Skip navigation, headers, footers, ads
            block_classes = ' '.join(block.get('class', [])).lower()
            if any(skip_class in block_classes for skip_class in ['nav', 'header', 'footer', 'sidebar', 'ad', 'menu']):
                continue
                
            text = block.get_text(separator='\n', strip=True)
            if len(text) > 2000:  # Only consider substantial content
                content_candidates.append((len(text), text))
        
        if content_candidates:
            # Return the longest content block
            content_candidates.sort(key=lambda x: x[0], reverse=True)
            logger.info(f"Successfully extracted content using largest text block: {len(content_candidates[0][1]):,} characters")
            return content_candidates[0][1]
        
        logger.warning("Failed to extract substantial transcript content")
        return ""
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str, symbol: str) -> Dict:
        """Extract metadata from the transcript page."""
        metadata = {
            'symbol': symbol,
            'company_name': '',
            'quarter': '',
            'year': datetime.now().year,
            'date': datetime.now()
        }
        
        # Extract title for company name and quarter info
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
            
            # Try to extract company name
            company_match = re.search(r'^([^(]+)', title)
            if company_match:
                metadata['company_name'] = company_match.group(1).strip()
            
            # Try to extract quarter and year
            quarter_match = re.search(r'Q([1-4])\s+(\d{4})', title, re.IGNORECASE)
            if quarter_match:
                metadata['quarter'] = f"Q{quarter_match.group(1)}"
                metadata['year'] = int(quarter_match.group(2))
        
        # Try to extract date from meta tags or content
        date_meta = soup.find('meta', {'property': 'article:published_time'}) or \
                   soup.find('meta', {'name': 'date'})
        
        if date_meta and date_meta.get('content'):
            try:
                metadata['date'] = datetime.fromisoformat(
                    date_meta['content'].replace('Z', '+00:00')
                )
            except ValueError:
                pass
        
        return metadata
    
    def _parse_transcript_sections(self, content: str) -> Tuple[str, str]:
        """Parse transcript into management presentation and Q&A sections."""
        content_lines = content.split('\n')
        
        # Look for section markers with more comprehensive patterns
        management_start = -1
        qa_start = -1
        
        for i, line in enumerate(content_lines):
            line_lower = line.lower().strip()
            
            # Skip very short lines
            if len(line_lower) < 5:
                continue
                
            # Look for management section indicators
            management_markers = [
                'prepared remarks',
                'management discussion', 
                'presentation',
                'opening statement',
                'management commentary',
                'company discussion',
                'executive summary'
            ]
            
            if any(marker in line_lower for marker in management_markers):
                if management_start == -1:
                    management_start = i
                    logger.debug(f"Found management section at line {i}: {line_lower[:50]}...")
            
            # Look for Q&A section indicators
            qa_markers = [
                'questions and answers',
                'q&a',
                'question and answer',
                'analyst questions',
                'questions from analysts',
                'operator',  # Often indicates start of Q&A
                'thank you. we will now begin the question'
            ]
            
            if any(marker in line_lower for marker in qa_markers):
                qa_start = i
                break
        
        # Extract sections
        if management_start >= 0 and qa_start > management_start:
            management_section = '\n'.join(content_lines[management_start:qa_start])
            qa_section = '\n'.join(content_lines[qa_start:])
        elif qa_start >= 0:
            management_section = '\n'.join(content_lines[:qa_start])
            qa_section = '\n'.join(content_lines[qa_start:])
        else:
            # No clear division found, split roughly in half
            mid_point = len(content_lines) // 2
            management_section = '\n'.join(content_lines[:mid_point])
            qa_section = '\n'.join(content_lines[mid_point:])
        
        return management_section.strip(), qa_section.strip()
    
    def _extract_key_metrics(self, content: str) -> Dict[str, str]:
        """Extract key financial metrics mentioned in the transcript."""
        metrics = {}
        
        # Common financial terms and their patterns
        metric_patterns = {
            'revenue': r'revenue[s]?\s+(?:of\s+)?[\$]?([\d,\.]+)\s*(?:billion|million|thousand)?',
            'net_income': r'net\s+income[s]?\s+(?:of\s+)?[\$]?([\d,\.]+)\s*(?:billion|million|thousand)?',
            'eps': r'earnings\s+per\s+share[s]?\s+(?:of\s+)?[\$]?([\d,\.]+)',
            'margin': r'(?:gross|operating|net)\s+margin[s]?\s+(?:of\s+)?([\d,\.]+)%?',
            'growth': r'(?:revenue|sales)\s+growth[s]?\s+(?:of\s+)?([\d,\.]+)%?'
        }
        
        content_lower = content.lower()
        
        for metric_name, pattern in metric_patterns.items():
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                # Take the first substantial match
                for match in matches:
                    if match and len(match.strip()) > 0:
                        metrics[metric_name] = match.strip()
                        break
        
        return metrics
    
    async def get_latest_transcript(self, symbol: str) -> Optional[EarningsTranscript]:
        """Get the most recent earnings call transcript for a symbol."""
        transcripts = await self.search_transcripts(symbol, days_back=120)
        
        if not transcripts:
            return None
        
        # Get the most recent transcript
        latest = transcripts[0]
        return await self.scrape_transcript(latest['url'], symbol)
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()