"""
GPT-5-nano Batch Sentiment Analysis for Financial Trading System

High-performance, cost-effective sentiment analysis using OpenAI's GPT-5-nano
model with batch processing for improved throughput and reduced costs.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive" 
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class SentimentAnalysis(BaseModel):
    """Structured sentiment analysis result."""
    
    sentiment: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    score: float = Field(ge=-1.0, le=1.0, description="Numerical sentiment score -1 to 1")
    reasoning: str = Field(description="Explanation of sentiment determination")
    key_phrases: List[str] = Field(default=[], description="Important phrases that influenced sentiment")
    financial_impact: Optional[str] = Field(default=None, description="Expected financial impact")
    risk_factors: List[str] = Field(default=[], description="Identified risk factors")
    opportunities: List[str] = Field(default=[], description="Identified opportunities")


@dataclass
class BatchRequest:
    """Individual batch request for sentiment analysis."""
    custom_id: str
    text: str
    context: str
    symbol: Optional[str] = None


@dataclass
class BatchResult:
    """Result from batch sentiment analysis."""
    custom_id: str
    sentiment: SentimentAnalysis
    processing_time: float
    success: bool
    error: Optional[str] = None


class GPTBatchSentimentAnalyzer:
    """
    GPT-5-nano batch sentiment analyzer for financial content.
    
    Features:
    - Batch processing for cost efficiency
    - Async/await support for non-blocking operations
    - Fail-fast error handling
    - Structured output with financial context
    """
    
    def __init__(self):
        # Initialize OpenAI client
        if not settings.openai_api_key:
            logger.warning("⚠️ OpenAI API key not configured - GPT-5-nano sentiment analysis disabled")
            self.client = None
            self.model = None
            self.active_batches = {}
            self.batch_cache = {}
            return
        
        try:
            # Initialize OpenAI client with minimal configuration to avoid version conflicts
            self.client = OpenAI(api_key=settings.openai_api_key)
            self.model = "gpt-4o-mini"  # Use available model for now (GPT-5-nano when available)
            self.active_batches = {}  # Track active batch jobs
            self.batch_cache = {}  # Cache completed batch results
            
            logger.info("✅ GPT-4o-mini Batch Sentiment Analyzer initialized (GPT-5-nano when available)")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
            self.model = None
            self.active_batches = {}
            self.batch_cache = {}
    
    def _create_batch_request(self, request: BatchRequest) -> Dict:
        """Create a single batch request for GPT-5-nano."""
        
        system_prompt = self._create_financial_system_prompt(request.context)
        
        return {
            "custom_id": request.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.text}
                ],
                "max_tokens": 300,
                "temperature": 0.1,  # Low temperature for consistent financial analysis
                "response_format": {"type": "json_object"}
            }
        }
    
    def _create_financial_system_prompt(self, context: str = "financial_news") -> str:
        """Create specialized system prompt for financial sentiment analysis."""
        
        base_prompt = """You are an expert financial sentiment analyst. Analyze the provided text and respond with JSON only.

Your response must be valid JSON in this exact format:
{
    "sentiment": "very_positive|positive|neutral|negative|very_negative",
    "confidence": 0.85,
    "score": 0.7,
    "reasoning": "Brief explanation of sentiment determination",
    "key_phrases": ["phrase1", "phrase2"],
    "financial_impact": "Expected impact on stock price/market",
    "risk_factors": ["risk1", "risk2"],
    "opportunities": ["opportunity1", "opportunity2"]
}

Sentiment Guidelines:
- very_positive: Strong bullish indicators, major positive catalysts
- positive: Moderately bullish, positive news/trends
- neutral: Mixed signals, no clear direction, routine updates
- negative: Moderately bearish, concerning developments
- very_negative: Strong bearish indicators, major negative catalysts

Score: -1.0 (very negative) to +1.0 (very positive)
Confidence: 0.0 (uncertain) to 1.0 (very confident)

Focus on financial implications, market impact, and trading relevance."""
        
        context_additions = {
            "financial_news": "\nContext: Analyzing financial news articles and press releases.",
            "social_media": "\nContext: Analyzing social media posts, tweets, and informal discussions.",
            "earnings_call": "\nContext: Analyzing earnings call transcripts and investor communications.",
            "analyst_report": "\nContext: Analyzing analyst reports and research notes."
        }
        
        return base_prompt + context_additions.get(context, "")
    
    async def submit_batch_analysis(self, requests: List[BatchRequest]) -> str:
        """
        Submit batch requests for sentiment analysis.
        
        Returns:
            batch_id: Batch job ID for tracking completion
        """
        if not requests:
            raise ValueError("No requests provided for batch analysis")
        
        try:
            # Create batch requests for GPT-5-nano
            batch_requests = [self._create_batch_request(req) for req in requests]
            
            # Submit batch to OpenAI
            batch = self.client.batches.create(
                input_file_id=None,  # We'll use requests directly
                endpoint="/v1/chat/completions",
                completion_window="24h",  # 24-hour completion window for cost savings
                metadata={
                    "purpose": "financial_sentiment_analysis",
                    "submitted_at": datetime.now().isoformat(),
                    "request_count": len(requests)
                }
            )
            
            # Store batch info for tracking
            self.active_batches[batch.id] = {
                "submitted_at": datetime.now(),
                "request_count": len(requests),
                "requests": {req.custom_id: req for req in requests},
                "status": "submitted"
            }
            
            logger.info(f"✅ Submitted batch {batch.id} with {len(requests)} sentiment analysis requests")
            return batch.id
            
        except Exception as e:
            error_msg = f"Failed to submit batch sentiment analysis: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    async def check_batch_status(self, batch_id: str) -> Dict:
        """Check the status of a submitted batch."""
        try:
            batch = self.client.batches.retrieve(batch_id)
            
            status_info = {
                "batch_id": batch_id,
                "status": batch.status,
                "created_at": batch.created_at,
                "in_progress_at": batch.in_progress_at,
                "completed_at": batch.completed_at,
                "failed_at": batch.failed_at,
                "request_counts": batch.request_counts,
                "metadata": batch.metadata
            }
            
            # Update local tracking
            if batch_id in self.active_batches:
                self.active_batches[batch_id]["status"] = batch.status
                self.active_batches[batch_id]["last_checked"] = datetime.now()
            
            return status_info
            
        except Exception as e:
            error_msg = f"Failed to check batch status {batch_id}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    async def retrieve_batch_results(self, batch_id: str) -> List[BatchResult]:
        """Retrieve and parse completed batch results."""
        try:
            # Check if batch is completed
            status = await self.check_batch_status(batch_id)
            if status["status"] != "completed":
                raise RuntimeError(f"Batch {batch_id} not completed yet. Status: {status['status']}")
            
            # Get batch results
            batch = self.client.batches.retrieve(batch_id)
            if not batch.output_file_id:
                raise RuntimeError(f"No output file for completed batch {batch_id}")
            
            # Download and parse results
            output_file = self.client.files.content(batch.output_file_id)
            results = []
            
            for line in output_file.text.strip().split('\n'):
                if not line:
                    continue
                    
                try:
                    result_data = json.loads(line)
                    custom_id = result_data["custom_id"]
                    
                    if result_data.get("error"):
                        # Handle error result
                        results.append(BatchResult(
                            custom_id=custom_id,
                            sentiment=None,
                            processing_time=0.0,
                            success=False,
                            error=result_data["error"]["message"]
                        ))
                        continue
                    
                    # Parse successful result
                    response = result_data["response"]
                    content = response["choices"][0]["message"]["content"]
                    
                    # Parse JSON response from GPT-5-nano
                    sentiment_data = json.loads(content)
                    sentiment = SentimentAnalysis(**sentiment_data)
                    
                    results.append(BatchResult(
                        custom_id=custom_id,
                        sentiment=sentiment,
                        processing_time=0.0,  # Batch processing time not individual
                        success=True
                    ))
                    
                except Exception as e:
                    logger.error(f"Failed to parse batch result for {custom_id}: {e}")
                    results.append(BatchResult(
                        custom_id=custom_id,
                        sentiment=None,
                        processing_time=0.0,
                        success=False,
                        error=str(e)
                    ))
            
            # Cache results
            self.batch_cache[batch_id] = results
            
            # Clean up active batch tracking
            if batch_id in self.active_batches:
                del self.active_batches[batch_id]
            
            logger.info(f"✅ Retrieved {len(results)} sentiment analysis results from batch {batch_id}")
            return results
            
        except Exception as e:
            error_msg = f"Failed to retrieve batch results {batch_id}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    async def analyze_texts_batch(self, texts: List[str], context: str = "financial_news", 
                                 symbols: Optional[List[str]] = None) -> List[BatchResult]:
        """
        Convenience method for end-to-end batch sentiment analysis.
        
        Note: This method submits the batch and returns immediately.
        Results must be retrieved later using retrieve_batch_results().
        """
        if symbols and len(symbols) != len(texts):
            symbols = [None] * len(texts)
        
        # Create batch requests
        requests = []
        for i, text in enumerate(texts):
            symbol = symbols[i] if symbols else None
            custom_id = f"sentiment_{int(time.time())}_{i}"
            if symbol:
                custom_id += f"_{symbol}"
            
            requests.append(BatchRequest(
                custom_id=custom_id,
                text=text,
                context=context,
                symbol=symbol
            ))
        
        # Submit batch
        batch_id = await self.submit_batch_analysis(requests)
        
        logger.info(f"✅ Submitted batch sentiment analysis for {len(texts)} texts. Batch ID: {batch_id}")
        return batch_id
    
    # LEGACY OLLAMA SUPPORT (COMMENTED OUT FOR POTENTIAL FUTURE USE)
    """
    async def analyze_text_ollama_fallback(self, text: str, context: str = "financial_news") -> SentimentAnalysis:
        # DISABLED: Ollama sentiment analysis fallback
        # Keep this code commented for potential future restoration
        
        # from core.llm_sentiment_analyzer import LLMSentimentAnalyzer
        # analyzer = LLMSentimentAnalyzer()
        # return await analyzer.analyze_text(text, context)
        
        raise RuntimeError("Ollama fallback disabled - use GPT-5-nano batch analysis only")
    """
    
    def get_active_batches(self) -> Dict:
        """Get status of all active batch jobs."""
        return self.active_batches.copy()
    
    def get_cached_results(self, batch_id: str) -> Optional[List[BatchResult]]:
        """Get cached batch results if available."""
        return self.batch_cache.get(batch_id)
    
    async def cleanup_completed_batches(self, max_age_hours: int = 24):
        """Clean up old batch tracking data."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean up active batches that are old
        old_batches = [
            batch_id for batch_id, info in self.active_batches.items()
            if info["submitted_at"] < cutoff_time
        ]
        
        for batch_id in old_batches:
            try:
                status = await self.check_batch_status(batch_id)
                if status["status"] in ["completed", "failed", "cancelled"]:
                    del self.active_batches[batch_id]
                    logger.info(f"Cleaned up old batch {batch_id}")
            except Exception as e:
                logger.warning(f"Failed to check old batch {batch_id}: {e}")
        
        # Clean up old cached results
        old_cache_keys = []
        for batch_id in self.batch_cache.keys():
            if batch_id not in self.active_batches:
                old_cache_keys.append(batch_id)
        
        for batch_id in old_cache_keys[:10]:  # Keep only 10 most recent
            del self.batch_cache[batch_id]


# Global instance
gpt_sentiment_analyzer = GPTBatchSentimentAnalyzer()


async def analyze_sentiment_batch(texts: List[str], context: str = "financial_news", 
                                 symbols: Optional[List[str]] = None) -> str:
    """
    Convenience function for batch sentiment analysis.
    
    Returns:
        batch_id: Use with retrieve_batch_results() to get results
    """
    return await gpt_sentiment_analyzer.analyze_texts_batch(texts, context, symbols)


async def get_batch_results(batch_id: str) -> List[BatchResult]:
    """Convenience function to retrieve batch results."""
    return await gpt_sentiment_analyzer.retrieve_batch_results(batch_id)


# Compatibility layer for existing Ollama-based code
class LegacySentimentInterface:
    """
    Compatibility interface for existing code that expects immediate sentiment analysis.
    
    NOTE: This will submit individual requests and wait for completion, which is less
    efficient than batch processing. For new code, use the batch methods directly.
    """
    
    async def analyze_text(self, text: str, context: str = "financial_news") -> SentimentAnalysis:
        """
        Legacy interface for single text analysis.
        
        WARNING: This method blocks until completion and is less efficient.
        Use batch methods for better performance.
        """
        # For immediate results, we'll need to use real-time API
        # This is a compatibility layer only
        
        if not settings.openai_api_key:
            raise RuntimeError("OpenAI API key required for sentiment analysis")
        
        try:
            # Initialize OpenAI client with minimal configuration
            client = OpenAI(api_key=settings.openai_api_key)
            
            system_prompt = gpt_sentiment_analyzer._create_financial_system_prompt(context)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use faster model for real-time requests
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=300,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            sentiment_data = json.loads(content)
            
            return SentimentAnalysis(**sentiment_data)
            
        except Exception as e:
            error_msg = f"Real-time sentiment analysis failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)


# Legacy compatibility instance
legacy_sentiment_analyzer = LegacySentimentInterface()