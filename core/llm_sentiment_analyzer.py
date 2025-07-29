"""
LLM-based sentiment analysis for financial texts.

Uses language models to analyze sentiment from news articles, earnings calls,
and social media content with structured output and detailed reasoning.
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

import aiohttp
from pydantic import BaseModel, Field

from .earnings_scraper import EarningsTranscript

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
    

class LLMSentimentAnalyzer:
    """LLM-based sentiment analyzer for financial content."""
    
    def __init__(self, anthropic_api_key: Optional[str] = None, openai_api_key: Optional[str] = None, ollama_base_url: str = "http://localhost:11434"):
        self.anthropic_api_key = anthropic_api_key
        self.openai_api_key = openai_api_key
        self.ollama_base_url = ollama_base_url
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Model preferences (try Anthropic first, then Ollama, then OpenAI)
        self.use_anthropic = bool(anthropic_api_key)
        self.use_openai = bool(openai_api_key)
        self.anthropic_model = "claude-3-5-sonnet-20241022"
        self.openai_model = "gpt-4o-mini"  # Use the more available model
        self.ollama_model = "llama3:8b"  # Match the available model
        
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
    
    def _create_sentiment_prompt(self, text: str, context: str = "financial_news") -> str:
        """Create a detailed prompt for sentiment analysis."""
        
        context_descriptions = {
            "financial_news": "financial news article",
            "earnings_call": "earnings call transcript",
            "social_media": "social media post about stocks/trading",
            "analyst_report": "financial analyst report"
        }
        
        context_desc = context_descriptions.get(context, "financial text")
        
        return f"""You are a financial sentiment analysis expert. Analyze the sentiment of the following {context_desc} and provide a detailed assessment.

Text to analyze:
{text}

Please provide your analysis in the following JSON format:
{{
    "sentiment": "very_positive|positive|neutral|negative|very_negative",
    "confidence": 0.85,
    "score": 0.6,
    "reasoning": "Detailed explanation of why you chose this sentiment",
    "key_phrases": ["phrase1", "phrase2", "phrase3"],
    "financial_impact": "Expected impact on stock price and company valuation",
    "risk_factors": ["risk1", "risk2"],
    "opportunities": ["opportunity1", "opportunity2"]
}}

Guidelines:
- sentiment: Choose the most appropriate sentiment level
- confidence: How confident you are (0.0-1.0)
- score: Numerical sentiment (-1.0 = very negative, 0.0 = neutral, 1.0 = very positive)
- reasoning: Explain your analysis considering financial context
- key_phrases: Extract 3-5 most sentiment-influencing phrases
- financial_impact: Assess likely impact on stock performance
- risk_factors: Identify potential risks mentioned
- opportunities: Identify potential opportunities mentioned

Focus on:
- Financial performance indicators
- Market outlook and guidance
- Competitive positioning
- Regulatory or industry changes
- Management confidence and outlook
- Revenue, profit, and growth trends

Respond only with valid JSON."""

    async def _call_openai(self, prompt: str) -> Optional[Dict]:
        """Call OpenAI API for sentiment analysis."""
        if not self.openai_api_key:
            return None
            
        await self._ensure_session()
        
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.openai_model,
                "messages": [
                    {"role": "system", "content": "You are a financial sentiment analysis expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            async with self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content'].strip()
                    
                    # Try to parse JSON response
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        # Try to extract JSON from markdown code blocks
                        if "```json" in content:
                            json_start = content.find("```json") + 7
                            json_end = content.find("```", json_start)
                            json_str = content[json_start:json_end].strip()
                            return json.loads(json_str)
                        raise
                        
                else:
                    logger.warning(f"OpenAI API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return None
    
    async def _call_anthropic(self, prompt: str) -> Optional[Dict]:
        """Call Anthropic Claude API for sentiment analysis."""
        if not self.anthropic_api_key:
            return None
            
        await self._ensure_session()
        
        try:
            headers = {
                "x-api-key": self.anthropic_api_key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": self.anthropic_model,
                "max_tokens": 1000,
                "temperature": 0.3,
                "system": "You are a financial sentiment analysis expert. Always respond with valid JSON only.",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            async with self.session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['content'][0]['text'].strip()
                    
                    # Try to parse JSON response
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        # Try to extract JSON from markdown code blocks
                        if "```json" in content:
                            json_start = content.find("```json") + 7
                            json_end = content.find("```", json_start)
                            json_str = content[json_start:json_end].strip()
                            return json.loads(json_str)
                        raise
                        
                else:
                    logger.warning(f"Anthropic API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return None
    
    async def _call_ollama(self, prompt: str) -> Optional[Dict]:
        """Call Ollama API for sentiment analysis."""
        await self._ensure_session()
        
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            }
            
            async with self.session.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data.get('response', '').strip()
                    
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse Ollama JSON response")
                        return None
                        
                else:
                    logger.warning(f"Ollama API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            return None
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict:
        """Fallback sentiment analysis using keyword-based approach."""
        positive_words = [
            'profit', 'growth', 'increase', 'positive', 'strong', 'beat', 'exceeded', 
            'outperform', 'bullish', 'upgrade', 'buy', 'optimistic', 'confident',
            'expansion', 'revenue', 'earnings', 'success', 'improvement', 'gain'
        ]
        
        negative_words = [
            'loss', 'decline', 'decrease', 'negative', 'weak', 'miss', 'failed',
            'underperform', 'bearish', 'downgrade', 'sell', 'pessimistic', 'concern',
            'contraction', 'debt', 'bankruptcy', 'crisis', 'deterioration', 'fall'
        ]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total_sentiment_words = pos_count + neg_count
        
        if total_sentiment_words == 0:
            sentiment = "neutral"
            score = 0.0
            confidence = 0.3
        else:
            score = (pos_count - neg_count) / max(total_sentiment_words, 1)
            confidence = min(0.8, total_sentiment_words / 10)  # Max 80% confidence
            
            if score > 0.5:
                sentiment = "positive"
            elif score > 0.2:
                sentiment = "neutral"
            elif score > -0.2:
                sentiment = "neutral"
            elif score > -0.5:
                sentiment = "negative"
            else:
                sentiment = "very_negative"
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "score": score,
            "reasoning": f"Keyword-based analysis: {pos_count} positive, {neg_count} negative words found",
            "key_phrases": [],
            "financial_impact": "Unable to determine without advanced analysis",
            "risk_factors": [],
            "opportunities": []
        }
    
    async def analyze_text(self, text: str, context: str = "financial_news") -> SentimentAnalysis:
        """Analyze sentiment of financial text using LLM."""
        
        # Truncate very long texts to avoid token limits
        if len(text) > 8000:
            text = text[:8000] + "... [truncated]"
        
        prompt = self._create_sentiment_prompt(text, context)
        
        # Try LLM analysis first (Anthropic -> Ollama -> OpenAI)
        result = None
        
        if self.use_anthropic:
            result = await self._call_anthropic(prompt)
        
        if not result:
            result = await self._call_ollama(prompt)
        
        if not result and self.use_openai:
            result = await self._call_openai(prompt)
        
        if not result:
            logger.warning("LLM analysis failed, using fallback method")
            result = self._fallback_sentiment_analysis(text)
        
        # Validate and create structured result
        try:
            return SentimentAnalysis(**result)
        except Exception as e:
            logger.error(f"Failed to create SentimentAnalysis: {e}")
            # Return safe fallback
            fallback_result = self._fallback_sentiment_analysis(text)
            return SentimentAnalysis(**fallback_result)
    
    async def analyze_news_articles(self, articles: List[Dict]) -> List[SentimentAnalysis]:
        """Analyze sentiment of multiple news articles."""
        if not articles:
            return []
        
        tasks = []
        for article in articles[:10]:  # Limit to 10 articles to avoid rate limits
            title = article.get('title', '')
            description = article.get('description', '')
            content = f"{title}. {description}"
            
            if content.strip():
                tasks.append(self.analyze_text(content, "financial_news"))
        
        if not tasks:
            return []
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def analyze_earnings_transcript(self, transcript: EarningsTranscript) -> Dict[str, SentimentAnalysis]:
        """Analyze sentiment of earnings call transcript sections."""
        results = {}
        
        # Analyze management presentation
        if transcript.management_section:
            results['management'] = await self.analyze_text(
                transcript.management_section, 
                "earnings_call"
            )
        
        # Analyze Q&A section
        if transcript.qa_section:
            results['qa'] = await self.analyze_text(
                transcript.qa_section,
                "earnings_call"
            )
        
        # Overall sentiment (first 3000 chars of full transcript)
        if transcript.full_text:
            summary_text = transcript.full_text[:3000]
            results['overall'] = await self.analyze_text(
                summary_text,
                "earnings_call"
            )
        
        return results
    
    def aggregate_sentiment_scores(self, analyses: List[SentimentAnalysis]) -> float:
        """Aggregate multiple sentiment analyses into a single score."""
        if not analyses:
            return 0.0
        
        # Filter out failed analyses
        valid_analyses = [a for a in analyses if isinstance(a, SentimentAnalysis)]
        
        if not valid_analyses:
            return 0.0
        
        # Weight by confidence and recency
        total_weight = 0.0
        weighted_score = 0.0
        
        for analysis in valid_analyses:
            weight = analysis.confidence
            weighted_score += analysis.score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()