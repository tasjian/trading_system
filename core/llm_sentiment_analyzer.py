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
    """LLM-based sentiment analyzer for financial content using unified LLM client."""
    
    def __init__(self, anthropic_api_key: Optional[str] = None, ollama_base_url: str = "http://localhost:11434"):
        # Initialize unified LLM client for consistency and FinGPT support
        from tools.llm_client import LLMClient
        self.llm_client = LLMClient()
        
        # Keep legacy parameters for backward compatibility
        self.anthropic_api_key = anthropic_api_key
        self.ollama_base_url = ollama_base_url
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)  # Greatly increased for CPU Ollama performance
            )
    
    def _create_sentiment_system_prompt(self, context: str = "financial_news") -> str:
        """Create fast, optimized system prompt for sentiment analysis."""
        
        return """Analyze financial sentiment. Respond with JSON only:
{
    "sentiment": "very_positive|positive|neutral|negative|very_negative",
    "confidence": 0.8,
    "score": 0.5,
    "reasoning": "brief explanation",
    "key_phrases": ["key", "phrases"],
    "financial_impact": "impact on stock price",
    "risk_factors": ["risk1"],
    "opportunities": ["opportunity1"]
}

Be fast and concise."""
    
    def _create_sentiment_prompt(self, text: str, context: str = "financial_news") -> str:
        """Create a detailed prompt for sentiment analysis (legacy method)."""
        
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

    # Legacy API methods removed - using unified LLM client (FinGPT -> Ollama) for local processing
    
    
    def _parse_llm_response(self, response_content: str) -> Optional[Dict]:
        """Parse LLM response and extract sentiment analysis data."""
        try:
            # Try to find JSON in the response
            import re
            
            # Look for JSON block in response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['sentiment', 'confidence', 'score']
                if all(field in result for field in required_fields):
                    # Ensure defaults for optional fields
                    result.setdefault('reasoning', 'LLM sentiment analysis')
                    result.setdefault('key_phrases', [])
                    
                    # Fix financial_impact if it's a dict (common LLM mistake)
                    financial_impact = result.get('financial_impact')
                    if isinstance(financial_impact, dict):
                        # Extract the main content from the dict
                        if 'expected_price_move' in financial_impact:
                            result['financial_impact'] = str(financial_impact['expected_price_move'])
                        else:
                            result['financial_impact'] = str(list(financial_impact.values())[0]) if financial_impact else None
                    elif financial_impact is None:
                        result['financial_impact'] = None
                    else:
                        result['financial_impact'] = str(financial_impact)
                    
                    result.setdefault('risk_factors', [])
                    result.setdefault('opportunities', [])
                    
                    return result
            
            # If no valid JSON found, try to extract sentiment from text
            response_lower = response_content.lower()
            
            if any(word in response_lower for word in ['very positive', 'strongly positive']):
                sentiment = 'very_positive'
                score = 0.8
            elif any(word in response_lower for word in ['positive', 'bullish', 'optimistic']):
                sentiment = 'positive'
                score = 0.4
            elif any(word in response_lower for word in ['very negative', 'strongly negative']):
                sentiment = 'very_negative'
                score = -0.8
            elif any(word in response_lower for word in ['negative', 'bearish', 'pessimistic']):
                sentiment = 'negative'
                score = -0.4
            else:
                sentiment = 'neutral'
                score = 0.0
            
            return {
                'sentiment': sentiment,
                'confidence': 0.6,
                'score': score,
                'reasoning': 'Extracted from LLM response text',
                'key_phrases': [],
                'financial_impact': None,
                'risk_factors': [],
                'opportunities': []
            }
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
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
        """Analyze sentiment of financial text using unified LLM client (including FinGPT)."""
        
        # Truncate very long texts to avoid token limits
        if len(text) > 8000:
            text = text[:8000] + "... [truncated]"
        
        system_prompt = self._create_sentiment_system_prompt(context)
        user_message = f"Analyze the sentiment of this financial content:\n\n{text}"
        
        try:
            # Use unified LLM client (FinGPT -> Claude -> Llama -> OpenAI)
            llm_response = await self.llm_client.generate_response(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=0.3,  # Lower temperature for consistent sentiment analysis
                max_tokens=500,
                model=None  # Use default/preferred model
            )
            
            # Parse JSON response
            result = self._parse_llm_response(llm_response.content)
            
            if result:
                # Validate and create structured result
                return SentimentAnalysis(**result)
        
        except Exception as e:
            logger.error(f"LLM sentiment analysis failed: {e}")
        
        # Fallback to rule-based analysis
        logger.warning("LLM analysis failed, using fallback method")
        result = self._fallback_sentiment_analysis(text)
        return SentimentAnalysis(**result)
    
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
    
    async def analyze_earnings_transcript(self, transcript) -> Dict[str, SentimentAnalysis]:
        """Analyze sentiment of earnings call transcript sections with enhanced prompts."""
        results = {}
        
        try:
            # Overall earnings sentiment analysis
            if transcript.full_text and len(transcript.full_text) > 500:
                overall_prompt = f"""
                Analyze the overall sentiment of this earnings call transcript for {transcript.symbol}:
                
                Company: {transcript.company_name}
                Quarter: {transcript.quarter} {transcript.year}
                Date: {transcript.date.strftime('%Y-%m-%d') if hasattr(transcript, 'date') else 'Recent'}
                
                Transcript Content (first 4000 characters):
                {transcript.full_text[:4000]}
                
                Provide comprehensive analysis focusing on:
                - Overall business performance vs expectations
                - Management confidence and forward guidance
                - Key financial metrics and growth trends
                - Market position and competitive outlook
                - Risk factors and operational challenges
                - Investment attractiveness and stock implications
                
                Consider both the content and tone of management communications.
                """
                
                results['overall'] = await self.analyze_text(overall_prompt, "earnings_call")
            
            # Management presentation analysis
            if transcript.management_section and len(transcript.management_section) > 200:
                mgmt_prompt = f"""
                Analyze the management presentation section of {transcript.symbol}'s earnings call:
                
                Management Presentation:
                {transcript.management_section[:3500]}
                
                Focus on:
                - Management tone and confidence level
                - Business performance highlights
                - Strategic initiatives and execution
                - Forward guidance and outlook
                - Key operational metrics discussed
                - Management's view on market conditions
                
                Assess the sentiment implications for stock performance.
                """
                
                results['management'] = await self.analyze_text(mgmt_prompt, "earnings_call")
            
            # Q&A section analysis
            if transcript.qa_section and len(transcript.qa_section) > 200:
                qa_prompt = f"""
                Analyze the Q&A section of {transcript.symbol}'s earnings call:
                
                Q&A Discussion:
                {transcript.qa_section[:3500]}
                
                Focus on:
                - Types of questions being asked by analysts
                - Management responsiveness and transparency
                - Areas of analyst concern or skepticism
                - Confidence in addressing future challenges
                - Clarity and specificity of answers
                - Any defensive or evasive responses
                
                Determine overall investor sentiment based on the Q&A dynamics.
                """
                
                results['qa'] = await self.analyze_text(qa_prompt, "earnings_call")
            
            # Key metrics analysis if available
            if hasattr(transcript, 'key_metrics') and transcript.key_metrics:
                metrics_text = ". ".join([f"{k}: {v}" for k, v in transcript.key_metrics.items()])
                metrics_prompt = f"""
                Analyze the sentiment implications of these key financial metrics from {transcript.symbol}'s earnings:
                
                Key Financial Metrics: {metrics_text}
                
                Evaluate:
                - Performance vs historical trends
                - Likely market reaction to these numbers
                - Strength of financial position
                - Growth trajectory implications
                """
                
                results['metrics'] = await self.analyze_text(metrics_prompt, "earnings_call")
            
            logger.info(f"Completed earnings transcript analysis for {transcript.symbol}: {len(results)} sections analyzed")
            return results
            
        except Exception as e:
            logger.error(f"Error in earnings transcript analysis: {e}")
            return {}
    
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