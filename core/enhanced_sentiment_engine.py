#!/usr/bin/env python3
"""
Enhanced Sentiment Analysis Engine
Combines FinGPT, Ollama, and traditional sentiment analysis for comprehensive market sentiment
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import numpy as np

from core.fingpt_sentiment_analyzer import get_fingpt_analyzer
from core.llm_sentiment_analyzer import LLMSentimentAnalyzer
from tools.llm_client import LLMClient

logger = logging.getLogger(__name__)

class EnhancedSentimentEngine:
    """Enhanced sentiment analysis combining multiple LLM approaches."""
    
    def __init__(self):
        """Initialize the enhanced sentiment engine."""
        self.fingpt_analyzer = None
        self.llm_analyzer = None
        self.llm_client = None
        
        # Sentiment weights for ensemble - prioritizing Ollama
        self.model_weights = {
            'ollama': 0.8,      # Local LLM (primary - fast and reliable)
            'fingpt': 0.1,      # Specialized financial model (fallback only)
            'traditional': 0.1  # Rule-based fallback
        }
        
        logger.info("✅ Enhanced Sentiment Engine initialized")
    
    async def initialize(self):
        """Initialize all sentiment analysis components."""
        
        try:
            # Initialize FinGPT
            logger.info("🤖 Initializing FinGPT analyzer...")
            self.fingpt_analyzer = get_fingpt_analyzer()
            await self.fingpt_analyzer.initialize_model()
            
            # Initialize traditional LLM analyzer
            logger.info("🔧 Initializing traditional LLM analyzer...")
            self.llm_analyzer = LLMSentimentAnalyzer()
            
            # Initialize LLM client for Ollama
            logger.info("🦙 Initializing Ollama client...")
            self.llm_client = LLMClient()
            
            logger.info("✅ Enhanced sentiment engine fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced sentiment engine: {e}")
            return False
    
    async def analyze_comprehensive_sentiment(self, 
                                           text: str, 
                                           symbol: str,
                                           use_ensemble: bool = True) -> Dict[str, Any]:
        """Perform comprehensive sentiment analysis using multiple models."""
        
        if not self.fingpt_analyzer:
            await self.initialize()
        
        results = {}
        ensemble_scores = []
        
        try:
            # 1. FinGPT Analysis (Primary - Financial Specialist)
            logger.debug(f"🤖 Running FinGPT analysis for {symbol}")
            try:
                fingpt_result = await self.fingpt_analyzer.analyze_sentiment(text, symbol)
                results['fingpt'] = fingpt_result
                
                if fingpt_result and 'score' in fingpt_result:
                    weighted_score = fingpt_result['score'] * self.model_weights['fingpt']
                    ensemble_scores.append(weighted_score)
                    logger.debug(f"✅ FinGPT score for {symbol}: {fingpt_result['score']:.3f}")
            except Exception as e:
                logger.warning(f"⚠️ FinGPT analysis failed for {symbol}: {e}")
                results['fingpt'] = None
            
            # 2. Ollama Analysis (Secondary - General Purpose)
            logger.debug(f"🦙 Running Ollama analysis for {symbol}")
            try:
                ollama_result = await self._analyze_with_ollama(text, symbol)
                results['ollama'] = ollama_result
                
                if ollama_result and 'score' in ollama_result:
                    weighted_score = ollama_result['score'] * self.model_weights['ollama']
                    ensemble_scores.append(weighted_score)
                    logger.debug(f"✅ Ollama score for {symbol}: {ollama_result['score']:.3f}")
            except Exception as e:
                logger.warning(f"⚠️ Ollama analysis failed for {symbol}: {e}")
                results['ollama'] = None
            
            # 3. Traditional Analysis (Fallback)
            logger.debug(f"🔧 Running traditional analysis for {symbol}")
            try:
                traditional_result = await self._analyze_traditional(text, symbol)
                results['traditional'] = traditional_result
                
                if traditional_result and 'score' in traditional_result:
                    weighted_score = traditional_result['score'] * self.model_weights['traditional']
                    ensemble_scores.append(weighted_score)
                    logger.debug(f"✅ Traditional score for {symbol}: {traditional_result['score']:.3f}")
            except Exception as e:
                logger.warning(f"⚠️ Traditional analysis failed for {symbol}: {e}")
                results['traditional'] = None
            
            # 4. Ensemble Results
            if use_ensemble and ensemble_scores:
                ensemble_score = np.mean(ensemble_scores)
                ensemble_confidence = self._calculate_ensemble_confidence(results)
                ensemble_sentiment = self._score_to_sentiment(ensemble_score)
                
                # Create comprehensive result
                comprehensive_result = {
                    'symbol': symbol,
                    'overall_sentiment': ensemble_sentiment,
                    'overall_score': ensemble_score,
                    'confidence': ensemble_confidence,
                    'reasoning': self._generate_ensemble_reasoning(results),
                    'model_results': results,
                    'ensemble_used': True,
                    'models_successful': len([r for r in results.values() if r is not None]),
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"📊 Comprehensive sentiment for {symbol}: {ensemble_sentiment} ({ensemble_score:+.3f}, conf: {ensemble_confidence:.2f})")
                
                return comprehensive_result
            else:
                # Use best available single result
                best_result = self._select_best_single_result(results, symbol)
                best_result.update({
                    'model_results': results,
                    'ensemble_used': False,
                    'analysis_timestamp': datetime.now().isoformat()
                })
                return best_result
                
        except Exception as e:
            logger.error(f"❌ Comprehensive sentiment analysis failed for {symbol}: {e}")
            return self._create_fallback_result(text, symbol)
    
    async def _analyze_with_ollama(self, text: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze sentiment using Ollama."""
        
        try:
            if not self.llm_client:
                return None
                
            prompt = f"""
Analyze the sentiment of this financial news about {symbol}:

"{text}"

Please provide:
1. Sentiment: positive, negative, or neutral
2. Score: -1.0 to +1.0 (negative to positive)
3. Confidence: 0.0 to 1.0
4. Brief reasoning

Format: Sentiment: [sentiment] | Score: [score] | Confidence: [confidence] | Reasoning: [reasoning]
"""
            
            response = await self.llm_client.generate_response(
                system_prompt="You are a financial analyst specializing in market sentiment.",
                user_message=prompt,
                max_tokens=150,
                temperature=0.3,
                provider="llama"  # Force Ollama usage
            )
            
            if response and response.content:
                return self._parse_ollama_response(response.content, symbol)
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Ollama sentiment analysis error for {symbol}: {e}")
            return None
    
    def _parse_ollama_response(self, response: str, symbol: str) -> Dict[str, Any]:
        """Parse Ollama response and extract sentiment data."""
        
        try:
            response_lower = response.lower()
            
            # Default values
            sentiment = 'neutral'
            score = 0.0
            confidence = 0.5
            reasoning = response
            
            # Extract sentiment
            if 'positive' in response_lower:
                sentiment = 'positive'
                score = 0.3
            elif 'negative' in response_lower:
                sentiment = 'negative'
                score = -0.3
            
            # Try to extract numerical score
            import re
            score_match = re.search(r'score[:\s]*([+-]?[0-9]*\.?[0-9]+)', response_lower)
            if score_match:
                extracted_score = float(score_match.group(1))
                if -1.0 <= extracted_score <= 1.0:
                    score = extracted_score
            
            # Try to extract confidence
            conf_match = re.search(r'confidence[:\s]*([0-9]*\.?[0-9]+)', response_lower)
            if conf_match:
                extracted_conf = float(conf_match.group(1))
                if extracted_conf > 1.0:
                    extracted_conf /= 100.0  # Convert percentage
                if 0.0 <= extracted_conf <= 1.0:
                    confidence = extracted_conf
            
            return {
                'sentiment': sentiment,
                'score': score,
                'confidence': confidence,
                'reasoning': reasoning,
                'model': 'ollama_llama3',
                'symbol': symbol
            }
            
        except Exception as e:
            logger.error(f"❌ Error parsing Ollama response: {e}")
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.3,
                'reasoning': 'Failed to parse response',
                'model': 'ollama_error',
                'symbol': symbol
            }
    
    async def _analyze_traditional(self, text: str, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment using traditional LLM analyzer."""
        
        try:
            if not self.llm_analyzer:
                self.llm_analyzer = LLMSentimentAnalyzer()
            
            # Use the existing traditional analyzer (check if it has async method)
            if hasattr(self.llm_analyzer, 'analyze_sentiment_async'):
                result = await self.llm_analyzer.analyze_sentiment_async(text, symbol)
            elif hasattr(self.llm_analyzer, 'analyze_text'):
                # Use the correct method name: analyze_text
                analysis_result = await self.llm_analyzer.analyze_text(text)
                result = {
                    'sentiment': analysis_result.sentiment.value if hasattr(analysis_result, 'sentiment') else 'neutral',
                    'overall_score': analysis_result.score if hasattr(analysis_result, 'score') else 0.0,
                    'confidence': analysis_result.confidence if hasattr(analysis_result, 'confidence') else 0.5,
                    'reasoning': analysis_result.reasoning if hasattr(analysis_result, 'reasoning') else 'Traditional LLM analysis'
                }
            else:
                # Fallback to a simple method
                result = None
            
            # Standardize the result format
            if result:
                return {
                    'sentiment': result.get('sentiment', 'neutral'),
                    'score': result.get('overall_score', 0.0),
                    'confidence': result.get('confidence', 0.5),
                    'reasoning': result.get('reasoning', 'Traditional LLM analysis'),
                    'model': 'traditional_llm',
                    'symbol': symbol
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Traditional analysis error for {symbol}: {e}")
            return None
    
    def _calculate_ensemble_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence for ensemble result."""
        
        confidences = []
        
        for model_name, result in results.items():
            if result and 'confidence' in result:
                confidences.append(result['confidence'])
        
        if not confidences:
            return 0.5
        
        # Use weighted average of confidences, with bonus for agreement
        avg_confidence = np.mean(confidences)
        
        # Agreement bonus: if models agree on sentiment, boost confidence
        sentiments = []
        for result in results.values():
            if result and 'sentiment' in result:
                sentiments.append(result['sentiment'])
        
        if len(set(sentiments)) == 1 and len(sentiments) > 1:  # All models agree
            avg_confidence = min(1.0, avg_confidence * 1.2)
        
        return avg_confidence
    
    def _score_to_sentiment(self, score: float) -> str:
        """Convert numerical score to sentiment label."""
        
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _generate_ensemble_reasoning(self, results: Dict[str, Any]) -> str:
        """Generate reasoning text for ensemble result."""
        
        active_models = [name for name, result in results.items() if result is not None]
        
        if not active_models:
            return "No models provided valid analysis"
        
        reasoning_parts = []
        
        for model_name in active_models:
            result = results[model_name]
            sentiment = result.get('sentiment', 'unknown')
            confidence = result.get('confidence', 0)
            reasoning_parts.append(f"{model_name}: {sentiment} ({confidence:.2f})")
        
        return f"Ensemble analysis from {len(active_models)} models: " + "; ".join(reasoning_parts)
    
    def _select_best_single_result(self, results: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Select the best single result when ensemble is not available."""
        
        # Priority order: FinGPT > Ollama > Traditional
        priority_order = ['fingpt', 'ollama', 'traditional']
        
        for model_name in priority_order:
            if model_name in results and results[model_name] is not None:
                result = results[model_name].copy()
                result.update({
                    'overall_sentiment': result.get('sentiment', 'neutral'),
                    'overall_score': result.get('score', 0.0),
                    'primary_model': model_name
                })
                logger.info(f"📊 Using {model_name} result for {symbol}: {result['overall_sentiment']}")
                return result
        
        # Fallback if no models worked
        return self._create_fallback_result("", symbol)
    
    def _create_fallback_result(self, text: str, symbol: str) -> Dict[str, Any]:
        """Create a fallback result when all models fail."""
        
        return {
            'symbol': symbol,
            'overall_sentiment': 'neutral',
            'overall_score': 0.0,
            'confidence': 0.3,
            'reasoning': 'Fallback analysis - all models failed',
            'model_results': {},
            'ensemble_used': False,
            'analysis_timestamp': datetime.now().isoformat(),
            'fallback': True
        }
    
    async def batch_analyze_sentiments(self, 
                                     news_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Batch analyze sentiments for multiple news items."""
        
        logger.info(f"📊 Batch analyzing sentiments for {len(news_data)} news items")
        
        results = {}
        
        # Process in smaller batches to avoid overwhelming the models
        batch_size = 3
        for i in range(0, len(news_data), batch_size):
            batch = news_data[i:i + batch_size]
            
            tasks = []
            for item in batch:
                symbol = item.get('symbol', 'UNKNOWN')
                text = item.get('content', item.get('text', ''))
                
                if text and symbol:
                    task = self.analyze_comprehensive_sentiment(text, symbol)
                    tasks.append((symbol, task))
            
            # Execute batch
            if tasks:
                batch_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                
                for (symbol, _), result in zip(tasks, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"❌ Batch analysis failed for {symbol}: {result}")
                        results[symbol] = self._create_fallback_result("", symbol)
                    else:
                        results[symbol] = result
        
        logger.info(f"✅ Batch sentiment analysis complete: {len(results)} results")
        return results
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get status information about the sentiment engine."""
        
        return {
            'fingpt_initialized': self.fingpt_analyzer is not None,
            'llm_analyzer_initialized': self.llm_analyzer is not None,
            'llm_client_initialized': self.llm_client is not None,
            'model_weights': self.model_weights,
            'fingpt_info': self.fingpt_analyzer.get_model_info() if self.fingpt_analyzer else None
        }

# Global instance
_enhanced_sentiment_engine = None

def get_enhanced_sentiment_engine() -> EnhancedSentimentEngine:
    """Get or create the global enhanced sentiment engine."""
    global _enhanced_sentiment_engine
    if _enhanced_sentiment_engine is None:
        _enhanced_sentiment_engine = EnhancedSentimentEngine()
    return _enhanced_sentiment_engine

# Test function
async def test_enhanced_sentiment():
    """Test the enhanced sentiment engine."""
    
    logging.basicConfig(level=logging.INFO)
    logger.info("🧪 Testing Enhanced Sentiment Engine...")
    
    engine = get_enhanced_sentiment_engine()
    await engine.initialize()
    
    # Test cases
    test_cases = [
        {
            'symbol': 'AAPL',
            'content': 'Apple reports record quarterly earnings, beating analyst expectations by 15% with strong iPhone sales and services growth.'
        },
        {
            'symbol': 'TSLA', 
            'content': 'Tesla stock plunges 20% after disappointing delivery numbers and production cuts at Shanghai factory.'
        },
        {
            'symbol': 'GOOGL',
            'content': 'Google announces breakthrough in quantum computing, potentially revolutionizing data processing capabilities.'
        }
    ]
    
    # Test comprehensive analysis
    for case in test_cases:
        result = await engine.analyze_comprehensive_sentiment(
            case['content'], 
            case['symbol']
        )
        logger.info(f"📈 {case['symbol']}: {result['overall_sentiment']} ({result['overall_score']:+.3f}) - {result['confidence']:.2f}")
    
    # Test batch analysis
    batch_results = await engine.batch_analyze_sentiments(test_cases)
    logger.info(f"📊 Batch analysis: {len(batch_results)} results")
    
    # Status check
    status = engine.get_engine_status()
    logger.info(f"🔧 Engine status: {status}")
    
    logger.info("✅ Enhanced sentiment engine test completed")

if __name__ == "__main__":
    asyncio.run(test_enhanced_sentiment())