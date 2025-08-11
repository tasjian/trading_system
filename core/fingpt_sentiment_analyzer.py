#!/usr/bin/env python3
"""
FinGPT Sentiment Analyzer
Integrates AI4Finance FinGPT for advanced financial sentiment analysis
"""

import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import requests
from datetime import datetime, timedelta
import re
import numpy as np
from pathlib import Path

# Add FinGPT to path
FINGPT_PATH = str(Path(__file__).parent.parent / "FinGPT")
if FINGPT_PATH not in sys.path:
    sys.path.insert(0, FINGPT_PATH)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from peft import PeftModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except (ImportError, AttributeError) as e:
    TRANSFORMERS_AVAILABLE = False

from config.settings import settings

logger = logging.getLogger(__name__)

class FinGPTSentimentAnalyzer:
    """Advanced financial sentiment analysis using AI4Finance FinGPT."""
    
    def __init__(self):
        """Initialize FinGPT sentiment analyzer."""
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_initialized = False
        self.fallback_mode = False
        
        # Model configurations
        self.model_configs = {
            'fingpt_sentiment_llama2_13b': {
                'model_name': 'FinGPT/fingpt-sentiment_llama2-13b_lora',
                'base_model': 'meta-llama/Llama-2-13b-hf',
                'description': 'FinGPT sentiment analysis based on LLaMA 2 13B'
            },
            'fingpt_forecaster': {
                'model_name': 'FinGPT/fingpt-forecaster_llama2-7b_lora',
                'base_model': 'meta-llama/Llama-2-7b-hf', 
                'description': 'FinGPT forecaster for market prediction'
            },
            'fingpt_mt_llama2_7b': {
                'model_name': 'FinGPT/fingpt-mt_llama2-7b_lora',
                'base_model': 'meta-llama/Llama-2-7b-hf',
                'description': 'FinGPT multi-task model'
            }
        }
        
        # Current model - using smaller 7B model for faster initialization
        self.current_model_key = 'fingpt_mt_llama2_7b'  # Smaller model for speed
        
        # Sentiment classification prompts
        self.sentiment_prompts = {
            'basic': """
Please analyze the sentiment of this financial text and classify it as one of: positive, negative, or neutral.
Focus on the financial implications and market impact.

Text: "{text}"

Classification:""",
            
            'advanced': """
You are a financial analyst specializing in market sentiment analysis. 
Analyze the following financial text and provide:
1. Sentiment classification (positive/negative/neutral)
2. Confidence score (0.0 to 1.0)
3. Key factors influencing the sentiment
4. Potential market impact

Text: "{text}"

Analysis:""",
            
            'trading_focused': """
As a quantitative trader, analyze this financial text for trading signals:
- Sentiment: positive/negative/neutral
- Strength: weak/moderate/strong  
- Time horizon: short-term/medium-term/long-term
- Confidence: 0.0 to 1.0
- Trading signal: buy/sell/hold

Text: "{text}"

Trading Analysis:"""
        }
        
        # Results cache
        self.analysis_cache = {}
        self.cache_expiry = timedelta(hours=1)  # 1 hour cache
        
        logger.info("✅ FinGPT Sentiment Analyzer initialized")
    
    async def initialize_model(self, model_key: Optional[str] = None) -> bool:
        """Initialize FinGPT model for inference."""
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️ Transformers not available, using fallback mode")
            self.fallback_mode = True
            return True
            
        if self.model_initialized:
            return True
            
        model_key = model_key or self.current_model_key
        
        try:
            if model_key not in self.model_configs:
                logger.error(f"❌ Unknown model key: {model_key}")
                return False
                
            config = self.model_configs[model_key]
            logger.info(f"🤖 Initializing {config['description']}...")
            
            # Check if running on CPU or GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🔧 Using device: {device}")
            
            # Try to load the model
            try:
                # First try to load the fine-tuned model directly
                self.model = AutoModelForCausalLM.from_pretrained(
                    config['model_name'],
                    trust_remote_code=True,
                    device_map="auto" if device == "cuda" else None,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    config['model_name'],
                    trust_remote_code=True
                )
                
                logger.info(f"✅ Loaded fine-tuned model: {config['model_name']}")
                
            except Exception as direct_load_error:
                logger.warning(f"⚠️ Could not load fine-tuned model directly: {direct_load_error}")
                logger.info("🔄 Trying to load base model with LoRA adapters...")
                
                # Try loading base model + LoRA adapters
                self.model = AutoModelForCausalLM.from_pretrained(
                    config['base_model'],
                    trust_remote_code=True,
                    device_map="auto" if device == "cuda" else None,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
                
                # Load LoRA adapters
                try:
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        config['model_name'],
                        trust_remote_code=True
                    )
                    logger.info(f"✅ Loaded base model + LoRA: {config['base_model']} + {config['model_name']}")
                except Exception as lora_error:
                    logger.warning(f"⚠️ Could not load LoRA adapters: {lora_error}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    config['base_model'],
                    trust_remote_code=True
                )
            
            # Configure tokenizer
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.tokenizer.padding_side = "left"
            
            # Create generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
            self.model_initialized = True
            self.current_model_key = model_key
            
            logger.info(f"✅ FinGPT model initialized successfully: {config['description']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize FinGPT model: {e}")
            logger.info("🔄 Falling back to FinGPT-enhanced Ollama analysis")
            self.fallback_mode = True
            return True  # Still return True to allow fallback operation
    
    async def analyze_sentiment(self, 
                              text: str, 
                              symbol: Optional[str] = None,
                              prompt_type: str = 'trading_focused',
                              use_cache: bool = True) -> Dict[str, Any]:
        """Analyze sentiment of financial text using FinGPT."""
        
        # Check cache first
        cache_key = f"{symbol}:{hash(text)}:{prompt_type}"
        if use_cache and cache_key in self.analysis_cache:
            cached_result, cached_time = self.analysis_cache[cache_key]
            if datetime.now() - cached_time < self.cache_expiry:
                logger.debug(f"📋 Using cached sentiment for {symbol}")
                return cached_result
        
        try:
            if self.fallback_mode or not self.model_initialized:
                # Use fallback analysis
                result = self._fallback_sentiment_analysis(text, symbol)
            else:
                # Use FinGPT model
                result = await self._fingpt_sentiment_analysis(text, symbol, prompt_type)
            
            # Cache result
            if use_cache:
                self.analysis_cache[cache_key] = (result, datetime.now())
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed for {symbol}: {e}")
            return self._fallback_sentiment_analysis(text, symbol)
    
    async def _fingpt_sentiment_analysis(self, 
                                       text: str, 
                                       symbol: Optional[str],
                                       prompt_type: str) -> Dict[str, Any]:
        """Perform sentiment analysis using FinGPT model."""
        
        if not self.model_initialized:
            await self.initialize_model()
            
        if not self.model_initialized or self.fallback_mode:
            return self._fallback_sentiment_analysis(text, symbol)
        
        try:
            # Prepare prompt
            if prompt_type not in self.sentiment_prompts:
                prompt_type = 'trading_focused'
                
            prompt = self.sentiment_prompts[prompt_type].format(text=text[:1000])  # Limit text length
            
            # Generate response
            logger.debug(f"🤖 Analyzing sentiment with FinGPT for {symbol}")
            
            response = self.pipeline(
                prompt,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
                num_return_sequences=1
            )
            
            # Extract and parse response
            generated_text = response[0]['generated_text'] if response else ""
            parsed_result = self._parse_fingpt_response(generated_text, symbol)
            
            # Add metadata
            parsed_result.update({
                'model': f'fingpt_{self.current_model_key}',
                'prompt_type': prompt_type,
                'raw_response': generated_text,
                'analysis_time': datetime.now().isoformat(),
                'symbol': symbol
            })
            
            logger.debug(f"✅ FinGPT sentiment analysis complete for {symbol}: {parsed_result.get('sentiment', 'unknown')}")
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"❌ FinGPT analysis failed for {symbol}: {e}")
            return self._fallback_sentiment_analysis(text, symbol)
    
    def _parse_fingpt_response(self, response: str, symbol: Optional[str]) -> Dict[str, Any]:
        """Parse FinGPT response and extract structured sentiment data."""
        
        try:
            # Initialize result
            result = {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'strength': 'moderate',
                'factors': [],
                'trading_signal': 'hold',
                'time_horizon': 'short-term',
                'score': 0.0,
                'reasoning': response
            }
            
            response_lower = response.lower()
            
            # Extract sentiment
            if any(word in response_lower for word in ['positive', 'bullish', 'optimistic', 'favorable']):
                result['sentiment'] = 'positive'
                result['score'] = 0.3
                result['trading_signal'] = 'buy'
            elif any(word in response_lower for word in ['negative', 'bearish', 'pessimistic', 'unfavorable']):
                result['sentiment'] = 'negative'  
                result['score'] = -0.3
                result['trading_signal'] = 'sell'
            else:
                result['sentiment'] = 'neutral'
                result['score'] = 0.0
                result['trading_signal'] = 'hold'
            
            # Extract confidence
            confidence_patterns = [
                r'confidence[:\s]*([0-9]*\.?[0-9]+)',
                r'([0-9]*\.?[0-9]+).*confidence',
                r'confidence.*?([0-9]{1,2})%'
            ]
            
            for pattern in confidence_patterns:
                match = re.search(pattern, response_lower)
                if match:
                    conf_value = float(match.group(1))
                    if conf_value > 1.0:  # Percentage format
                        conf_value = conf_value / 100.0
                    result['confidence'] = min(max(conf_value, 0.0), 1.0)
                    break
            
            # Extract strength
            if any(word in response_lower for word in ['strong', 'high', 'significant']):
                result['strength'] = 'strong'
                result['score'] *= 1.5  # Amplify score for strong sentiment
            elif any(word in response_lower for word in ['weak', 'low', 'mild']):
                result['strength'] = 'weak'  
                result['score'] *= 0.7  # Reduce score for weak sentiment
                
            # Extract time horizon
            if any(word in response_lower for word in ['long-term', 'long term', 'quarterly', 'annual']):
                result['time_horizon'] = 'long-term'
            elif any(word in response_lower for word in ['medium-term', 'medium term', 'monthly']):
                result['time_horizon'] = 'medium-term'
            else:
                result['time_horizon'] = 'short-term'
            
            # Adjust score based on confidence
            result['score'] = result['score'] * result['confidence']
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error parsing FinGPT response: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'score': 0.0,
                'reasoning': 'Failed to parse response',
                'trading_signal': 'hold'
            }
    
    def _fallback_sentiment_analysis(self, text: str, symbol: Optional[str]) -> Dict[str, Any]:
        """Fallback rule-based sentiment analysis when FinGPT is not available."""
        
        logger.debug(f"🔧 Using fallback sentiment analysis for {symbol}")
        
        # Simple rule-based sentiment
        text_lower = text.lower()
        
        positive_words = [
            'gain', 'profit', 'growth', 'increase', 'rise', 'up', 'bullish', 'positive',
            'strong', 'beat', 'exceed', 'outperform', 'boost', 'surge', 'rally',
            'upgrade', 'buy', 'recommend', 'optimistic', 'favorable', 'good'
        ]
        
        negative_words = [
            'loss', 'decline', 'decrease', 'fall', 'down', 'bearish', 'negative',
            'weak', 'miss', 'underperform', 'drop', 'plunge', 'crash', 'sell',
            'downgrade', 'cut', 'pessimistic', 'unfavorable', 'bad', 'concern'
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate sentiment
        if pos_count > neg_count:
            sentiment = 'positive'
            score = min(0.5, (pos_count - neg_count) * 0.1)
            trading_signal = 'buy'
        elif neg_count > pos_count:
            sentiment = 'negative'
            score = max(-0.5, -(neg_count - pos_count) * 0.1)
            trading_signal = 'sell'
        else:
            sentiment = 'neutral'
            score = 0.0
            trading_signal = 'hold'
            
        confidence = min(0.8, abs(pos_count - neg_count) * 0.1 + 0.3)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'score': score,
            'trading_signal': trading_signal,
            'reasoning': f'Rule-based analysis: {pos_count} positive, {neg_count} negative words',
            'model': 'fallback_rule_based',
            'symbol': symbol,
            'analysis_time': datetime.now().isoformat()
        }
    
    async def batch_analyze_sentiments(self, 
                                     texts: List[Tuple[str, str]], 
                                     prompt_type: str = 'trading_focused') -> Dict[str, Dict[str, Any]]:
        """Analyze sentiment for multiple texts in batch."""
        
        logger.info(f"📊 Batch analyzing sentiment for {len(texts)} texts")
        
        results = {}
        
        # Process in batches to avoid memory issues
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self.analyze_sentiment(text, symbol, prompt_type, use_cache=True)
                for text, symbol in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for (text, symbol), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Batch analysis failed for {symbol}: {result}")
                    results[symbol] = self._fallback_sentiment_analysis(text, symbol)
                else:
                    results[symbol] = result
        
        logger.info(f"✅ Batch sentiment analysis complete: {len(results)} results")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model configuration."""
        
        return {
            'model_initialized': self.model_initialized,
            'fallback_mode': self.fallback_mode,
            'current_model': self.current_model_key,
            'available_models': list(self.model_configs.keys()),
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'device': 'cuda' if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else 'cpu',
            'cache_size': len(self.analysis_cache)
        }
    
    async def clear_cache(self):
        """Clear analysis cache."""
        self.analysis_cache.clear()
        logger.info("🧹 Analysis cache cleared")

# Global instance
_fingpt_analyzer = None

def get_fingpt_analyzer() -> FinGPTSentimentAnalyzer:
    """Get or create the global FinGPT analyzer instance."""
    global _fingpt_analyzer
    if _fingpt_analyzer is None:
        _fingpt_analyzer = FinGPTSentimentAnalyzer()
    return _fingpt_analyzer

# Test function
async def test_fingpt_analyzer():
    """Test the FinGPT sentiment analyzer."""
    
    logger.info("🧪 Testing FinGPT Sentiment Analyzer...")
    
    analyzer = get_fingpt_analyzer()
    await analyzer.initialize_model()
    
    # Test texts
    test_cases = [
        ("Apple reports record quarterly earnings, beating analyst expectations by 15%.", "AAPL"),
        ("Tesla stock plunges 20% after disappointing delivery numbers and production cuts.", "TSLA"),
        ("Microsoft announces new AI partnership, expanding cloud services globally.", "MSFT"),
        ("Federal Reserve raises interest rates by 0.75%, markets show mixed reaction.", "SPY")
    ]
    
    # Single analysis test
    for text, symbol in test_cases:
        result = await analyzer.analyze_sentiment(text, symbol)
        logger.info(f"📈 {symbol}: {result['sentiment']} ({result['confidence']:.2f}) - {result['trading_signal']}")
    
    # Batch analysis test  
    batch_results = await analyzer.batch_analyze_sentiments(test_cases)
    logger.info(f"📊 Batch analysis complete: {len(batch_results)} results")
    
    # Model info
    info = analyzer.get_model_info()
    logger.info(f"🤖 Model info: {info}")
    
    logger.info("✅ FinGPT analyzer test completed")

if __name__ == "__main__":
    # Test the analyzer
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_fingpt_analyzer())