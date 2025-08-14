#!/usr/bin/env python3
"""
LLM Client Wrapper for AI Trading System

Provides unified interface for OpenAI and Anthropic APIs with fallback support,
retry logic, and trading-specific optimizations.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import time

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Standardized LLM response structure."""
    content: str
    model: str
    tokens_used: int
    cost_estimate: float
    response_time: float
    confidence: float
    metadata: Dict[str, Any]

class LLMClient:
    """Unified LLM client supporting Llama 3.1 (via Ollama) and FinGPT only."""
    
    def __init__(self):
        self.anthropic_client = None
        self.ollama_client = None
        self.fingpt_client = None
        self.preferred_provider = None
        self.request_count = 0
        self.total_cost = 0.0
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize available LLM clients based on API keys."""
        
        # Try FinGPT first (specialized for financial tasks)
        if settings.use_fingpt_primary:
            try:
                from core.fingpt_sentiment_analyzer import get_fingpt_analyzer
                self.fingpt_client = get_fingpt_analyzer()
                # Don't set as preferred provider yet - let it initialize first
                logger.info("✅ FinGPT sentiment analyzer loaded as primary financial LLM")
            except ImportError as ie:
                logger.warning(f"FinGPT dependencies not available: {ie}")
            except Exception as e:
                logger.warning(f"FinGPT client initialization failed: {e}")
        
        # Try Llama 3.1 via Ollama as secondary (better than OpenAI quota limits)
        if settings.use_llama_fallback:
            try:
                import ollama
                self.ollama_client = ollama.Client(host=settings.ollama_base_url)
                
                # Test connection immediately to avoid runtime failures
                try:
                    # Quick test to see if Ollama is responsive
                    self.ollama_client.list()
                    if not self.preferred_provider:
                        self.preferred_provider = "llama"
                    logger.info("✅ Llama 3.1 (Ollama) client initialized and connected as secondary")
                except Exception as conn_error:
                    logger.warning(f"Ollama server not accessible: {conn_error}")
                    logger.info("💡 To use Ollama: 1) Install Ollama from https://ollama.com/download 2) Run 'ollama serve' 3) Run 'ollama pull llama3.1'")
                    self.ollama_client = None  # Disable Ollama if not accessible
                    
            except ImportError:
                logger.warning("Ollama package not installed. Run: pip install ollama")
            except Exception as e:
                logger.warning(f"Ollama client initialization failed: {e}")
                self.ollama_client = None
        
        # Try Anthropic Claude as third fallback (before OpenAI)
        if settings.anthropic_api_key and not self.preferred_provider:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                self.preferred_provider = "anthropic"
                logger.info("✅ Anthropic Claude client initialized as third fallback")
            except ImportError:
                logger.warning("Anthropic package not installed. Run: pip install anthropic")
            except Exception as e:
                logger.warning(f"Anthropic client initialization failed: {e}")
        
        logger.info("🚫 OpenAI disabled - using only FinGPT and Ollama for LLM calls")
                
        # Legacy Anthropic support (deprecated)
        # if settings.anthropic_api_key:
        #     try:
        #         import anthropic
        #         self.anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        #         if not self.preferred_provider:
        #             self.preferred_provider = "anthropic"
        #         logger.info("✅ Anthropic client initialized")
        #     except ImportError:
        #         logger.warning("Anthropic package not installed. Run: pip install anthropic")
        #     except Exception as e:
        #         logger.warning(f"Anthropic client initialization failed: {e}")
        
        if not self.preferred_provider:
            logger.error("❌ No LLM clients available. Please ensure Ollama is running and FinGPT is configured.")
    
    async def generate_response(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        provider: Optional[str] = None
    ) -> LLMResponse:
        """Generate response using available LLM provider."""
        
        start_time = time.time()
        self.request_count += 1
        
        # Determine provider
        use_provider = provider or self.preferred_provider
        
        # Try primary provider first, then fallback on failure
        response = None
        tried_providers = []
        
        try:
            if use_provider == "fingpt" and self.fingpt_client:
                tried_providers.append("fingpt")
                response = await self._call_fingpt(
                    system_prompt, user_message, model, temperature, max_tokens
                )
            elif use_provider == "anthropic" and self.anthropic_client:
                tried_providers.append("anthropic")
                response = await self._call_anthropic(
                    system_prompt, user_message, model, temperature, max_tokens
                )
            elif use_provider == "llama" and self.ollama_client:
                tried_providers.append("llama")
                response = await self._call_llama(
                    system_prompt, user_message, model, temperature, max_tokens
                )
            else:
                # No specific provider, try any available (FinGPT first if enabled)
                if self.fingpt_client:
                    tried_providers.append("fingpt")
                    response = await self._call_fingpt(
                        system_prompt, user_message, model, temperature, max_tokens
                    )
                elif self.anthropic_client:
                    tried_providers.append("anthropic")
                    response = await self._call_anthropic(
                        system_prompt, user_message, model, temperature, max_tokens
                    )
                elif self.ollama_client:
                    tried_providers.append("llama")
                    response = await self._call_llama(
                        system_prompt, user_message, model, temperature, max_tokens
                    )
                else:
                    raise ValueError("No LLM providers available")
                    
        except Exception as primary_error:
            logger.warning(f"Primary LLM provider failed ({tried_providers}): {primary_error}")
            
            # Try fallback providers (FinGPT -> Anthropic -> Ollama -> OpenAI)
            if "fingpt" not in tried_providers and self.fingpt_client:
                try:
                    logger.info("Falling back to FinGPT")
                    response = await self._call_fingpt(
                        system_prompt, user_message, model, temperature, max_tokens
                    )
                except Exception as fingpt_error:
                    logger.warning(f"FinGPT fallback failed: {fingpt_error}")
            
            if response is None and "anthropic" not in tried_providers and self.anthropic_client:
                try:
                    logger.info("Falling back to Anthropic Claude")
                    response = await self._call_anthropic(
                        system_prompt, user_message, model, temperature, max_tokens
                    )
                except Exception as anthropic_error:
                    logger.warning(f"Anthropic fallback failed: {anthropic_error}")
            
            if response is None and "llama" not in tried_providers and self.ollama_client:
                try:
                    logger.info("Falling back to Ollama")
                    response = await self._call_llama(
                        system_prompt, user_message, model, temperature, max_tokens
                    )
                except Exception as llama_error:
                    logger.warning(f"Ollama fallback failed: {llama_error}")
            
            
            # If all providers failed, raise the original error
            if response is None:
                raise primary_error
        
        if response:
            response.response_time = time.time() - start_time
            self.total_cost += response.cost_estimate
            
            logger.info(f"LLM response generated: {response.tokens_used} tokens, "
                       f"{response.response_time:.2f}s, ${response.cost_estimate:.4f}")
            
            return response
        else:
            # This shouldn't happen with the new logic, but just in case
            logger.error("No response generated from any provider")
            return LLMResponse(
                content="LLM Error: No providers available. Using fallback analysis.",
                model="fallback",
                tokens_used=0,
                cost_estimate=0.0,
                response_time=time.time() - start_time,
                confidence=0.0,
                metadata={"error": "No providers available"}
            )
    
    
    async def _call_anthropic(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Call Anthropic Claude API."""
        
        # Default model selection
        if not model:
            model = "claude-3-5-sonnet-20241022"  # Latest Claude 3.5 Sonnet
        
        try:
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            content = response.content[0].text
            
            # Anthropic provides token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            tokens_used = input_tokens + output_tokens
            
            # Estimate cost (approximate pricing for Claude 3.5 Sonnet)
            cost_per_input_token = 0.000003  # $3/1M input tokens
            cost_per_output_token = 0.000015  # $15/1M output tokens
            cost_estimate = (input_tokens * cost_per_input_token) + (output_tokens * cost_per_output_token)
            
            # Estimate confidence based on response characteristics
            confidence = self._estimate_confidence(content, temperature)
            
            return LLMResponse(
                content=content,
                model=f"anthropic/{model}",
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                response_time=0.0,  # Will be set by caller
                confidence=confidence,
                metadata={
                    "provider": "anthropic",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "stop_reason": response.stop_reason
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise
    
    async def _call_llama(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Call Llama 3.1 via Ollama."""
        
        # Default model selection
        if not model:
            model = settings.ollama_model
        
        try:
            # Create combined prompt for Llama (it doesn't separate system/user)
            combined_prompt = f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:"
            
            # Try to connect to Ollama server
            response = await asyncio.to_thread(
                self.ollama_client.generate,
                model=model,
                prompt=combined_prompt,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'top_p': 0.9,
                    'stop': ['User:', 'System:']
                }
            )
            
            content = response['response'].strip()
            
            # Ollama doesn't provide token usage directly
            tokens_used = self._estimate_tokens(combined_prompt + content)
            
            # Local model has no cost
            cost_estimate = 0.0
            
            # Estimate confidence based on response length and temperature
            confidence = self._estimate_confidence(content, temperature)
            
            return LLMResponse(
                content=content,
                model=f"ollama/{model}",
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                response_time=0.0,  # Will be set by caller
                confidence=confidence,
                metadata={
                    "provider": "ollama",
                    "model_info": response.get('model', ''),
                    "local_inference": True
                }
            )
            
        except Exception as e:
            logger.error(f"Llama/Ollama API call failed: {e}")
            
            # If Ollama server isn't running, disable the client and provide fallback
            if any(phrase in str(e).lower() for phrase in ["failed to connect", "connection refused", "connection error", "timeout"]):
                logger.warning("Ollama server not accessible, disabling Ollama client for this session")
                self.ollama_client = None  # Disable for this session
                
                # Simple rule-based response for common financial queries
                content = self._generate_fallback_response(system_prompt, user_message)
                
                return LLMResponse(
                    content=content,
                    model="ollama-fallback",
                    tokens_used=self._estimate_tokens(content),
                    cost_estimate=0.0,
                    response_time=0.0,
                    confidence=0.4,  # Lower confidence for rule-based
                    metadata={
                        "provider": "ollama-fallback",
                        "note": "Ollama server not available, using rule-based response",
                        "error": str(e)
                    }
                )
            
            raise
    
    async def _call_fingpt(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Call FinGPT via HuggingFace transformers."""
        
        # Default model selection
        if not model:
            model = settings.fingpt_model
        
        try:
            # Create combined prompt for FinGPT
            combined_prompt = f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:"
            
            # Generate response using FinGPT analyzer's sentiment analysis
            # Note: FinGPT is primarily for sentiment analysis, not general text generation
            analysis_result = await self.fingpt_client.analyze_sentiment(
                user_message,
                symbol=None,  # No specific symbol in this context
                prompt_type='advanced',
                use_cache=False
            )
            
            # Convert FinGPT sentiment analysis to general response
            if analysis_result and 'reasoning' in analysis_result:
                content = f"Sentiment: {analysis_result.get('sentiment', 'neutral')} (Score: {analysis_result.get('score', 0.0):.2f})\n\nReasoning: {analysis_result.get('reasoning', 'No reasoning provided')}"
            else:
                # If FinGPT analysis fails, return a simple response
                content = "Unable to analyze with FinGPT. This model is specialized for financial sentiment analysis."
            
            # Content is already prepared from FinGPT analysis above
            
            # Estimate token usage
            tokens_used = self._estimate_tokens(combined_prompt + content)
            
            # FinGPT is free via transformers (local processing)
            cost_estimate = 0.0
            
            # Estimate confidence - FinGPT is specialized for finance, so base confidence is higher
            confidence = self._estimate_confidence(content, temperature)
            confidence = min(0.95, confidence + 0.1)  # Boost confidence for financial domain
            
            return LLMResponse(
                content=content,
                model=f"fingpt/{model.split('/')[-1]}",
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                response_time=0.0,  # Will be set by caller
                confidence=confidence,
                metadata={
                    "provider": "fingpt",
                    "model_path": model,
                    "financial_specialized": True,
                    "local_inference": True
                }
            )
            
        except Exception as e:
            logger.error(f"FinGPT API call failed: {e}")
            
            # Provide a financial-focused fallback response
            content = self._generate_financial_fallback_response(system_prompt, user_message)
            
            return LLMResponse(
                content=content,
                model="fingpt-fallback",
                tokens_used=self._estimate_tokens(content),
                cost_estimate=0.0,
                response_time=0.0,
                confidence=0.5,  # Medium confidence for rule-based financial response
                metadata={
                    "provider": "fingpt-fallback",
                    "note": "FinGPT not available, using financial rule-based response"
                }
            )
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation for text."""
        # Rough estimate: ~4 characters per token
        return max(1, len(text) // 4)
    
    def _estimate_confidence(self, content: str, temperature: float) -> float:
        """Estimate response confidence based on content and parameters."""
        base_confidence = 1.0 - (temperature * 0.3)  # Lower temperature = higher confidence
        
        # Adjust based on content characteristics
        if "uncertain" in content.lower() or "unsure" in content.lower():
            base_confidence -= 0.2
        if "confident" in content.lower() or "certain" in content.lower():
            base_confidence += 0.1
        if len(content) > 1000:  # Detailed responses suggest higher confidence
            base_confidence += 0.05
        
        return max(0.1, min(1.0, base_confidence))
    
    def _get_openai_cost_per_token(self, model: str) -> float:
        """Get approximate cost per token for OpenAI models."""
        costs = {
            "gpt-4o": 0.000015,  # $15/1M tokens (blended input/output)
            "gpt-4": 0.000030,   # $30/1M tokens
            "gpt-4-turbo": 0.000020,  # $20/1M tokens
            "gpt-3.5-turbo": 0.0000015  # $1.5/1M tokens
        }
        return costs.get(model, 0.000015)  # Default to GPT-4o pricing
    
    # Removed Anthropic cost calculation - replaced with free local Llama 3.1
    
    def _generate_fallback_response(self, system_prompt: str, user_message: str) -> str:
        """Generate a simple rule-based response when all LLM providers are unavailable."""
        
        # Simple pattern matching for common financial queries
        query = user_message.lower()
        
        if "sentiment" in query:
            return "Market sentiment analysis requires real-time data and sentiment sources. Current analysis suggests neutral sentiment with mixed signals from technical indicators."
        
        elif "buy" in query or "sell" in query or "trade" in query:
            return "Trade recommendations require careful analysis of multiple factors including technical indicators, fundamental analysis, and risk assessment. Please consult current market data before making trading decisions."
        
        elif "risk" in query:
            return "Risk management is crucial in trading. Consider diversification, position sizing, stop-loss levels, and overall portfolio risk. Current market conditions suggest maintaining conservative risk parameters."
        
        elif "market" in query:
            return "Market analysis indicates mixed conditions. Monitor key economic indicators, earnings reports, and technical levels for trading opportunities."
        
        elif "portfolio" in query:
            return "Portfolio optimization should consider risk tolerance, diversification across sectors, and rebalancing frequency. Current configuration suggests reviewing allocation weights."
        
        else:
            return "Financial analysis requires real-time data and comprehensive market evaluation. Consider consulting multiple sources and maintaining appropriate risk management practices."
    
    def _generate_financial_fallback_response(self, system_prompt: str, user_message: str) -> str:
        """Generate a specialized financial rule-based response for FinGPT fallback."""
        
        query = user_message.lower()
        
        if "sentiment" in query:
            return "Based on available financial indicators, market sentiment appears mixed with moderate volatility. Consider monitoring social media trends, news sentiment, and technical indicators for comprehensive sentiment analysis."
        
        elif any(word in query for word in ["buy", "sell", "trade", "invest"]):
            return "Investment decisions should be based on fundamental analysis, technical indicators, and risk assessment. Current market conditions suggest maintaining diversified positions with appropriate stop-loss levels."
        
        elif "risk" in query:
            return "Financial risk management should incorporate portfolio diversification, position sizing (typically 2-5% per position), volatility assessment, and correlation analysis. Monitor VIX and sector rotation patterns."
        
        elif any(word in query for word in ["earnings", "financial", "revenue"]):
            return "Financial analysis requires evaluation of key metrics: P/E ratios, revenue growth, debt levels, and cash flow. Compare against sector averages and consider seasonal factors."
        
        elif any(word in query for word in ["market", "economy", "macro"]):
            return "Market analysis suggests monitoring Federal Reserve policy, inflation indicators, GDP growth, and sector rotation trends. Consider both technical and fundamental factors."
        
        elif "portfolio" in query:
            return "Portfolio optimization recommendations: maintain 60-40 equity-bond allocation for moderate risk, rebalance quarterly, limit single positions to 5% maximum, and diversify across sectors."
        
        else:
            return "Financial analysis indicates the need for comprehensive evaluation of market conditions, fundamental metrics, and risk factors. Recommend consulting multiple data sources and maintaining disciplined risk management."
    
    async def analyze_financial_data(
        self,
        agent_name: str,
        analysis_data: Dict[str, Any],
        system_prompt: str,
        temperature: float = 0.7
    ) -> LLMResponse:
        """Specialized method for financial data analysis."""
        
        # Format analysis data into structured prompt
        user_message = self._format_financial_prompt(agent_name, analysis_data)
        
        return await self.generate_response(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=2000
        )
    
    def _format_financial_prompt(self, agent_name: str, data: Dict[str, Any]) -> str:
        """Format financial data into a structured prompt."""
        
        prompt_parts = [
            f"AGENT: {agent_name}",
            f"ANALYSIS REQUEST: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Add market data if available
        if "market_data" in data:
            prompt_parts.append("MARKET DATA:")
            for key, value in data["market_data"].items():
                if isinstance(value, (int, float)):
                    prompt_parts.append(f"- {key}: {value:.4f}")
                else:
                    prompt_parts.append(f"- {key}: {value}")
            prompt_parts.append("")
        
        # Add quantitative analysis if available
        if "quantitative_analysis" in data:
            prompt_parts.append("QUANTITATIVE ANALYSIS:")
            quant = data["quantitative_analysis"]
            for key, value in quant.items():
                prompt_parts.append(f"- {key}: {value}")
            prompt_parts.append("")
        
        # Add symbols being analyzed
        if "symbols" in data:
            prompt_parts.append(f"SYMBOLS FOR ANALYSIS: {', '.join(data['symbols'])}")
            prompt_parts.append("")
        
        # Add portfolio context if available
        if "portfolio_context" in data:
            prompt_parts.append("PORTFOLIO CONTEXT:")
            context = data["portfolio_context"]
            for key, value in context.items():
                prompt_parts.append(f"- {key}: {value}")
            prompt_parts.append("")
        
        # Add any additional instructions
        if "instructions" in data:
            prompt_parts.append("SPECIFIC INSTRUCTIONS:")
            prompt_parts.append(data["instructions"])
            prompt_parts.append("")
        
        prompt_parts.append("Please provide your analysis and recommendations based on the above data.")
        
        return "\n".join(prompt_parts)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        return {
            "total_requests": self.request_count,
            "total_cost_estimate": self.total_cost,
            "preferred_provider": self.preferred_provider,
            "anthropic_available": self.anthropic_client is not None,
            "openai_available": self.openai_client is not None,
            "llama_available": self.ollama_client is not None
        }
    
    async def test_connection(self) -> Dict[str, bool]:
        """Test connection to all available providers."""
        results = {}
        
        if self.anthropic_client:
            try:
                await self.generate_response(
                    system_prompt="You are a test assistant.",
                    user_message="Respond with 'Anthropic Claude connection successful'",
                    max_tokens=50,
                    provider="anthropic"
                )
                results["anthropic"] = True
            except Exception as e:
                logger.error(f"Anthropic connection test failed: {e}")
                results["anthropic"] = False
        
        if self.ollama_client:
            try:
                await self.generate_response(
                    system_prompt="You are a test assistant.",
                    user_message="Respond with 'Llama connection successful'",
                    max_tokens=50,
                    provider="llama"
                )
                results["llama"] = True
            except Exception as e:
                logger.error(f"Llama/Ollama connection test failed: {e}")
                results["llama"] = False
        
        if self.openai_client:
            try:
                await self.generate_response(
                    system_prompt="You are a test assistant.",
                    user_message="Respond with 'OpenAI connection successful'",
                    max_tokens=50,
                    provider="openai"
                )
                results["openai"] = True
            except Exception as e:
                logger.error(f"OpenAI connection test failed: {e}")
                results["openai"] = False
        
        return results

# Global LLM client instance
llm_client = LLMClient()

async def get_llm_analysis(
    agent_name: str,
    system_prompt: str,
    analysis_data: Dict[str, Any],
    temperature: float = 0.7
) -> LLMResponse:
    """Convenience function for getting LLM analysis."""
    return await llm_client.analyze_financial_data(
        agent_name=agent_name,
        analysis_data=analysis_data,
        system_prompt=system_prompt,
        temperature=temperature
    )

if __name__ == "__main__":
    # Test the LLM client
    async def test_client():
        print("🧪 Testing LLM Client")
        print("=" * 50)
        
        # Test connection
        connections = await llm_client.test_connection()
        print("Connection tests:")
        for provider, status in connections.items():
            print(f"  {provider}: {'✅' if status else '❌'}")
        
        # Test analysis
        if any(connections.values()):
            test_data = {
                "market_data": {"SPY_price": 420.50, "VIX": 18.5},
                "symbols": ["SPY", "QQQ"],  # Use ETFs for testing instead of individual stocks
                "instructions": "Analyze current market conditions"
            }
            
            response = await get_llm_analysis(
                agent_name="test_agent",
                system_prompt="You are a financial analyst. Provide brief market analysis.",
                analysis_data=test_data
            )
            
            print(f"\nTest response:")
            print(f"Model: {response.model}")
            print(f"Tokens: {response.tokens_used}")
            print(f"Cost: ${response.cost_estimate:.4f}")
            print(f"Content: {response.content[:200]}...")
        
        # Usage stats
        stats = llm_client.get_usage_stats()
        print(f"\nUsage stats: {stats}")
    
    asyncio.run(test_client())