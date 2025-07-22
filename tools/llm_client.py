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
    """Unified LLM client supporting OpenAI and Llama 3.1 (via Ollama)."""
    
    def __init__(self):
        self.openai_client = None
        self.ollama_client = None
        # self.anthropic_client = None  # Deprecated
        self.preferred_provider = None
        self.request_count = 0
        self.total_cost = 0.0
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize available LLM clients based on API keys."""
        
        # Try OpenAI first
        if settings.openai_api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
                self.preferred_provider = "openai"
                logger.info("✅ OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI package not installed. Run: pip install openai")
            except Exception as e:
                logger.warning(f"OpenAI client initialization failed: {e}")
        
        # Try Llama 3.1 via Ollama as fallback
        if settings.use_llama_fallback:
            try:
                import ollama
                self.ollama_client = ollama.Client(host=settings.ollama_base_url)
                if not self.preferred_provider:
                    self.preferred_provider = "llama"
                logger.info("✅ Llama 3.1 (Ollama) client initialized")
            except ImportError:
                logger.warning("Ollama package not installed. Run: pip install ollama")
            except Exception as e:
                logger.warning(f"Ollama client initialization failed: {e}")
                
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
            logger.error("❌ No LLM clients available. Please configure API keys.")
    
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
        
        try:
            if use_provider == "openai" and self.openai_client:
                response = await self._call_openai(
                    system_prompt, user_message, model, temperature, max_tokens
                )
            elif use_provider == "llama" and self.ollama_client:
                response = await self._call_llama(
                    system_prompt, user_message, model, temperature, max_tokens
                )
            # elif use_provider == "anthropic" and self.anthropic_client:  # Deprecated
            #     response = await self._call_anthropic(
            #         system_prompt, user_message, model, temperature, max_tokens
            #     )
            else:
                # Fallback to any available provider
                if self.openai_client:
                    response = await self._call_openai(
                        system_prompt, user_message, model, temperature, max_tokens
                    )
                elif self.ollama_client:
                    response = await self._call_llama(
                        system_prompt, user_message, model, temperature, max_tokens
                    )
                # elif self.anthropic_client:  # Deprecated
                #     response = await self._call_anthropic(
                #         system_prompt, user_message, model, temperature, max_tokens
                #     )
                else:
                    raise ValueError("No LLM providers available")
            
            response.response_time = time.time() - start_time
            self.total_cost += response.cost_estimate
            
            logger.info(f"LLM response generated: {response.tokens_used} tokens, "
                       f"{response.response_time:.2f}s, ${response.cost_estimate:.4f}")
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Return fallback response
            return LLMResponse(
                content=f"LLM Error: {str(e)}. Using fallback analysis.",
                model="fallback",
                tokens_used=0,
                cost_estimate=0.0,
                response_time=time.time() - start_time,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    async def _call_openai(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Call OpenAI API."""
        
        # Default model selection
        if not model:
            model = "gpt-4o" if "gpt-4" in str(settings.openai_api_key) else "gpt-3.5-turbo"
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=60
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Estimate cost (approximate pricing)
            cost_per_token = self._get_openai_cost_per_token(model)
            cost_estimate = tokens_used * cost_per_token
            
            # Estimate confidence based on response characteristics
            confidence = self._estimate_confidence(content, temperature)
            
            return LLMResponse(
                content=content,
                model=f"openai/{model}",
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                response_time=0.0,  # Will be set by caller
                confidence=confidence,
                metadata={
                    "provider": "openai",
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
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
            raise
    
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
            "openai_available": self.openai_client is not None,
            "llama_available": self.ollama_client is not None
        }
    
    async def test_connection(self) -> Dict[str, bool]:
        """Test connection to all available providers."""
        results = {}
        
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
                "symbols": ["AAPL", "GOOGL"],
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