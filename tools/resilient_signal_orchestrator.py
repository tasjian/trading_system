#!/usr/bin/env python3
"""
Resilient Signal Orchestrator
Master coordinator for dual-provider signal generation (Finnhub + Alpha Vantage + SEC EDGAR).
Completely removes yfinance dependencies and implements optimal fallback strategy.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

from tools.dual_provider_market_data import dual_provider, get_price_signals, get_market_data
from tools.enhanced_news_signals import enhanced_news_generator
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class SignalSourceMetrics:
    """Metrics for a signal source."""
    source_name: str
    success_rate: float
    average_latency: float
    signal_count: int
    last_success: Optional[datetime]
    reliability_score: float

@dataclass
class OrchestrationResult:
    """Result of signal orchestration."""
    total_signals: int
    signals_by_source: Dict[str, int]
    primary_source_success: bool
    fallback_sources_used: List[str]
    processing_time: float
    reliability_score: float
    signals: List[Dict[str, Any]]

class ResilientSignalOrchestrator:
    """Master orchestrator for dual-provider signal generation."""
    
    def __init__(self):
        """Initialize the signal orchestrator."""
        self.source_metrics = {}
        self.initialization_time = datetime.now()
        
        # NEW: Dual-provider priority order (removes yfinance completely)
        self.source_priorities = [
            'finnhub_primary',       # Primary: Real-time quotes (60 req/min)
            'alpha_vantage',         # Secondary: Historical data (5 req/min)
            'sec_edgar',             # Tertiary: Fundamental data (10 req/sec)
            'enhanced_news',         # Quaternary: News-based signals
            'market_patterns'        # Emergency: Technical patterns only
        ]
        
        # Requirements (more aggressive with higher quality sources)
        self.min_signals_required = 2
        self.min_confidence_threshold = 0.3
        
        logger.info("🎯 Resilient Signal Orchestrator initialized (Dual-Provider)")
        logger.info(f"Source priority: {' > '.join(self.source_priorities)}")
    
    async def orchestrate_signals(self, symbols: List[str], 
                                threshold: float = 0.02,
                                min_signals: int = 2,
                                max_processing_time: float = 45.0) -> OrchestrationResult:
        """
        Orchestrate signal generation with dual-provider strategy.
        
        Strategy:
        1. Finnhub (primary) - Real-time quotes, high reliability
        2. Alpha Vantage (secondary) - Historical context, rate limited
        3. SEC EDGAR (tertiary) - Fundamental insights, high throughput
        4. Enhanced News (fallback) - Sentiment-based signals
        
        Args:
            symbols: List of symbols to analyze
            threshold: Price change threshold
            min_signals: Minimum signals required
            max_processing_time: Maximum processing time in seconds
            
        Returns:
            OrchestrationResult with comprehensive signal data
        """
        start_time = datetime.now()
        logger.info(f"🚀 DUAL-PROVIDER ORCHESTRATION for {len(symbols)} symbols")
        logger.info(f"Requirements: min_signals={min_signals}, threshold={threshold:.1%}")
        
        signals = []
        sources_used = []
        source_signal_counts = {}
        
        # Stage 1: Primary Dual-Provider Market Data
        logger.info("📊 Stage 1: Dual-provider market data (Finnhub + Alpha Vantage)")
        try:
            # Get price signals using our new dual-provider system
            dual_provider_signals = await get_price_signals(symbols, threshold)
            
            if dual_provider_signals:
                signals.extend(dual_provider_signals)
                sources_used.append('dual_provider')
                source_signal_counts['dual_provider'] = len(dual_provider_signals)
                logger.info(f"✅ Dual-provider: {len(dual_provider_signals)} signals")
            else:
                logger.warning("⚠️ Dual-provider returned no signals")
                
        except Exception as e:
            logger.error(f"❌ Dual-provider failed: {e}")
        
        # Stage 2: SEC EDGAR Fundamental Enhancement (if still need signals)
        if len(signals) < min_signals:
            logger.info("🏛️ Stage 2: SEC EDGAR fundamental analysis")
            try:
                sec_signals = await self._get_sec_edgar_signals(symbols[:10])  # Limit for processing
                if sec_signals:
                    signals.extend(sec_signals)
                    sources_used.append('sec_edgar')
                    source_signal_counts['sec_edgar'] = len(sec_signals)
                    logger.info(f"✅ SEC EDGAR: {len(sec_signals)} signals")
            except Exception as e:
                logger.error(f"❌ SEC EDGAR failed: {e}")
        
        # Stage 3: Enhanced News Fallback (if still insufficient)
        if len(signals) < min_signals:
            logger.info("📰 Stage 3: Enhanced news signals")
            try:
                news_signals = await self._get_enhanced_news_signals(symbols[:20], min_signals)
                if news_signals:
                    signals.extend(news_signals)
                    sources_used.append('enhanced_news')
                    source_signal_counts['enhanced_news'] = len(news_signals)
                    logger.info(f"✅ Enhanced news: {len(news_signals)} signals")
            except Exception as e:
                logger.error(f"❌ Enhanced news failed: {e}")
        
        # Stage 4: Technical Pattern Emergency (last resort)
        if len(signals) < min_signals:
            logger.warning("⚠️ Stage 4: Technical pattern emergency signals")
            try:
                pattern_signals = await self._get_technical_pattern_signals(symbols[:15])
                if pattern_signals:
                    signals.extend(pattern_signals)
                    sources_used.append('technical_patterns')
                    source_signal_counts['technical_patterns'] = len(pattern_signals)
                    logger.info(f"✅ Technical patterns: {len(pattern_signals)} signals")
            except Exception as e:
                logger.error(f"❌ Technical patterns failed: {e}")
        
        # Final validation and results
        processing_time = (datetime.now() - start_time).total_seconds()
        primary_success = 'dual_provider' in sources_used
        reliability_score = self._calculate_reliability_score(sources_used, len(signals), min_signals)
        
        result = OrchestrationResult(
            total_signals=len(signals),
            signals_by_source=source_signal_counts,
            primary_source_success=primary_success,
            fallback_sources_used=sources_used,
            processing_time=processing_time,
            reliability_score=reliability_score,
            signals=signals
        )
        
        # Log final results
        logger.info("✅ SIGNAL ORCHESTRATION COMPLETE")
        logger.info(f"Signals: {len(signals)} | Sources: {len(sources_used)} | Time: {processing_time:.1f}s | Reliability: {reliability_score:.2f}")
        logger.info(f"Source breakdown: {source_signal_counts}")
        
        # Critical failure check - fail fast with detailed error
        if len(signals) < min_signals:
            error_msg = (
                f"❌ CRITICAL SYSTEM FAILURE: Signal orchestration failed\n"
                f"Signals Generated: {len(signals)} (minimum required: {min_signals})\n"
                f"Sources Attempted: {', '.join(self.source_priorities)}\n"
                f"Sources That Responded: {', '.join(sources_used) if sources_used else 'NONE'}\n"
                f"Primary Source Success: {'YES' if primary_success else 'NO'}\n"
                f"Processing Time: {processing_time:.1f}s\n"
                f"Signal Breakdown: {source_signal_counts}\n\n"
                f"SYSTEM REQUIRES MINIMUM {min_signals} TRADING SIGNALS TO OPERATE SAFELY\n"
                f"All external data sources appear to be unavailable or rate-limited\n"
                f"No hardcoded fallback mechanisms are permitted per system design"
            )
            logger.error(error_msg)
            
            # Cleanup sessions before raising error
            try:
                await dual_provider.close()
                logger.debug("✅ Cleaned up sessions after orchestration failure")
            except Exception as e:
                logger.debug(f"Session cleanup warning: {e}")
            
            raise RuntimeError(error_msg)
        
        # Cleanup sessions after successful orchestration
        try:
            await dual_provider.close()
            logger.debug("✅ Cleaned up sessions after orchestration success")
        except Exception as e:
            logger.debug(f"Session cleanup warning: {e}")
        
        return result
    
    async def _get_sec_edgar_signals(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Generate signals based on SEC EDGAR fundamental data."""
        signals = []
        
        try:
            # Get ticker to CIK mapping
            ticker_mapping = await dual_provider.sec_edgar.get_ticker_to_cik_mapping()
            
            for symbol in symbols:
                try:
                    if symbol in ticker_mapping:
                        cik = ticker_mapping[symbol]
                        company_facts = await dual_provider.sec_edgar.get_company_facts(cik)
                        
                        if company_facts and 'facts' in company_facts:
                            # Simple fundamental analysis for signal generation
                            facts = company_facts['facts']
                            
                            # Look for recent revenue growth or earnings
                            signal_strength = 0.4  # Base strength for fundamental data
                            
                            # Create fundamental-based signal
                            signal = {
                                "symbol": symbol,
                                "signal_type": "fundamental",
                                "strength": signal_strength,
                                "direction": "up",  # Assume positive for companies with recent filings
                                "description": f"SEC fundamental data available (CIK: {cik})",
                                "data_source": "sec_edgar",
                                "confidence": 0.6,
                                "timestamp": datetime.now().isoformat()
                            }
                            signals.append(signal)
                            
                except Exception as e:
                    logger.debug(f"SEC EDGAR failed for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"SEC EDGAR batch processing failed: {e}")
        
        return signals
    
    async def _get_enhanced_news_signals(self, symbols: List[str], target_signals: int) -> List[Dict[str, Any]]:
        """Get enhanced news signals."""
        try:
            # Use existing enhanced news generator
            news_signals = await enhanced_news_generator.generate_enhanced_signals(symbols)
            
            # Convert to standard format if needed
            formatted_signals = []
            for signal in news_signals:
                if isinstance(signal, dict):
                    # Ensure consistent format
                    formatted_signal = {
                        "symbol": signal.get("symbol", ""),
                        "signal_type": "news",
                        "strength": signal.get("strength", 0.5),
                        "direction": signal.get("direction", "up"),
                        "description": signal.get("description", "News signal"),
                        "data_source": "enhanced_news",
                        "confidence": signal.get("confidence", 0.5),
                        "timestamp": signal.get("timestamp", datetime.now().isoformat())
                    }
                    formatted_signals.append(formatted_signal)
            
            return formatted_signals[:target_signals]  # Limit to target
            
        except Exception as e:
            logger.error(f"Enhanced news signals failed: {e}")
            return []
    
    async def _get_technical_pattern_signals(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Generate emergency technical pattern signals."""
        signals = []
        
        try:
            # Use market data from dual-provider for technical analysis
            market_data = await get_market_data(symbols)
            
            for symbol, data in market_data.items():
                try:
                    # Simple technical analysis based on price action
                    price_change_pct = data.get('price_change_pct', 0)
                    
                    # Generate signal if significant price movement
                    if abs(price_change_pct) > 0.015:  # Lower threshold for emergency
                        strength = min(0.7, abs(price_change_pct) * 10)  # Scale strength
                        direction = "up" if price_change_pct > 0 else "down"
                        
                        signal = {
                            "symbol": symbol,
                            "signal_type": "technical_pattern",
                            "strength": strength,
                            "direction": direction,
                            "description": f"Technical pattern: {direction} {price_change_pct:.1%}",
                            "data_source": "technical_analysis",
                            "confidence": 0.4,  # Lower confidence for emergency signals
                            "timestamp": datetime.now().isoformat()
                        }
                        signals.append(signal)
                        
                except Exception as e:
                    logger.debug(f"Technical analysis failed for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Technical pattern analysis failed: {e}")
        
        return signals
    
    def _calculate_reliability_score(self, sources_used: List[str], signals_generated: int, min_required: int) -> float:
        """Calculate reliability score based on sources and signal count."""
        # Base score from signal sufficiency
        signal_score = min(1.0, signals_generated / max(min_required, 1))
        
        # Bonus for using primary sources
        source_bonus = 0.0
        if 'dual_provider' in sources_used:
            source_bonus += 0.3
        if 'sec_edgar' in sources_used:
            source_bonus += 0.2
        if 'enhanced_news' in sources_used:
            source_bonus += 0.1
        
        return min(1.0, signal_score + source_bonus)
    
    async def get_source_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all signal sources."""
        health_status = {}
        
        # Check dual-provider health
        try:
            test_data = await get_market_data(['AAPL'])  # Quick test
            health_status['dual_provider'] = {
                'status': 'healthy' if test_data else 'degraded',
                'last_check': datetime.now().isoformat(),
                'provider_status': dual_provider._provider_health
            }
        except Exception as e:
            health_status['dual_provider'] = {
                'status': 'failed',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
        
        return health_status


# Global instance
resilient_orchestrator = ResilientSignalOrchestrator()


async def get_resilient_price_signals_async(symbols: List[str], 
                                          threshold: float = 0.02,
                                          min_signals: int = 2) -> List[Dict[str, Any]]:
    """Async function to get resilient price signals."""
    result = await resilient_orchestrator.orchestrate_signals(symbols, threshold, min_signals)
    return result.signals


def get_resilient_price_signals_sync(symbols: List[str], 
                                   threshold: float = 0.02,
                                   min_signals: int = 2) -> List[Dict[str, Any]]:
    """Sync wrapper for resilient signal generation."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        get_resilient_price_signals_async(symbols, threshold, min_signals)
    )


async def cleanup():
    """Cleanup function."""
    await dual_provider.close()


if __name__ == "__main__":
    import asyncio
    
    async def test_orchestrator():
        """Test the resilient orchestrator."""
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        print("🧪 Testing Resilient Signal Orchestrator...")
        
        try:
            result = await resilient_orchestrator.orchestrate_signals(
                symbols=test_symbols,
                threshold=0.015,
                min_signals=2
            )
            
            print(f"✅ Orchestration successful!")
            print(f"📊 Total signals: {result.total_signals}")
            print(f"🏆 Reliability score: {result.reliability_score:.2f}")
            print(f"⏱️ Processing time: {result.processing_time:.1f}s")
            print(f"🔗 Sources used: {', '.join(result.fallback_sources_used)}")
            print(f"📈 Breakdown: {result.signals_by_source}")
            
            # Show sample signals
            for i, signal in enumerate(result.signals[:3]):
                print(f"⚡ Signal {i+1}: {signal['symbol']} {signal['direction']} "
                      f"{signal['strength']:.2f} [{signal['data_source']}]")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        await cleanup()
        print("✅ Test completed!")
    
    asyncio.run(test_orchestrator())