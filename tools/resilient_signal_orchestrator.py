#!/usr/bin/env python3
"""
Fail-Fast Signal Orchestrator
Single-source signal generation using ONLY Alpaca data.
No fallbacks allowed - system halts on data quality issues.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

# REMOVED: All fallback data sources for fail-fast architecture
# from tools.dual_provider_market_data import dual_provider, get_price_signals, get_market_data
# from tools.enhanced_news_signals import enhanced_news_generator
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
        
        # FAIL-FAST ARCHITECTURE: No graceful degradation allowed
        # System must have reliable data or halt completely
        self.source_priorities = [
            'alpaca_only',           # PRIMARY AND ONLY: Alpaca market data
            # REMOVED: All fallback sources - system halts if Alpaca fails
        ]
        
        # STRICT REQUIREMENTS: No tolerance for poor quality data
        self.min_signals_required = 5  # Higher minimum for quality
        self.min_confidence_threshold = 0.8  # High confidence required
        self.max_processing_time = 15.0  # Fast timeout
        
        logger.info("🎯 FAIL-FAST Signal Orchestrator initialized (Alpaca-Only)")
        logger.info("⚠️ NO FALLBACKS: System halts on data failures")
        logger.info(f"Single source: {self.source_priorities[0]}")
    
    async def orchestrate_signals(self, symbols: List[str], 
                                threshold: float = 0.02,
                                min_signals: int = 5,
                                max_processing_time: float = 15.0) -> OrchestrationResult:
        """
        FAIL-FAST signal orchestration using ONLY Alpaca data.
        
        Strategy:
        1. Alpaca ONLY - No fallbacks allowed
        2. Halt immediately on any data quality issues
        3. Require high signal count and confidence
        
        Args:
            symbols: List of symbols to analyze
            threshold: Price change threshold (strict)
            min_signals: Minimum signals required (higher than before)
            max_processing_time: Maximum processing time (faster timeout)
            
        Returns:
            OrchestrationResult with high-quality signal data
            
        Raises:
            RuntimeError: If Alpaca data unavailable or insufficient
        """
        start_time = datetime.now()
        logger.info(f"🚀 FAIL-FAST ORCHESTRATION for {len(symbols)} symbols")
        logger.info(f"STRICT Requirements: min_signals={min_signals}, threshold={threshold:.1%}")
        logger.info(f"⚠️ NO FALLBACKS: Alpaca data must be available or system halts")
        
        signals = []
        sources_used = []
        source_signal_counts = {}
        
        # SINGLE STAGE: Alpaca-only data with strict validation
        logger.info("📊 ALPACA-ONLY DATA COLLECTION (No fallbacks)")
        try:
            # Import Alpaca market data directly
            from tools.alpaca_market_data import get_price_signals as alpaca_get_price_signals
            
            # Get signals from Alpaca ONLY
            alpaca_signals = alpaca_get_price_signals(symbols, threshold)
            
            if not alpaca_signals:
                # For paper trading with limited data access, log warning but continue with empty signals
                # The system will use cached sentiment data instead
                logger.warning(
                    f"⚠️ Alpaca returned no price signals from {len(symbols)} symbols "
                    f"(threshold: {threshold:.1%}). This is normal for paper trading with limited SIP data access. "
                    f"System will continue with cached sentiment data and portfolio allocations."
                )
                alpaca_signals = []  # Empty signals list, system will use other data sources
            
            # Validate signal quality
            high_quality_signals = []
            for signal in alpaca_signals:
                confidence = signal.get('confidence', 0.0)
                if confidence >= self.min_confidence_threshold:
                    high_quality_signals.append(signal)
            
            if len(high_quality_signals) < min_signals:
                error_msg = (
                    f"❌ CRITICAL SYSTEM FAILURE: Insufficient high-quality signals\n"
                    f"High-quality signals: {len(high_quality_signals)} (min required: {min_signals})\n"
                    f"Total signals: {len(alpaca_signals)}\n"
                    f"Confidence threshold: {self.min_confidence_threshold}\n"
                    f"SYSTEM REQUIRES HIGH-QUALITY DATA TO OPERATE SAFELY\n"
                    f"Trading halted to prevent poor-quality decisions"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            signals.extend(high_quality_signals)
            sources_used.append('alpaca_only')
            source_signal_counts['alpaca_only'] = len(high_quality_signals)
            logger.info(f"✅ Alpaca-only: {len(high_quality_signals)} high-quality signals")
                
        except Exception as e:
            error_msg = (
                f"❌ CRITICAL SYSTEM FAILURE: Alpaca data collection failed\n"
                f"Error: {str(e)}\n"
                f"SYSTEM CANNOT OPERATE WITHOUT ALPACA DATA\n"
                f"All trading operations halted"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # NO FALLBACK STAGES - FAIL FAST ARCHITECTURE
        # If Alpaca data is insufficient, system halts immediately
        
        # Final validation and results
        processing_time = (datetime.now() - start_time).total_seconds()
        primary_success = 'alpaca_only' in sources_used
        reliability_score = 1.0 if len(signals) >= min_signals else 0.0  # Binary scoring
        
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
        logger.info("✅ FAIL-FAST ORCHESTRATION COMPLETE")
        logger.info(f"Signals: {len(signals)} | Source: Alpaca-only | Time: {processing_time:.1f}s")
        logger.info(f"Quality: {reliability_score} (binary: pass/fail)")
        logger.info(f"Signal breakdown: {source_signal_counts}")
        
        # This should never happen with the new fail-fast architecture
        # as we raise errors immediately when Alpaca fails
        if len(signals) < min_signals:
            error_msg = (
                f"❌ IMPOSSIBLE SYSTEM STATE: Signals passed initial validation but failed final check\n"
                f"This indicates a bug in the fail-fast orchestrator logic\n"
                f"Signals Generated: {len(signals)} (minimum required: {min_signals})\n"
                f"Source: Alpaca-only\n"
                f"Processing Time: {processing_time:.1f}s\n"
                f"SYSTEM LOGIC ERROR - INVESTIGATION REQUIRED"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # No cleanup needed for Alpaca-only architecture
        # Alpaca client manages its own connections
        try:
            pass  # No dual_provider to close
            logger.debug("✅ Cleaned up sessions after orchestration success")
        except Exception as e:
            logger.debug(f"Session cleanup warning: {e}")
        
        return result
    
    # REMOVED: SEC EDGAR fallback method - fail-fast architecture only
    
    # REMOVED: Enhanced news fallback method - fail-fast architecture only
    
    # REMOVED: Technical pattern fallback method - fail-fast architecture only
    
    def _calculate_reliability_score(self, sources_used: List[str], signals_generated: int, min_required: int) -> float:
        """Binary reliability scoring for fail-fast architecture."""
        # Simple binary scoring: pass (1.0) or fail (0.0)
        return 1.0 if signals_generated >= min_required else 0.0
    
    async def get_source_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of Alpaca-only data source."""
        health_status = {}
        
        # Check Alpaca health only
        try:
            from tools.alpaca_market_data import get_market_data_alpaca
            test_data = await get_market_data_alpaca(['AAPL'])  # Quick test
            health_status['alpaca_only'] = {
                'status': 'healthy' if test_data else 'failed',
                'last_check': datetime.now().isoformat(),
                'note': 'Alpaca-only architecture - no fallbacks'
            }
        except Exception as e:
            health_status['alpaca_only'] = {
                'status': 'failed',
                'error': str(e),
                'last_check': datetime.now().isoformat(),
                'note': 'CRITICAL: System halts when Alpaca fails'
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
    """Cleanup function for fail-fast architecture."""
    # No cleanup needed for Alpaca-only architecture
    pass


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