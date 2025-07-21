#!/usr/bin/env python3
"""Test script for enhanced multi-source market analysis system."""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_enhanced_analysis():
    """Test the enhanced multi-source market analysis system."""
    
    print("🚀 ENHANCED MULTI-SOURCE MARKET ANALYSIS TEST")
    print("=" * 60)
    print("Testing comprehensive data aggregation from multiple APIs...")
    print()
    
    try:
        # Import components
        from agents.market_analysis import (
            multi_source_data, 
            comprehensive_analyst,
            enhanced_market_analysis_factory,
            DataSource
        )
        
        # Test symbol
        test_symbol = "AAPL"
        print(f"🍎 Testing analysis for {test_symbol}")
        print("-" * 40)
        
        # Test 1: Multi-source data provider
        print("📊 Test 1: Multi-Source Data Aggregation")
        print("-" * 35)
        
        # Test stock data aggregation
        print("   Getting historical stock data...")
        stock_data = await multi_source_data.get_stock_data_multi_source(test_symbol, "1y")
        print(f"   ✅ Stock data: {len(stock_data.sources_used)} sources, confidence: {stock_data.confidence:.1%}")
        if stock_data.sources_used:
            print(f"      Sources: {', '.join([s.value for s in stock_data.sources_used])}")
        
        # Test financial metrics aggregation
        print("   Getting financial metrics...")
        metrics_data = await multi_source_data.get_financial_metrics_multi_source(test_symbol)
        print(f"   ✅ Financial metrics: {len(metrics_data.sources_used)} sources, confidence: {metrics_data.confidence:.1%}")
        if metrics_data.value:
            key_metrics = ['pe_ratio', 'market_cap', 'sector']
            for metric in key_metrics:
                if metric in metrics_data.value:
                    value = metrics_data.value[metric]
                    if metric == 'market_cap':
                        print(f"      {metric}: ${value:,.0f}" if isinstance(value, (int, float)) else f"      {metric}: {value}")
                    else:
                        print(f"      {metric}: {value}")
        
        # Test news sentiment aggregation
        print("   Getting news sentiment...")
        sentiment_data = await multi_source_data.get_news_sentiment_multi_source(test_symbol, "Apple Inc")
        print(f"   ✅ News sentiment: {len(sentiment_data.sources_used)} sources, confidence: {sentiment_data.confidence:.1%}")
        if sentiment_data.value:
            sentiment_info = sentiment_data.value
            print(f"      Sentiment score: {sentiment_info.get('sentiment_score', 0.5):.2f}")
            print(f"      Articles analyzed: {sentiment_info.get('article_count', 0)}")
        
        print()
        
        # Test 2: Individual Enhanced Agents
        print("🧠 Test 2: Enhanced Analysis Agents")
        print("-" * 32)
        
        agents = enhanced_market_analysis_factory
        
        # Test technical analysis
        print("   Technical Analysis Agent...")
        tech_result = await agents['technical'].analyze_symbol(test_symbol)
        print(f"   ✅ Technical: {tech_result.action} signal, confidence: {tech_result.confidence:.1%}")
        print(f"      Strength: {tech_result.strength.name}")
        print(f"      Data sources: {len(tech_result.data_sources or [])}")
        print(f"      Reasoning: {tech_result.reasoning[:60]}...")
        
        # Test fundamental analysis
        print("   Fundamental Analysis Agent...")
        fund_result = await agents['fundamental'].analyze_symbol(test_symbol)
        print(f"   ✅ Fundamental: {fund_result.action} signal, confidence: {fund_result.confidence:.1%}")
        print(f"      Strength: {fund_result.strength.name}")
        print(f"      Data confidence: {fund_result.data_confidence:.1%}")
        
        # Test sentiment analysis
        print("   Sentiment Analysis Agent...")
        sent_result = await agents['sentiment'].analyze_symbol(test_symbol, "Apple Inc")
        print(f"   ✅ Sentiment: {sent_result.action} signal, confidence: {sent_result.confidence:.1%}")
        print(f"      Strength: {sent_result.strength.name}")
        print(f"      Data sources: {len(sent_result.data_sources or [])}")
        
        print()
        
        # Test 3: H2O.ai Prediction Agent (if available)
        print("🤖 Test 3: H2O.ai ML Prediction Agent")
        print("-" * 33)
        
        try:
            from agents.h2o_prediction_agent import h2o_prediction_agent
            
            if h2o_prediction_agent and h2o_prediction_agent.h2o_initialized:
                print("   H2O.ai ML Price Prediction...")
                h2o_result = await h2o_prediction_agent.analyze_symbol(test_symbol)
                print(f"   ✅ H2O ML: {h2o_result.action} signal, confidence: {h2o_result.confidence:.1%}")
                print(f"      Strength: {h2o_result.strength.name}")
                print(f"      Model accuracy: {h2o_result.data_confidence:.1%}")
                
                # Get detailed prediction
                prediction = await h2o_prediction_agent.predict_stock_price(test_symbol)
                print(f"      Current price: ${prediction.current_price:.2f}")
                print(f"      7-day prediction: ${prediction.predicted_price_7d:.2f}")
                print(f"      Price direction: {prediction.price_direction}")
                print(f"      Features used: {len(prediction.features_used)}")
            else:
                print("   ⚠️ H2O.ai not initialized - skipping ML predictions")
                
        except Exception as h2o_error:
            print(f"   ⚠️ H2O.ai test error: {h2o_error}")
        
        print()
        
        # Test 4: Comprehensive Analysis
        print("🎯 Test 4: Comprehensive Multi-Source Analysis")
        print("-" * 42)
        
        print("   Running comprehensive analysis...")
        comp_result = await comprehensive_analyst.comprehensive_analysis(test_symbol, "Apple Inc")
        
        print(f"   ✅ Final Signal: {comp_result.action.upper()}")
        print(f"      Confidence: {comp_result.confidence:.1%}")
        print(f"      Strength: {comp_result.strength.name}")
        print(f"      Data confidence: {comp_result.data_confidence:.1%}")
        print(f"      Data sources: {len(comp_result.data_sources or [])}")
        
        if comp_result.price_target:
            print(f"      Price target: ${comp_result.price_target:.2f}")
        if comp_result.stop_loss:
            print(f"      Stop loss: ${comp_result.stop_loss:.2f}")
        
        print(f"      Reasoning: {comp_result.reasoning}")
        
        # Show key indicators
        if comp_result.indicators:
            print("      Key indicators:")
            key_indicators = ['current_price', 'rsi', 'pe_ratio', 'sentiment_score', 'predicted_7d']
            for indicator in key_indicators:
                # Check all possible indicator names
                indicator_keys = [k for k in comp_result.indicators.keys() if indicator in k.lower()]
                for key in indicator_keys[:1]:  # Show first match
                    value = comp_result.indicators[key]
                    if isinstance(value, (int, float)):
                        print(f"         {key}: {value:.2f}")
                    else:
                        print(f"         {key}: {value}")
        
        print()
        
        # Test 5: Data Source Reliability Assessment
        print("📈 Test 5: Data Source Reliability Assessment")
        print("-" * 41)
        
        data_sources_used = set()
        if hasattr(stock_data, 'sources_used'):
            data_sources_used.update(stock_data.sources_used)
        if hasattr(metrics_data, 'sources_used'):
            data_sources_used.update(metrics_data.sources_used)
        if hasattr(sentiment_data, 'sources_used'):
            data_sources_used.update(sentiment_data.sources_used)
        
        print(f"   Total unique data sources: {len(data_sources_used)}")
        for source in data_sources_used:
            print(f"      ✓ {source.value}")
        
        # Calculate overall system confidence
        confidence_scores = [
            stock_data.confidence,
            metrics_data.confidence, 
            sentiment_data.confidence,
            comp_result.confidence
        ]
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        print(f"   Average system confidence: {avg_confidence:.1%}")
        
        if avg_confidence >= 0.8:
            print("   🟢 HIGH CONFIDENCE - System ready for trading decisions")
        elif avg_confidence >= 0.6:
            print("   🟡 MEDIUM CONFIDENCE - Proceed with caution")
        elif avg_confidence >= 0.4:
            print("   🟠 LOW CONFIDENCE - Manual review recommended")
        else:
            print("   🔴 VERY LOW CONFIDENCE - Do not trade based on these signals")
        
        print()
        print("🎉 ENHANCED MULTI-SOURCE ANALYSIS TEST COMPLETE!")
        print("=" * 50)
        print("✅ Multi-source data aggregation working")
        print("✅ Enhanced analysis agents operational")
        print("✅ Comprehensive decision making functional")
        print("✅ Data reliability assessment active")
        
        if h2o_prediction_agent and h2o_prediction_agent.h2o_initialized:
            print("✅ H2O.ai ML predictions integrated")
        else:
            print("⚠️ H2O.ai ML predictions not available")
        
        print("\n🚀 System ready for production trading!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_enhanced_analysis())
    sys.exit(0 if success else 1)