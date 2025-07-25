#!/usr/bin/env python3
"""
Comprehensive Portfolio Rebalancing with End-to-End Enhanced Functionality

Uses the complete enhanced trading system:
- LLM-powered sentiment analysis from multiple sources
- Technical analysis with advanced indicators
- Fundamental analysis with financial metrics
- Risk assessment and diversification
- Market structure analysis
- Comprehensive stock screening and selection
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from core.market_intelligence import UnifiedMarketIntelligence
from tools.alpaca_client import alpaca_client
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensivePortfolioRebalancer:
    """Complete portfolio rebalancing system with all enhanced features."""
    
    def __init__(self):
        self.market_intel = UnifiedMarketIntelligence()
        self.target_positions = 8  # Target number of positions for diversification
        self.max_position_size = 0.15  # 15% max per position
        
        # Stock universe for screening
        self.stock_universe = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'DHR', 'ABT', 'BMY',
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'COF', 'AXP',
            # Consumer
            'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX',
            # Industrial
            'BA', 'CAT', 'GE', 'RTX', 'LMT', 'HON', 'UPS', 'MMM',
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'VLO', 'MPC',
            # Utilities & REITs
            'AMT', 'CCI', 'EQIX', 'PLD', 'SPG', 'NEE', 'SO', 'DUK'
        ]
    
    async def run_comprehensive_rebalance(self):
        """Execute complete portfolio rebalancing with all enhanced features."""
        
        print("🚀 COMPREHENSIVE PORTFOLIO REBALANCING")
        print("Using Complete Enhanced Trading System")
        print("=" * 70)
        
        # Step 1: Current Portfolio Analysis
        await self._analyze_current_portfolio()
        
        # Step 2: Comprehensive Stock Screening
        top_candidates = await self._comprehensive_stock_screening()
        
        # Step 3: Portfolio Construction
        new_portfolio = await self._construct_optimal_portfolio(top_candidates)
        
        # Step 4: Execute Rebalancing
        await self._execute_rebalancing(new_portfolio)
        
        print("\n🎉 COMPREHENSIVE REBALANCING COMPLETE!")
    
    async def _analyze_current_portfolio(self):
        """Analyze current portfolio with enhanced intelligence."""
        
        print("\n📊 STEP 1: CURRENT PORTFOLIO ANALYSIS")
        print("-" * 50)
        
        account_info = alpaca_client.get_account_info()
        positions = alpaca_client.get_positions()
        
        portfolio_value = float(account_info['equity'])
        cash_available = float(account_info['cash'])
        buying_power = float(account_info['buying_power'])
        
        print(f"💰 Portfolio Value: ${portfolio_value:,.2f}")
        print(f"💵 Cash Available: ${cash_available:,.2f}")
        print(f"🔥 Buying Power: ${buying_power:,.2f}")
        print(f"📈 Current Positions: {len(positions)}")
        
        if positions:
            print("\nCurrent Holdings Analysis:")
            total_pnl = 0
            
            for pos in positions:
                symbol = pos.get('symbol', '')
                qty = float(pos.get('qty', 0))
                market_value = float(pos.get('market_value', 0))
                unrealized_pl = float(pos.get('unrealized_pl', 0))
                total_pnl += unrealized_pl
                
                print(f"  {symbol}: ${market_value:,.2f} (P&L: ${unrealized_pl:,.2f})")
                
                # Quick sentiment check on existing positions
                try:
                    sentiment = await self.market_intel._analyze_sentiment(symbol)
                    if sentiment is not None:
                        sentiment_status = "📈" if sentiment > 0.2 else "📉" if sentiment < -0.2 else "⚖️"
                        print(f"    Sentiment: {sentiment_status} {sentiment:.3f}")
                except:
                    pass
            
            print(f"\n💰 Total Unrealized P&L: ${total_pnl:,.2f}")
        
        return portfolio_value, buying_power
    
    async def _comprehensive_stock_screening(self) -> List[Tuple[str, float, Dict]]:
        """Screen stocks using complete enhanced analysis system."""
        
        print("\n🔍 STEP 2: COMPREHENSIVE STOCK SCREENING")
        print("Using: Technical + Fundamental + Sentiment + Market Structure Analysis")
        print("-" * 70)
        
        candidates = []
        total_stocks = len(self.stock_universe)
        
        print(f"Analyzing {total_stocks} stocks across all sectors...")
        
        # Analyze stocks in batches to manage API limits
        batch_size = 10
        for i in range(0, len(self.stock_universe), batch_size):
            batch = self.stock_universe[i:i+batch_size]
            batch_number = (i // batch_size) + 1
            total_batches = (total_stocks + batch_size - 1) // batch_size
            
            print(f"\n📈 Batch {batch_number}/{total_batches}: {', '.join(batch)}")
            
            batch_tasks = []
            for symbol in batch:
                batch_tasks.append(self._analyze_single_stock(symbol))
            
            # Process batch concurrently but with delays
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    print(f"  {symbol}: ❌ Analysis failed - {result}")
                elif result:
                    score, analysis = result
                    candidates.append((symbol, score, analysis))
                    
                    # Show top-level results
                    signal_emoji = "🚀" if score > 0.6 else "📈" if score > 0.3 else "⚖️" if score > -0.1 else "📉"
                    print(f"  {symbol}: {signal_emoji} Score: {score:.3f} | Signal: {analysis.get('signal', 'N/A')}")
                else:
                    print(f"  {symbol}: ⚠️ No analysis data")
            
            # Small delay between batches to respect API limits
            if i + batch_size < len(self.stock_universe):
                print("  ⏳ Brief pause to respect API limits...")
                await asyncio.sleep(2)
        
        # Sort by comprehensive score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 TOP 15 CANDIDATES (from {len(candidates)} analyzed):")
        print("-" * 50)
        
        for i, (symbol, score, analysis) in enumerate(candidates[:15]):
            components = analysis.get('components', {})
            technical = components.get('technical', 0)
            fundamental = components.get('fundamental', 0)
            sentiment = components.get('sentiment', 0)  # Our new LLM sentiment!
            market_structure = components.get('market_structure', 0)
            
            print(f"{i+1:2d}. {symbol}: {score:.3f} | "
                  f"T:{technical:.2f} F:{fundamental:.2f} S:{sentiment:.2f} M:{market_structure:.2f}")
        
        return candidates[:20]  # Return top 20 for portfolio construction
    
    async def _analyze_single_stock(self, symbol: str) -> Tuple[float, Dict]:
        """Analyze a single stock with complete enhanced system."""
        try:
            # Use the enhanced market intelligence system
            analysis = await self.market_intel.analyze_stock(symbol)
            
            if analysis:
                return analysis.score, {
                    'signal': analysis.signal.value if hasattr(analysis.signal, 'value') else str(analysis.signal),
                    'confidence': analysis.confidence.value if hasattr(analysis.confidence, 'value') else str(analysis.confidence),
                    'components': getattr(analysis, 'components', {})
                }
            
            return 0.0, {}
            
        except Exception as e:
            logger.warning(f"Analysis failed for {symbol}: {e}")
            return None
    
    async def _construct_optimal_portfolio(self, candidates: List[Tuple[str, float, Dict]]) -> Dict[str, Dict]:
        """Construct optimal portfolio with diversification."""
        
        print(f"\n🏗️ STEP 3: OPTIMAL PORTFOLIO CONSTRUCTION")
        print("-" * 50)
        
        account_info = alpaca_client.get_account_info()
        total_capital = float(account_info['buying_power'])
        
        print(f"💰 Available Capital: ${total_capital:,.2f}")
        print(f"🎯 Target Positions: {self.target_positions}")
        print(f"📊 Max Position Size: {self.max_position_size:.1%}")
        
        # Select best candidates with sector diversification
        portfolio = {}
        sector_allocation = {}
        total_weight = 0
        
        # Sector mapping (simplified)
        sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
            'NVDA': 'Technology', 'META': 'Technology', 'TSLA': 'Technology', 'NFLX': 'Technology',
            'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'GS': 'Financial',
            'PG': 'Consumer', 'KO': 'Consumer', 'PEP': 'Consumer', 'WMT': 'Consumer',
            'BA': 'Industrial', 'CAT': 'Industrial', 'RTX': 'Industrial', 'LMT': 'Industrial',
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
            'AMT': 'REIT', 'CCI': 'REIT', 'NEE': 'Utilities', 'SO': 'Utilities'
        }
        
        for symbol, score, analysis in candidates:
            if len(portfolio) >= self.target_positions:
                break
            
            sector = sector_map.get(symbol, 'Other')
            sector_current = sector_allocation.get(sector, 0)
            
            # Diversification constraint: max 30% per sector
            if sector_current < 0.30:
                position_weight = min(self.max_position_size, (1.0 - total_weight) / (self.target_positions - len(portfolio)))
                position_value = total_capital * position_weight
                
                portfolio[symbol] = {
                    'weight': position_weight,
                    'value': position_value,
                    'score': score,
                    'sector': sector,
                    'analysis': analysis
                }
                
                sector_allocation[sector] = sector_current + position_weight
                total_weight += position_weight
        
        print(f"\n📋 CONSTRUCTED PORTFOLIO:")
        print("-" * 30)
        
        for symbol, data in portfolio.items():
            components = data['analysis'].get('components', {})
            sentiment_score = components.get('sentiment', 0)
            sentiment_emoji = "🚀" if sentiment_score > 0.5 else "📈" if sentiment_score > 0.2 else "⚖️"
            
            print(f"{symbol}: ${data['value']:,.0f} ({data['weight']:.1%}) | "
                  f"Score: {data['score']:.3f} | Sector: {data['sector']} | "
                  f"Sentiment: {sentiment_emoji} {sentiment_score:.2f}")
        
        print(f"\n🎯 SECTOR DIVERSIFICATION:")
        for sector, allocation in sector_allocation.items():
            print(f"  {sector}: {allocation:.1%}")
        
        return portfolio
    
    async def _execute_rebalancing(self, new_portfolio: Dict[str, Dict]):
        """Execute the portfolio rebalancing trades."""
        
        print(f"\n💫 STEP 4: EXECUTING REBALANCING TRADES")
        print("-" * 50)
        
        # First, sell existing positions that aren't in new portfolio
        current_positions = alpaca_client.get_positions()
        
        print("🔥 SELLING EXISTING POSITIONS:")
        for pos in current_positions:
            symbol = pos.get('symbol', '')
            qty = float(pos.get('qty', 0))
            
            if symbol not in new_portfolio:
                print(f"  Selling {symbol} ({qty} shares)...")
                try:
                    result = alpaca_client.place_order(
                        symbol=symbol,
                        qty=abs(qty),
                        side='sell',
                        order_type='market',
                        time_in_force='day'
                    )
                    if result:
                        print(f"    ✅ Sell order placed")
                    else:
                        print(f"    ❌ Sell order failed")
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                
                await asyncio.sleep(0.5)
        
        # Wait for sells to settle
        if current_positions:
            print("⏳ Waiting 5 seconds for sell orders to settle...")  
            await asyncio.sleep(5)
        
        # Now buy new positions
        print(f"\n🛒 BUYING NEW POSITIONS:")
        successful_buys = 0
        total_buys = len(new_portfolio)
        
        for symbol, data in new_portfolio.items():
            try:
                position_value = data['value']
                
                # Get current price to calculate shares
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                current_price = ticker.info.get('currentPrice', ticker.info.get('regularMarketPrice', 100))
                
                shares_to_buy = max(1, int(position_value / current_price))
                actual_cost = shares_to_buy * current_price
                
                print(f"  Buying {symbol}: {shares_to_buy} shares @ ${current_price:.2f} = ${actual_cost:.2f}")
                
                result = alpaca_client.place_order(
                    symbol=symbol,
                    qty=shares_to_buy,
                    side='buy',
                    order_type='market',
                    time_in_force='day'
                )
                
                if result:
                    successful_buys += 1
                    print(f"    ✅ Buy order placed successfully")
                    
                    # Show why we bought this stock
                    analysis = data['analysis']
                    components = analysis.get('components', {})
                    sentiment = components.get('sentiment', 0)
                    
                    reasons = []
                    if sentiment > 0.3:
                        reasons.append(f"Strong sentiment ({sentiment:.2f})")
                    if data['score'] > 0.5:
                        reasons.append(f"High overall score ({data['score']:.2f})")
                    if analysis.get('signal') in ['buy', 'strong_buy']:
                        reasons.append(f"Buy signal")
                    
                    if reasons:
                        print(f"    💡 Reason: {', '.join(reasons)}")
                else:
                    print(f"    ❌ Buy order failed")
                    
            except Exception as e:
                print(f"    ❌ Error buying {symbol}: {e}")
            
            await asyncio.sleep(0.5)
        
        print(f"\n📊 REBALANCING SUMMARY:")
        print(f"✅ Successful purchases: {successful_buys}/{total_buys}")
        print(f"📈 Success rate: {successful_buys/total_buys*100:.1f}%")
        
        # Final portfolio check
        await asyncio.sleep(3)
        final_positions = alpaca_client.get_positions()
        final_account = alpaca_client.get_account_info()
        
        print(f"\n🎉 FINAL PORTFOLIO STATUS:")
        print(f"💰 Portfolio Value: ${float(final_account['equity']):,.2f}")
        print(f"📈 Active Positions: {len(final_positions)}")
        print(f"💵 Remaining Cash: ${float(final_account['cash']):,.2f}")
        
        if final_positions:
            print(f"\nNew Holdings:")
            for pos in final_positions:
                symbol = pos.get('symbol', '')
                qty = float(pos.get('qty', 0))
                market_value = float(pos.get('market_value', 0))
                print(f"  {symbol}: {qty} shares, ${market_value:,.2f}")

async def main():
    """Run comprehensive portfolio rebalancing."""
    
    print("🎯 COMPREHENSIVE END-TO-END PORTFOLIO REBALANCING")
    print("Using Complete Enhanced Trading System:")
    print("✅ LLM-Powered Sentiment Analysis")
    print("✅ Advanced Technical Analysis") 
    print("✅ Fundamental Analysis")
    print("✅ Market Structure Analysis")
    print("✅ Risk Assessment & Diversification")
    print("=" * 70)
    
    rebalancer = ComprehensivePortfolioRebalancer()
    await rebalancer.run_comprehensive_rebalance()

if __name__ == "__main__":
    asyncio.run(main())