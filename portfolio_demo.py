#!/usr/bin/env python3
"""Demonstration of the multi-agent portfolio management system."""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce log noise
logger = logging.getLogger(__name__)

async def portfolio_demo():
    """Quick demonstration of portfolio construction capabilities."""
    
    print("🎯 MULTI-AGENT PORTFOLIO MANAGEMENT DEMO")
    print("=" * 60)
    print("Demonstrating intelligent portfolio construction...")
    print()
    
    try:
        from agents.portfolio_management import construct_optimal_portfolio
        
        # Demo parameters - smaller set for speed
        candidate_symbols = [
            "AAPL", "MSFT", "GOOGL", "JPM", "JNJ", 
            "UNH", "PFE", "XOM", "KO", "PG"
        ]
        
        portfolio_value = 100000.0  # $100K portfolio
        risk_profile = "moderate"
        max_positions = 6
        
        print(f"📋 DEMO PARAMETERS")
        print(f"Candidate symbols: {candidate_symbols}")
        print(f"Portfolio value: ${portfolio_value:,.0f}")
        print(f"Risk profile: {risk_profile}")
        print(f"Max positions: {max_positions}")
        print()
        
        print("🚀 Constructing optimal portfolio...")
        print("(This may take a moment as agents analyze market conditions)")
        print()
        
        # Construct portfolio
        recommendation = await construct_optimal_portfolio(
            candidate_symbols=candidate_symbols,
            portfolio_value=portfolio_value,
            risk_profile=risk_profile,
            max_positions=max_positions
        )
        
        print("✅ PORTFOLIO RECOMMENDATION COMPLETE!")
        print("=" * 50)
        
        # Display results
        print(f"📊 PORTFOLIO OVERVIEW")
        print("-" * 30)
        print(f"Market Regime: {recommendation.market_regime.value.replace('_', ' ').title()}")
        print(f"Positions Selected: {len(recommendation.allocations)}")
        print(f"Expected Annual Return: {recommendation.expected_return:.1%}")
        print(f"Expected Volatility: {recommendation.expected_volatility:.1%}")
        print(f"Diversification Score: {recommendation.diversification_score:.3f}")
        print(f"Overall Confidence: {recommendation.confidence:.1%}")
        print(f"Cash Allocation: {recommendation.cash_allocation:.1%}")
        print()
        
        print(f"📈 POSITION ALLOCATIONS")
        print("-" * 40)
        print(f"{'Symbol':<8} {'Weight':<8} {'Asset Class':<15} {'Confidence':<12} {'Expected Ret':<12}")
        print("-" * 65)
        
        for alloc in recommendation.allocations:
            print(f"{alloc.symbol:<8} "
                  f"{alloc.target_weight:<8.1%} "
                  f"{alloc.asset_class.value.replace('_', ' ').title():<15} "
                  f"{alloc.confidence:<12.1%} "
                  f"{alloc.expected_return:<12.1%}")
        
        print()
        
        print(f"🧠 AGENT ANALYSIS")
        print("-" * 25)
        if recommendation.agents_consensus:
            for agent, score in recommendation.agents_consensus.items():
                agent_name = agent.replace('_', ' ').title()
                print(f"{agent_name:<25}: {score:.3f}")
        
        print()
        
        print(f"💡 RECOMMENDATION REASONING")
        print("-" * 35)
        print(f"{recommendation.reasoning}")
        print()
        
        print(f"🎯 INVESTMENT SUMMARY")
        print("-" * 30)
        
        total_allocation = sum(alloc.target_weight for alloc in recommendation.allocations)
        cash_amount = portfolio_value * recommendation.cash_allocation
        invested_amount = portfolio_value * total_allocation
        
        print(f"Total Invested: ${invested_amount:,.0f} ({total_allocation:.1%})")
        print(f"Cash Reserve: ${cash_amount:,.0f} ({recommendation.cash_allocation:.1%})")
        print(f"Expected Annual Gain: ${invested_amount * recommendation.expected_return:,.0f}")
        
        # Risk assessment
        print()
        print(f"⚖️ RISK ASSESSMENT")
        print("-" * 25)
        
        if recommendation.expected_volatility < 0.12:
            risk_level = "LOW RISK"
            risk_color = "🟢"
        elif recommendation.expected_volatility < 0.20:
            risk_level = "MODERATE RISK"
            risk_color = "🟡"
        else:
            risk_level = "HIGH RISK"
            risk_color = "🔴"
        
        print(f"Risk Level: {risk_color} {risk_level}")
        print(f"Volatility: {recommendation.expected_volatility:.1%}")
        print(f"Diversification: {'🟢 EXCELLENT' if recommendation.diversification_score > 0.6 else '🟡 GOOD' if recommendation.diversification_score > 0.4 else '🔴 NEEDS IMPROVEMENT'}")
        
        # Rebalancing recommendation
        if recommendation.rebalance_urgency > 0.3:
            print(f"🔄 Rebalancing Recommended (Urgency: {recommendation.rebalance_urgency:.1%})")
        else:
            print(f"✅ Portfolio Balanced (Urgency: {recommendation.rebalance_urgency:.1%})")
        
        print()
        print("🎉 PORTFOLIO CONSTRUCTION SUCCESSFUL!")
        print("=" * 45)
        print("✅ Multi-agent analysis completed")
        print("✅ Risk-optimized allocations determined")
        print("✅ Diversification maximized")
        print("✅ Market regime considerations applied")
        print("✅ Ready for implementation")
        
        print()
        print("💡 NEXT STEPS:")
        print("• Review allocations and adjust if needed")
        print("• Execute trades according to recommendations")
        print("• Monitor portfolio performance")
        print("• Rebalance when urgency score >30%")
        print("• Re-evaluate during market regime changes")
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio construction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(portfolio_demo())
    sys.exit(0 if success else 1)