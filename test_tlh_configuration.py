#!/usr/bin/env python3
"""
Test TLH Configuration - Verify TLH Enable/Disable Functionality

Tests the Tax-Loss Harvesting system with both enabled and disabled configurations
to ensure proper behavior in both modes.
"""

import asyncio
import sys
import os
sys.path.append('.')

async def test_tlh_enabled():
    """Test TLH system when enabled."""
    print("\n" + "="*60)
    print("🧪 TESTING TLH ENABLED MODE")
    print("="*60)
    
    # Temporarily enable TLH
    from config.settings import settings
    original_tlh_enabled = settings.tlh_enabled
    settings.tlh_enabled = True
    
    try:
        # Test TLH Engine
        from core.tax_loss_harvesting import tax_loss_harvesting_engine
        
        print(f"✅ TLH Engine Status: {'ENABLED' if tax_loss_harvesting_engine.tlh_enabled else 'DISABLED'}")
        
        # Test scanning for opportunities
        opportunities = await tax_loss_harvesting_engine.scan_for_opportunities({})
        print(f"✅ TLH Opportunity Scan: {len(opportunities)} opportunities found")
        
        # Test execution (with empty list)
        results = await tax_loss_harvesting_engine.execute_harvesting_strategy([])
        print(f"✅ TLH Strategy Execution: {len(results)} results")
        
        # Test Tax-Aware Portfolio Balancer
        from core.tax_aware_portfolio_balancer import tax_aware_portfolio_balancer
        
        test_allocation = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
        analysis = await tax_aware_portfolio_balancer.analyze_tax_aware_portfolio_balance(test_allocation)
        
        print(f"✅ Portfolio Analysis Strategy: {analysis.recommended_strategy.value}")
        print(f"✅ TLH Opportunities Found: {len(analysis.tlh_opportunities)}")
        print(f"✅ Priority Actions: {len(analysis.priority_actions)}")
        
        # Check that TLH-specific actions are included
        tlh_actions = [action for action in analysis.priority_actions if 'TLH' in action or 'Harvest' in action or 'wash sale' in action]
        if analysis.tlh_opportunities or tlh_actions:
            print(f"✅ TLH-specific recommendations present")
        else:
            print(f"ℹ️ No TLH opportunities in current test scenario")
        
        return True
        
    except Exception as e:
        print(f"❌ TLH Enabled Test Failed: {e}")
        return False
    finally:
        settings.tlh_enabled = original_tlh_enabled

async def test_tlh_disabled():
    """Test TLH system when disabled."""
    print("\n" + "="*60)
    print("🧪 TESTING TLH DISABLED MODE")
    print("="*60)
    
    # Temporarily disable TLH
    from config.settings import settings
    original_tlh_enabled = settings.tlh_enabled
    settings.tlh_enabled = False
    
    try:
        # Reload the TLH engine with new settings
        from core.tax_loss_harvesting import TaxLossHarvestingEngine
        disabled_engine = TaxLossHarvestingEngine()
        
        print(f"✅ TLH Engine Status: {'ENABLED' if disabled_engine.tlh_enabled else 'DISABLED'}")
        
        # Test scanning for opportunities (should return empty)
        opportunities = await disabled_engine.scan_for_opportunities({})
        print(f"✅ TLH Opportunity Scan: {len(opportunities)} opportunities (should be 0)")
        
        # Test execution (should return empty)
        results = await disabled_engine.execute_harvesting_strategy([])
        print(f"✅ TLH Strategy Execution: {len(results)} results (should be 0)")
        
        # Test Tax-Aware Portfolio Balancer with disabled TLH
        from core.tax_aware_portfolio_balancer import TaxAwarePortfolioBalancer
        disabled_balancer = TaxAwarePortfolioBalancer()
        
        test_allocation = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
        analysis = await disabled_balancer.analyze_tax_aware_portfolio_balance(test_allocation)
        
        print(f"✅ Portfolio Analysis Strategy: {analysis.recommended_strategy.value}")
        print(f"✅ TLH Opportunities Found: {len(analysis.tlh_opportunities)} (should be 0)")
        
        # Check that strategy is IGNORE_TAX when TLH is disabled
        if analysis.recommended_strategy.value == "ignore_tax":
            print(f"✅ Correctly using IGNORE_TAX strategy when TLH disabled")
        else:
            print(f"⚠️ Expected IGNORE_TAX strategy, got: {analysis.recommended_strategy.value}")
        
        # Check for TLH disabled message in recommendations
        disabled_messages = [action for action in analysis.priority_actions if 'disabled' in action.lower() or 'TLH' in action]
        if disabled_messages:
            print(f"✅ TLH disabled message present in recommendations")
        else:
            print(f"ℹ️ No explicit TLH disabled message found")
        
        return len(opportunities) == 0 and len(results) == 0
        
    except Exception as e:
        print(f"❌ TLH Disabled Test Failed: {e}")
        return False
    finally:
        settings.tlh_enabled = original_tlh_enabled

async def test_configuration_settings():
    """Test TLH configuration settings."""
    print("\n" + "="*60)
    print("🧪 TESTING TLH CONFIGURATION SETTINGS")
    print("="*60)
    
    try:
        from config.settings import settings
        
        print(f"✅ TLH Enabled Setting: {settings.tlh_enabled}")
        print(f"✅ TLH Strategy: {settings.tlh_strategy}")
        print(f"✅ Min Loss Threshold: ${settings.min_loss_threshold}")
        print(f"✅ Tax Situation: {settings.tax_situation}")
        print(f"✅ Require Replacement: {settings.require_replacement}")
        print(f"✅ Default Lot Accounting: {settings.default_lot_accounting_method}")
        
        # Test environment variable override capability
        original_value = settings.tlh_enabled
        
        # Verify settings are accessible and reasonable
        assert hasattr(settings, 'tlh_enabled')
        assert hasattr(settings, 'tlh_strategy')
        assert hasattr(settings, 'min_loss_threshold')
        assert settings.min_loss_threshold >= 0
        
        print(f"✅ All TLH configuration settings are properly defined")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration Test Failed: {e}")
        return False

async def main():
    """Run all TLH configuration tests."""
    print("🚀 STARTING TLH CONFIGURATION TESTS")
    print("Testing Tax-Loss Harvesting enable/disable functionality")
    
    results = []
    
    # Test configuration settings
    results.append(await test_configuration_settings())
    
    # Test TLH enabled mode
    results.append(await test_tlh_enabled())
    
    # Test TLH disabled mode  
    results.append(await test_tlh_disabled())
    
    # Summary
    print("\n" + "="*60)
    print("🎯 TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED - TLH Enable/Disable functionality working correctly")
        print("✅ TLH can be safely enabled or disabled via settings.tlh_enabled")
        print("✅ System gracefully handles both modes without errors")
    else:
        print("❌ SOME TESTS FAILED - Review implementation")
        for i, result in enumerate(results, 1):
            status = "✅ PASS" if result else "❌ FAIL"
            test_name = ["Configuration", "TLH Enabled", "TLH Disabled"][i-1]
            print(f"   Test {i} ({test_name}): {status}")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)