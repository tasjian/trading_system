#!/usr/bin/env python3
"""
Test Continuous Rebalancing System
Quick test of the continuous system with shorter intervals for demonstration.
"""

import asyncio
import logging
from datetime import datetime
from continuous_rebalancer import ContinuousRebalancer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_continuous_system():
    """Test the continuous rebalancing system with shorter intervals."""
    
    print("🧪 TESTING CONTINUOUS REBALANCING SYSTEM")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This is a quick test with shortened intervals for demonstration")
    print("=" * 60)
    
    # Create a test instance with shorter intervals
    rebalancer = ContinuousRebalancer()
    
    # Initialize running state
    rebalancer.running = True
    
    # Override intervals for testing
    rebalancer.min_interval_minutes = 1      # 1 minute minimum
    rebalancer.standard_interval_minutes = 2 # 2 minutes standard
    rebalancer.after_hours_interval_minutes = 3  # 3 minutes after hours
    
    print(f"⚡ Test Configuration:")
    print(f"   Min Interval: {rebalancer.min_interval_minutes} minute(s)")
    print(f"   Standard Interval: {rebalancer.standard_interval_minutes} minute(s)")
    print(f"   After Hours Interval: {rebalancer.after_hours_interval_minutes} minute(s)")
    print()
    
    try:
        # Run for a limited time (5 minutes max)
        print("🚀 Starting test run (will stop after 3 cycles or 5 minutes)...")
        
        start_time = datetime.now()
        max_runtime_minutes = 5
        max_cycles = 3
        cycles_completed = 0
        
        # Override the continuous loop for testing
        while rebalancer.running and cycles_completed < max_cycles:
            # Check runtime limit
            runtime_minutes = (datetime.now() - start_time).total_seconds() / 60
            if runtime_minutes > max_runtime_minutes:
                print(f"⏰ Reached {max_runtime_minutes} minute time limit")
                break
            
            # Check if we should run (force first run for testing)
            should_run = await rebalancer._should_run_now()
            if not should_run and cycles_completed == 0:
                print("🔧 Forcing first test run...")
                should_run = True
            
            if not should_run:
                print("⏳ Waiting for next run opportunity...")
                await asyncio.sleep(30)  # Check every 30 seconds
                continue
            
            cycles_completed += 1
            print(f"\n🔄 TEST CYCLE #{cycles_completed}")
            print("-" * 40)
            
            # Run single rebalancing cycle
            result = await rebalancer._run_rebalancing_pipeline()
            
            # Process results
            rebalancer._process_result(result)
            
            # Show results
            if result.success:
                print(f"✅ Cycle {cycles_completed} completed successfully")
                print(f"   Duration: {result.duration_seconds:.1f}s")
                print(f"   Signals: {result.signals_generated}")
                print(f"   Orders: {result.orders_executed}")
                print(f"   Portfolio Value: ${result.portfolio_value:,.2f}")
            else:
                print(f"❌ Cycle {cycles_completed} failed")
                print(f"   Error: {result.error_message}")
                print(f"   Failed at: {result.pipeline_stage}")
            
            # Short delay before next cycle
            if cycles_completed < max_cycles:
                print(f"⏳ Waiting {rebalancer.standard_interval_minutes} minute(s) before next cycle...")
                await asyncio.sleep(rebalancer.standard_interval_minutes * 60)
        
        # Final results
        print(f"\n🏆 TEST RESULTS")
        print("=" * 40)
        print(f"Total Cycles: {cycles_completed}")
        print(f"Successful: {rebalancer.health.successful_runs}")
        print(f"Failed: {rebalancer.health.failed_runs}")
        
        if cycles_completed > 0:
            success_rate = (rebalancer.health.successful_runs / cycles_completed) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        print(f"Total Runtime: {(datetime.now() - start_time).total_seconds():.1f}s")
        
        # Show system health
        status = rebalancer.get_status_report()
        print(f"\n📊 FINAL SYSTEM STATUS:")
        print(f"   Average Runtime: {status['health']['avg_runtime_seconds']:.1f}s")
        print(f"   API Rate Limit Hits: {status['health']['api_rate_limit_hits']}")
        print(f"   Consecutive Failures: {status['health']['consecutive_failures']}")
        
        return cycles_completed > 0 and rebalancer.health.successful_runs > 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Test stopped by user after {cycles_completed} cycles")
        return cycles_completed > 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Ensure cleanup
        rebalancer.running = False

if __name__ == "__main__":
    print("Starting continuous rebalancing system test...")
    print("This will run 3 quick cycles to demonstrate the system.")
    print("Press Ctrl+C to stop early.")
    print()
    
    success = asyncio.run(test_continuous_system())
    
    if success:
        print(f"\n🎉 TEST COMPLETED: Continuous system working successfully!")
        print("\nTo run the full system continuously:")
        print("   python manage_rebalancer.py start")
        print("\nTo monitor in real-time:")
        print("   python manage_rebalancer.py monitor")
    else:
        print(f"\n❌ TEST FAILED: Check logs for issues")