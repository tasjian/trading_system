#!/bin/bash

echo "🔄 RESTARTING TRADING SYSTEM WITH PERFORMANCE FIXES"
echo "=================================================="

# Stop existing system
echo "🛑 Stopping existing continuous rebalancer..."
pkill -f "continuous_rebalancer.py" || echo "No existing process found"

# Wait for clean shutdown
sleep 3

# Apply RL training fixes
echo "🧠 Applying RL training fixes..."
python3 force_rl_training.py

# Test daily summary email
echo "📧 Testing daily summary email..."
python3 -c "
import asyncio
from utils.daily_summary_scheduler import DailySummaryScheduler

async def test_email():
    scheduler = DailySummaryScheduler()
    success = await scheduler.test_daily_summary()
    if success:
        print('✅ Email system working')
    else:
        print('❌ Email system failed')

asyncio.run(test_email())
" || echo "⚠️ Email test failed (may be system-specific)"

# Restart system
echo "🚀 Starting enhanced trading system..."
nohup python3 continuous_rebalancer.py > logs/system_restart_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Get new process ID
sleep 2
NEW_PID=$(pgrep -f "continuous_rebalancer.py")

if [ ! -z "$NEW_PID" ]; then
    echo "✅ Trading system restarted successfully (PID: $NEW_PID)"
    echo ""
    echo "🔧 FIXES APPLIED:"
    echo "  • Email: Expanded sending window (3:30 PM - 11:59 PM)"
    echo "  • RL: Increased position change limits (10% → 25%)"  
    echo "  • RL: Relaxed safety constraints for more aggressive trading"
    echo "  • RL: Reduced training thresholds to encourage learning"
    echo "  • RL: Force-triggered training restart"
    echo ""
    echo "📊 Monitor system with: tail -f logs/continuous_rebalancer.log"
else
    echo "❌ Failed to restart trading system"
    exit 1
fi