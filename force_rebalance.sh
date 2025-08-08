#!/bin/bash
# Force an immediate rebalance of the ML4T Trading System
# This script triggers an immediate rebalancing cycle

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/trading_system.pid"
LOG_FILE="$SCRIPT_DIR/logs/continuous_rebalancer.log"

echo "⚡ Force Rebalancing ML4T Trading System..."

# Check if system is running
if [ ! -f "$PID_FILE" ]; then
    echo "❌ Trading system is not running"
    echo "   Start it first with: ./start.sh"
    exit 1
fi

PID=$(cat "$PID_FILE")

# Verify process is actually running
if ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ Process $PID is not running (stale PID file)"
    echo "   Clean up with: rm $PID_FILE"
    echo "   Then start with: ./start.sh"
    exit 1
fi

echo "📝 Found running system (PID: $PID)"

# Option 1: Send SIGUSR1 signal to trigger immediate rebalance (if implemented)
# This is a common pattern for triggering immediate actions
echo "🔄 Method 1: Sending SIGUSR1 signal for immediate rebalance..."
kill -USR1 $PID 2>/dev/null || {
    echo "⚠️  SIGUSR1 not handled by the process (not implemented)"
    echo ""
    
    # Option 2: Create a trigger file that the system monitors
    echo "🔄 Method 2: Creating force rebalance trigger file..."
    TRIGGER_FILE="$SCRIPT_DIR/.force_rebalance_trigger"
    echo "$(date +%s)" > "$TRIGGER_FILE"
    echo "   Created trigger file: $TRIGGER_FILE"
    echo "   The system will detect this file and trigger rebalancing"
    echo ""
    
    # Option 3: Direct API call if available
    echo "🔄 Method 3: Manual immediate execution..."
    echo "   Starting manual rebalancing process..."
    
    # Execute the rebalance logic directly in background
    (
        cd "$SCRIPT_DIR"
        python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')
from continuous_rebalancer import ContinuousRebalancer

async def force_rebalance():
    print('🔄 Executing forced rebalance...')
    rebalancer = ContinuousRebalancer()
    result = await rebalancer._run_rebalancing_cycle()
    if result.success:
        print(f'✅ Forced rebalance completed successfully')
        print(f'   Duration: {result.duration_seconds:.1f}s')
        print(f'   Signals: {result.signals_generated}')
        print(f'   Orders: {result.orders_executed}')
        print(f'   Portfolio Value: \${result.portfolio_value:,.2f}')
    else:
        print(f'❌ Forced rebalance failed: {result.error_message}')

if __name__ == '__main__':
    asyncio.run(force_rebalance())
" 2>&1 | tee -a "$LOG_FILE"
    ) &
    
    FORCE_PID=$!
    echo "   Manual rebalance started (PID: $FORCE_PID)"
}

echo ""
echo "📊 Monitoring recent activity..."
echo "   Watch logs: tail -f $LOG_FILE"
echo ""

# Show recent log activity
if [ -f "$LOG_FILE" ]; then
    echo "📈 Recent system activity:"
    tail -15 "$LOG_FILE" 2>/dev/null || echo "   (No recent log entries)"
else
    echo "   (No log file found yet)"
fi

echo ""
echo "✅ Force rebalance initiated"
echo "   The system should begin rebalancing shortly"
echo "   Monitor the log file for progress updates"