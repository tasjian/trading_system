#!/bin/bash
# Enhanced ML4T Trading System Stop Script
# Gracefully shuts down all trading system components

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/trading_system.pid"
LOG_FILE="$SCRIPT_DIR/logs/continuous_rebalancer.log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}🛑 Stopping Enhanced ML4T Trading System${NC}"
echo -e "${CYAN}==========================================${NC}"

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "❌ No PID file found - system may not be running"
    echo "   PID file expected at: $PID_FILE"
    
    # Check for any running processes anyway
    RUNNING_PIDS=$(pgrep -f "python.*continuous_rebalancer.py" || true)
    if [ -n "$RUNNING_PIDS" ]; then
        echo "🔍 Found running trading processes:"
        for pid in $RUNNING_PIDS; do
            echo "   PID: $pid"
        done
        echo ""
        read -p "Do you want to stop these processes? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "🛑 Stopping found processes..."
            kill $RUNNING_PIDS
            sleep 2
            echo "✅ Processes stopped"
        fi
    else
        echo "   No running trading processes found"
    fi
    exit 0
fi

# Read PID
PID=$(cat "$PID_FILE")
echo "📝 Found PID: $PID"

# Check if process is actually running
if ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ Process $PID is not running"
    echo "🧹 Cleaning up PID file"
    rm -f "$PID_FILE"
    exit 0
fi

# Get process info
PROCESS_INFO=$(ps -p $PID -o pid,ppid,comm,args --no-headers 2>/dev/null || echo "Process info unavailable")
echo "🔍 Process info: $PROCESS_INFO"

# Graceful shutdown
echo "🤝 Sending graceful shutdown signal (SIGTERM)..."
kill -TERM $PID

# Wait for graceful shutdown
echo "⏳ Waiting for graceful shutdown (max 30 seconds)..."
for i in {1..30}; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "✅ Process stopped gracefully"
        rm -f "$PID_FILE"
        echo "🧹 PID file cleaned up"
        
        # Show final log entries
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "📊 Final log entries:"
            tail -10 "$LOG_FILE" 2>/dev/null || echo "   (No recent log entries)"
        fi
        
        echo ""
        echo -e "${GREEN}✅ Enhanced ML4T Trading System stopped successfully${NC}"
        echo -e "${BLUE}   All components gracefully shut down${NC}"
        exit 0
    fi
    sleep 1
done

# Force kill if graceful shutdown failed
echo "⚠️  Graceful shutdown timed out, forcing stop..."
kill -KILL $PID 2>/dev/null || true

# Wait a bit more
sleep 2

# Final check
if ps -p $PID > /dev/null 2>&1; then
    echo "❌ Failed to stop process $PID"
    echo "   You may need to manually kill it: kill -9 $PID"
    exit 1
else
    echo "✅ Process forcefully stopped"
    rm -f "$PID_FILE"
    echo "🧹 PID file cleaned up"
    echo ""
    echo -e "${GREEN}✅ Enhanced ML4T Trading System stopped successfully${NC}"
fi