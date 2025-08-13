#!/bin/bash
# Start the ML4T Trading System
# This script launches the continuous rebalancer with proper logging and process management

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$SCRIPT_DIR/trading_system.pid"
LOG_FILE="$LOG_DIR/continuous_rebalancer.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

echo "🚀 Starting ML4T Trading System..."
echo "📁 Working directory: $SCRIPT_DIR"
echo "📝 Log file: $LOG_FILE"
echo "🔧 PID file: $PID_FILE"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "❌ Trading system is already running (PID: $PID)"
        echo "   Use ./stop.sh to stop it first"
        exit 1
    else
        echo "🧹 Cleaning up stale PID file"
        rm -f "$PID_FILE"
    fi
fi

# Check Python dependencies
echo "🔍 Checking Python environment..."
cd "$SCRIPT_DIR"
if ! $(pyenv which python3) -c "import pandas, numpy, yfinance, alpaca_trade_api" 2>/dev/null; then
    echo "❌ Missing required Python packages"
    echo "   Run: pip install -r requirements.txt"
    exit 1
fi

# Start the trading system
echo "🎯 Launching continuous rebalancer..."
echo "   Press Ctrl+C to stop gracefully"
echo "   Or use ./stop.sh from another terminal"
echo ""

# Start with nohup for background execution
nohup $(pyenv which python3) "$SCRIPT_DIR/continuous_rebalancer.py" >> "$LOG_FILE" 2>&1 &
PID=$!

# Save PID
echo $PID > "$PID_FILE"

echo "✅ Trading system started successfully!"
echo "   PID: $PID"
echo "   Monitor logs: tail -f $LOG_FILE"
echo "   Stop system: ./stop.sh"
echo ""

# Show initial log output
echo "📊 Initial system output:"
sleep 2
tail -20 "$LOG_FILE" 2>/dev/null || echo "   (Log file not ready yet)"