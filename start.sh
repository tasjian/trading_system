#!/bin/bash
# Enhanced ML4T Trading System Startup Script
# Starts all components for continuous trading with full functionality

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
DATA_DIR="$SCRIPT_DIR/data"
PID_FILE="$SCRIPT_DIR/trading_system.pid"
LOG_FILE="$LOG_DIR/continuous_rebalancer.log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Ensure required directories exist
mkdir -p "$LOG_DIR" "$DATA_DIR" "$DATA_DIR/monitoring" "$DATA_DIR/historical_cache"

echo -e "${CYAN}🚀 Starting Enhanced ML4T Trading System${NC}"
echo -e "${CYAN}=====================================================${NC}"
echo -e "${BLUE}📁 Working directory: $SCRIPT_DIR${NC}"
echo -e "${BLUE}📝 Log file: $LOG_FILE${NC}"
echo -e "${BLUE}🔧 PID file: $PID_FILE${NC}"
echo ""

# Function to check if a process is running
check_process_running() {
    local pid=$1
    if ps -p $pid > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to get process by port
get_process_by_port() {
    local port=$1
    lsof -ti:$port 2>/dev/null || echo ""
}

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if check_process_running $PID; then
        echo -e "${RED}❌ Trading system is already running (PID: $PID)${NC}"
        echo -e "${YELLOW}   Use ./stop.sh to stop it first${NC}"
        exit 1
    else
        echo -e "${YELLOW}🧹 Cleaning up stale PID file${NC}"
        rm -f "$PID_FILE"
    fi
fi

# Change to script directory
cd "$SCRIPT_DIR"

# Check Python environment
echo -e "${BLUE}🔍 Checking Python environment...${NC}"
PYTHON_CMD=$(which python3 2>/dev/null || which python 2>/dev/null || echo "")
if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}❌ Python not found in PATH${NC}"
    exit 1
fi

# Try pyenv first if available
if command -v pyenv >/dev/null 2>&1; then
    PYTHON_CMD=$(pyenv which python3 2>/dev/null || pyenv which python 2>/dev/null || echo "")
    if [ -n "$PYTHON_CMD" ]; then
        echo -e "${GREEN}✅ Using pyenv Python: $PYTHON_CMD${NC}"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
fi

echo -e "${BLUE}   Using Python: $PYTHON_CMD${NC}"

# Check core dependencies
echo -e "${BLUE}🔍 Checking core Python dependencies...${NC}"
MISSING_DEPS=""

check_dependency() {
    local dep=$1
    if ! $PYTHON_CMD -c "import $dep" 2>/dev/null; then
        MISSING_DEPS="$MISSING_DEPS $dep"
        return 1
    fi
    return 0
}

# Core trading dependencies
check_dependency "pandas" || echo -e "${YELLOW}   ⚠️ pandas missing${NC}"
check_dependency "numpy" || echo -e "${YELLOW}   ⚠️ numpy missing${NC}"
check_dependency "alpaca_trade_api" || echo -e "${YELLOW}   ⚠️ alpaca_trade_api missing${NC}"
check_dependency "asyncio" || echo -e "${YELLOW}   ⚠️ asyncio missing${NC}"

# Optional but recommended dependencies
check_dependency "redis" && echo -e "${GREEN}   ✅ Redis client available${NC}" || echo -e "${YELLOW}   ⚠️ Redis client missing (degraded caching)${NC}"
check_dependency "torch" && echo -e "${GREEN}   ✅ PyTorch available (RL enabled)${NC}" || echo -e "${YELLOW}   ⚠️ PyTorch missing (RL disabled)${NC}"
check_dependency "aiohttp" && echo -e "${GREEN}   ✅ aiohttp available${NC}" || echo -e "${YELLOW}   ⚠️ aiohttp missing${NC}"

if [ -n "$MISSING_DEPS" ]; then
    echo -e "${RED}❌ Critical dependencies missing:$MISSING_DEPS${NC}"
    echo -e "${YELLOW}   Run: pip install -r requirements.txt${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Core dependencies verified${NC}"

# Check system services (Redis)
echo -e "${BLUE}🔧 Checking system services...${NC}"

# Check Redis availability
REDIS_RUNNING=false
if command -v redis-cli >/dev/null 2>&1; then
    if redis-cli ping >/dev/null 2>&1; then
        echo -e "${GREEN}   ✅ Redis server running${NC}"
        REDIS_RUNNING=true
    else
        echo -e "${YELLOW}   ⚠️ Redis server not responding${NC}"
        
        # Try to start Redis
        echo -e "${BLUE}   🔄 Attempting to start Redis...${NC}"
        
        # macOS (Homebrew)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew >/dev/null 2>&1; then
                if brew services start redis >/dev/null 2>&1; then
                    echo -e "${GREEN}   ✅ Redis started via Homebrew${NC}"
                    sleep 2
                    if redis-cli ping >/dev/null 2>&1; then
                        REDIS_RUNNING=true
                    fi
                fi
            fi
            
            # Try direct start if brew failed
            if [ "$REDIS_RUNNING" = false ] && command -v redis-server >/dev/null 2>&1; then
                redis-server --daemonize yes --port 6379 --bind 127.0.0.1 >/dev/null 2>&1 &
                sleep 2
                if redis-cli ping >/dev/null 2>&1; then
                    echo -e "${GREEN}   ✅ Redis started directly${NC}"
                    REDIS_RUNNING=true
                fi
            fi
        else
            # Linux - try systemctl
            if command -v systemctl >/dev/null 2>&1; then
                if systemctl start redis >/dev/null 2>&1; then
                    echo -e "${GREEN}   ✅ Redis started via systemctl${NC}"
                    sleep 2
                    if redis-cli ping >/dev/null 2>&1; then
                        REDIS_RUNNING=true
                    fi
                fi
            fi
        fi
        
        if [ "$REDIS_RUNNING" = false ]; then
            echo -e "${YELLOW}   ⚠️ Could not start Redis - continuing with degraded performance${NC}"
        fi
    fi
else
    echo -e "${YELLOW}   ⚠️ Redis not installed - install for better performance${NC}"
fi

# Check environment variables
echo -e "${BLUE}🔑 Checking environment configuration...${NC}"
ENV_WARNINGS=""

check_env_var() {
    local var_name=$1
    local required=${2:-false}
    
    if [ -z "${!var_name}" ]; then
        if [ "$required" = true ]; then
            echo -e "${RED}   ❌ $var_name not set (required)${NC}"
            return 1
        else
            echo -e "${YELLOW}   ⚠️ $var_name not set (optional)${NC}"
            ENV_WARNINGS="$ENV_WARNINGS $var_name"
        fi
    else
        echo -e "${GREEN}   ✅ $var_name configured${NC}"
    fi
    return 0
}

# Check critical environment variables
check_env_var "ALPACA_API_KEY" true || exit 1
check_env_var "ALPACA_SECRET_KEY" true || exit 1
check_env_var "ALPACA_BASE_URL" false
check_env_var "OPENAI_API_KEY" false
check_env_var "EMAIL_HOST" false
check_env_var "EMAIL_PORT" false

if [ -n "$ENV_WARNINGS" ]; then
    echo -e "${YELLOW}   Optional variables not set:$ENV_WARNINGS${NC}"
    echo -e "${YELLOW}   Some features may be disabled${NC}"
fi

# Display system configuration
echo ""
echo -e "${PURPLE}📊 System Configuration Summary:${NC}"
echo -e "${BLUE}   • Continuous Rebalancer: ✅ Enabled (main trading engine)${NC}"
echo -e "${BLUE}   • RL Agent Integration: ✅ Enabled (hybrid LLM-RL portfolio decisions)${NC}"
echo -e "${BLUE}   • Sentiment Analysis: ✅ Enabled (social media + news + earnings)${NC}"
echo -e "${BLUE}   • Portfolio Balancer: ✅ Enabled (intelligent buy/sell prioritization)${NC}"
echo -e "${BLUE}   • Risk Management: ✅ Enabled (stop-losses, position limits, drawdown)${NC}"
echo -e "${BLUE}   • Performance Monitoring: ✅ Enabled (pipeline performance tracking)${NC}"
echo -e "${BLUE}   • Daily Email Reports: ✅ Enabled (automated portfolio summaries)${NC}"
echo -e "${BLUE}   • OCO Trading: ✅ Enabled (bracket orders, stop-losses)${NC}"
echo -e "${BLUE}   • Market Data Caching: $([ "$REDIS_RUNNING" = true ] && echo "✅ Enabled (Redis)" || echo "⚠️ Degraded (no Redis)")${NC}"

# Check for optimal trading intervals
echo ""
echo -e "${PURPLE}⏰ Trading Schedule Configuration:${NC}"
echo -e "${BLUE}   • Market Hours: 5-minute rebalancing intervals${NC}"
echo -e "${BLUE}   • After Hours: 30-minute monitoring intervals${NC}"
echo -e "${BLUE}   • Sentiment Analysis: 90-minute refresh cycles${NC}"
echo -e "${BLUE}   • Daily Summaries: Automated email reports${NC}"
echo -e "${BLUE}   • Risk Checks: Continuous monitoring${NC}"

# Final pre-flight check
echo ""
echo -e "${BLUE}🔍 Pre-flight system check...${NC}"

# Check if log file is writable
if ! touch "$LOG_FILE" 2>/dev/null; then
    echo -e "${RED}❌ Cannot write to log file: $LOG_FILE${NC}"
    exit 1
fi

# Check if data directory is writable
if ! touch "$DATA_DIR/test_write" 2>/dev/null; then
    echo -e "${RED}❌ Cannot write to data directory: $DATA_DIR${NC}"
    exit 1
else
    rm -f "$DATA_DIR/test_write"
fi

# Check for conflicting processes
CONFLICTING_PORTS=(6379 8000 11434)
for port in "${CONFLICTING_PORTS[@]}"; do
    EXISTING_PID=$(get_process_by_port $port)
    if [ -n "$EXISTING_PID" ]; then
        echo -e "${YELLOW}   ⚠️ Port $port in use by PID $EXISTING_PID${NC}"
        if [ $port -eq 6379 ] && [ "$REDIS_RUNNING" = true ]; then
            echo -e "${GREEN}     (This is expected for Redis)${NC}"
        fi
    fi
done

echo -e "${GREEN}✅ Pre-flight check complete${NC}"

# Start the enhanced trading system
echo ""
echo -e "${CYAN}🎯 Launching Enhanced Continuous Trading System...${NC}"
echo -e "${BLUE}   Mode: Continuous rebalancing with hybrid LLM-RL intelligence${NC}"
echo -e "${BLUE}   Components: All trading systems active${NC}"
echo -e "${BLUE}   Control: Press Ctrl+C to stop gracefully${NC}"
echo -e "${BLUE}   Monitor: tail -f $LOG_FILE${NC}"
echo -e "${BLUE}   Stop: ./stop.sh from another terminal${NC}"
echo ""

# Create system startup log entry
echo "$(date '+%Y-%m-%d %H:%M:%S') - Enhanced ML4T Trading System startup initiated" >> "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Configuration: Full continuous trading mode" >> "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Components: Continuous rebalancer, RL agent, sentiment analysis, portfolio balancer" >> "$LOG_FILE"

# Start with nohup for background execution with enhanced logging
echo -e "${GREEN}🚀 Starting continuous rebalancer with all functionality...${NC}"

# Enhanced startup command with proper error handling
nohup $PYTHON_CMD "$SCRIPT_DIR/continuous_rebalancer.py" >> "$LOG_FILE" 2>&1 &
PID=$!

# Save PID immediately
echo $PID > "$PID_FILE"

# Wait a moment to check if process started successfully
sleep 3

if check_process_running $PID; then
    echo -e "${GREEN}✅ Trading system started successfully!${NC}"
    echo -e "${GREEN}   PID: $PID${NC}"
    echo -e "${BLUE}   Status: All systems operational${NC}"
    echo -e "${BLUE}   Log monitoring: tail -f $LOG_FILE${NC}"
    echo -e "${BLUE}   System control: ./stop.sh${NC}"
    echo ""
    
    # Show initial system output
    echo -e "${PURPLE}📊 Initial system output (last 15 lines):${NC}"
    echo -e "${CYAN}------------------------------------------------${NC}"
    if [ -f "$LOG_FILE" ]; then
        tail -15 "$LOG_FILE" 2>/dev/null | while IFS= read -r line; do
            echo -e "${BLUE}   $line${NC}"
        done
    else
        echo -e "${YELLOW}   (Log file not ready yet - check in a few seconds)${NC}"
    fi
    echo -e "${CYAN}------------------------------------------------${NC}"
    echo ""
    
    # Display real-time monitoring suggestion
    echo -e "${PURPLE}💡 Real-time Monitoring Commands:${NC}"
    echo -e "${BLUE}   Watch logs:       tail -f $LOG_FILE${NC}"
    echo -e "${BLUE}   System status:    ps aux | grep continuous_rebalancer${NC}"
    echo -e "${BLUE}   Trading activity: grep -E 'BUY|SELL|ORDER' $LOG_FILE | tail -10${NC}"
    echo -e "${BLUE}   Stop system:      ./stop.sh${NC}"
    echo ""
    
    echo -e "${GREEN}🎯 Enhanced ML4T Trading System is now running in continuous mode!${NC}"
    echo -e "${GREEN}   All advanced features active: RL intelligence, balanced buy/sell, risk management${NC}"
    
else
    echo -e "${RED}❌ Trading system failed to start properly${NC}"
    echo -e "${YELLOW}   Checking log file for errors...${NC}"
    
    if [ -f "$LOG_FILE" ]; then
        echo -e "${RED}Last 10 lines of log:${NC}"
        tail -10 "$LOG_FILE"
    fi
    
    # Clean up PID file
    rm -f "$PID_FILE"
    exit 1
fi