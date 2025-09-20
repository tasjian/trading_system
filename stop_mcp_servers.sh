#!/bin/bash
"""
Stop All MCP Servers for Trading System
Cleanly shuts down all background MCP server processes
"""

echo "🛑 Stopping All MCP Servers for Trading System"
echo "=" * 60

# Base directory
BASE_DIR="/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system"
PIDS_FILE="$BASE_DIR/mcp_server_pids.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if process is running
check_process() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to gracefully stop process
stop_process() {
    local name=$1
    local pid=$2
    
    if check_process $pid; then
        echo "   🔄 Stopping $name (PID: $pid)..."
        
        # Try graceful shutdown first
        kill -TERM $pid 2>/dev/null
        sleep 3
        
        # Check if still running
        if check_process $pid; then
            echo "   ⚠️  Process still running, forcing shutdown..."
            kill -KILL $pid 2>/dev/null
            sleep 1
            
            if check_process $pid; then
                echo -e "${RED}   ❌ Failed to stop $name (PID: $pid)${NC}"
                return 1
            else
                echo -e "${GREEN}   ✅ $name stopped (forced)${NC}"
                return 0
            fi
        else
            echo -e "${GREEN}   ✅ $name stopped gracefully${NC}"
            return 0
        fi
    else
        echo -e "${YELLOW}   ⚠️  $name (PID: $pid) was not running${NC}"
        return 0
    fi
}

echo -e "${BLUE}💾 1. Stopping Semantic Memory Server...${NC}"

# Stop processes from PID file
if [ -f "$PIDS_FILE" ]; then
    while IFS=':' read -r name pid; do
        if [ ! -z "$name" ] && [ ! -z "$pid" ]; then
            stop_process "$name" "$pid"
        fi
    done < "$PIDS_FILE"
    
    # Remove PID file
    rm -f "$PIDS_FILE"
    echo "   🗑️  Cleaned up PID file"
else
    echo "   ℹ️  No PID file found, checking for running processes..."
fi

# Also check for any remaining semantic memory servers by port
if curl -s --max-time 2 "http://127.0.0.1:8080/health" > /dev/null 2>&1; then
    echo "   🔍 Found semantic memory server still running on port 8080"
    
    # Find process by port
    MEMORY_PID=$(lsof -ti:8080 2>/dev/null)
    if [ ! -z "$MEMORY_PID" ]; then
        stop_process "semantic_memory_by_port" "$MEMORY_PID"
    fi
fi

echo -e "${BLUE}🧠 2. Sequential Thinking MCP Server...${NC}"
echo "   ℹ️  Sequential thinking server managed by Claude Code (no action needed)"

echo -e "${BLUE}📁 3. Filesystem MCP Server...${NC}"
echo "   ℹ️  Filesystem server managed by Claude Code (no action needed)"

echo ""
echo -e "${BLUE}🔍 4. Verifying Shutdown...${NC}"

# Check if semantic memory server is still accessible
if curl -s --max-time 2 "http://127.0.0.1:8080/health" > /dev/null 2>&1; then
    echo -e "${RED}   ❌ Semantic memory server still responding on port 8080${NC}"
else
    echo -e "${GREEN}   ✅ Semantic memory server is no longer accessible${NC}"
fi

# Check for any remaining Python processes running our MCP servers
REMAINING_PROCS=$(ps aux | grep -E "(semantic_memory|start_semantic_memory)" | grep -v grep | wc -l)
if [ "$REMAINING_PROCS" -gt 0 ]; then
    echo -e "${YELLOW}   ⚠️  Found $REMAINING_PROCS remaining MCP-related processes${NC}"
    ps aux | grep -E "(semantic_memory|start_semantic_memory)" | grep -v grep | sed 's/^/      /'
    
    echo "   🔧 To force kill all remaining processes:"
    echo "      pkill -f semantic_memory"
else
    echo -e "${GREEN}   ✅ No remaining MCP server processes found${NC}"
fi

# Check Claude MCP status
echo ""
echo -e "${BLUE}📊 5. Claude MCP Status After Shutdown...${NC}"
claude mcp list > mcp_status.tmp 2>&1

if grep -q "✗ Failed" mcp_status.tmp; then
    echo -e "${GREEN}   ✅ Expected failures (servers stopped):${NC}"
    grep "✗ Failed" mcp_status.tmp | sed 's/^/      /'
fi

if grep -q "✓ Connected" mcp_status.tmp; then
    echo -e "${BLUE}   ℹ️  Still connected (managed servers):${NC}"
    grep "✓ Connected" mcp_status.tmp | sed 's/^/      /'
fi

rm -f mcp_status.tmp

echo ""
echo -e "${GREEN}🎉 MCP Server Shutdown Complete!${NC}"
echo ""
echo -e "${BLUE}🔧 Cleanup Actions Performed:${NC}"
echo "   • Stopped background semantic memory server"
echo "   • Removed PID tracking file"
echo "   • Verified port 8080 is freed"
echo "   • Claude-managed servers remain configured"
echo ""
echo -e "${BLUE}💡 To restart all servers: ./start_mcp_servers.sh${NC}"