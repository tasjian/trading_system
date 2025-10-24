#!/bin/bash
"""
Check Status of All MCP Servers
Quick health check for all running MCP services
"""

echo "📊 MCP Servers Status Check"
echo "=" * 40

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check HTTP endpoint
check_http_endpoint() {
    local url=$1
    local timeout=${2:-3}
    if curl -s --max-time $timeout "$url" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

echo -e "${BLUE}1. Claude MCP Servers:${NC}"
claude mcp list 2>/dev/null | grep -E "(✓|✗)" | sed 's/^/   /'

echo ""
echo -e "${BLUE}2. Semantic Memory Server:${NC}"
if check_http_endpoint "http://127.0.0.1:8080/health"; then
    echo -e "${GREEN}   ✅ Running on http://127.0.0.1:8080${NC}"
    
    # Get detailed stats
    STATS=$(curl -s http://127.0.0.1:8080/memory/stats 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "   📊 $(echo $STATS | python3 -c "import sys,json; data=json.load(sys.stdin); print(f\"Collection: {data.get('collection_name', 'N/A')}, Memories: {data.get('total_memories', 0)}, Status: {data.get('status', 'unknown')}\")" 2>/dev/null || echo "Stats: Available")"
    fi
else
    echo -e "${RED}   ❌ Not accessible on http://127.0.0.1:8080${NC}"
fi

echo ""
echo -e "${BLUE}3. Background Processes:${NC}"
PIDS_FILE="/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system/mcp_server_pids.txt"

if [ -f "$PIDS_FILE" ]; then
    echo "   📝 Tracked processes:"
    while IFS=':' read -r name pid; do
        if [ ! -z "$name" ] && [ ! -z "$pid" ]; then
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "${GREEN}      ✅ $name (PID: $pid)${NC}"
            else
                echo -e "${RED}      ❌ $name (PID: $pid) - Not running${NC}"
            fi
        fi
    done < "$PIDS_FILE"
else
    echo "   ℹ️  No PID file found"
fi

# Check for any MCP-related processes
MCP_PROCS=$(ps aux | grep -E "(semantic_memory|sequentialthinking|mcp.*server)" | grep -v grep | wc -l)
if [ "$MCP_PROCS" -gt 0 ]; then
    echo "   🔍 Found $MCP_PROCS MCP-related processes:"
    ps aux | grep -E "(semantic_memory|sequentialthinking|mcp.*server)" | grep -v grep | awk '{print "      " $2 " " $11}' | head -5
fi

echo ""
echo -e "${BLUE}4. Network Ports:${NC}"
echo "   🌐 Port usage:"
for port in 8080; do
    if lsof -ti:$port > /dev/null 2>&1; then
        PROC_INFO=$(lsof -ti:$port | head -1 | xargs ps -p 2>/dev/null | tail -1 | awk '{print $4}')
        echo -e "${GREEN}      ✅ Port $port: In use ($PROC_INFO)${NC}"
    else
        echo -e "${YELLOW}      ⚪ Port $port: Available${NC}"
    fi
done

echo ""
echo -e "${BLUE}💡 Quick Commands:${NC}"
echo "   • Start all: ./start_mcp_servers.sh"
echo "   • Stop all: ./stop_mcp_servers.sh"
echo "   • This check: ./check_mcp_status.sh"
echo "   • Semantic memory test: curl http://127.0.0.1:8080/health"