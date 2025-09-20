#!/bin/bash
"""
Start All MCP Servers for Trading System
Starts filesystem, sequential thinking, and semantic memory MCP servers
"""

echo "🚀 Starting All MCP Servers for Trading System"
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

# Function to check HTTP endpoint
check_http_endpoint() {
    local url=$1
    local timeout=${2:-5}
    if curl -s --max-time $timeout "$url" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Clear any existing PID file
> "$PIDS_FILE"

echo -e "${BLUE}📁 1. Starting Filesystem MCP Server...${NC}"
# Filesystem server is managed by Claude Code directly
claude mcp add trading-fs npx @modelcontextprotocol/server-filesystem "$BASE_DIR" --scope local 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}   ✅ Filesystem MCP server configured${NC}"
else
    echo -e "${YELLOW}   ⚠️  Filesystem MCP server already configured${NC}"
fi

echo -e "${BLUE}🧠 2. Starting Sequential Thinking MCP Server...${NC}"
# Sequential thinking server is managed by Claude Code directly  
claude mcp add sequential-thinking node "$BASE_DIR/mcp-sequentialthinking-tools/dist/index.js" --scope local 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}   ✅ Sequential thinking MCP server configured${NC}"
else
    echo -e "${YELLOW}   ⚠️  Sequential thinking MCP server already configured${NC}"
fi

echo -e "${BLUE}💾 3. Starting ChromaDB Semantic Memory Server...${NC}"
# Check if semantic memory server is already running
if check_http_endpoint "http://127.0.0.1:8080/health"; then
    echo -e "${GREEN}   ✅ Semantic memory server already running on port 8080${NC}"
    MEMORY_STATS=$(curl -s http://127.0.0.1:8080/memory/stats 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "   📊 $(echo $MEMORY_STATS | python3 -c "import sys,json; data=json.load(sys.stdin); print(f\"Collection: {data.get('collection_name', 'N/A')}, Memories: {data.get('total_memories', 0)}\")" 2>/dev/null || echo "Stats unavailable")"
    fi
else
    echo "   🚀 Starting semantic memory server..."
    cd "$BASE_DIR/claude_dev_kit"
    
    # Start in background and capture PID
    nohup python start_semantic_memory.py > semantic_memory.log 2>&1 &
    MEMORY_PID=$!
    echo "semantic_memory:$MEMORY_PID" >> "$PIDS_FILE"
    
    # Wait for server to start
    echo "   ⏳ Waiting for server to start..."
    sleep 5
    
    # Verify it started
    if check_http_endpoint "http://127.0.0.1:8080/health"; then
        echo -e "${GREEN}   ✅ Semantic memory server started successfully (PID: $MEMORY_PID)${NC}"
        
        # Get stats
        MEMORY_STATS=$(curl -s http://127.0.0.1:8080/memory/stats 2>/dev/null)
        if [ $? -eq 0 ]; then
            echo "   📊 $(echo $MEMORY_STATS | python3 -c "import sys,json; data=json.load(sys.stdin); print(f\"Collection: {data.get('collection_name', 'N/A')}, Memories: {data.get('total_memories', 0)}\")" 2>/dev/null || echo "Stats: Available")"
        fi
    else
        echo -e "${RED}   ❌ Failed to start semantic memory server${NC}"
        if check_process $MEMORY_PID; then
            kill $MEMORY_PID 2>/dev/null
        fi
    fi
    
    cd "$BASE_DIR"
fi

# Add semantic memory server to Claude MCP configuration
claude mcp add semantic-memory --transport http http://127.0.0.1:8080 --scope local 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}   ✅ Semantic memory server configured in Claude MCP${NC}"
else
    echo -e "${YELLOW}   ⚠️  Semantic memory server already configured in Claude MCP${NC}"
fi

echo ""
echo -e "${BLUE}🔍 4. Verifying All MCP Servers...${NC}"
echo "   Running Claude MCP health check..."

# Check MCP server status
claude mcp list > mcp_status.tmp 2>&1

if grep -q "✓ Connected" mcp_status.tmp; then
    echo -e "${GREEN}   ✅ MCP servers status:${NC}"
    grep "✓ Connected" mcp_status.tmp | sed 's/^/      /'
else
    echo -e "${YELLOW}   ⚠️  Some MCP servers may not be connected:${NC}"
    cat mcp_status.tmp | sed 's/^/      /'
fi

if grep -q "✗ Failed" mcp_status.tmp; then
    echo -e "${RED}   ❌ Failed connections:${NC}"
    grep "✗ Failed" mcp_status.tmp | sed 's/^/      /'
fi

rm -f mcp_status.tmp

echo ""
echo -e "${GREEN}🎉 MCP Server Startup Complete!${NC}"
echo ""
echo -e "${BLUE}📊 Available MCP Services:${NC}"
echo "   🗂️  Filesystem MCP: Enhanced file operations"
echo "   🧠 Sequential Thinking: Intelligent problem-solving workflow"
echo "   💾 Semantic Memory: ChromaDB-powered context storage"
echo ""
echo -e "${BLUE}🔧 Management Commands:${NC}"
echo "   • Health check: claude mcp list"
echo "   • Stop all: ./stop_mcp_servers.sh"
echo "   • Semantic memory: http://127.0.0.1:8080/health"
echo "   • View logs: cat claude_dev_kit/semantic_memory.log"

if [ -f "$PIDS_FILE" ] && [ -s "$PIDS_FILE" ]; then
    echo ""
    echo -e "${BLUE}📝 Running background processes saved to: $PIDS_FILE${NC}"
fi

echo ""
echo -e "${YELLOW}💡 Pro tip: All MCP tools now available with 'mcp__' prefix${NC}"