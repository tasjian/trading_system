#!/bin/bash
"""
Start ChromaDB Semantic Memory MCP Server
"""

echo "🧠 Starting ChromaDB Semantic Memory MCP Server..."

# Navigate to claude_dev_kit directory
cd "/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system/claude_dev_kit"

# Check if server is already running
if curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
    echo "✅ Semantic memory server already running on http://127.0.0.1:8080"
    curl -s http://127.0.0.1:8080/memory/stats | python -m json.tool
else
    echo "🚀 Starting semantic memory server..."
    python start_semantic_memory.py &
    
    # Wait for server to start
    sleep 3
    
    # Verify it's running
    if curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
        echo "✅ Semantic memory server started successfully!"
        curl -s http://127.0.0.1:8080/memory/stats | python -m json.tool
    else
        echo "❌ Failed to start semantic memory server"
        exit 1
    fi
fi

echo ""
echo "📊 Available endpoints:"
echo "   - Health: http://127.0.0.1:8080/health"
echo "   - Store Memory: POST http://127.0.0.1:8080/memory/store"
echo "   - Query Memory: POST http://127.0.0.1:8080/memory/query" 
echo "   - Memory Stats: GET http://127.0.0.1:8080/memory/stats"
echo "   - Clear Memory: DELETE http://127.0.0.1:8080/memory/clear"