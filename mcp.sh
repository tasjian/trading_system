#!/bin/bash
"""
MCP Management Shortcuts
Quick access to MCP server management commands
"""

case "$1" in
    "start"|"up")
        echo "🚀 Starting all MCP servers..."
        ./start_mcp_servers.sh
        ;;
    "stop"|"down")
        echo "🛑 Stopping all MCP servers..."
        ./stop_mcp_servers.sh
        ;;
    "status"|"check")
        echo "📊 Checking MCP server status..."
        ./check_mcp_status.sh
        ;;
    "restart"|"reload")
        echo "🔄 Restarting all MCP servers..."
        ./stop_mcp_servers.sh
        sleep 2
        ./start_mcp_servers.sh
        ;;
    "memory")
        echo "💾 Testing semantic memory server..."
        curl -s http://127.0.0.1:8080/health | python3 -m json.tool
        ;;
    "list"|"ls")
        echo "📋 Listing Claude MCP configuration..."
        claude mcp list
        ;;
    *)
        echo "🧠 MCP Server Management"
        echo "Usage: ./mcp.sh {start|stop|status|restart|memory|list}"
        echo ""
        echo "Commands:"
        echo "  start    Start all MCP servers"
        echo "  stop     Stop all MCP servers"
        echo "  status   Check server status"
        echo "  restart  Restart all servers"
        echo "  memory   Test semantic memory"
        echo "  list     Show Claude MCP config"
        echo ""
        echo "Current status:"
        ./check_mcp_status.sh | grep -E "(✅|❌|⚠️)" | head -5
        ;;
esac