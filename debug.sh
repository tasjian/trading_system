#!/bin/bash
"""
Quick Debug Utility for ML4T Trading System
Integrates with Claude Code debugging workflow
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

case "$1" in
    "logs"|"l")
        echo -e "${CYAN}📋 Recent Trading System Logs${NC}"
        echo "=================================="
        echo -e "${BLUE}Continuous Rebalancer (last 10 lines):${NC}"
        tail -10 logs/continuous_rebalancer.log 2>/dev/null || echo "  Log not found"
        echo ""
        echo -e "${BLUE}RL System (last 10 lines):${NC}"
        tail -10 logs/rl_only_trading_system.log 2>/dev/null || echo "  Log not found"
        echo ""
        echo -e "${BLUE}Errors (last 5):${NC}"
        grep -E "(ERROR|CRITICAL|Exception)" logs/*.log 2>/dev/null | tail -5 || echo "  No recent errors"
        ;;
    "status"|"s")
        echo -e "${CYAN}🔍 Trading System Debug Status${NC}"
        echo "=================================="
        echo -e "${BLUE}1. Trading Processes:${NC}"
        ps aux | grep -E "(continuous_rebalancer|start_rl_only)" | grep -v grep | sed 's/^/   /'
        echo ""
        echo -e "${BLUE}2. MCP Servers:${NC}"
        ./check_mcp_status.sh | grep -E "(✅|❌|⚠️)" | head -5 | sed 's/^/   /'
        echo ""
        echo -e "${BLUE}3. Recent Orders (last 5):${NC}"
        grep -E "(BUY|SELL|ORDER)" logs/rl_only_trading_system.log 2>/dev/null | tail -5 | sed 's/^/   /' || echo "  No recent orders"
        ;;
    "errors"|"e")
        echo -e "${CYAN}🚨 Recent Error Analysis${NC}"
        echo "=================================="
        echo -e "${BLUE}Python Exceptions:${NC}"
        grep -E "(Exception|Error|Traceback)" logs/*.log 2>/dev/null | tail -10 | sed 's/^/   /'
        echo ""
        echo -e "${BLUE}API Errors:${NC}"
        grep -E "(HTTP|API.*error|Failed.*request)" logs/*.log 2>/dev/null | tail -5 | sed 's/^/   /'
        echo ""
        echo -e "${BLUE}Order Failures:${NC}"
        grep -E "(Order.*failed|rejected|insufficient)" logs/*.log 2>/dev/null | tail -5 | sed 's/^/   /'
        ;;
    "performance"|"p")
        echo -e "${CYAN}⚡ Performance Analysis${NC}"
        echo "=================================="
        echo -e "${BLUE}Memory Usage:${NC}"
        ps aux | grep -E "(continuous_rebalancer|python)" | grep -v grep | awk '{print "   PID: " $2 ", Memory: " $4 "%, CPU: " $3 "%"}'
        echo ""
        echo -e "${BLUE}Disk Space:${NC}"
        df -h . | tail -1 | awk '{print "   Used: " $3 "/" $2 " (" $5 ")"}'
        echo ""
        echo -e "${BLUE}Log File Sizes:${NC}"
        ls -lh logs/*.log 2>/dev/null | awk '{print "   " $9 ": " $5}' || echo "  No log files found"
        ;;
    "orders"|"o")
        echo -e "${CYAN}📊 Order Execution Analysis${NC}"
        echo "=================================="
        echo -e "${BLUE}Recent Orders:${NC}"
        grep -E "Transaction added to batch" logs/rl_only_trading_system.log 2>/dev/null | tail -10 | sed 's/^/   /'
        echo ""
        echo -e "${BLUE}Order Success Rate:${NC}"
        TOTAL_ORDERS=$(grep -c "Transaction added to batch" logs/rl_only_trading_system.log 2>/dev/null || echo "0")
        FAILED_ORDERS=$(grep -c "Order.*failed\|rejected" logs/*.log 2>/dev/null || echo "0")
        if [ "$TOTAL_ORDERS" -gt 0 ]; then
            SUCCESS_RATE=$(echo "scale=1; (($TOTAL_ORDERS - $FAILED_ORDERS) / $TOTAL_ORDERS) * 100" | bc 2>/dev/null || echo "N/A")
            echo "   Total Orders: $TOTAL_ORDERS, Failed: $FAILED_ORDERS, Success Rate: ${SUCCESS_RATE}%"
        else
            echo "   No order data available"
        fi
        ;;
    "prompt"|"debug")
        echo -e "${CYAN}🛠 Generate Debug Prompt${NC}"
        echo "=================================="
        echo "Running debug assistant..."
        python utils/debug_assistant.py
        echo ""
        echo -e "${GREEN}📋 Debug prompt ready for Claude Code!${NC}"
        echo -e "${YELLOW}💡 Copy the generated prompt from debug_output/ to analyze issues${NC}"
        ;;
    "mcp"|"m")
        echo -e "${CYAN}🧠 MCP Server Debug${NC}"
        echo "=================================="
        ./mcp.sh status
        echo ""
        echo -e "${BLUE}Semantic Memory Test:${NC}"
        curl -s http://127.0.0.1:8080/health | python3 -m json.tool 2>/dev/null | sed 's/^/   /' || echo "   Semantic memory not accessible"
        ;;
    "full"|"f")
        echo -e "${CYAN}🔍 Full System Debug Report${NC}"
        echo "=================================="
        echo ""
        $0 status
        echo ""
        $0 logs
        echo ""
        $0 errors
        echo ""
        $0 orders
        echo ""
        $0 mcp
        ;;
    *)
        echo -e "${PURPLE}🛠 ML4T Trading System Debug Utility${NC}"
        echo "======================================"
        echo ""
        echo -e "${BLUE}Usage: ./debug.sh {command}${NC}"
        echo ""
        echo -e "${BLUE}Commands:${NC}"
        echo "  logs, l        Show recent trading system logs"
        echo "  status, s      Check system status and processes"
        echo "  errors, e      Analyze recent errors and exceptions"
        echo "  performance, p Show performance metrics"
        echo "  orders, o      Analyze order execution"
        echo "  prompt, debug  Generate debugging prompt for Claude Code"
        echo "  mcp, m         Check MCP server status"
        echo "  full, f        Full debug report (all commands)"
        echo ""
        echo -e "${YELLOW}💡 Quick Commands:${NC}"
        echo "  ./debug.sh s     # Quick status check"
        echo "  ./debug.sh e     # Check for errors"
        echo "  ./debug.sh debug # Generate Claude Code debug prompt"
        echo ""
        echo -e "${GREEN}🧠 MCP Integration: Sequential thinking and semantic memory available${NC}"
        ;;
esac