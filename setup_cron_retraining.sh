#!/bin/bash
#
# Setup Cron Job for Automatic FinRL Retraining
#
# This script helps you set up automatic retraining on a schedule using cron.
#
# Usage:
#   ./setup_cron_retraining.sh
#
# Schedules available:
#   1. Weekly (Sunday 2:00 AM) - Recommended for production
#   2. Bi-weekly (1st and 15th, 2:00 AM) - Conservative approach
#   3. Daily (2:00 AM) - Aggressive, not recommended for production
#   4. Custom - Specify your own cron expression

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get trading system directory
TRADING_SYSTEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PATH=$(which python3)

echo "=================================="
echo "FinRL Automatic Retraining Setup"
echo "=================================="
echo ""
echo "Trading System: $TRADING_SYSTEM_DIR"
echo "Python: $PYTHON_PATH"
echo ""

# Check if scheduled_retraining.py exists
if [ ! -f "$TRADING_SYSTEM_DIR/scheduled_retraining.py" ]; then
    echo -e "${RED}Error: scheduled_retraining.py not found${NC}"
    exit 1
fi

# Check current crontab
echo "Current cron jobs for FinRL retraining:"
crontab -l 2>/dev/null | grep "scheduled_retraining" || echo "  None found"
echo ""

# Schedule options
echo "Select retraining schedule:"
echo "  1. Weekly (Every Sunday at 2:00 AM) - Recommended"
echo "  2. Bi-weekly (1st and 15th at 2:00 AM) - Conservative"
echo "  3. Weekly (Every Saturday at 3:00 AM) - Alternative"
echo "  4. Custom cron expression"
echo "  5. Remove existing cron job"
echo "  6. View sample cron expressions"
echo "  0. Exit without changes"
echo ""
read -p "Enter choice [0-6]: " choice

case $choice in
    1)
        CRON_SCHEDULE="0 2 * * 0"
        DESCRIPTION="Weekly (Sunday 2:00 AM)"
        ;;
    2)
        CRON_SCHEDULE="0 2 1,15 * *"
        DESCRIPTION="Bi-weekly (1st and 15th, 2:00 AM)"
        ;;
    3)
        CRON_SCHEDULE="0 3 * * 6"
        DESCRIPTION="Weekly (Saturday 3:00 AM)"
        ;;
    4)
        echo ""
        echo "Enter custom cron expression (e.g., '0 2 * * 0' for Sunday 2am):"
        read -p "Cron expression: " CRON_SCHEDULE
        DESCRIPTION="Custom schedule: $CRON_SCHEDULE"
        ;;
    5)
        echo ""
        echo "Removing existing FinRL retraining cron jobs..."
        crontab -l 2>/dev/null | grep -v "scheduled_retraining" | crontab -
        echo -e "${GREEN}✓ Removed existing cron jobs${NC}"
        exit 0
        ;;
    6)
        echo ""
        echo "Sample Cron Expressions:"
        echo "  0 2 * * 0     - Every Sunday at 2:00 AM"
        echo "  0 2 * * 6     - Every Saturday at 2:00 AM"
        echo "  0 2 1 * *     - First day of month at 2:00 AM"
        echo "  0 2 1,15 * *  - 1st and 15th at 2:00 AM"
        echo "  0 2 * * 1-5   - Monday-Friday at 2:00 AM"
        echo "  0 */6 * * *   - Every 6 hours"
        echo ""
        echo "Format: MIN HOUR DAY MONTH WEEKDAY"
        echo ""
        exit 0
        ;;
    0)
        echo "Exiting without changes."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# Create log directory
LOG_DIR="$TRADING_SYSTEM_DIR/logs"
mkdir -p "$LOG_DIR"

# Construct cron command
CRON_COMMAND="cd $TRADING_SYSTEM_DIR && $PYTHON_PATH scheduled_retraining.py >> $LOG_DIR/cron_retraining.log 2>&1"

# Show what will be added
echo ""
echo -e "${YELLOW}The following cron job will be added:${NC}"
echo "  Schedule: $DESCRIPTION"
echo "  Command: $CRON_COMMAND"
echo ""
read -p "Continue? [y/N]: " confirm

if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Add to crontab
(crontab -l 2>/dev/null | grep -v "scheduled_retraining"; echo "$CRON_SCHEDULE $CRON_COMMAND") | crontab -

echo ""
echo -e "${GREEN}✓ Cron job added successfully!${NC}"
echo ""
echo "Cron Schedule: $DESCRIPTION"
echo "Log file: $LOG_DIR/cron_retraining.log"
echo ""
echo "To verify:"
echo "  crontab -l | grep scheduled_retraining"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_DIR/cron_retraining.log"
echo ""
echo "To remove:"
echo "  ./setup_cron_retraining.sh (select option 5)"
echo ""

# Test the script
echo "Testing retraining script..."
echo "Running: python scheduled_retraining.py --help"
cd "$TRADING_SYSTEM_DIR"
$PYTHON_PATH -c "import scheduled_retraining; print('✓ Script can be imported')" && \
    echo -e "${GREEN}✓ Script test passed${NC}" || \
    echo -e "${RED}✗ Script test failed - check dependencies${NC}"

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
