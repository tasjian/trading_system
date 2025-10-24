#!/usr/bin/env python3
"""
RL_ONLY Branch Startup Script
Starts the continuous rebalancer with RL-only configuration and dedicated logging
"""

import os
import sys
import logging
from datetime import datetime

def setup_rl_only_logging():
    """Configure logging specifically for RL_ONLY branch"""
    log_file = "logs/rl_only_trading_system.log"
    
    # Ensure logs directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("🤖 RL_ONLY BRANCH STARTUP")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info(f"Log File: {log_file}")
    logger.info("Configuration: Pure FinRL-driven trading")
    logger.info("- Universe Filter: DISABLED")
    logger.info("- Sentiment Analysis: DISABLED") 
    logger.info("- Social Media: DISABLED")
    logger.info("- Stock Limits: REMOVED")
    logger.info("- FinRL: FULL CONTROL")
    logger.info("=" * 80)

if __name__ == "__main__":
    logger = None
    try:
        # Setup RL_ONLY logging
        setup_rl_only_logging()
        logger = logging.getLogger(__name__)
        
        # Import and start the continuous rebalancer
        logger.info("🚀 Starting RL_ONLY Continuous Rebalancer...")
        
        # Execute the continuous rebalancer module directly
        import subprocess
        import sys
        
        # Run continuous_rebalancer.py as a subprocess with the current Python environment
        result = subprocess.run([
            sys.executable, 
            "continuous_rebalancer.py"
        ], check=True)
        
    except KeyboardInterrupt:
        if logger:
            logger.info("🛑 RL_ONLY system stopped by user")
    except Exception as e:
        if logger:
            logger.error(f"💥 RL_ONLY system failed: {e}")
        else:
            print(f"💥 RL_ONLY system failed: {e}")
        raise