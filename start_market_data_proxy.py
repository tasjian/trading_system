#!/usr/bin/env python3
"""
Market Data Proxy Service Starter
Starts the WebSocket-based market data proxy service as a background service.
"""

import asyncio
import logging
import signal
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tools.market_data_proxy import market_data_proxy

# Configure logging
def setup_logging():
    """Setup logging for the service."""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/market_data_proxy.log', mode='a')
        ]
    )
    
    # Suppress noisy WebSocket logs
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)

async def main():
    """Main service function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Market Data Proxy Service...")
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        # Cancel all running tasks
        for task in asyncio.all_tasks():
            task.cancel()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the market data proxy
        await market_data_proxy.start()
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service crashed: {e}")
        raise
    finally:
        logger.info("Market Data Proxy Service stopped")

def start_service():
    """Start the service (can be called from other modules)."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMarket Data Proxy Service stopped by user")
    except Exception as e:
        print(f"Service failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_service()