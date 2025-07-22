#!/usr/bin/env python3
"""Launch script for the Gradio trading interface."""

import sys
import os
import gradio as gr
import logging
from simple_ui import create_simple_interface

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to launch the Gradio interface."""
    try:
        logger.info("🚀 Starting AI Trading System Gradio Interface...")
        
        # Create the interface
        interface = create_simple_interface()
        
        # Launch with explicit settings
        logger.info("📱 Launching interface on http://localhost:7863")
        logger.info("🔗 You can access the trading system through your web browser")
        
        # Launch the interface
        interface.launch(
            server_port=7863,
            server_name="localhost", 
            share=False,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to launch Gradio interface: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 AI TRADING SYSTEM - GRADIO INTERFACE")
    print("=" * 60)
    print("📊 Portfolio Construction & Trading Dashboard")
    print("🔗 Access URL: http://localhost:7863")
    print("=" * 60)
    
    main()