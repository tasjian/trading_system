"""Simplified Gradio UI for streamlined portfolio management."""

import asyncio
import gradio as gr
import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging to reduce noise
logging.basicConfig(level=logging.WARNING)

def construct_portfolio_simple(symbols_text: str, portfolio_value: float, 
                             risk_level: str, max_positions: int) -> str:
    """Simplified portfolio construction with clean output."""
    try:
        # Parse symbols
        if not symbols_text.strip():
            return "❌ Please provide candidate symbols"
        
        symbols = [s.strip().upper() for s in symbols_text.replace(',', ' ').split() if s.strip()]
        
        if not symbols:
            return "❌ No valid symbols provided"
        
        if portfolio_value <= 0:
            return "❌ Portfolio value must be positive"
        
        # Import and run simplified system
        from agents.simplified_portfolio import build_simple_portfolio, format_portfolio_output
        
        # Run async portfolio construction
        portfolio = asyncio.run(build_simple_portfolio(
            symbols=symbols,
            portfolio_value=portfolio_value,
            risk_level=risk_level,
            max_positions=max_positions
        ))
        
        return format_portfolio_output(portfolio)
        
    except Exception as e:
        return f"❌ Portfolio construction failed: {str(e)}"

def create_simple_interface():
    """Create simplified Gradio interface."""
    
    with gr.Blocks(title="🎯 Simple Portfolio Manager", theme=gr.themes.Default()) as interface:
        
        gr.Markdown("""
        # 🎯 Simple AI Portfolio Manager
        
        **Streamlined portfolio construction with essential features only**
        
        Enter stock symbols and get an optimized portfolio recommendation in seconds.
        """)
        
        with gr.Row():
            with gr.Column():
                symbols_input = gr.Textbox(
                    label="Stock Symbols",
                    placeholder="AAPL MSFT GOOGL JPM JNJ UNH",
                    info="Enter symbols separated by spaces",
                    value="AAPL MSFT GOOGL JPM JNJ UNH KO PG"
                )
                
                portfolio_value = gr.Number(
                    label="Portfolio Value ($)",
                    value=100000,
                    minimum=1000
                )
                
                with gr.Row():
                    risk_level = gr.Dropdown(
                        label="Risk Level",
                        choices=["conservative", "moderate", "aggressive"],
                        value="moderate"
                    )
                    
                    max_positions = gr.Slider(
                        label="Max Positions",
                        minimum=3,
                        maximum=15,
                        value=6,
                        step=1
                    )
                
                construct_btn = gr.Button("🚀 Build Portfolio", variant="primary")
            
            with gr.Column():
                result_output = gr.Textbox(
                    label="Portfolio Recommendation",
                    lines=20,
                    interactive=False
                )
        
        # Event handler
        construct_btn.click(
            fn=construct_portfolio_simple,
            inputs=[symbols_input, portfolio_value, risk_level, max_positions],
            outputs=[result_output]
        )
        
        gr.Markdown("""
        ## ✨ How It Works
        
        1. **📊 Analysis**: Analyzes each stock using technical indicators (momentum, RSI, volatility)
        2. **⚖️ Optimization**: Creates risk-balanced portfolio using inverse volatility weighting
        3. **🎯 Selection**: Chooses top-scoring stocks within position limits
        4. **💰 Allocation**: Calculates optimal position sizes with cash reserve
        
        **Simple, fast, effective portfolio management in under 10 seconds!**
        """)
    
    return interface

if __name__ == "__main__":
    # Create and launch the simplified interface
    interface = create_simple_interface()
    interface.launch(server_port=7863, share=False)