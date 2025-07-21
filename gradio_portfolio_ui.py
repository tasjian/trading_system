"""Gradio UI for Multi-Agent Portfolio Management with Chain of Thought Visualization."""

import asyncio
import gradio as gr
import pandas as pd
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class PortfolioUIManager:
    """Manager for the Gradio portfolio interface."""
    
    def __init__(self):
        self.current_recommendation = None
        self.chain_of_thought_history = []
        self.agent_decisions_history = []
        
    async def construct_portfolio_async(self, candidate_symbols_text: str, 
                                      portfolio_value: float,
                                      risk_profile: str, 
                                      max_positions: int) -> Tuple[str, str, str, str]:
        """Async wrapper for portfolio construction."""
        try:
            # Parse candidate symbols
            if not candidate_symbols_text.strip():
                return ("❌ Error: Please provide candidate symbols", "", "", "")
            
            symbols = [s.strip().upper() for s in candidate_symbols_text.replace(',', ' ').split()]
            symbols = [s for s in symbols if s]  # Remove empty strings
            
            if not symbols:
                return ("❌ Error: No valid symbols provided", "", "", "")
            
            if portfolio_value <= 0:
                return ("❌ Error: Portfolio value must be positive", "", "", "")
            
            # Try LangGraph orchestration first, fallback to regular construction
            try:
                from agents.langgraph_portfolio_engine import construct_portfolio_with_orchestration
                result = await construct_portfolio_with_orchestration(
                    candidate_symbols=symbols,
                    portfolio_value=portfolio_value,
                    risk_profile=risk_profile,
                    max_positions=max_positions
                )
                
                recommendation = result["recommendation"]
                chain_of_thought = result.get("chain_of_thought", [])
                agent_decisions = result.get("agent_decisions", {})
                langgraph_used = result.get("langgraph_used", False)
                
            except Exception as e:
                logger.warning(f"LangGraph construction failed, using fallback: {e}")
                from agents.portfolio_management import construct_optimal_portfolio
                recommendation = await construct_optimal_portfolio(
                    candidate_symbols=symbols,
                    portfolio_value=portfolio_value,
                    risk_profile=risk_profile,
                    max_positions=max_positions
                )
                chain_of_thought = []
                agent_decisions = {}
                langgraph_used = False
            
            # Store results
            self.current_recommendation = recommendation
            self.chain_of_thought_history = chain_of_thought
            self.agent_decisions_history = agent_decisions
            
            # Generate outputs
            overview = self._generate_portfolio_overview(recommendation, langgraph_used)
            allocations_table = self._generate_allocations_table(recommendation)
            chain_of_thought_display = self._generate_chain_of_thought_display(chain_of_thought)
            agent_decisions_display = self._generate_agent_decisions_display(agent_decisions)
            
            return overview, allocations_table, chain_of_thought_display, agent_decisions_display
            
        except Exception as e:
            error_msg = f"❌ Portfolio construction failed: {str(e)}"
            logger.error(error_msg)
            return (error_msg, "", "", "")
    
    def _generate_portfolio_overview(self, recommendation, langgraph_used: bool = False) -> str:
        """Generate portfolio overview HTML."""
        if not recommendation:
            return "No recommendation available"
        
        # Determine risk level color
        volatility = recommendation.expected_volatility
        if volatility < 0.12:
            risk_color = "🟢"
            risk_level = "LOW RISK"
        elif volatility < 0.20:
            risk_color = "🟡" 
            risk_level = "MODERATE RISK"
        else:
            risk_color = "🔴"
            risk_level = "HIGH RISK"
        
        # Diversification assessment
        div_score = recommendation.diversification_score
        if div_score > 0.6:
            div_status = "🟢 EXCELLENT"
        elif div_score > 0.4:
            div_status = "🟡 GOOD"
        else:
            div_status = "🔴 NEEDS IMPROVEMENT"
        
        orchestration_badge = "🤖 LangGraph" if langgraph_used else "🔧 Standard"
        
        html = f"""
        <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin: 10px 0;">
            <h2>🎯 Portfolio Recommendation {orchestration_badge}</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">
                
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                    <h3>📊 Performance Metrics</h3>
                    <p><strong>Expected Return:</strong> {recommendation.expected_return:.1%}</p>
                    <p><strong>Volatility:</strong> {recommendation.expected_volatility:.1%}</p>
                    <p><strong>Confidence:</strong> {recommendation.confidence:.1%}</p>
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                    <h3>⚖️ Risk Assessment</h3>
                    <p><strong>Risk Level:</strong> {risk_color} {risk_level}</p>
                    <p><strong>Diversification:</strong> {div_status}</p>
                    <p><strong>Score:</strong> {div_score:.3f}</p>
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                    <h3>🏛️ Market Context</h3>
                    <p><strong>Regime:</strong> {recommendation.market_regime.value.replace('_', ' ').title()}</p>
                    <p><strong>Positions:</strong> {len(recommendation.allocations)}</p>
                    <p><strong>Cash Reserve:</strong> {recommendation.cash_allocation:.1%}</p>
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                    <h3>🔄 Rebalancing</h3>
                    <p><strong>Urgency:</strong> {recommendation.rebalance_urgency:.1%}</p>
                    <p><strong>Status:</strong> {'🔄 Needed' if recommendation.rebalance_urgency > 0.3 else '✅ Balanced'}</p>
                    <p><strong>Timestamp:</strong> {recommendation.timestamp.strftime('%H:%M:%S')}</p>
                </div>
                
            </div>
            
            <div style="margin-top: 20px; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                <h3>💡 Reasoning</h3>
                <p>{recommendation.reasoning}</p>
            </div>
        </div>
        """
        
        return html
    
    def _generate_allocations_table(self, recommendation) -> str:
        """Generate allocations table HTML."""
        if not recommendation or not recommendation.allocations:
            return "<p>No allocations available</p>"
        
        # Create DataFrame for better display
        data = []
        total_value = 0
        
        for alloc in recommendation.allocations:
            position_value = alloc.metadata.get('position_value', 0)
            total_value += position_value
            
            data.append({
                'Symbol': alloc.symbol,
                'Weight': f"{alloc.target_weight:.1%}",
                'Value': f"${position_value:,.0f}",
                'Asset Class': alloc.asset_class.value.replace('_', ' ').title(),
                'Expected Return': f"{alloc.expected_return:.1%}",
                'Confidence': f"{alloc.confidence:.1%}",
                'Risk Score': f"{alloc.risk_score:.2f}",
                'Action': alloc.recommended_action.title()
            })
        
        df = pd.DataFrame(data)
        
        # Convert to HTML with styling
        table_html = df.to_html(index=False, escape=False, classes="portfolio-table")
        
        styled_html = f"""
        <div style="margin: 20px 0;">
            <h3>📈 Portfolio Allocations (Total: ${total_value:,.0f})</h3>
            <div style="overflow-x: auto;">
                <style>
                .portfolio-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-family: Arial, sans-serif;
                }}
                .portfolio-table th {{
                    background-color: #4CAF50;
                    color: white;
                    padding: 12px;
                    text-align: left;
                    border: 1px solid #ddd;
                }}
                .portfolio-table td {{
                    padding: 10px;
                    border: 1px solid #ddd;
                    text-align: center;
                }}
                .portfolio-table tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .portfolio-table tr:hover {{
                    background-color: #e8f4fd;
                }}
                </style>
                {table_html}
            </div>
        </div>
        """
        
        return styled_html
    
    def _generate_chain_of_thought_display(self, chain_of_thought: List[Dict]) -> str:
        """Generate chain of thought visualization."""
        if not chain_of_thought:
            return "<p>No chain of thought data available (using standard engine)</p>"
        
        html_parts = []
        html_parts.append("""
        <div style="margin: 20px 0;">
            <h3>🧠 Chain of Thought - Agent Decision Process</h3>
            <div style="max-height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; background: #f9f9f9;">
        """)
        
        for i, entry in enumerate(chain_of_thought):
            timestamp = entry.get('timestamp', '')
            agent = entry.get('agent', 'unknown')
            message = entry.get('message', '')
            
            # Color code by agent
            agent_colors = {
                'market_regime_agent': '#FF6B6B',
                'sector_rotation_agent': '#4ECDC4', 
                'risk_parity_agent': '#45B7D1',
                'momentum_agent': '#96CEB4',
                'value_agent': '#FFEAA7',
                'ml_agent': '#DDA0DD',
                'portfolio_optimizer': '#98D8C8',
                'risk_validator': '#F7DC6F',
                'final_recommendation': '#BB8FCE'
            }
            
            color = agent_colors.get(agent, '#BDC3C7')
            
            html_parts.append(f"""
            <div style="margin: 10px 0; padding: 10px; background: {color}; border-radius: 5px; border-left: 4px solid #333;">
                <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 5px;">
                    <strong style="color: #333;">🤖 {agent.replace('_', ' ').title()}</strong>
                    <small style="color: #666; margin-left: auto;">{timestamp.split('T')[1][:8] if 'T' in timestamp else timestamp}</small>
                </div>
                <div style="color: #333; margin-left: 20px;">{message}</div>
            </div>
            """)
        
        html_parts.append("</div></div>")
        
        return ''.join(html_parts)
    
    def _generate_agent_decisions_display(self, agent_decisions: Dict) -> str:
        """Generate agent decisions display."""
        if not agent_decisions:
            return "<p>No agent decisions data available</p>"
        
        html_parts = []
        html_parts.append("""
        <div style="margin: 20px 0;">
            <h3>🔍 Agent Decisions & Analysis</h3>
        """)
        
        for agent_name, decision in agent_decisions.items():
            if not isinstance(decision, dict):
                continue
                
            html_parts.append(f"""
            <div style="margin: 15px 0; padding: 15px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px;">
                <h4 style="color: #495057; margin-top: 0;">🤖 {agent_name.replace('_', ' ').title()}</h4>
            """)
            
            # Display key decision points
            for key, value in decision.items():
                if key == 'reasoning':
                    html_parts.append(f"<p><strong>💡 Reasoning:</strong> {value}</p>")
                elif key == 'confidence':
                    html_parts.append(f"<p><strong>🎯 Confidence:</strong> {value:.1%}</p>")
                elif isinstance(value, (int, float)):
                    if key.endswith('_score') or 'score' in key:
                        html_parts.append(f"<p><strong>📊 {key.replace('_', ' ').title()}:</strong> {value:.3f}</p>")
                    else:
                        html_parts.append(f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>")
                elif isinstance(value, list) and len(value) <= 5:
                    html_parts.append(f"<p><strong>{key.replace('_', ' ').title()}:</strong> {', '.join(map(str, value))}</p>")
                elif isinstance(value, bool):
                    icon = "✅" if value else "❌"
                    html_parts.append(f"<p><strong>{key.replace('_', ' ').title()}:</strong> {icon} {value}</p>")
                elif isinstance(value, str) and len(value) < 100:
                    html_parts.append(f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>")
            
            html_parts.append("</div>")
        
        html_parts.append("</div>")
        
        return ''.join(html_parts)

# Initialize the UI manager
ui_manager = PortfolioUIManager()

def construct_portfolio_sync(*args):
    """Synchronous wrapper for portfolio construction."""
    return asyncio.run(ui_manager.construct_portfolio_async(*args))

def export_recommendation():
    """Export current recommendation as JSON."""
    if not ui_manager.current_recommendation:
        return "No recommendation to export"
    
    try:
        # Convert recommendation to dict for JSON serialization
        export_data = {
            "timestamp": ui_manager.current_recommendation.timestamp.isoformat(),
            "market_regime": ui_manager.current_recommendation.market_regime.value,
            "expected_return": ui_manager.current_recommendation.expected_return,
            "expected_volatility": ui_manager.current_recommendation.expected_volatility,
            "confidence": ui_manager.current_recommendation.confidence,
            "diversification_score": ui_manager.current_recommendation.diversification_score,
            "cash_allocation": ui_manager.current_recommendation.cash_allocation,
            "allocations": [
                {
                    "symbol": alloc.symbol,
                    "target_weight": alloc.target_weight,
                    "asset_class": alloc.asset_class.value,
                    "expected_return": alloc.expected_return,
                    "confidence": alloc.confidence,
                    "risk_score": alloc.risk_score
                }
                for alloc in ui_manager.current_recommendation.allocations
            ],
            "reasoning": ui_manager.current_recommendation.reasoning
        }
        
        filename = f"portfolio_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return f"✅ Recommendation exported to {filename}"
        
    except Exception as e:
        return f"❌ Export failed: {e}"

# Create Gradio interface
def create_portfolio_interface():
    """Create the main Gradio interface."""
    
    with gr.Blocks(title="🎯 Multi-Agent Portfolio Management", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🎯 Multi-Agent Portfolio Management System
        
        **Intelligent portfolio construction using AI agents with chain of thought visualization**
        
        This system uses multiple specialized AI agents to analyze market conditions, evaluate stocks, 
        and construct optimal portfolios. Watch the agents work together in real-time!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📝 Portfolio Parameters")
                
                symbols_input = gr.Textbox(
                    label="Candidate Symbols",
                    placeholder="AAPL, MSFT, GOOGL, JPM, JNJ, UNH, PFE, XOM, KO, PG",
                    info="Enter stock symbols separated by commas or spaces",
                    value="AAPL, MSFT, GOOGL, JPM, JNJ, UNH"
                )
                
                portfolio_value = gr.Number(
                    label="Portfolio Value ($)",
                    value=100000,
                    minimum=1000,
                    info="Total portfolio value in USD"
                )
                
                risk_profile = gr.Dropdown(
                    label="Risk Profile",
                    choices=["conservative", "moderate", "aggressive", "tactical"],
                    value="moderate",
                    info="Risk preference for portfolio construction"
                )
                
                max_positions = gr.Slider(
                    label="Maximum Positions",
                    minimum=3,
                    maximum=20,
                    value=8,
                    step=1,
                    info="Maximum number of positions in portfolio"
                )
                
                construct_btn = gr.Button("🚀 Construct Portfolio", variant="primary", size="lg")
                export_btn = gr.Button("💾 Export Recommendation", variant="secondary")
                
                export_status = gr.Textbox(label="Export Status", interactive=False)
        
        with gr.Column(scale=2):
            gr.Markdown("## 📊 Portfolio Results")
            
            with gr.Tabs():
                with gr.TabItem("📈 Portfolio Overview"):
                    portfolio_overview = gr.HTML(label="Portfolio Overview")
                    
                with gr.TabItem("📋 Allocations"):
                    allocations_table = gr.HTML(label="Portfolio Allocations")
                    
                with gr.TabItem("🧠 Chain of Thought"):
                    chain_of_thought = gr.HTML(label="Agent Decision Process")
                    
                with gr.TabItem("🔍 Agent Decisions"):
                    agent_decisions = gr.HTML(label="Detailed Agent Analysis")
        
        # Event handlers
        construct_btn.click(
            fn=construct_portfolio_sync,
            inputs=[symbols_input, portfolio_value, risk_profile, max_positions],
            outputs=[portfolio_overview, allocations_table, chain_of_thought, agent_decisions]
        )
        
        export_btn.click(
            fn=export_recommendation,
            inputs=[],
            outputs=[export_status]
        )
        
        # Example section
        gr.Markdown("""
        ## 💡 How It Works
        
        1. **🎯 Input Parameters**: Specify candidate stocks, portfolio value, and risk preferences
        2. **🤖 Agent Analysis**: Watch AI agents analyze market conditions, sectors, momentum, value, and ML predictions
        3. **⚖️ Risk Management**: Agents optimize allocations based on diversification and risk constraints
        4. **📊 Portfolio Construction**: Final recommendation with detailed reasoning and confidence scores
        
        ### 🧠 Agent Specializations:
        - **Market Regime Agent**: Identifies current market conditions (bull/bear/sideways/volatility)
        - **Sector Rotation Agent**: Analyzes sector opportunities and rotations
        - **Risk Parity Agent**: Calculates risk-weighted allocations and diversification
        - **Momentum Agent**: Evaluates momentum signals across multiple timeframes
        - **Value Agent**: Identifies value opportunities using fundamental analysis
        - **ML Agent**: Provides machine learning predictions using H2O.ai
        """)
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_portfolio_interface()
    
    # Launch with sharing enabled for remote access
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        debug=True,
        show_error=True
    )