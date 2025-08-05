#!/usr/bin/env python3
"""
Enhanced ICIJ Offshore Leaks Investigation Interface
Advanced Gradio UI for financial investigations with improved UX
"""

import gradio as gr
import requests
import json
import time
import subprocess
import os
import webbrowser
import threading

# Global variables
server_process = None
server_running = False
base_url = "http://localhost:9012"

def start_server_if_needed():
    """Start ICIJ server if not running"""
    global server_process, server_running
    
    if server_running:
        return True, "Server already running"
        
    try:
        # Set working directory and environment
        os.chdir("/Users/zac/Desktop/Education/GATech/CS8001/CS8001-public/RAG/Lab-7-8-Assessment")
        os.environ['NVIDIA_API_KEY'] = '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'
        
        # Start server process
        server_process = subprocess.Popen(
            ['python', 'icij_server_app.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(10)
        
        # Test if server is responsive
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
            if response.status_code == 200:
                server_running = True
                return True, "✅ ICIJ Server started successfully!"
        except:
            pass
            
        return False, "❌ Server failed to respond"
        
    except Exception as e:
        return False, f"❌ Error starting server: {str(e)}"

def icij_investigation_chat(message, history):
    """Enhanced ICIJ investigation chat with status updates"""
    
    # Start server if needed
    success, status_msg = start_server_if_needed()
    if not success:
        yield history + [[message, status_msg]]
        return
    
    try:
        # Step 1: Show retrieval progress
        yield history + [[message, "🔍 Searching offshore leaks database..."]]
        
        # Retrieve offshore documents
        retrieval_response = requests.post(
            f"{base_url}/retriever/invoke",
            json={"input": {"input": message}},
            timeout=30
        )
        
        if retrieval_response.status_code != 200:
            yield history + [[message, f"❌ Database search failed: {retrieval_response.status_code}"]]
            return
        
        docs = retrieval_response.json()['output']
        
        # Step 2: Show analysis progress
        yield history + [[message, f"📊 Analyzing {len(docs)} offshore documents..."]]
        
        # Format context with enhanced ICIJ information
        context = ""
        doc_types = {"entity": 0, "officer": 0, "investigation": 0}
        
        for doc in docs:
            title = doc.get('metadata', {}).get('title', 'Offshore Document')
            doc_type = doc.get('metadata', {}).get('type', 'document')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            if doc_type == 'entity':
                jurisdiction = doc.get('metadata', {}).get('jurisdiction', 'Unknown')
                source = doc.get('metadata', {}).get('source', 'Unknown')
                entity_type = doc.get('metadata', {}).get('entity_type', 'Unknown')
                context += f"[{source} - {entity_type}: {title} in {jurisdiction}] "
            elif doc_type == 'officer':
                country = doc.get('metadata', {}).get('country', 'Unknown')
                role = doc.get('metadata', {}).get('role', 'Unknown')
                context += f"[Individual: {title} - {role} from {country}] "
            elif doc_type == 'investigation':
                source = doc.get('metadata', {}).get('source', 'Unknown')
                context += f"[Investigation: {source}] "
            else:
                context += f"[{title}] "
                
            content = doc.get('page_content', str(doc))
            context += content + "\n\n"
        
        # Step 3: Show generation progress
        doc_summary = f"Found {doc_types.get('entity', 0)} entities, {doc_types.get('officer', 0)} individuals, {doc_types.get('investigation', 0)} investigations"
        yield history + [[message, f"🤖 Generating investigation report based on {doc_summary}..."]]
        
        # Generate enhanced investigation response
        generation_response = requests.post(
            f"{base_url}/generator/invoke",
            json={
                "input": {
                    "input": message,
                    "context": context
                }
            },
            timeout=30
        )
        
        if generation_response.status_code == 200:
            answer = generation_response.json()['output']
            
            # Add investigation metadata to response
            investigation_header = f"📋 **Investigation Report**\n"
            investigation_header += f"*Query:* {message}\n"
            investigation_header += f"*Sources analyzed:* {doc_summary}\n"
            investigation_header += f"*Database:* ICIJ Offshore Leaks\n\n---\n\n"
            
            full_response = investigation_header + answer
            
            # Stream the response with typing effect
            words = full_response.split()
            current_response = ""
            
            for word in words:
                current_response += word + " "
                yield history + [[message, current_response]]
                time.sleep(0.03)
                
        else:
            yield history + [[message, f"❌ Investigation analysis failed: {generation_response.status_code}"]]
            
    except Exception as e:
        yield history + [[message, f"❌ Investigation error: {str(e)}"]]

def get_detailed_stats():
    """Get enhanced server statistics with investigation focus"""
    try:
        if not server_running:
            return "🔴 **Server Status:** Not running\n\nClick 'Start Server' to begin investigations."
            
        response = requests.get(f"{base_url}/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            
            output = "# 🕵️ **ICIJ Offshore Leaks Investigation Dashboard**\n\n"
            output += "## 📊 **Database Overview**\n\n"
            
            # Entities section
            entities = stats['entities']
            output += f"### 🏢 **Offshore Entities: {entities['total']}**\n"
            output += f"**Top Jurisdictions:**\n"
            for jurisdiction, count in list(entities['by_jurisdiction'].items())[:5]:
                output += f"- 🏝️ {jurisdiction}: {count} entities\n"
            
            output += f"\n**Entity Types:**\n"
            for entity_type, count in entities['by_type'].items():
                icon = {"Company": "🏢", "Trust": "🏛️", "Foundation": "🏛️", "Other": "📄"}.get(entity_type, "📄")
                output += f"- {icon} {entity_type}: {count}\n"
            
            output += f"\n**Investigation Sources:**\n"
            for source, count in entities['by_source'].items():
                icon = {"Panama Papers": "📰", "Paradise Papers": "📑", "Pandora Papers": "📋", "Offshore Leaks": "📊", "Bahamas Leaks": "📈"}.get(source, "📄")
                output += f"- {icon} {source}: {count} entities\n"
            
            # Officers section
            officers = stats['officers']
            output += f"\n### 👥 **Individuals & Officers: {officers['total']}**\n"
            output += f"**Top Countries:**\n"
            for country, count in list(officers['by_country'].items())[:5]:
                flag = {"UK": "🇬🇧", "USA": "🇺🇸", "Russia": "🇷🇺", "China": "🇨🇳", "Germany": "🇩🇪", "France": "🇫🇷", "Brazil": "🇧🇷", "India": "🇮🇳"}.get(country, "🌍")
                output += f"- {flag} {country}: {count} individuals\n"
            
            output += f"\n**Roles:**\n"
            for role, count in officers['by_role'].items():
                icon = {"Director": "👨‍💼", "Beneficial Owner": "💰", "Shareholder": "📈", "Nominee": "📝", "Secretary": "📋"}.get(role, "👤")
                output += f"- {icon} {role}: {count}\n"
            
            # Network analysis
            graph = stats['graph']
            output += f"\n### 🔗 **Network Analysis**\n"
            output += f"- **Total Nodes:** {graph['nodes']:,}\n"
            output += f"- **Total Relationships:** {graph['edges']:,}\n"
            output += f"- **Network Density:** {(graph['edges'] / max(graph['nodes'], 1)):.2f} connections per node\n"
            
            # Investigation tips
            output += f"\n### 💡 **Investigation Tips**\n"
            output += f"- Search by **jurisdiction** (e.g., 'Panama', 'British Virgin Islands')\n"
            output += f"- Look for **roles** (e.g., 'beneficial owners', 'directors')\n"
            output += f"- Explore **investigations** (e.g., 'Paradise Papers', 'Panama Papers')\n"
            output += f"- Find **connections** (e.g., 'entities connected to [person name]')\n"
            
            return output
        else:
            return f"❌ **Error:** Could not fetch statistics (Status: {response.status_code})"
            
    except Exception as e:
        return f"❌ **Connection Error:** {str(e)}\n\nPlease ensure the server is running."

def create_enhanced_interface():
    """Create enhanced ICIJ investigation interface"""
    
    # Enhanced CSS with investigation theme
    css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: auto !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .investigation-header {
        background: linear-gradient(135deg, #1a2a6c 0%, #b21f1f 50%, #fdbb2d 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        text-align: center;
    }
    
    .stats-panel {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    
    .investigation-examples {
        background: #f8f9ff;
        border-left: 4px solid #4f46e5;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    .status-indicator {
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: bold;
    }
    
    .status-success {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    .status-warning {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    
    .chatbot-container .message.user {
        background: #e0f2fe !important;
        border-left: 3px solid #0288d1;
    }
    
    .chatbot-container .message.bot {
        background: #f3e5f5 !important;
        border-left: 3px solid #7b1fa2;
    }
    """
    
    with gr.Blocks(css=css, title="ICIJ Offshore Leaks Investigation", theme=gr.themes.Soft()) as demo:
        
        gr.HTML("""
        <div class="investigation-header">
            <h1>🕵️‍♂️ ICIJ Offshore Leaks Investigation Center</h1>
            <h2>AI-Powered Financial Investigation Platform</h2>
            <p style="font-size: 1.1em; margin-top: 10px;">
                🌐 Explore offshore financial networks • 🔗 Analyze entity relationships • 📊 Investigate suspicious patterns
            </p>
            <p style="opacity: 0.9; margin-top: 15px;">
                Powered by Graph-Enhanced RAG | Data: ICIJ Offshore Leaks Database
            </p>
        </div>
        """)
        
        with gr.Tabs():
            # Main Investigation Tab
            with gr.TabItem("🔍 Investigation Center", id="investigation"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 🕵️ Start Your Investigation")
                        gr.Markdown("Ask specific questions about offshore entities, beneficial owners, or financial networks")
                        
                        investigation_chatbot = gr.Chatbot(
                            height=600,
                            label="Investigation Assistant",
                            show_label=False,
                            container=True,
                            bubble_full_width=False
                        )
                        
                        with gr.Row():
                            investigation_msg = gr.Textbox(
                                placeholder="e.g., 'Find all beneficial owners from the UK connected to Panama entities'",
                                label="Investigation Query",
                                lines=2,
                                max_lines=5
                            )
                            send_btn = gr.Button("🔍 Investigate", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Investigation Toolkit")
                        
                        # Quick investigation templates
                        gr.Markdown("#### 🎯 Quick Investigations")
                        quick_investigations = [
                            ("🏢 Entity Search", "Show me offshore companies in [jurisdiction]"),
                            ("👥 People Search", "Find beneficial owners from [country]"),
                            ("🔗 Network Analysis", "Show connections for [entity/person name]"),
                            ("📊 Investigation Review", "What was revealed in [investigation name]"),
                            ("🏝️ Jurisdiction Analysis", "Analyze entities in [tax haven]"),
                            ("📈 Pattern Detection", "Find suspicious offshore structures"),
                        ]
                        
                        for title, template in quick_investigations:
                            gr.Button(title, size="sm").click(
                                lambda t=template: t,
                                outputs=investigation_msg
                            )
                        
                        # Server status
                        gr.Markdown("#### 🖥️ Server Status")
                        server_status = gr.Markdown("🔴 **Status:** Not started")
                        
                        start_server_btn = gr.Button("🚀 Start Investigation Server", variant="secondary")
                        
                # Investigation Examples
                with gr.Row():
                    gr.Examples(
                        examples=[
                            "What offshore companies are incorporated in Panama with US beneficial owners?",
                            "Show me directors and shareholders from the UK in the Paradise Papers",
                            "Find entities in the British Virgin Islands connected to shell companies",
                            "Analyze the network of relationships around offshore trusts in the Cayman Islands",
                            "What patterns emerge from the Pandora Papers investigation?",
                            "Compare offshore structures revealed in Panama Papers vs Paradise Papers",
                            "Find the most connected individuals in the offshore network",
                            "Show me entities with multiple beneficial owners from different countries"
                        ],
                        inputs=investigation_msg,
                        label="🎯 Investigation Examples - Click to Try"
                    )
            
            # Database Analytics Tab
            with gr.TabItem("📊 Database Analytics"):
                gr.Markdown("### 📈 ICIJ Offshore Leaks Database Analytics")
                
                with gr.Row():
                    refresh_stats_btn = gr.Button("🔄 Refresh Analytics", variant="primary", size="lg")
                    export_btn = gr.Button("📤 Export Report", variant="secondary")
                
                stats_display = gr.Markdown(value="Click 'Refresh Analytics' to load database statistics...")
            
            # Investigation Guide Tab
            with gr.TabItem("📚 Investigation Guide"):
                gr.Markdown("""
                # 🕵️ **Offshore Investigation Guide**
                
                ## 🎯 **How to Conduct Effective Investigations**
                
                ### **1. Entity Investigations**
                - **Search by jurisdiction:** "Show me companies in Panama"
                - **Filter by entity type:** "Find all trusts in British Virgin Islands"
                - **Status analysis:** "List dissolved entities in the Cayman Islands"
                
                ### **2. People & Relationships**
                - **Beneficial owner search:** "Find beneficial owners from [country]"
                - **Role-based queries:** "Show all directors connected to multiple entities"
                - **Cross-reference individuals:** "Find people appearing in multiple investigations"
                
                ### **3. Network Analysis**
                - **Connection mapping:** "Show the network around [entity name]"
                - **Multi-hop relationships:** "Find entities 2 degrees away from [person]"
                - **Cluster analysis:** "Identify groups of related offshore structures"
                
                ### **4. Investigation Sources**
                - **Panama Papers:** Focus on Mossack Fonseca client data
                - **Paradise Papers:** Appleby, Estera, and other offshore service providers
                - **Pandora Papers:** 14 offshore service providers
                - **Offshore Leaks:** ICIJ's original offshore investigation
                - **Bahamas Leaks:** Corporate registries from the Bahamas
                
                ### **5. Advanced Queries**
                - **Temporal analysis:** "Show entities incorporated between 2010-2015"
                - **Pattern detection:** "Find common structures across investigations"
                - **Compliance checks:** "Identify entities with similar addresses"
                
                ## 🔍 **Investigation Keywords**
                
                **Jurisdictions:** Panama, British Virgin Islands, Cayman Islands, Bermuda, Malta, Cyprus
                
                **Entity Types:** Company, Trust, Foundation, Partnership
                
                **Roles:** Beneficial Owner, Director, Shareholder, Nominee, Secretary
                
                **Investigations:** Panama Papers, Paradise Papers, Pandora Papers, Offshore Leaks, Bahamas Leaks
                
                ## ⚠️ **Important Notes**
                
                - This system uses **mock data** for educational purposes
                - All entities and individuals are **fictional**
                - Real ICIJ data requires proper authorization and ethical use
                - Always verify information through multiple sources
                """)
        
        # Event handlers
        def update_server_status():
            if server_running:
                return "🟢 **Status:** Investigation server running and ready"
            else:
                return "🔴 **Status:** Server not started - Click 'Start Investigation Server'"
        
        # Investigation chat
        send_btn.click(
            fn=icij_investigation_chat,
            inputs=[investigation_msg, investigation_chatbot],
            outputs=investigation_chatbot,
            show_progress=True
        ).then(
            lambda: "",
            outputs=investigation_msg
        )
        
        investigation_msg.submit(
            fn=icij_investigation_chat,
            inputs=[investigation_msg, investigation_chatbot],
            outputs=investigation_chatbot,
            show_progress=True
        ).then(
            lambda: "",
            outputs=investigation_msg
        )
        
        # Server management
        start_server_btn.click(
            fn=lambda: start_server_if_needed()[1],
            outputs=server_status
        )
        
        # Stats refresh
        refresh_stats_btn.click(
            fn=get_detailed_stats,
            outputs=stats_display
        )
        
        # Auto-update server status
        demo.load(
            fn=update_server_status,
            outputs=server_status
        )
    
    return demo

def open_in_firefox(url):
    """Open URL in Firefox after a delay"""
    time.sleep(2)
    try:
        webbrowser.get('firefox').open(url)
        print(f"🦊 Opened {url} in Firefox")
    except:
        try:
            webbrowser.open(url)
            print(f"🌐 Opened {url} in default browser")
        except:
            print(f"⚠️ Could not open browser. Please visit: {url}")

if __name__ == "__main__":
    # Set working directory
    os.chdir("/Users/zac/Desktop/Education/GATech/CS8001/CS8001-public/RAG/Lab-7-8-Assessment")
    
    print("🕵️ Starting Enhanced ICIJ Investigation Interface...")
    print("📍 Interface will be available at: http://127.0.0.1:7865")
    print("🦊 Opening in Firefox...")
    print("🏛️ Investigation server will auto-start when needed")
    print("💡 This system uses graph-enhanced RAG for offshore financial investigations!")
    
    # Start Firefox opener in background
    url = "http://127.0.0.1:7865"
    threading.Thread(target=open_in_firefox, args=(url,), daemon=True).start()
    
    demo = create_enhanced_interface()
    demo.launch(
        share=False,
        server_name="127.0.0.1", 
        server_port=7865,
        inbrowser=False,
        show_api=False
    )