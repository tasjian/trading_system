#!/usr/bin/env python3
"""
ICIJ Offshore Leaks Chat Interface
Specialized Gradio interface for investigating offshore financial data
"""

import gradio as gr
import requests
import json
import time
import subprocess
import os

# Global variables
server_process = None
server_running = False
base_url = "http://localhost:9012"

def start_server_if_needed():
    """Start ICIJ server if not running"""
    global server_process, server_running
    
    if server_running:
        return True
        
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
        time.sleep(8)
        
        # Test if server is responsive
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                server_running = True
                print("✅ ICIJ Server started successfully!")
                return True
        except:
            pass
            
        return False
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def icij_rag_chat_stream(message, history):
    """Generate ICIJ RAG response for chat interface"""
    
    # Start server if needed
    if not start_server_if_needed():
        yield history + [[message, "❌ Could not start ICIJ server. Please check the logs."]]
        return
    
    try:
        # Step 1: Retrieve offshore documents
        retrieval_response = requests.post(
            f"{base_url}/retriever/invoke",
            json={"input": message},
            timeout=30
        )
        
        if retrieval_response.status_code != 200:
            yield history + [[message, f"❌ Retrieval failed: {retrieval_response.status_code}"]]
            return
        
        docs = retrieval_response.json()['output']
        
        # Step 2: Format context with ICIJ-specific information
        context = ""
        for doc in docs:
            title = doc.get('metadata', {}).get('title', 'Offshore Document')
            doc_type = doc.get('metadata', {}).get('type', 'document')
            
            if doc_type == 'entity':
                jurisdiction = doc.get('metadata', {}).get('jurisdiction', 'Unknown')
                source = doc.get('metadata', {}).get('source', 'Unknown')
                context += f"[{source} - {title} in {jurisdiction}] "
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
        
        # Step 3: Generate response
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
            
            # Stream the response word by word
            words = answer.split()
            current_response = ""
            
            for word in words:
                current_response += word + " "
                yield history + [[message, current_response]]
                time.sleep(0.04)
                
        else:
            yield history + [[message, f"❌ Generation failed: {generation_response.status_code}"]]
            
    except Exception as e:
        yield history + [[message, f"❌ Error: {str(e)}"]]

def basic_chat_stream(message, history):
    """Generate basic chat response"""
    
    # Start server if needed
    if not start_server_if_needed():
        yield history + [[message, "❌ Could not start server. Please check the logs."]]
        return
    
    try:
        response = requests.post(
            f"{base_url}/basic_chat/invoke",
            json={"input": message},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()['output']
            if isinstance(result, dict):
                answer = result.get('content', str(result))
            else:
                answer = str(result)
            
            # Stream the response
            words = answer.split()
            current_response = ""
            
            for word in words:
                current_response += word + " "
                yield history + [[message, current_response]]
                time.sleep(0.03)
                
        else:
            yield history + [[message, f"❌ Error {response.status_code}: {response.text}"]]
            
    except Exception as e:
        yield history + [[message, f"❌ Request failed: {str(e)}"]]

def get_server_stats():
    """Get server statistics"""
    try:
        if not server_running:
            return "Server not running"
            
        response = requests.get(f"{base_url}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            
            output = "📊 **ICIJ Offshore Leaks Database Statistics**\n\n"
            
            # Entities
            entities = stats['entities']
            output += f"**🏢 Offshore Entities:** {entities['total']}\n"
            output += f"- **Top Jurisdictions:** {', '.join(list(entities['by_jurisdiction'].keys())[:5])}\n"
            output += f"- **Entity Types:** {', '.join(entities['by_type'].keys())}\n"
            output += f"- **Data Sources:** {', '.join(entities['by_source'].keys())}\n\n"
            
            # Officers
            officers = stats['officers']
            output += f"**👥 Officers/Individuals:** {officers['total']}\n"
            output += f"- **Top Countries:** {', '.join(list(officers['by_country'].keys())[:5])}\n"
            output += f"- **Roles:** {', '.join(officers['by_role'].keys())}\n\n"
            
            # Graph
            graph = stats['graph']
            output += f"**🔗 Network Analysis:**\n"
            output += f"- **Total Nodes:** {graph['nodes']}\n"
            output += f"- **Total Connections:** {graph['edges']}\n"
            
            return output
        else:
            return f"Error fetching stats: {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def create_icij_interface():
    """Create the ICIJ investigation interface"""
    
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .investigation-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css, title="ICIJ Offshore Leaks Investigation") as demo:
        
        gr.HTML("""
        <div class="investigation-header">
            <h1>🕵️ ICIJ Offshore Leaks Investigation Assistant</h1>
            <h3>Explore offshore financial networks with AI-powered analysis</h3>
            <p>Based on the International Consortium of Investigative Journalists (ICIJ) Offshore Leaks database</p>
        </div>
        """)
        
        with gr.Tabs():
            # ICIJ Investigation Tab
            with gr.TabItem("🔍 Offshore Investigation"):
                gr.Markdown("### Investigate offshore financial networks")
                gr.Markdown("Ask questions about offshore entities, beneficial owners, shell companies, and financial structures")
                
                icij_chatbot = gr.Chatbot(height=500, label="Investigation Assistant")
                icij_msg = gr.Textbox(
                    placeholder="e.g., 'Show me offshore companies in Panama with beneficial owners from the US'",
                    label="Investigation Query"
                )
                
                with gr.Row():
                    gr.Examples(
                        examples=[
                            "What offshore companies are incorporated in Panama?",
                            "Show me beneficial owners and directors from the UK",
                            "Tell me about entities in the British Virgin Islands",
                            "What was revealed in the Paradise Papers investigation?",
                            "Find shell companies with connections to the Cayman Islands",
                            "Show me the network of relationships around offshore entities",
                            "What types of offshore structures are most common?",
                            "Which investigations exposed the most entities?"
                        ],
                        inputs=icij_msg,
                        label="Investigation Examples"
                    )
            
            # Basic Chat Tab  
            with gr.TabItem("💬 General Chat"):
                gr.Markdown("### General AI Assistant")
                gr.Markdown("General conversation without offshore data context")
                
                basic_chatbot = gr.Chatbot(height=500)
                basic_msg = gr.Textbox(
                    placeholder="Ask any general question...",
                    label="Your message"
                )
                
                gr.Examples(
                    examples=[
                        "What is offshore banking?",
                        "Explain shell companies and their purposes",
                        "What are tax havens?",
                        "How do offshore financial structures work?",
                        "What is financial journalism?"
                    ],
                    inputs=basic_msg
                )
            
            # Database Statistics Tab
            with gr.TabItem("📊 Database Stats"):
                gr.Markdown("### ICIJ Offshore Leaks Database Statistics")
                
                stats_button = gr.Button("🔄 Refresh Statistics", variant="primary")
                stats_output = gr.Markdown(value="Click 'Refresh Statistics' to load database information")
        
        # Status
        gr.Markdown("**Status:** Server will auto-start on first query. Initial startup takes ~10 seconds.")
        
        # Event handlers for ICIJ investigation chat
        icij_msg.submit(
            fn=icij_rag_chat_stream,
            inputs=[icij_msg, icij_chatbot],
            outputs=icij_chatbot,
            show_progress=True
        ).then(
            lambda: "",
            outputs=icij_msg
        )
        
        # Event handlers for basic chat
        basic_msg.submit(
            fn=basic_chat_stream,
            inputs=[basic_msg, basic_chatbot],
            outputs=basic_chatbot,
            show_progress=True
        ).then(
            lambda: "",
            outputs=basic_msg
        )
        
        # Stats refresh
        stats_button.click(
            fn=get_server_stats,
            outputs=stats_output
        )
    
    return demo

if __name__ == "__main__":
    # Set working directory
    os.chdir("/Users/zac/Desktop/Education/GATech/CS8001/CS8001-public/RAG/Lab-7-8-Assessment")
    
    print("🕵️ Starting ICIJ Offshore Leaks Investigation Interface...")
    print("📍 Interface will be available at: http://127.0.0.1:7864")
    print("🏛️ Server will auto-start when you send your first investigation query")
    print("💡 This system uses graph-enhanced RAG for offshore financial investigations!")
    
    demo = create_icij_interface()
    demo.launch(
        share=False,
        server_name="127.0.0.1", 
        server_port=7864,
        inbrowser=False
    )