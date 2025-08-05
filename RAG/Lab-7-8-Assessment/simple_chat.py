#!/usr/bin/env python3
"""
Simple chat interface for the RAG system
Similar to the original course frontend
"""

import gradio as gr
import requests
import json
import time
import subprocess
import os
from typing import Iterator

# Global variables
server_process = None
server_running = False
base_url = "http://localhost:9012"

def start_server_if_needed():
    """Start server if not running"""
    global server_process, server_running
    
    if server_running:
        return True
        
    try:
        # Set working directory and environment
        os.chdir("/Users/zac/Desktop/Education/GATech/CS8001/CS8001-public/RAG/Lab-7-8-Assessment")
        os.environ['NVIDIA_API_KEY'] = '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'
        
        # Start server process
        server_process = subprocess.Popen(
            ['python', 'server_app.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(5)
        
        # Test if server is responsive
        try:
            response = requests.get(f"{base_url}/docs", timeout=5)
            if response.status_code == 200:
                server_running = True
                print("✅ Server started successfully!")
                return True
        except:
            pass
            
        return False
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def rag_chat_response(message: str, history: list) -> Iterator[str]:
    """Generate RAG response with streaming"""
    
    # Start server if needed
    if not start_server_if_needed():
        yield "❌ Could not start server. Please check the logs."
        return
    
    try:
        # Step 1: Retrieve documents
        retrieval_response = requests.post(
            f"{base_url}/retriever/invoke",
            json={"input": message},
            timeout=30
        )
        
        if retrieval_response.status_code != 200:
            yield f"❌ Retrieval failed: {retrieval_response.status_code}"
            return
        
        docs = retrieval_response.json()['output']
        
        # Step 2: Format context
        context = ""
        for doc in docs:
            title = doc.get('metadata', {}).get('Title', 'Document')
            content = doc.get('page_content', str(doc))
            context += f"[Quote from {title}] {content}\n"
        
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
            # Simulate streaming by yielding chunks
            answer = generation_response.json()['output']
            
            # Stream the response word by word for better UX
            words = answer.split()
            current_response = ""
            
            for word in words:
                current_response += word + " "
                yield current_response
                time.sleep(0.05)  # Small delay for streaming effect
                
        else:
            yield f"❌ Generation failed: {generation_response.status_code}"
            
    except Exception as e:
        yield f"❌ Error: {str(e)}"

def basic_chat_response(message: str, history: list) -> Iterator[str]:
    """Generate basic chat response"""
    
    # Start server if needed
    if not start_server_if_needed():
        yield "❌ Could not start server. Please check the logs."
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
                yield current_response
                time.sleep(0.03)  # Faster streaming for basic chat
                
        else:
            yield f"❌ Error {response.status_code}: {response.text}"
            
    except Exception as e:
        yield f"❌ Request failed: {str(e)}"

def create_chat_interface():
    """Create the main chat interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .chat-message {
        padding: 10px !important;
        margin: 5px !important;
        border-radius: 10px !important;
    }
    """
    
    with gr.Blocks(css=css, title="RAG Chat Assistant", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🤖 RAG Chat Assistant
        ### Powered by NVIDIA AI and LangChain
        
        Choose between **Basic Chat** (direct LLM) or **RAG Chat** (document-enhanced responses)
        """)
        
        with gr.Tabs():
            # RAG Chat Tab
            with gr.TabItem("📚 RAG Chat", id="rag_chat"):
                gr.Markdown("### Document-Enhanced Chat")
                gr.Markdown("Ask questions and get responses based on the document knowledge base")
                
                rag_chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                    avatar_images=("🧑‍💻", "🤖"),
                    bubble_full_width=False
                )
                
                with gr.Row():
                    rag_msg = gr.Textbox(
                        placeholder="Ask me about transformers, BERT, RAG, or any topic in the documents...",
                        container=False,
                        scale=7
                    )
                    rag_submit = gr.Button("Send", variant="primary", scale=1)
                
                gr.Examples(
                    examples=[
                        "What are transformers and how do they work?",
                        "Explain BERT and its bidirectional training",
                        "How does retrieval-augmented generation work?",
                        "What's the difference between BERT and traditional transformers?",
                        "Tell me about attention mechanisms in deep learning"
                    ],
                    inputs=rag_msg
                )
            
            # Basic Chat Tab  
            with gr.TabItem("💬 Basic Chat", id="basic_chat"):
                gr.Markdown("### Direct LLM Chat")
                gr.Markdown("Direct conversation with the language model (no document context)")
                
                basic_chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                    avatar_images=("🧑‍💻", "🤖"),
                    bubble_full_width=False
                )
                
                with gr.Row():
                    basic_msg = gr.Textbox(
                        placeholder="Chat with the AI assistant...",
                        container=False,
                        scale=7
                    )
                    basic_submit = gr.Button("Send", variant="primary", scale=1)
                
                gr.Examples(
                    examples=[
                        "Hello! How are you today?",
                        "Can you help me with a coding problem?",
                        "What's the weather like?",
                        "Tell me a joke",
                        "Explain quantum computing"
                    ],
                    inputs=basic_msg
                )
        
        # Status indicator
        with gr.Row():
            status = gr.Markdown("🔄 **Status:** Starting up... Server will auto-start on first message.")
        
        # Event handlers for RAG chat
        rag_submit.click(
            fn=rag_chat_response,
            inputs=[rag_msg, rag_chatbot],
            outputs=rag_chatbot,
            show_progress=True
        ).then(
            lambda: "",
            outputs=rag_msg
        )
        
        rag_msg.submit(
            fn=rag_chat_response,
            inputs=[rag_msg, rag_chatbot],
            outputs=rag_chatbot,
            show_progress=True
        ).then(
            lambda: "",
            outputs=rag_msg
        )
        
        # Event handlers for basic chat
        basic_submit.click(
            fn=basic_chat_response,
            inputs=[basic_msg, basic_chatbot],
            outputs=basic_chatbot,
            show_progress=True
        ).then(
            lambda: "",
            outputs=basic_msg
        )
        
        basic_msg.submit(
            fn=basic_chat_response,
            inputs=[basic_msg, basic_chatbot],
            outputs=basic_chatbot,
            show_progress=True
        ).then(
            lambda: "",
            outputs=basic_msg
        )
    
    return demo

if __name__ == "__main__":
    # Set working directory
    os.chdir("/Users/zac/Desktop/Education/GATech/CS8001/CS8001-public/RAG/Lab-7-8-Assessment")
    
    print("🚀 Starting RAG Chat Interface...")
    print("📍 Interface will be available at: http://127.0.0.1:7861")
    print("🤖 Server will auto-start when you send your first message")
    print("💡 Try both Basic Chat and RAG Chat to see the difference!")
    
    demo = create_chat_interface()
    demo.launch(
        share=False,
        server_name="127.0.0.1", 
        server_port=7861,
        show_api=False,
        inbrowser=True
    )