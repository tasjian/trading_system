#!/usr/bin/env python3
"""
Updated Gradio interface to test the RAG functionality
This provides a web UI to interact with the LangServe endpoints
"""

import gradio as gr
import requests
import json
import time
import threading
import subprocess
import os
from typing import Iterator

class RAGTester:
    def __init__(self):
        self.server_process = None
        self.server_running = False
        self.base_url = "http://localhost:9012"
        
    def start_server(self):
        """Start the LangServe server in background"""
        if self.server_running:
            return "Server already running"
            
        try:
            # Set working directory and environment
            os.chdir("/Users/zac/Desktop/Education/GATech/CS8001/CS8001-public/RAG/Lab-7-8-Assessment")
            os.environ['NVIDIA_API_KEY'] = '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'
            
            # Start server process
            self.server_process = subprocess.Popen(
                ['python', 'server_app.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(3)
            
            # Test if server is responsive
            try:
                response = requests.get(f"{self.base_url}/docs", timeout=5)
                if response.status_code == 200:
                    self.server_running = True
                    return "✅ Server started successfully! Ready for testing."
                else:
                    return f"❌ Server started but not responsive: {response.status_code}"
            except requests.exceptions.RequestException as e:
                return f"❌ Server failed to start: {str(e)}"
                
        except Exception as e:
            return f"❌ Error starting server: {str(e)}"
    
    def stop_server(self):
        """Stop the LangServe server"""
        if not self.server_running:
            return "Server not running"
            
        try:
            if self.server_process:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            self.server_running = False
            return "✅ Server stopped successfully"
        except Exception as e:
            try:
                if self.server_process:
                    self.server_process.kill()
                self.server_running = False
                return "✅ Server force-stopped"
            except:
                return f"❌ Error stopping server: {str(e)}"
    
    def test_basic_chat(self, message: str) -> str:
        """Test the basic chat endpoint"""
        if not self.server_running:
            return "❌ Server not running. Please start the server first."
            
        try:
            response = requests.post(
                f"{self.base_url}/basic_chat/invoke",
                json={"input": message},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()['output']
                if isinstance(result, dict):
                    return result.get('content', str(result))
                return str(result)
            else:
                return f"❌ Error {response.status_code}: {response.text}"
                
        except Exception as e:
            return f"❌ Request failed: {str(e)}"
    
    def test_retriever(self, query: str) -> str:
        """Test the document retriever endpoint"""
        if not self.server_running:
            return "❌ Server not running. Please start the server first."
            
        try:
            response = requests.post(
                f"{self.base_url}/retriever/invoke",
                json={"input": query},
                timeout=30
            )
            
            if response.status_code == 200:
                docs = response.json()['output']
                
                if not docs:
                    return "No documents retrieved"
                
                result = f"📚 Retrieved {len(docs)} documents:\n\n"
                for i, doc in enumerate(docs[:3], 1):  # Show first 3 docs
                    title = doc.get('metadata', {}).get('Title', f'Document {i}')
                    content = doc.get('page_content', str(doc))
                    result += f"**{i}. {title}**\n"
                    result += f"{content[:200]}...\n\n"
                
                if len(docs) > 3:
                    result += f"... and {len(docs) - 3} more documents"
                
                return result
            else:
                return f"❌ Error {response.status_code}: {response.text}"
                
        except Exception as e:
            return f"❌ Request failed: {str(e)}"
    
    def test_generator(self, question: str, context: str) -> str:
        """Test the generator endpoint"""
        if not self.server_running:
            return "❌ Server not running. Please start the server first."
            
        try:
            response = requests.post(
                f"{self.base_url}/generator/invoke",
                json={
                    "input": {
                        "input": question,
                        "context": context
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['output']
            else:
                return f"❌ Error {response.status_code}: {response.text}"
                
        except Exception as e:
            return f"❌ Request failed: {str(e)}"
    
    def test_full_rag(self, question: str) -> Iterator[str]:
        """Test the complete RAG pipeline"""
        if not self.server_running:
            yield "❌ Server not running. Please start the server first."
            return
            
        try:
            # Step 1: Retrieve documents
            yield "🔍 Retrieving relevant documents..."
            
            retrieval_response = requests.post(
                f"{self.base_url}/retriever/invoke",
                json={"input": question},
                timeout=30
            )
            
            if retrieval_response.status_code != 200:
                yield f"❌ Retrieval failed: {retrieval_response.status_code}"
                return
            
            docs = retrieval_response.json()['output']
            yield f"📚 Retrieved {len(docs)} documents\n\n"
            
            # Step 2: Format context
            context = ""
            for doc in docs:
                title = doc.get('metadata', {}).get('Title', 'Document')
                content = doc.get('page_content', str(doc))
                context += f"[Quote from {title}] {content}\n"
            
            yield "🤖 Generating response...\n\n"
            
            # Step 3: Generate response
            generation_response = requests.post(
                f"{self.base_url}/generator/invoke",
                json={
                    "input": {
                        "input": question,
                        "context": context
                    }
                },
                timeout=30
            )
            
            if generation_response.status_code == 200:
                answer = generation_response.json()['output']
                yield f"**Question:** {question}\n\n**Answer:** {answer}"
            else:
                yield f"❌ Generation failed: {generation_response.status_code}"
                
        except Exception as e:
            yield f"❌ RAG pipeline failed: {str(e)}"

def create_interface():
    """Create the Gradio interface"""
    rag_tester = RAGTester()
    
    with gr.Blocks(title="RAG System Tester", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 RAG System Testing Interface")
        gr.Markdown("Test the complete Retrieval-Augmented Generation system with LangServe endpoints")
        
        # Server Control Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🖥️ Server Control")
                start_btn = gr.Button("🚀 Start Server", variant="primary")
                stop_btn = gr.Button("🛑 Stop Server", variant="secondary")
                server_status = gr.Textbox(label="Server Status", interactive=False)
        
        # Testing Tabs
        with gr.Tabs():
            # Full RAG Test
            with gr.TabItem("🎯 Complete RAG Test"):
                gr.Markdown("### Test the full RAG pipeline")
                with gr.Row():
                    with gr.Column(scale=2):
                        rag_question = gr.Textbox(
                            label="Ask a question",
                            placeholder="e.g., What are transformers and how do they work?",
                            value="Tell me about transformers and attention mechanisms"
                        )
                        rag_test_btn = gr.Button("🔍 Run RAG Pipeline", variant="primary")
                    with gr.Column(scale=3):
                        rag_output = gr.Textbox(
                            label="RAG Response",
                            lines=15,
                            max_lines=20
                        )
            
            # Individual Component Tests
            with gr.TabItem("🧪 Component Tests"):
                with gr.Row():
                    # Basic Chat Test
                    with gr.Column():
                        gr.Markdown("### 💬 Basic Chat")
                        basic_input = gr.Textbox(
                            label="Message",
                            placeholder="Hello, how are you?",
                            value="Hello! Tell me about yourself."
                        )
                        basic_test_btn = gr.Button("Test Basic Chat")
                        basic_output = gr.Textbox(label="Response", lines=5)
                    
                    # Retriever Test
                    with gr.Column():
                        gr.Markdown("### 📚 Document Retriever")
                        retriever_input = gr.Textbox(
                            label="Search Query",
                            placeholder="What are transformers?",
                            value="machine learning transformers"
                        )
                        retriever_test_btn = gr.Button("Test Retriever")
                        retriever_output = gr.Textbox(label="Retrieved Documents", lines=10)
                
                # Generator Test
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ⚡ Response Generator")
                        gen_question = gr.Textbox(
                            label="Question",
                            placeholder="What are transformers?",
                            value="What are transformers?"
                        )
                        gen_context = gr.Textbox(
                            label="Context",
                            placeholder="Provide context documents...",
                            value="[Quote from Attention Is All You Need] Transformers are neural network architectures based on attention mechanisms.",
                            lines=3
                        )
                        gen_test_btn = gr.Button("Test Generator")
                        gen_output = gr.Textbox(label="Generated Response", lines=8)
            
            # API Documentation
            with gr.TabItem("📖 API Docs"):
                gr.Markdown("""
                ### Available Endpoints
                
                Once the server is running, these endpoints are available:
                
                - **Basic Chat**: `POST http://localhost:9012/basic_chat/invoke`
                  - Input: `{"input": "your message"}`
                  - Output: LLM response
                
                - **Retriever**: `POST http://localhost:9012/retriever/invoke`
                  - Input: `{"input": "search query"}`
                  - Output: List of relevant documents
                
                - **Generator**: `POST http://localhost:9012/generator/invoke`
                  - Input: `{"input": {"input": "question", "context": "context text"}}`
                  - Output: Generated response based on context
                
                - **Interactive Docs**: `GET http://localhost:9012/docs`
                  - FastAPI interactive documentation
                
                ### Testing the Complete RAG Pipeline
                
                1. Start the server using the "Start Server" button
                2. Use the "Complete RAG Test" tab to test end-to-end functionality
                3. Or test individual components in the "Component Tests" tab
                4. Check server logs and responses for debugging
                """)
        
        # Event handlers
        start_btn.click(
            fn=rag_tester.start_server,
            outputs=server_status
        )
        
        stop_btn.click(
            fn=rag_tester.stop_server,
            outputs=server_status
        )
        
        # Full RAG test with streaming
        rag_test_btn.click(
            fn=rag_tester.test_full_rag,
            inputs=rag_question,
            outputs=rag_output,
            show_progress=True
        )
        
        # Component tests
        basic_test_btn.click(
            fn=rag_tester.test_basic_chat,
            inputs=basic_input,
            outputs=basic_output
        )
        
        retriever_test_btn.click(
            fn=rag_tester.test_retriever,
            inputs=retriever_input,
            outputs=retriever_output
        )
        
        gen_test_btn.click(
            fn=rag_tester.test_generator,
            inputs=[gen_question, gen_context],
            outputs=gen_output
        )
    
    return demo

if __name__ == "__main__":
    # Set working directory
    os.chdir("/Users/zac/Desktop/Education/GATech/CS8001/CS8001-public/RAG/Lab-7-8-Assessment")
    
    print("Starting Gradio RAG Testing Interface...")
    print("This interface allows you to:")
    print("1. Start/stop the LangServe server")
    print("2. Test individual endpoints")
    print("3. Test the complete RAG pipeline")
    print("4. View API documentation")
    
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_api=False,
        inbrowser=True
    )