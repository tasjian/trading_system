#!/usr/bin/env python3
"""
Quick test interface for the RAG system
Simple forms to test functionality
"""

import gradio as gr
import requests
import subprocess
import time
import os

def test_rag_pipeline(question):
    """Test the complete RAG pipeline"""
    
    # Set up environment
    os.chdir("/Users/zac/Desktop/Education/GATech/CS8001/CS8001-public/RAG/Lab-7-8-Assessment")
    os.environ['NVIDIA_API_KEY'] = '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'
    
    # Start server in background
    server_process = subprocess.Popen(['python', 'server_app.py'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
    
    result = "🔄 Starting server...\n\n"
    yield result
    
    # Wait for server to start
    time.sleep(8)
    
    try:
        base_url = "http://localhost:9012"
        
        # Test server health
        try:
            response = requests.get(f"{base_url}/docs", timeout=5)
            if response.status_code == 200:
                result += "✅ Server started successfully!\n\n"
                yield result
            else:
                result += "❌ Server not responding\n\n"
                yield result
                return
        except:
            result += "❌ Could not connect to server\n\n"
            yield result
            return
        
        # Step 1: Test retrieval
        result += "🔍 Step 1: Retrieving documents...\n"
        yield result
        
        retrieval_response = requests.post(
            f"{base_url}/retriever/invoke",
            json={"input": question},
            timeout=30
        )
        
        if retrieval_response.status_code == 200:
            docs = retrieval_response.json()['output']
            result += f"✅ Retrieved {len(docs)} documents\n\n"
            
            # Show retrieved documents
            result += "📚 **Retrieved Documents:**\n"
            for i, doc in enumerate(docs[:2], 1):  # Show first 2
                title = doc.get('metadata', {}).get('Title', f'Document {i}')
                content = doc.get('page_content', str(doc))[:150]
                result += f"{i}. **{title}**: {content}...\n\n"
            
            yield result
            
            # Step 2: Format context and generate
            result += "🤖 Step 2: Generating response...\n"
            yield result
            
            context = ""
            for doc in docs:
                title = doc.get('metadata', {}).get('Title', 'Document')
                content = doc.get('page_content', str(doc))
                context += f"[Quote from {title}] {content}\n"
            
            generation_response = requests.post(
                f"{base_url}/generator/invoke",
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
                result += "✅ Response generated!\n\n"
                result += "🎯 **Final Answer:**\n"
                result += f"{answer}\n\n"
                result += "🎉 **RAG Pipeline Complete!**"
                yield result
            else:
                result += f"❌ Generation failed: {generation_response.status_code}\n"
                yield result
                
        else:
            result += f"❌ Retrieval failed: {retrieval_response.status_code}\n"
            yield result
            
    except Exception as e:
        result += f"❌ Error: {str(e)}\n"
        yield result
    
    finally:
        # Stop server
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
        except:
            server_process.kill()

def create_test_interface():
    """Create simple test interface"""
    
    with gr.Blocks(title="RAG Quick Test") as demo:
        gr.Markdown("# 🧪 RAG System Quick Test")
        gr.Markdown("Enter a question to test the complete RAG pipeline")
        
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="What are transformers and attention mechanisms?",
            value="Tell me about transformers and BERT",
            lines=2
        )
        
        test_button = gr.Button("🚀 Test RAG Pipeline", variant="primary", size="lg")
        
        output_box = gr.Textbox(
            label="Test Results",
            lines=20,
            max_lines=30,
            show_copy_button=True
        )
        
        gr.Examples(
            examples=[
                "What are transformers and how do they work?",
                "Explain BERT and its training process",
                "How does retrieval-augmented generation improve AI responses?",
                "What is the difference between BERT and GPT models?",
                "Tell me about attention mechanisms in neural networks"
            ],
            inputs=question_input
        )
        
        test_button.click(
            fn=test_rag_pipeline,
            inputs=question_input,
            outputs=output_box,
            show_progress=True
        )
    
    return demo

if __name__ == "__main__":
    print("🧪 Starting RAG Quick Test Interface...")
    print("This will test the complete pipeline step by step")
    
    demo = create_test_interface()
    demo.launch(
        share=False,
        server_port=7863,
        inbrowser=True
    )