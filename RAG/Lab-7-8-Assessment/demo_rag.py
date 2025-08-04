#!/usr/bin/env python3
"""
Command line demo of the RAG system
"""

import subprocess
import time
import requests
import os

def main():
    print("🚀 RAG System Demo")
    print("=" * 50)
    
    # Set up environment
    os.chdir("/Users/zac/Desktop/Education/GATech/CS8001/CS8001-public/RAG/Lab-7-8-Assessment")
    os.environ['NVIDIA_API_KEY'] = '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'
    
    # Start server
    print("Starting LangServe server...")
    server_process = subprocess.Popen(['python', 'server_app.py'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
    
    # Wait for server to start
    print("Waiting for server to initialize...")
    time.sleep(8)
    
    try:
        base_url = "http://localhost:9012"
        
        # Test connection
        print("Testing server connection...")
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
        else:
            print("❌ Server not responding")
            return
        
        # Demo questions
        questions = [
            "What are transformers?",
            "Tell me about BERT",
            "How does RAG work?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n🤔 Question {i}: {question}")
            print("-" * 30)
            
            # Retrieve documents
            print("🔍 Retrieving documents...")
            retrieval_response = requests.post(
                f"{base_url}/retriever/invoke",
                json={"input": question},
                timeout=30
            )
            
            if retrieval_response.status_code == 200:
                docs = retrieval_response.json()['output']
                print(f"📚 Found {len(docs)} relevant documents")
                
                # Format context
                context = ""
                for doc in docs:
                    title = doc.get('metadata', {}).get('Title', 'Document')
                    content = doc.get('page_content', str(doc))
                    context += f"[Quote from {title}] {content}\n"
                
                # Generate response
                print("🤖 Generating response...")
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
                    print("💬 Answer:")
                    print(answer[:300] + "..." if len(answer) > 300 else answer)
                else:
                    print(f"❌ Generation failed: {generation_response.status_code}")
            else:
                print(f"❌ Retrieval failed: {retrieval_response.status_code}")
        
        print("\n🎉 Demo completed successfully!")
        print(f"💡 Server running at: {base_url}")
        print("📖 API docs at: http://localhost:9012/docs")
        
        # Keep server running
        print("\n⏸️  Server will continue running...")
        print("   Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
        except:
            server_process.kill()
        print("✅ Server stopped")

if __name__ == "__main__":
    main()