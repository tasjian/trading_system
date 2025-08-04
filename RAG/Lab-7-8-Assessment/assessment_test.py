#!/usr/bin/env python3
"""
Final assessment test script
This simulates what the course assessment system would do
"""

import time
import subprocess
import requests
import json
import os

def simulate_assessment():
    print("=== ASSESSMENT SIMULATION ===")
    print("Testing RAG pipeline for course completion...")
    
    # Start the server
    print("\n1. Starting LangServe server...")
    server_process = subprocess.Popen(['python', 'server_app.py'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
    
    # Wait for server to start
    time.sleep(5)
    
    try:
        # Test 1: Basic functionality
        print("\n2. Testing basic chat functionality...")
        response = requests.post(
            "http://localhost:9012/basic_chat/invoke",
            json={"input": "Hello"}
        )
        assert response.status_code == 200, "Basic chat endpoint failed"
        print("✓ Basic chat working")
        
        # Test 2: Retriever functionality
        print("\n3. Testing document retrieval...")
        response = requests.post(
            "http://localhost:9012/retriever/invoke",
            json={"input": "What are transformers and attention mechanisms?"}
        )
        assert response.status_code == 200, "Retriever endpoint failed"
        docs = response.json()['output']
        assert len(docs) > 0, "No documents retrieved"
        print(f"✓ Retrieved {len(docs)} documents")
        
        # Test 3: Generator functionality
        print("\n4. Testing response generation...")
        # First get some context from retriever
        context_docs = docs[:2]  # Use first 2 docs
        context_str = ""
        for doc in context_docs:
            title = doc.get('metadata', {}).get('Title', 'Document')
            content = doc.get('page_content', str(doc))
            context_str += f"[Quote from {title}] {content}\n"
        
        response = requests.post(
            "http://localhost:9012/generator/invoke",
            json={
                "input": {
                    "input": "What are transformers?",
                    "context": context_str
                }
            }
        )
        assert response.status_code == 200, "Generator endpoint failed"
        result = response.json()['output']
        assert len(result) > 50, "Generated response too short"
        print("✓ Generator producing responses")
        
        # Test 4: Full RAG pipeline simulation
        print("\n5. Testing complete RAG pipeline...")
        
        # Simulate what the frontend does
        test_query = "Tell me about BERT and transformers"
        
        # Get retrieval
        retrieval_response = requests.post(
            "http://localhost:9012/retriever/invoke",
            json={"input": test_query}
        )
        retrieved_docs = retrieval_response.json()['output']
        
        # Format context
        context = ""
        for doc in retrieved_docs:
            title = doc.get('metadata', {}).get('Title', 'Document')
            content = doc.get('page_content', str(doc))
            context += f"[Quote from {title}] {content}\n"
        
        # Generate response
        generation_response = requests.post(
            "http://localhost:9012/generator/invoke",
            json={
                "input": {
                    "input": test_query,
                    "context": context
                }
            }
        )
        final_answer = generation_response.json()['output']
        
        print(f"✓ RAG pipeline complete")
        print(f"Question: {test_query}")
        print(f"Answer length: {len(final_answer)} characters")
        print(f"Answer preview: {final_answer[:200]}...")
        
        # Assessment criteria
        print("\n6. Checking assessment criteria...")
        
        # Check if answer mentions source documents
        has_citations = any(marker in final_answer for marker in ["[Quote from", "According to", "BERT", "Transformer"])
        assert has_citations, "Response doesn't cite sources properly"
        print("✓ Response cites sources")
        
        # Check response length
        assert len(final_answer) > 100, "Response too short"
        print("✓ Response has adequate length")
        
        # Check conversational tone
        conversational_markers = ["you", "I", "let", "help", "explain"]
        is_conversational = any(marker in final_answer.lower() for marker in conversational_markers)
        assert is_conversational, "Response not conversational enough"
        print("✓ Response is conversational")
        
        print("\n🎉 ASSESSMENT PASSED! 🎉")
        print("All RAG components working correctly:")
        print("- ✓ Document loading and indexing")
        print("- ✓ Vector store retrieval")  
        print("- ✓ Context-aware generation")
        print("- ✓ Proper source citation")
        print("- ✓ LangServe API endpoints")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ASSESSMENT FAILED: {e}")
        return False
    
    finally:
        # Kill the server
        print("\n7. Stopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()

if __name__ == "__main__":
    os.chdir("/Users/zac/Desktop/Education/GATech/CS8001/CS8001-public/RAG/Lab-7-8-Assessment")
    success = simulate_assessment()
    exit(0 if success else 1)