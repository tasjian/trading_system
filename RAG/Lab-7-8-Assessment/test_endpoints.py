#!/usr/bin/env python3
import time
import subprocess
import requests
import json
import signal
import os

def test_endpoints():
    # Start the server
    print("Starting LangServe server...")
    server_process = subprocess.Popen(['python', 'server_app.py'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
    
    # Wait for server to start
    time.sleep(5)
    
    try:
        # Test basic_chat endpoint
        print("Testing /basic_chat endpoint...")
        response = requests.post(
            "http://localhost:9012/basic_chat/invoke",
            json={"input": "Hello, how are you?"}
        )
        print(f"Basic chat status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()['output']
            print(f"Basic chat response: {str(result)[:100]}...")
        
        # Test retriever endpoint
        print("\nTesting /retriever endpoint...")
        response = requests.post(
            "http://localhost:9012/retriever/invoke",
            json={"input": "Tell me about transformers"}
        )
        print(f"Retriever status: {response.status_code}")
        if response.status_code == 200:
            docs = response.json()['output']
            print(f"Retrieved {len(docs)} documents")
            if docs:
                print(f"First doc title: {docs[0].get('metadata', {}).get('Title', 'Unknown')}")
        
        # Test generator endpoint
        print("\nTesting /generator endpoint...")
        response = requests.post(
            "http://localhost:9012/generator/invoke",
            json={
                "input": {
                    "input": "What are transformers?",
                    "context": "Transformers are neural network architectures based on attention mechanisms."
                }
            }
        )
        print(f"Generator status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()['output'] 
            print(f"Generator response: {str(result)[:100]}...")
        
        print("\nAll endpoints tested successfully!")
        
    except Exception as e:
        print(f"Error testing endpoints: {e}")
    
    finally:
        # Kill the server
        print("Stopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()

if __name__ == "__main__":
    os.chdir("/Users/zac/Desktop/Education/GATech/CS8001/CS8001-public/RAG/Lab-7-8-Assessment")
    test_endpoints()