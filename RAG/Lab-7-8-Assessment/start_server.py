#!/usr/bin/env python3
"""
Simple script to start the LangServe server
Run this to start the assessment server on port 9012
"""

import os
import uvicorn

# Set working directory
os.chdir("/Users/zac/Desktop/Education/GATech/CS8001/CS8001-public/RAG/Lab-7-8-Assessment")

# Set environment variable
os.environ['NVIDIA_API_KEY'] = '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'

print("Starting LangServe server on http://localhost:9012")
print("Available endpoints:")
print("- http://localhost:9012/basic_chat/")
print("- http://localhost:9012/retriever/") 
print("- http://localhost:9012/generator/")
print("\nPress Ctrl+C to stop the server")

# Import and run the app
from server_app import app
uvicorn.run(app, host="0.0.0.0", port=9012)