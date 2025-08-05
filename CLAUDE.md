# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a machine learning education repository containing multiple course modules focused on Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Agentic AI applications. The repository is structured around NVIDIA's Deep Learning Institute (DLI) courses.

## Course Modules

### AAI/ - Agentic AI Applications
Course on building agentic AI applications with LLMs including:
- Basic chat implementations
- Structured output and tool calling
- LangGraph workflow orchestration
- Custom personas and retrievers

### RAD/ - Retrieval-Augmented Development
Course covering LLM fundamentals and multimodal applications:
- LLM introductions and intake
- Encoder tasks and sequence-to-sequence models
- Multimodal processing (audio, images)
- Agentic applications with LangGraph

### RAG/ - Retrieval-Augmented Generation
Multi-lab course on RAG systems:
- Lab-0-1-2: Microservices and LLM basics
- Lab-3-4: LangChain introduction and state management
- Lab-5-6: Document processing and embeddings
- Lab-7-8-Assessment: Vector stores, evaluation, and final assessment

### PRMPT/ - Prompting Fundamentals
Workshop on prompt engineering covering:
- Introduction to prompting with NVIDIA NIM
- LangChain Expression Language (LCEL)
- Message-based prompting and few-shot examples
- Structured output and tool calling

## Development Environment

### Docker-Based Architecture
Each course module uses Docker Compose with the following services:
- **lab**: Jupyter notebook server with NVIDIA GPU runtime
- **nginx**: Reverse proxy for web services
- **docker_router**: Container orchestration helper (port 8070)
- **nim/nim-llm**: NVIDIA NIM LLM inference services
- **embedding/ranking**: Specialized NIM services for embeddings and ranking
- **chatbot**: Interactive chat interfaces
- **assessment**: Automated assessment services

### Starting the Environment
```bash
# Navigate to specific course composer directory
cd AAI/composer/  # or RAD/composer/ or RAG/Lab-*/composer/
docker-compose up -d
```

### Key Ports
- Jupyter Lab: Usually proxied through nginx (port 80)
- Docker Router: 8070
- Chatbot Interfaces: 8990-8991
- LLM Client: 9000
- NIM Services: 8000, 7007
- LangServe APIs: 9012 (RAG assessment)

## Key Development Commands

### Jupyter Notebook Usage
- Primary development interface through JupyterLab
- Notebooks are auto-saved and include rich content (markdown, images)
- Use `Table_of_Contents.ipynb` as navigation entry point

### Python Environment
Dependencies managed via `requirements.txt` files in each service:
```bash
# Core dependencies include:
# langchain, langchain_nvidia_ai_endpoints, langgraph, langserve
# transformers, torch, datasets, diffusers
# gradio (for UI interfaces)
```

### Assessment and Testing
For RAG Lab-7-8-Assessment specifically:
```bash
cd RAG/Lab-7-8-Assessment/
python assessment_test.py  # Full automated assessment
python server_app.py       # Start LangServe API server
python fixed_chat.py       # Launch Gradio chat interface
```

## Architecture Patterns

### Microservices Architecture
- Each course uses containerized microservices
- Service discovery through Docker networking
- Nginx reverse proxy for external access
- Shared volumes for data persistence

### LLM Integration Patterns
- NVIDIA NIM endpoints for inference
- LangChain for orchestration and chaining
- LangServe for API deployment
- Custom client wrappers for API access

### RAG Pipeline Architecture
1. **Document Loading**: Unstructured document processing
2. **Chunking**: Text segmentation for embeddings
3. **Vector Storage**: FAISS for similarity search
4. **Retrieval**: Context-aware document retrieval
5. **Generation**: LLM-based response synthesis
6. **Evaluation**: RAGAS and custom metrics

### Agentic Patterns
- LangGraph for state machine workflows
- Tool calling and function integration
- Multi-agent coordination patterns
- Streaming response handling

## File Structure Conventions

### Notebook Organization
- Numbered sequence (00, 01, 02, etc.)
- Table_of_Contents.ipynb as main navigation
- Solution notebooks in dedicated `solutions/` directories
- Assessment notebooks typically end with high numbers (99)

### Service Directories
- **chatbot/**: Interactive interfaces with Gradio
- **composer/**: Docker composition and environment setup
- **docker_router/**: Container management utilities
- **llm_client/**: API client implementations
- **frontend/**: Web interface components

### Asset Management
- **imgs/**: Course images and diagrams
- **slides/**: Presentation materials
- **audio-files/**, **img-files/**: Sample media for multimodal tasks
- **nim-cache/**: Model caching for performance

## Common Workflows

### Developing RAG Applications
1. Start with basic document loading (Notebook 05-06)
2. Implement vector storage and retrieval (Notebook 07)
3. Add evaluation metrics (Notebook 08)
4. Deploy via LangServe (Notebook 09)
5. Run assessment validation

### Building Agentic Applications
1. Design state machine with LangGraph
2. Implement tool calling capabilities
3. Add streaming and visualization
4. Test with conversation flows

### Multimodal Processing
1. Load media files from provided directories
2. Use appropriate transformers models
3. Implement cross-modal understanding
4. Evaluate on provided test sets

## NVIDIA Integration

### API Keys and Configuration
- NGC_API_KEY for NVIDIA services
- NVIDIA_API_KEY for cloud endpoints
- Configuration via `.env` files

### Model Access Patterns
- Local NIM containers for development
- build.nvidia.com for cloud inference
- Model-specific containers (Llama, embedding, ranking)

### GPU Utilization
- NVIDIA runtime in Docker Compose
- CUDA_VISIBLE_DEVICES for GPU assignment
- Shared memory configuration for model loading

## Assessment and Evaluation

### Automated Assessment
- Scripts validate implementation completeness
- Endpoint testing for API functionality
- Response quality evaluation
- Performance benchmarking

### Testing Patterns
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Interactive testing via chat interfaces
- Automated assessment simulation