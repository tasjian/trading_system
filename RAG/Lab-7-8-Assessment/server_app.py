# https://python.langchain.com/docs/langserve#server
from fastapi import FastAPI
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langserve import add_routes

## May be useful later
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_community.document_transformers import LongContextReorder
from functools import partial
from operator import itemgetter

from langchain_community.vectorstores import FAISS
import os

# Set up environment
os.environ['NVIDIA_API_KEY'] = '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'

## Initialize models
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
instruct_llm = ChatNVIDIA(model="meta/llama3-8b-instruct")

# Load the vector store we created
import subprocess
subprocess.run(['tar', 'xzvf', 'docstore_index.tgz'], capture_output=True)
docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

## PRE-ASSESSMENT: Run as-is and see the basic chain in action
add_routes(
    app,
    instruct_llm,
    path="/basic_chat",
)

## ASSESSMENT: Implement these components

# Create retriever chain
retriever_chain = docstore.as_retriever()

# Create generator chain with proper RAG prompt
def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name: 
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

# Generator that expects dict input with 'input' and 'context' keys
generator_prompt = ChatPromptTemplate.from_template(
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User question: {input}\n\n"
    " The following information may be useful for your response: "
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational)"
    "\n\nUser Question: {input}"
)

generator_chain = generator_prompt | instruct_llm | StrOutputParser()

add_routes(
    app,
    generator_chain,
    path="/generator",
)

add_routes(
    app,
    retriever_chain,
    path="/retriever",
)

## Might be encountered if this were for a standalone python file...
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9012)