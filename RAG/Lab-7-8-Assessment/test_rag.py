#!/usr/bin/env python3

import os
os.environ['NVIDIA_API_KEY'] = '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'

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_community.vectorstores import FAISS
from langchain_community.document_transformers import LongContextReorder
from operator import itemgetter
import subprocess

# Load models
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
instruct_llm = ChatNVIDIA(model="meta/llama3-8b-instruct")

# Load vector store
subprocess.run(['tar', 'xzvf', 'docstore_index.tgz'], capture_output=True)
docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name: 
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

# Create RAG components
retriever = docstore.as_retriever()
long_reorder = RunnableLambda(LongContextReorder().transform_documents)

# Retrieval chain - returns dict with input and context
retrieval_chain = (
    {'input': lambda x: x}
    | RunnableAssign({
        'context': itemgetter('input') | retriever | long_reorder | docs2str
    })
)

# Generator chain - expects dict with input and context
generator_prompt = ChatPromptTemplate.from_template(
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User question: {input}\n\n"
    " The following information may be useful for your response: "
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational)"
    "\n\nUser Question: {input}"
)

generator_chain = generator_prompt | instruct_llm | StrOutputParser()

# Output puller function
def output_puller(inputs):
    """Output generator. Useful if your chain returns a dictionary with key 'output'"""
    if isinstance(inputs, dict):
        inputs = [inputs]
    for token in inputs:
        if token.get('output'):
            yield token.get('output')

# Complete RAG chain
output_chain = RunnableAssign({"output": generator_chain}) | RunnableLambda(output_puller)
rag_chain = retrieval_chain | output_chain

print("Testing RAG chain...")
print("Question: Tell me about transformers and attention mechanisms")
print("\nAnswer:")

for token in rag_chain.stream("Tell me about transformers and attention mechanisms"):
    print(token, end="")

print("\n\nRAG chain working successfully!")