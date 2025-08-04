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
from graph_retriever import ICIJGraphRetriever
import os

# Set up environment
os.environ['NVIDIA_API_KEY'] = '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'

## Initialize models
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
instruct_llm = ChatNVIDIA(model="meta/llama3-8b-instruct")

# Load the ICIJ vector store we created
import subprocess
subprocess.run(['tar', 'xzvf', 'icij_docstore_index.tgz'], capture_output=True)

# Create the graph retriever for enhanced functionality
graph_retriever = ICIJGraphRetriever(embedder)
graph_retriever.load_icij_data()

# Load the pre-built vector store
docstore = FAISS.load_local("icij_docstore_index", embedder, allow_dangerous_deserialization=True)
graph_retriever.document_store = docstore

app = FastAPI(
  title="ICIJ Offshore Leaks RAG Server",
  version="1.0",
  description="Graph-enhanced RAG system for ICIJ Offshore Leaks data using LangChain and Neo4j concepts",
)

## Basic chat endpoint (unchanged)
add_routes(
    app,
    instruct_llm,
    path="/basic_chat",
)

## ICIJ-specific implementations

def docs2str(docs, title="Document"):
    """Utility for making chunks into context string with ICIJ-specific formatting"""
    out_str = ""
    for doc in docs:
        doc_title = doc.metadata.get('title', title)
        doc_type = doc.metadata.get('type', 'document')
        
        if doc_type == 'entity':
            jurisdiction = doc.metadata.get('jurisdiction', 'Unknown')
            source = doc.metadata.get('source', 'Unknown')
            out_str += f"[{source} - {doc_title} in {jurisdiction}] "
        elif doc_type == 'officer':
            country = doc.metadata.get('country', 'Unknown')
            role = doc.metadata.get('role', 'Unknown')
            out_str += f"[Individual: {doc_title} - {role} from {country}] "
        elif doc_type == 'investigation':
            source = doc.metadata.get('source', 'Unknown')
            out_str += f"[Investigation: {source}] "
        else:
            out_str += f"[{doc_title}] "
            
        out_str += doc.page_content + "\n\n"
    return out_str

# Enhanced retriever that uses graph context
def enhanced_retriever_chain(query_dict):
    """Enhanced retrieval with graph context"""
    query = query_dict["input"] if isinstance(query_dict, dict) else query_dict
    
    # Use the graph retriever for enhanced results
    docs = graph_retriever.retrieve(query, k=4)
    
    return docs

# Create retriever chain
retriever_chain = RunnableLambda(enhanced_retriever_chain)

# Generator that expects dict input with 'input' and 'context' keys
generator_prompt = ChatPromptTemplate.from_template(
    "You are an investigative assistant specialized in offshore financial data from the ICIJ Offshore Leaks database. "
    "Help users understand complex offshore financial networks, shell companies, and connections between entities and individuals.\n\n"
    "User question: {input}\n\n"
    "Relevant offshore financial data:\n{context}\n\n"
    "Instructions:\n"
    "- Focus on factual information from the offshore leaks data\n"
    "- Explain relationships between entities, officers, and jurisdictions\n"
    "- Highlight important connections and patterns\n"
    "- Be precise about jurisdictions, dates, and entity types\n"
    "- Cite specific sources (Panama Papers, Paradise Papers, etc.) when mentioned\n"
    "- Use clear, investigative language appropriate for financial analysis\n\n"
    "Response:"
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

## Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": "ICIJ Offshore Leaks (Mock Data)",
        "total_entities": len(graph_retriever.entities),
        "total_officers": len(graph_retriever.officers),
        "graph_nodes": graph_retriever.graph.number_of_nodes(),
        "graph_edges": graph_retriever.graph.number_of_edges()
    }

## Data statistics endpoint
@app.get("/stats")
async def get_stats():
    """Get statistics about the ICIJ data"""
    
    # Calculate stats
    entity_types = {}
    jurisdictions = {}
    sources = {}
    
    for entity in graph_retriever.entities.values():
        entity_types[entity['entity_type']] = entity_types.get(entity['entity_type'], 0) + 1
        jurisdictions[entity['jurisdiction']] = jurisdictions.get(entity['jurisdiction'], 0) + 1
        sources[entity['source']] = sources.get(entity['source'], 0) + 1
    
    officer_countries = {}
    officer_roles = {}
    
    for officer in graph_retriever.officers.values():
        officer_countries[officer['country']] = officer_countries.get(officer['country'], 0) + 1
        officer_roles[officer['role']] = officer_roles.get(officer['role'], 0) + 1
    
    return {
        "entities": {
            "total": len(graph_retriever.entities),
            "by_type": entity_types,
            "by_jurisdiction": jurisdictions,
            "by_source": sources
        },
        "officers": {
            "total": len(graph_retriever.officers),
            "by_country": officer_countries,
            "by_role": officer_roles
        },
        "relationships": {
            "total": graph_retriever.graph.number_of_edges()
        },
        "graph": {
            "nodes": graph_retriever.graph.number_of_nodes(),
            "edges": graph_retriever.graph.number_of_edges()
        }
    }

## Might be encountered if this were for a standalone python file...
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9012)