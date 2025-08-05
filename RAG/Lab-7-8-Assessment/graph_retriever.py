#!/usr/bin/env python3
"""
Graph-based retriever for ICIJ Offshore Leaks data
Uses NetworkX for graph operations and similarity search
"""

import networkx as nx
import json
import pandas as pd
import os
from typing import List, Dict, Any, Tuple
import re
from langchain.schema import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np

class ICIJGraphRetriever:
    """Graph-based retriever for ICIJ offshore leaks data"""
    
    def __init__(self, embedder=None):
        self.graph = nx.MultiDiGraph()
        self.embedder = embedder or NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
        self.entities = {}
        self.officers = {}
        self.relationships = {}
        self.addresses = {}
        self.document_store = None
        
    def load_icij_data(self):
        """Load ICIJ data from JSON files into graph structure"""
        print("Loading ICIJ data into graph...")
        
        # Load data files
        with open('icij_entities.json', 'r') as f:
            entities_data = json.load(f)
        with open('icij_officers.json', 'r') as f:
            officers_data = json.load(f)
        with open('icij_relationships.json', 'r') as f:
            relationships_data = json.load(f)
        with open('icij_addresses.json', 'r') as f:
            addresses_data = json.load(f)
        
        # Store data in dictionaries for quick lookup
        self.entities = {e['entity_id']: e for e in entities_data}
        self.officers = {o['officer_id']: o for o in officers_data}
        self.addresses = {a['address_id']: a for a in addresses_data}
        
        # Add nodes to graph
        print("Adding entities to graph...")
        for entity in entities_data:
            node_attrs = {
                'node_type': 'entity',
                'entity_name': entity['name'],
                'jurisdiction': entity['jurisdiction'],
                'entity_type': entity['entity_type'],
                'status': entity['status'],
                'incorporation_date': entity['incorporation_date'],
                'source': entity['source'],
                'description': entity['description'],
                'address': entity['address']
            }
            self.graph.add_node(entity['entity_id'], **node_attrs)
        
        print("Adding officers to graph...")
        for officer in officers_data:
            node_attrs = {
                'node_type': 'officer',
                'officer_name': officer['name'],
                'country': officer['country'],
                'role': officer['role'],
                'source': officer['source'],
                'description': officer['description']
            }
            self.graph.add_node(officer['officer_id'], **node_attrs)
        
        print("Adding addresses to graph...")
        for address in addresses_data:
            node_attrs = {
                'node_type': 'address',
                'address_text': address['address'],
                'city': address['city'],
                'country': address['country'],
                'postal_code': address['postal_code'],
                'source': address['source']
            }
            self.graph.add_node(address['address_id'], **node_attrs)
        
        # Add relationships (edges)
        print("Adding relationships to graph...")
        for rel in relationships_data:
            edge_attrs = {
                'relationship_type': rel['relationship_type'],
                'start_date': rel['start_date'],
                'end_date': rel.get('end_date'),
                'source': rel['source'],
                'description': rel['description']
            }
            self.graph.add_edge(rel['from_entity'], rel['to_entity'], **edge_attrs)
        
        print(f"✅ Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
    def create_documents_from_graph(self) -> List[Document]:
        """Convert graph data into documents for vector store"""
        documents = []
        
        # Create documents for entities
        for entity_id, entity in self.entities.items():
            # Get connected officers
            connected_officers = []
            if entity_id in self.graph:
                for neighbor in self.graph.neighbors(entity_id):
                    if neighbor.startswith('OFF_'):
                        officer = self.officers.get(neighbor, {})
                        edge_data = self.graph.get_edge_data(entity_id, neighbor)
                        if edge_data:
                            rel_type = list(edge_data.values())[0].get('relationship_type', 'connected')
                            connected_officers.append(f"{officer.get('name', 'Unknown')} ({rel_type})")
            
            # Create document content
            content = f"""
Offshore Entity: {entity['name']}
Entity ID: {entity_id}
Jurisdiction: {entity['jurisdiction']}
Type: {entity['entity_type']}
Status: {entity['status']}
Incorporation Date: {entity['incorporation_date']}
Address: {entity['address']}
Source: {entity['source']}

Connected Officers: {', '.join(connected_officers) if connected_officers else 'None listed'}

Description: {entity['description']}

This entity was revealed in the {entity['source']} investigation and is incorporated in {entity['jurisdiction']}.
            """.strip()
            
            doc = Document(
                page_content=content,
                metadata={
                    'entity_id': entity_id,
                    'name': entity['name'],
                    'type': 'entity',
                    'jurisdiction': entity['jurisdiction'],
                    'entity_type': entity['entity_type'],
                    'source': entity['source'],
                    'title': f"Offshore Entity: {entity['name']}"
                }
            )
            documents.append(doc)
        
        # Create documents for officers
        for officer_id, officer in self.officers.items():
            # Get connected entities
            connected_entities = []
            if officer_id in self.graph:
                for neighbor in self.graph.neighbors(officer_id):
                    if neighbor.startswith('ENT_'):
                        entity = self.entities.get(neighbor, {})
                        # Check incoming edges (entity -> officer relationships)
                        for pred in self.graph.predecessors(officer_id):
                            if pred == neighbor:
                                edge_data = self.graph.get_edge_data(pred, officer_id)
                                if edge_data:
                                    rel_type = list(edge_data.values())[0].get('relationship_type', 'connected')
                                    connected_entities.append(f"{entity.get('name', 'Unknown')} ({rel_type})")
            
            content = f"""
Individual: {officer['name']}
Officer ID: {officer_id}
Country: {officer['country']}
Role: {officer['role']}
Source: {officer['source']}

Connected Entities: {', '.join(connected_entities) if connected_entities else 'None listed'}

Description: {officer['description']}

This individual was identified in the {officer['source']} investigation and has connections to offshore entities.
            """.strip()
            
            doc = Document(
                page_content=content,
                metadata={
                    'officer_id': officer_id,
                    'name': officer['name'],
                    'type': 'officer',
                    'country': officer['country'],
                    'role': officer['role'],
                    'source': officer['source'],
                    'title': f"Individual: {officer['name']}"
                }
            )
            documents.append(doc)
        
        # Create summary documents for investigations
        sources = set(entity['source'] for entity in self.entities.values())
        for source in sources:
            source_entities = [e for e in self.entities.values() if e['source'] == source]
            source_officers = [o for o in self.officers.values() if o['source'] == source]
            
            content = f"""
Investigation: {source}
Type: Offshore Financial Investigation
Total Entities: {len(source_entities)}
Total Officers: {len(source_officers)}

Overview: The {source} investigation revealed a network of offshore entities and individuals involved in complex financial structures. This investigation exposed entities across multiple jurisdictions including tax havens and offshore financial centers.

Key Jurisdictions: {', '.join(set(e['jurisdiction'] for e in source_entities))}

This investigation is part of the broader ICIJ Offshore Leaks database, which contains information about offshore entities, their officers, and the relationships between them.
            """.strip()
            
            doc = Document(
                page_content=content,
                metadata={
                    'type': 'investigation',
                    'source': source,
                    'entity_count': len(source_entities),
                    'officer_count': len(source_officers),
                    'title': f"Investigation: {source}"
                }
            )
            documents.append(doc)
        
        print(f"✅ Created {len(documents)} documents from graph data")
        return documents
    
    def build_vector_store(self):
        """Build vector store from graph documents"""
        print("Building vector store from graph data...")
        documents = self.create_documents_from_graph()
        
        # Create FAISS vector store
        self.document_store = FAISS.from_documents(documents, self.embedder)
        print(f"✅ Vector store built with {len(documents)} documents")
        
    def graph_search(self, query: str, max_hops: int = 2) -> List[Dict]:
        """Search graph using entity/person names and relationships"""
        results = []
        query_lower = query.lower()
        
        # Find matching nodes by name
        matching_nodes = []
        for node_id, node_data in self.graph.nodes(data=True):
            # Check both entity_name and officer_name fields
            name = (node_data.get('entity_name', '') + ' ' + node_data.get('officer_name', '')).lower()
            if any(term in name for term in query_lower.split() if len(term) > 2):
                matching_nodes.append(node_id)
        
        # For each matching node, explore neighborhood
        for node_id in matching_nodes:
            node_data = self.graph.nodes[node_id]
            
            # Get direct neighbors
            neighbors = list(self.graph.neighbors(node_id))
            
            # Get 2-hop neighbors for more context
            two_hop_neighbors = set()
            for neighbor in neighbors:
                two_hop_neighbors.update(self.graph.neighbors(neighbor))
            
            # Create result with context
            result = {
                'node_id': node_id,
                'node_data': node_data,
                'direct_neighbors': len(neighbors),
                'extended_network': len(two_hop_neighbors),
                'connections': []
            }
            
            # Add connection details
            for neighbor in neighbors[:5]:  # Limit to top 5 connections
                neighbor_data = self.graph.nodes[neighbor]
                edge_data = self.graph.get_edge_data(node_id, neighbor)
                if edge_data:
                    rel_info = list(edge_data.values())[0]
                    neighbor_name = neighbor_data.get('entity_name') or neighbor_data.get('officer_name') or 'Unknown'
                    result['connections'].append({
                        'neighbor_id': neighbor,
                        'neighbor_name': neighbor_name,
                        'neighbor_type': neighbor_data.get('node_type', 'unknown'),
                        'relationship': rel_info.get('relationship_type', 'connected'),
                        'source': rel_info.get('source', 'unknown')
                    })
            
            results.append(result)
        
        return results
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Main retrieval method combining vector search and graph search"""
        if not self.document_store:
            self.build_vector_store()
        
        # Vector-based retrieval
        vector_docs = self.document_store.similarity_search(query, k=k)
        
        # Graph-based search for additional context
        graph_results = self.graph_search(query)
        
        # Enhance vector docs with graph context
        enhanced_docs = []
        for doc in vector_docs:
            enhanced_content = doc.page_content
            
            # Add graph context if relevant
            doc_id = doc.metadata.get('entity_id') or doc.metadata.get('officer_id')
            if doc_id and graph_results:
                for graph_result in graph_results:
                    if graph_result['node_id'] == doc_id:
                        graph_info = f"\n\nGraph Context:\n"
                        graph_info += f"Network size: {graph_result['direct_neighbors']} direct connections, {graph_result['extended_network']} in extended network\n"
                        if graph_result['connections']:
                            graph_info += "Key connections:\n"
                            for conn in graph_result['connections'][:3]:
                                graph_info += f"- {conn['neighbor_name']} ({conn['relationship']})\n"
                        enhanced_content += graph_info
                        break
            
            enhanced_doc = Document(
                page_content=enhanced_content,
                metadata=doc.metadata
            )
            enhanced_docs.append(enhanced_doc)
        
        return enhanced_docs

def test_graph_retriever():
    """Test the graph retriever"""
    import os
    os.environ['NVIDIA_API_KEY'] = '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'
    
    print("Testing ICIJ Graph Retriever...")
    
    retriever = ICIJGraphRetriever()
    retriever.load_icij_data()
    retriever.build_vector_store()
    
    # Test queries
    test_queries = [
        "offshore companies in Panama",
        "directors and beneficial owners",
        "entities in British Virgin Islands",
        "Paradise Papers investigation"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        docs = retriever.retrieve(query, k=3)
        for i, doc in enumerate(docs, 1):
            print(f"\n{i}. {doc.metadata.get('title', 'Document')}")
            print(f"   {doc.page_content[:200]}...")

if __name__ == "__main__":
    os.chdir("/Users/zac/Desktop/Education/GATech/CS8001/CS8001-public/RAG/Lab-7-8-Assessment")
    test_graph_retriever()