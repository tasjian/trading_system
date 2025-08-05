#!/usr/bin/env python3
"""
Create mock ICIJ Offshore Leaks data structure
This simulates the structure of the actual ICIJ data dump
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
import random

def create_mock_icij_data():
    """Create mock offshore leaks data similar to ICIJ structure"""
    
    # Sample data based on ICIJ Offshore Leaks structure
    countries = ['Bahamas', 'Panama', 'British Virgin Islands', 'Bermuda', 'Cayman Islands', 
                'Malta', 'Cyprus', 'Luxembourg', 'Switzerland', 'Monaco']
    
    jurisdictions = ['Panama Papers', 'Paradise Papers', 'Pandora Papers', 'Offshore Leaks', 
                    'Bahamas Leaks']
    
    entity_types = ['Company', 'Trust', 'Foundation', 'Other']
    
    # Generate entities (offshore companies, trusts, etc.)
    entities = []
    for i in range(500):
        entity = {
            'entity_id': f'ENT_{i:05d}',
            'name': f'Offshore Entity {i+1}',
            'jurisdiction': random.choice(countries),
            'incorporation_date': (datetime.now() - timedelta(days=random.randint(365, 7300))).strftime('%Y-%m-%d'),
            'entity_type': random.choice(entity_types),
            'source': random.choice(jurisdictions),
            'status': random.choice(['Active', 'Inactive', 'Dissolved']),
            'description': f'This is an offshore entity incorporated in {random.choice(countries)} for business purposes.',
            'address': f'{random.randint(1, 999)} Financial District, {random.choice(countries)}'
        }
        entities.append(entity)
    
    # Generate officers (people connected to entities)
    officers = []
    for i in range(300):
        officer = {
            'officer_id': f'OFF_{i:05d}',
            'name': f'Person {i+1}',
            'country': random.choice(['USA', 'UK', 'Russia', 'China', 'Germany', 'France', 'Brazil', 'India']),
            'role': random.choice(['Director', 'Shareholder', 'Beneficial Owner', 'Nominee', 'Secretary']),
            'source': random.choice(jurisdictions),
            'description': f'Individual associated with offshore entities through various roles and connections.'
        }
        officers.append(officer)
    
    # Generate relationships (connections between entities and officers)
    relationships = []
    for i in range(800):
        relationship = {
            'relationship_id': f'REL_{i:05d}',
            'from_entity': random.choice(entities)['entity_id'],
            'to_entity': random.choice(officers)['officer_id'],
            'relationship_type': random.choice(['officer_of', 'shareholder_of', 'director_of', 'beneficiary_of']),
            'start_date': (datetime.now() - timedelta(days=random.randint(365, 5475))).strftime('%Y-%m-%d'),
            'end_date': random.choice([None, (datetime.now() - timedelta(days=random.randint(1, 1095))).strftime('%Y-%m-%d')]),
            'source': random.choice(jurisdictions),
            'description': f'Professional relationship connecting offshore entities with individuals.'
        }
        relationships.append(relationship)
    
    # Generate addresses
    addresses = []
    for i in range(200):
        address = {
            'address_id': f'ADDR_{i:05d}',
            'address': f'{random.randint(1, 9999)} {random.choice(["Main St", "Financial Ave", "Corporate Blvd", "Business Rd"])}',
            'city': random.choice(['George Town', 'Road Town', 'Nassau', 'Panama City', 'Hamilton']),
            'country': random.choice(countries),
            'postal_code': f'{random.randint(10000, 99999)}',
            'source': random.choice(jurisdictions)
        }
        addresses.append(address)
    
    return {
        'entities': entities,
        'officers': officers, 
        'relationships': relationships,
        'addresses': addresses
    }

def save_icij_data():
    """Save mock ICIJ data to files"""
    print("Creating mock ICIJ Offshore Leaks data...")
    
    data = create_mock_icij_data()
    
    # Save as JSON files
    for key, value in data.items():
        filename = f'icij_{key}.json'
        with open(filename, 'w') as f:
            json.dump(value, f, indent=2)
        print(f"✅ Saved {len(value)} {key} to {filename}")
    
    # Save as CSV files for easier processing
    for key, value in data.items():
        filename = f'icij_{key}.csv'
        df = pd.DataFrame(value)
        df.to_csv(filename, index=False)
        print(f"✅ Saved {key} CSV to {filename}")
    
    # Create summary document
    summary = {
        'dataset': 'ICIJ Offshore Leaks (Mock Data)',
        'description': 'Simulated offshore financial data based on ICIJ Offshore Leaks structure',
        'total_entities': len(data['entities']),
        'total_officers': len(data['officers']),
        'total_relationships': len(data['relationships']),
        'total_addresses': len(data['addresses']),
        'data_sources': ['Panama Papers', 'Paradise Papers', 'Pandora Papers', 'Offshore Leaks', 'Bahamas Leaks'],
        'jurisdictions': ['Bahamas', 'Panama', 'British Virgin Islands', 'Bermuda', 'Cayman Islands'],
        'created': datetime.now().isoformat()
    }
    
    with open('icij_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\\n📊 ICIJ Data Summary:")
    print(f"- {summary['total_entities']} offshore entities")
    print(f"- {summary['total_officers']} officers/individuals") 
    print(f"- {summary['total_relationships']} relationships")
    print(f"- {summary['total_addresses']} addresses")
    print(f"\\n✅ Mock ICIJ data created successfully!")
    
    return data

if __name__ == "__main__":
    os.chdir("/Users/zac/Desktop/Education/GATech/CS8001/CS8001-public/RAG/Lab-7-8-Assessment")
    save_icij_data()