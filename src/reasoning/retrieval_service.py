import torch
from fuzzywuzzy import fuzz
from collections import defaultdict

def enhanced_entity_linking(question, entity_list, threshold=85):
    """Matches question tokens to KG entities using fuzzy logic."""
    tokens = question.lower().replace("?", "").split()
    matched_entities = []
    for entity in entity_list:
        # Check for multi-word entity matches in the question
        if entity.lower() in question.lower():
            matched_entities.append(entity)
        else:
            # Fuzzy fallback for spelling variations
            for token in tokens:
                if len(token) > 3 and fuzz.ratio(token, entity.lower()) > threshold:
                    matched_entities.append(entity)
    return list(set(matched_entities))

def extract_multi_hop_subgraph(seed_entities, kg_adj, max_hops=2):
    """Performs BFS to capture 2-hop reasoning paths as per SubgraphRAG principles."""
    subgraph = []
    visited_nodes = set(seed_entities)
    current_layer = seed_entities

    for hop in range(max_hops):
        next_layer = []
        for node in current_layer:
            if node in kg_adj:
                for rel, neighbor in kg_adj[node]:
                    subgraph.append((node, rel, neighbor))
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        next_layer.append(neighbor)
        current_layer = next_layer
    return subgraph[:100] # Limit to paper's K=100 budget
