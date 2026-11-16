# association_engine.py
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from sklearn.cluster import DBSCAN

def associate_text_to_characters(
    texts: List[Dict[str, Any]],
    characters: List[Dict[str, Any]],
    relations: List[Dict[str, Any]]
) -> List[Tuple[int, int]]:
    """Associate text to characters using predicted relations"""
    associations = []
    for rel in relations:
        text_idx = rel['text_index']
        char_idx = rel['character_index']
        if text_idx < len(texts) and char_idx < len(characters):
            associations.append((text_idx, char_idx))
    return associations

def cluster_characters(embeddings: np.ndarray, eps=0.4, min_samples=2) -> Dict[int, int]:
    """Improved clustering with tuned parameters"""
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='cosine'
    ).fit(embeddings)
    
    return {i: int(label) for i, label in enumerate(clustering.labels_)}

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0