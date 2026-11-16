# transcript_generator.py - FIXED
from character_mapper import CharacterMapper
from typing import List, Dict, Tuple
import os
import numpy as np

def generate_transcript(
    ocr_results: List[str],
    text_char_associations: List[Tuple[int, int]],
    char_clusters: Dict[int, int],
    output_file: str = None,
    character_mapper: CharacterMapper = None
) -> str:
    """Generate formatted transcript with speaker associations"""
    if character_mapper is None:
        character_mapper = CharacterMapper()
    
    transcript = []
    
    for text_idx, text in enumerate(ocr_results):
        # Find associated character
        char_idx = next((c for t, c in text_char_associations if t == text_idx), None)
        
        # Determine speaker
        if char_idx is not None and char_idx in char_clusters:
            cluster_id = char_clusters[char_idx]
            cluster_type = "face"  # or "body" depending on your data/context

            speaker = character_mapper.get_speaker(cluster_type, cluster_id)
        else:
            speaker = "NARRATOR"
        
        transcript.append(f"{speaker}: {text}")
    
    # Format final output
    formatted = "### MANGA TRANSCRIPT ###\n" + "\n".join(transcript)
    
    # Save to file if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted)
    
    return formatted

def visualize_results(
    image: np.ndarray,
    detections: Dict[str, List[List[float]]],
    associations: List[Tuple[int, int]],
    output_path: str
):
    """Visualize detections and associations (pseudo-code)"""
    import cv2
    # Visualization implementation would go here
    # This would draw bounding boxes and association lines
    pass