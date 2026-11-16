# bubble_tracker.py
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import uuid

@dataclass
class BubbleInfo:
    """Enhanced bubble tracking information"""
    bubble_id: str
    original_text: str
    current_text: str
    source_language: str
    target_language: str
    bbox: List[float]  # [x1, y1, x2, y2]
    container_info: str
    image_src: str
    speaker: Optional[str]
    created_time: float
    last_updated: float
    translation_attempts: int
    translation_successes: int
    is_translated: bool
    confidence: float
    hidden: bool

class BubbleTracker:
    """Comprehensive bubble tracking system"""
    
    def __init__(self):
        self.bubbles: Dict[str, BubbleInfo] = {}
        self.character_bubbles: Dict[str, List[str]] = {}  # character_id -> bubble_ids
        self.panel_bubbles: Dict[str, List[str]] = {}  # panel_id -> bubble_ids
        self.translation_history: List[Dict] = []
    
    def register_bubble(
        self,
        original_text: str,
        bbox: List[float],
        container_info: str,
        image_src: str,
        source_language: str,
        speaker: Optional[str] = None
    ) -> str:
        """Register a new bubble and return its ID"""
        bubble_id = str(uuid.uuid4())
        
        bubble_info = BubbleInfo(
            bubble_id=bubble_id,
            original_text=original_text,
            current_text=original_text,
            source_language=source_language,
            target_language="en",  # default
            bbox=bbox,
            container_info=container_info,
            image_src=image_src,
            speaker=speaker,
            created_time=time.time(),
            last_updated=time.time(),
            translation_attempts=0,
            translation_successes=0,
            is_translated=False,
            confidence=0.0,
            hidden=False
        )
        
        self.bubbles[bubble_id] = bubble_info
        return bubble_id
    
    def update_bubble_translation(
        self,
        bubble_id: str,
        translated_text: str,
        target_language: str,
        confidence: float,
        success: bool
    ):
        """Update bubble after translation attempt"""
        if bubble_id not in self.bubbles:
            return
        
        bubble = self.bubbles[bubble_id]
        bubble.current_text = translated_text
        bubble.target_language = target_language
        bubble.confidence = confidence
        bubble.is_translated = success
        bubble.last_updated = time.time()
        bubble.translation_attempts += 1
        
        if success:
            bubble.translation_successes += 1
        
        # Record in history
        self.translation_history.append({
            'bubble_id': bubble_id,
            'timestamp': time.time(),
            'original_text': bubble.original_text,
            'translated_text': translated_text,
            'source_lang': bubble.source_language,
            'target_lang': target_language,
            'confidence': confidence,
            'success': success
        })
    
    def get_bubble_stats(self) -> Dict:
        """Get comprehensive bubble statistics"""
        total_bubbles = len(self.bubbles)
        translated_bubbles = sum(1 for b in self.bubbles.values() if b.is_translated)
        hidden_bubbles = sum(1 for b in self.bubbles.values() if b.hidden)
        
        success_rate = 0
        if total_bubbles > 0:
            success_rate = translated_bubbles / total_bubbles
        
        return {
            'total_bubbles': total_bubbles,
            'translated_bubbles': translated_bubbles,
            'translation_success_rate': success_rate,
            'hidden_bubbles': hidden_bubbles,
            'average_confidence': sum(b.confidence for b in self.bubbles.values()) / max(total_bubbles, 1),
            'characters_with_bubbles': len(self.character_bubbles),
            'panels_with_bubbles': len(self.panel_bubbles)
        }
    
    def get_bubbles_by_language(self, language: str) -> List[BubbleInfo]:
        """Get all bubbles for a specific source language"""
        return [b for b in self.bubbles.values() if b.source_language == language]
    
    def get_untranlsated_bubbles(self) -> List[BubbleInfo]:
        """Get all bubbles that haven't been translated yet"""
        return [b for b in self.bubbles.values() if not b.is_translated]
    
    def associate_bubble_with_character(self, bubble_id: str, character_id: str):
        """Associate a bubble with a character"""
        if character_id not in self.character_bubbles:
            self.character_bubbles[character_id] = []
        
        if bubble_id not in self.character_bubbles[character_id]:
            self.character_bubbles[character_id].append(bubble_id)
    
    def get_character_dialogue(self, character_id: str) -> List[str]:
        """Get all dialogue for a specific character"""
        if character_id not in self.character_bubbles:
            return []
        
        return [self.bubbles[b_id].current_text 
                for b_id in self.character_bubbles[character_id] 
                if b_id in self.bubbles]
    
    def clear_bubbles_for_image(self, image_src: str):
        """Clear all bubbles for a specific image (when image is removed)"""
        bubbles_to_remove = [
            bubble_id for bubble_id, bubble in self.bubbles.items() 
            if bubble.image_src == image_src
        ]
        
        for bubble_id in bubbles_to_remove:
            self.remove_bubble(bubble_id)
    
    def remove_bubble(self, bubble_id: str):
        """Remove a bubble from tracking"""
        if bubble_id in self.bubbles:
            del self.bubbles[bubble_id]
        
        # Remove from character associations
        for character_id, bubble_ids in self.character_bubbles.items():
            if bubble_id in bubble_ids:
                bubble_ids.remove(bubble_id)
        
        # Remove from panel associations
        for panel_id, bubble_ids in self.panel_bubbles.items():
            if bubble_id in bubble_ids:
                bubble_ids.remove(bubble_id)

# Global instance
bubble_tracker = BubbleTracker()