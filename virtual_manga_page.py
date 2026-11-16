# virtual_manga_page.py
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import math
from enum import Enum

class ReadingOrder(Enum):
    MANGA = "manga"      # Right to left, top to bottom
    WESTERN = "western"  # Left to right, top to bottom
    VERTICAL = "vertical" # Top to bottom, right to left

@dataclass
class VirtualBubble:
    """Represents a virtual speech bubble on the manga page"""
    id: str
    bbox: List[float]  # [x1, y1, x2, y2] normalized coordinates
    original_text: str
    translated_text: str
    language: str
    speaker: Optional[str]
    reading_order_index: int
    panel_id: Optional[str]
    confidence: float
    visual_style: str  # 'normal', 'scream', 'thought', 'whisper'

@dataclass
class VirtualPanel:
    """Represents a virtual panel on the manga page"""
    id: str
    bbox: List[float]  # [x1, y1, x2, y2]
    bubbles: List[VirtualBubble]
    reading_order: int
    dominant_language: str

class VirtualMangaPage:
    """
    Creates a virtual representation of a manga page with proper bubble ordering
    and placement for context-aware processing.
    """
    
    def __init__(self, image_width: int, image_height: int, reading_order: ReadingOrder = ReadingOrder.MANGA):
        self.image_width = image_width
        self.image_height = image_height
        self.reading_order = reading_order
        self.panels: List[VirtualPanel] = []
        self.bubbles: List[VirtualBubble] = []
        self.character_positions: Dict[str, List[Tuple[float, float]]] = {}
        self.context_flow: List[str] = []  # Ordered bubble IDs for reading flow
        
    def add_panel(self, panel_bbox: List[float], panel_id: Optional[str] = None) -> str:
        """Add a panel to the virtual page"""
        if panel_id is None:
            panel_id = f"panel_{len(self.panels) + 1}"
        
        panel = VirtualPanel(
            id=panel_id,
            bbox=panel_bbox,
            bubbles=[],
            reading_order=0,  # Will be set during ordering
            dominant_language="unknown"
        )
        self.panels.append(panel)
        return panel_id
    
    def add_bubble(
        self,
        bubble_bbox: List[float],
        original_text: str,
        translated_text: str = "",
        language: str = "auto",
        speaker: Optional[str] = None,
        panel_id: Optional[str] = None,
        confidence: float = 1.0,
        visual_style: str = "normal"
    ) -> str:
        """Add a bubble to the virtual page with automatic panel assignment"""
        bubble_id = f"bubble_{len(self.bubbles) + 1}"
        
        # Auto-assign to panel if not specified
        if panel_id is None:
            panel_id = self._find_containing_panel(bubble_bbox)
        
        # Create the bubble
        bubble = VirtualBubble(
            id=bubble_id,
            bbox=bubble_bbox,
            original_text=original_text,
            translated_text=translated_text,
            language=language,
            speaker=speaker,
            reading_order_index=0,  # Will be set during ordering
            panel_id=panel_id,
            confidence=confidence,
            visual_style=visual_style
        )
        
        self.bubbles.append(bubble)
        
        # Add to panel if found
        if panel_id:
            for panel in self.panels:
                if panel.id == panel_id:
                    panel.bubbles.append(bubble)
                    break
        
        return bubble_id
    
    def _find_containing_panel(self, bubble_bbox: List[float]) -> Optional[str]:
        """Find which panel contains the bubble (center point check)"""
        bubble_center = self._get_bbox_center(bubble_bbox)
        
        for panel in self.panels:
            if self._is_point_in_bbox(bubble_center, panel.bbox):
                return panel.id
        return None
    
    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of a bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _is_point_in_bbox(self, point: Tuple[float, float], bbox: List[float]) -> bool:
        """Check if a point is inside a bounding box"""
        px, py = point
        x1, y1, x2, y2 = bbox
        return (x1 <= px <= x2) and (y1 <= py <= y2)
    
    def calculate_reading_order(self):
        """Calculate the proper reading order for all elements"""
        # First, order panels
        self._order_panels()
        
        # Then, order bubbles within each panel
        self._order_bubbles_in_panels()
        
        # Build the overall reading flow
        self._build_reading_flow()
    
    def _order_panels(self):
        """Order panels based on reading direction"""
        if self.reading_order == ReadingOrder.MANGA:
            # Manga: right to left, top to bottom
            self.panels.sort(key=lambda p: (-self._get_bbox_center(p.bbox)[0], self._get_bbox_center(p.bbox)[1]))
        elif self.reading_order == ReadingOrder.WESTERN:
            # Western: left to right, top to bottom
            self.panels.sort(key=lambda p: (self._get_bbox_center(p.bbox)[0], self._get_bbox_center(p.bbox)[1]))
        elif self.reading_order == ReadingOrder.VERTICAL:
            # Vertical: top to bottom, right to left
            self.panels.sort(key=lambda p: (-self._get_bbox_center(p.bbox)[0], self._get_bbox_center(p.bbox)[1]))
        
        # Assign reading order indices
        for i, panel in enumerate(self.panels):
            panel.reading_order = i + 1
    
    def _order_bubbles_in_panels(self):
        """Order bubbles within each panel based on reading direction"""
        for panel in self.panels:
            if self.reading_order == ReadingOrder.MANGA:
                # Right to left, top to bottom within panel
                panel.bubbles.sort(key=lambda b: (-self._get_bbox_center(b.bbox)[0], self._get_bbox_center(b.bbox)[1]))
            else:
                # Left to right, top to bottom within panel
                panel.bubbles.sort(key=lambda b: (self._get_bbox_center(b.bbox)[0], self._get_bbox_center(b.bbox)[1]))
            
            # Assign reading order indices within panel
            for i, bubble in enumerate(panel.bubbles):
                bubble.reading_order_index = i + 1
    
    def _build_reading_flow(self):
        """Build the overall reading flow across all panels"""
        self.context_flow = []
        
        for panel in sorted(self.panels, key=lambda p: p.reading_order):
            for bubble in sorted(panel.bubbles, key=lambda b: b.reading_order_index):
                self.context_flow.append(bubble.id)
    
    def get_context_window(self, bubble_id: str, window_size: int = 3) -> List[VirtualBubble]:
        """Get contextual bubbles around a specific bubble for better translation"""
        if bubble_id not in self.context_flow:
            return []
        
        index = self.context_flow.index(bubble_id)
        start = max(0, index - window_size)
        end = min(len(self.context_flow), index + window_size + 1)
        
        context_bubble_ids = self.context_flow[start:end]
        return [b for b in self.bubbles if b.id in context_bubble_ids]
    
    def get_character_dialogue_flow(self, character_id: str) -> List[VirtualBubble]:
        """Get all dialogue from a specific character in reading order"""
        character_bubbles = [b for b in self.bubbles if b.speaker == character_id]
        return sorted(character_bubbles, key=lambda b: self.context_flow.index(b.id))
    
    def create_virtual_overlay(
        self, 
        base_image: Optional[np.ndarray] = None,
        show_reading_order: bool = True,
        show_panels: bool = True,
        show_bubbles: bool = True
    ) -> np.ndarray:
        """Create a visual representation of the virtual page with overlays"""
        if base_image is None:
            # Create a blank canvas
            overlay = np.ones((self.image_height, self.image_width, 3), dtype=np.uint8) * 255
        else:
            overlay = base_image.copy()
        
        # Convert normalized coordinates to pixel coordinates
        def to_pixels(bbox):
            x1, y1, x2, y2 = bbox
            return (
                int(x1 * self.image_width),
                int(y1 * self.image_height),
                int(x2 * self.image_width),
                int(y2 * self.image_height)
            )
        
        # Draw panels
        if show_panels:
            for panel in self.panels:
                x1, y1, x2, y2 = to_pixels(panel.bbox)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Panel number
                cv2.putText(
                    overlay, f"P{panel.reading_order}", (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
        
        # Draw bubbles and reading order
        if show_bubbles:
            for bubble in self.bubbles:
                x1, y1, x2, y2 = to_pixels(bubble.bbox)
                
                # Bubble rectangle
                color = self._get_bubble_color(bubble.visual_style)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                
                # Reading order indicator
                if show_reading_order:
                    # Find panel reading order
                    panel_order = next((p.reading_order for p in self.panels if p.id == bubble.panel_id), 0)
                    order_text = f"{panel_order}.{bubble.reading_order_index}"
                    
                    cv2.putText(
                        overlay, order_text, (x1 + 5, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                    )
                
                # Speaker label
                if bubble.speaker:
                    cv2.putText(
                        overlay, bubble.speaker, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
                    )
        
        return overlay
    
    def _get_bubble_color(self, visual_style: str) -> Tuple[int, int, int]:
        """Get color for different bubble types"""
        colors = {
            'normal': (0, 0, 255),      # Red
            'scream': (0, 165, 255),    # Orange
            'thought': (128, 0, 128),   # Purple
            'whisper': (255, 0, 0),     # Blue
            'narration': (0, 128, 0)    # Green
        }
        return colors.get(visual_style, (0, 0, 255))
    
    def analyze_reading_patterns(self) -> Dict:
        """Analyze reading patterns and bubble distribution"""
        if not self.context_flow:
            self.calculate_reading_order()
        
        # Calculate statistics
        total_bubbles = len(self.bubbles)
        bubbles_per_panel = [len(panel.bubbles) for panel in self.panels]
        
        # Character speaking frequency
        character_freq = {}
        for bubble in self.bubbles:
            if bubble.speaker:
                character_freq[bubble.speaker] = character_freq.get(bubble.speaker, 0) + 1
        
        # Language distribution
        language_dist = {}
        for bubble in self.bubbles:
            language_dist[bubble.language] = language_dist.get(bubble.language, 0) + 1
        
        return {
            "total_panels": len(self.panels),
            "total_bubbles": total_bubbles,
            "bubbles_per_panel": {
                "max": max(bubbles_per_panel) if bubbles_per_panel else 0,
                "min": min(bubbles_per_panel) if bubbles_per_panel else 0,
                "average": sum(bubbles_per_panel) / len(bubbles_per_panel) if bubbles_per_panel else 0
            },
            "character_frequency": character_freq,
            "language_distribution": language_dist,
            "reading_order": self.context_flow,
            "reading_direction": self.reading_order.value
        }
    
    def export_to_json(self) -> str:
        """Export virtual page data to JSON"""
        data = {
            "image_dimensions": {
                "width": self.image_width,
                "height": self.image_height
            },
            "reading_order": self.reading_order.value,
            "panels": [
                {
                    "id": panel.id,
                    "bbox": panel.bbox,
                    "reading_order": panel.reading_order,
                    "bubble_count": len(panel.bubbles)
                }
                for panel in self.panels
            ],
            "bubbles": [
                {
                    "id": bubble.id,
                    "bbox": bubble.bbox,
                    "original_text": bubble.original_text,
                    "translated_text": bubble.translated_text,
                    "language": bubble.language,
                    "speaker": bubble.speaker,
                    "reading_order_index": bubble.reading_order_index,
                    "panel_id": bubble.panel_id,
                    "confidence": bubble.confidence,
                    "visual_style": bubble.visual_style
                }
                for bubble in self.bubbles
            ],
            "reading_flow": self.context_flow
        }
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    @classmethod
    def import_from_json(cls, json_data: str) -> 'VirtualMangaPage':
        """Import virtual page data from JSON"""
        data = json.loads(json_data)
        
        page = cls(
            image_width=data["image_dimensions"]["width"],
            image_height=data["image_dimensions"]["height"],
            reading_order=ReadingOrder(data["reading_order"])
        )
        
        # Recreate panels
        for panel_data in data["panels"]:
            page.panels.append(VirtualPanel(
                id=panel_data["id"],
                bbox=panel_data["bbox"],
                bubbles=[],
                reading_order=panel_data["reading_order"],
                dominant_language="unknown"
            ))
        
        # Recreate bubbles
        for bubble_data in data["bubbles"]:
            bubble = VirtualBubble(
                id=bubble_data["id"],
                bbox=bubble_data["bbox"],
                original_text=bubble_data["original_text"],
                translated_text=bubble_data["translated_text"],
                language=bubble_data["language"],
                speaker=bubble_data["speaker"],
                reading_order_index=bubble_data["reading_order_index"],
                panel_id=bubble_data["panel_id"],
                confidence=bubble_data["confidence"],
                visual_style=bubble_data["visual_style"]
            )
            page.bubbles.append(bubble)
            
            # Add to panel
            for panel in page.panels:
                if panel.id == bubble.panel_id:
                    panel.bubbles.append(bubble)
                    break
        
        page.context_flow = data["reading_flow"]
        return page

# Integration with existing translator
class ContextAwareMangaProcessor:
    """Enhanced processor that uses virtual pages for context-aware translation"""
    
    def __init__(self):
        self.virtual_pages: Dict[str, VirtualMangaPage] = {}  # image_path -> VirtualMangaPage
        self.current_context: Dict[str, any] = {}
    
    def create_virtual_page_from_detection(
        self,
        image_path: str,
        detection_results: Dict,
        reading_order: ReadingOrder = ReadingOrder.MANGA
    ) -> VirtualMangaPage:
        """Create virtual page from MAGI detection results"""
        # Extract image dimensions (you might need to load the image)
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Create virtual page
        virtual_page = VirtualMangaPage(width, height, reading_order)
        
        # Add panels
        for i, panel in enumerate(detection_results.get("panels", [])):
            panel_id = virtual_page.add_panel(panel["bbox"], f"panel_{i+1}")
        
        # Add bubbles with text
        for i, text in enumerate(detection_results.get("texts", [])):
            # Find associated character if any
            speaker = None
            for association in detection_results.get("associations", []):
                if association["text_index"] == i:
                    char_index = association["char_index"]
                    speaker = detection_results.get("speakerMap", {}).get(str(char_index))
                    break
            
            bubble_id = virtual_page.add_bubble(
                bubble_bbox=text["bbox"],
                original_text=detection_results.get("ocrResults", [""] * (i+1))[i],
                language=detection_results.get("detected_language", "auto"),
                speaker=speaker,
                confidence=text.get("confidence", 1.0)
            )
        
        # Calculate reading order
        virtual_page.calculate_reading_order()
        
        # Store the virtual page
        self.virtual_pages[image_path] = virtual_page
        self.current_context["current_page"] = image_path
        
        return virtual_page
    
    def get_context_for_translation(self, bubble_id: str, window_size: int = 2) -> List[Dict]:
        """Get contextual information for better translation"""
        current_page = self.virtual_pages.get(self.current_context.get("current_page"))
        if not current_page:
            return []
        
        context_bubbles = current_page.get_context_window(bubble_id, window_size)
        
        return [
            {
                "text": bubble.original_text,
                "translated_text": bubble.translated_text,
                "speaker": bubble.speaker,
                "position": bubble.reading_order_index,
                "panel": bubble.panel_id
            }
            for bubble in context_bubbles
        ]
    
    def update_bubble_translation(self, bubble_id: str, translated_text: str, confidence: float = 1.0):
        """Update a bubble's translation in the virtual page"""
        current_page = self.virtual_pages.get(self.current_context.get("current_page"))
        if not current_page:
            return
        
        for bubble in current_page.bubbles:
            if bubble.id == bubble_id:
                bubble.translated_text = translated_text
                bubble.confidence = confidence
                break
    
    def get_reading_flow_analysis(self, image_path: str) -> Dict:
        """Get analysis of reading flow and patterns"""
        virtual_page = self.virtual_pages.get(image_path)
        if not virtual_page:
            return {}
        
        return virtual_page.analyze_reading_patterns()